//! GPU-accelerated retrieval via wgpu compute shaders.
//!
//! Moves the `dot_key()` inner loop to the GPU. Each workgroup thread
//! handles one cached position: reads 3-bit angles + radius from a
//! storage buffer, does the LUT dot product against the pre-rotated
//! query, writes the score.
//!
//! The CPU pre-rotates the query once (matrix-vector multiply), uploads
//! it, then dispatches N threads where N = cached_len. The GPU returns
//! a score per position. Softmax + top-k stays on CPU.

use crate::cache::quantized::QuantizedKvCache;
use crate::ops::polar::{self, AngleLUT};

/// WGSL compute shader for compressed-domain dot product.
///
/// Each thread computes dot(rotated_query, compressed_key[pos]) for one position.
/// Reads: angles (u32 packed), radius (f32), LUT cos/sin (8 each), rotated query.
/// Writes: one f32 score per position.
const SHADER_SOURCE: &str = r#"
// Angle LUT packed as vec4s for uniform alignment.
// cos_01 = vec4(cos0, cos1, cos2, cos3), cos_45 = vec4(cos4, cos5, cos6, cos7)
// sin_01 = vec4(sin0, sin1, sin2, sin3), sin_45 = vec4(sin4, sin5, sin6, sin7)
struct AngleLUT {
    cos_lo: vec4<f32>,  // cos[0..4]
    cos_hi: vec4<f32>,  // cos[4..8]
    sin_lo: vec4<f32>,  // sin[0..4]
    sin_hi: vec4<f32>,  // sin[4..8]
};

fn lut_cos(lut: AngleLUT, idx: u32) -> f32 {
    if (idx < 4u) { return lut.cos_lo[idx]; }
    return lut.cos_hi[idx - 4u];
}

fn lut_sin(lut: AngleLUT, idx: u32) -> f32 {
    if (idx < 4u) { return lut.sin_lo[idx]; }
    return lut.sin_hi[idx - 4u];
}

@group(0) @binding(0) var<storage, read> angles: array<u32>;       // packed angle bytes
@group(0) @binding(1) var<storage, read> radius: array<f32>;       // per-position radius
@group(0) @binding(2) var<uniform> lut: AngleLUT;                  // angle lookup table
@group(0) @binding(3) var<storage, read> rotated_query: array<f32>; // pre-rotated query
@group(0) @binding(4) var<storage, read_write> scores: array<f32>; // output scores
@group(0) @binding(5) var<uniform> params: vec4<u32>;              // [n_pairs, n_positions, kv_head, n_kv_heads]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pos = gid.x;
    let n_pairs = params.x;
    let n_positions = params.y;
    let kv_head = params.z;
    let n_kv_heads = params.w;

    if (pos >= n_positions) {
        return;
    }

    let angle_base = (pos * n_kv_heads + kv_head) * n_pairs;
    let r = radius[pos * n_kv_heads + kv_head];

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < n_pairs; i = i + 1u) {
        let word_idx = (angle_base + i) / 4u;
        let byte_idx = (angle_base + i) % 4u;
        let word = angles[word_idx];
        let bucket = (word >> (byte_idx * 8u)) & 0x7u;  // 3-bit: 0..7

        let cos_val = lut_cos(lut, bucket);
        let sin_val = lut_sin(lut, bucket);

        sum += rotated_query[2u * i] * cos_val + rotated_query[2u * i + 1u] * sin_val;
    }

    scores[pos] = sum * r;
}
"#;

/// GPU retrieval context — holds device, queue, pipeline, and reusable buffers.
#[cfg(feature = "gpu")]
pub struct GpuRetriever {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// GPU adapter name for diagnostics.
    pub adapter_name: String,
}

#[cfg(feature = "gpu")]
impl GpuRetriever {
    /// Initialize the GPU retrieval pipeline.
    pub fn new() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))?;

        let adapter_name = adapter.get_info().name.clone();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("engram-retrieval"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            },
            None,
        ))
        .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dot_key_shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dot_key_layout"),
            entries: &[
                // angles (storage, read)
                bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                // radius (storage, read)
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                // lut (uniform)
                bgl_entry(2, wgpu::BufferBindingType::Uniform),
                // rotated_query (storage, read)
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                // scores (storage, read_write)
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                // params (uniform)
                bgl_entry(5, wgpu::BufferBindingType::Uniform),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dot_key_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dot_key_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            adapter_name,
        })
    }

    /// Score all cached positions against a query for one KV head.
    ///
    /// Returns a score per position (length = cache.len()).
    /// The query should be in original (non-rotated) space — rotation
    /// is done CPU-side once before upload.
    pub fn score_positions(
        &self,
        cache: &QuantizedKvCache,
        kv_head: usize,
        query: &[f32],
    ) -> Vec<f32> {
        let head_dim = cache.head_dim();
        let n_pairs = head_dim / 2;
        let n_positions = cache.len();
        let n_kv_heads = cache.n_kv_heads();

        if n_positions == 0 {
            return vec![];
        }

        // Pre-rotate query on CPU (one matrix-vector multiply).
        let rotation_matrix = polar::generate_rotation_matrix(head_dim, 42); // same seed as cache
        let mut rq = vec![0.0f32; head_dim];
        polar::rotate(&rotation_matrix, query, &mut rq);

        // Build angle LUT data: [cos_lo(4), cos_hi(4), sin_lo(4), sin_hi(4)]
        let lut = AngleLUT::new();
        let mut lut_data = [0.0f32; 16];
        lut_data[0..4].copy_from_slice(&lut.cos[0..4]);
        lut_data[4..8].copy_from_slice(&lut.cos[4..8]);
        lut_data[8..12].copy_from_slice(&lut.sin[0..4]);
        lut_data[12..16].copy_from_slice(&lut.sin[4..8]);

        // Upload angles as raw bytes (packed into u32 for GPU access).
        let angle_bytes = cache.k_angles_slice();
        // Pad to 4-byte alignment.
        let mut angle_padded = angle_bytes.to_vec();
        while angle_padded.len() % 4 != 0 {
            angle_padded.push(0);
        }

        let angles_buf = self.create_buffer_init("angles", bytemuck_cast_slice(&angle_padded), true);
        let radius_buf = self.create_buffer_init("radius", bytemuck_cast_slice(cache.k_radius_slice()), true);
        let lut_buf = self.create_buffer_init("lut", bytemuck_cast_slice(&lut_data), false);
        let query_buf = self.create_buffer_init("query", bytemuck_cast_slice(&rq), true);

        let scores_size = (n_positions * 4) as u64;
        let scores_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size: scores_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params: [u32; 4] = [n_pairs as u32, n_positions as u32, kv_head as u32, n_kv_heads as u32];
        let params_buf = self.create_buffer_init("params", bytemuck_cast_slice(&params), false);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dot_key_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: angles_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: radius_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lut_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: query_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: scores_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: params_buf.as_entire_binding() },
            ],
        });

        // Dispatch.
        let workgroups = (n_positions as u32 + 255) / 256;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Readback buffer.
        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: scores_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&scores_buf, 0, &readback_buf, 0, scores_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read results.
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let scores: Vec<f32> = bytemuck_cast_slice_from(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        scores
    }

    fn create_buffer_init(&self, label: &str, data: &[u8], storage: bool) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let usage = if storage {
            wgpu::BufferUsages::STORAGE
        } else {
            wgpu::BufferUsages::UNIFORM
        };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        })
    }
}

/// Full retrieval using GPU for the dot_key scan.
///
/// Replaces `retrieve::retrieve()` for the GPU path. Softmax + aggregation
/// still on CPU (they're cheap — O(cached_len) scalar ops).
#[cfg(feature = "gpu")]
pub fn gpu_retrieve(
    gpu: &GpuRetriever,
    queries: &[f32],
    n_query_tokens: usize,
    n_heads: usize,
    cache: &QuantizedKvCache,
) -> super::AttentionScores {
    let head_dim = cache.head_dim();
    let n_kv_heads = cache.n_kv_heads();
    let cached_len = cache.len();
    let heads_per_group = n_heads / n_kv_heads;

    let mut position_scores = vec![0.0f32; cached_len];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for qt in 0..n_query_tokens {
        for qh in 0..n_heads {
            let kv_h = qh / heads_per_group;
            let q_off = (qt * n_heads + qh) * head_dim;
            let q_vec = &queries[q_off..q_off + head_dim];

            // GPU scores all positions for this (query_token, head) pair.
            let mut scores = gpu.score_positions(cache, kv_h, q_vec);

            // Scale.
            for s in &mut scores {
                *s *= scale;
            }

            // Softmax (CPU — trivial).
            softmax_inplace(&mut scores);

            // Accumulate.
            for (pos, &weight) in scores.iter().enumerate() {
                position_scores[pos] += weight;
            }
        }
    }

    // Normalize.
    let normalizer = (n_query_tokens * n_heads) as f32;
    for s in &mut position_scores {
        *s /= normalizer;
    }

    super::AttentionScores::from_scores(position_scores, cached_len)
}

#[cfg(feature = "gpu")]
fn softmax_inplace(values: &mut [f32]) {
    if values.is_empty() { return; }
    if values.len() == 1 { values[0] = 1.0; return; }
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in values.iter_mut() {
        *v *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Helpers: safe byte casting without the bytemuck crate
// ---------------------------------------------------------------------------

fn bytemuck_cast_slice<T>(data: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    }
}

fn bytemuck_cast_slice_from(data: &[u8]) -> &[f32] {
    assert_eq!(data.len() % 4, 0);
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const f32,
            data.len() / 4,
        )
    }
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
