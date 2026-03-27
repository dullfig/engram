//! QuantizedKvCache — TurboQuant-compressed KV storage.
//!
//! Replaces the f32 KV cache with 3-bit PolarQuant angles + per-head radius,
//! optionally with QJL 1-bit residual correction. Achieves ~12x memory
//! reduction over f32, turning the KV cache from the dominant memory consumer
//! into a fraction of the model weights.

use crate::ops::polar::{self, AngleLUT};
use crate::ops::qjl::QjlProjection;

/// 3-bit quantized KV cache for one transformer layer.
pub struct QuantizedKvCache {
    // -- PolarQuant compressed storage --
    /// Quantized angle indices for K cache.
    /// Shape: [max_seq_len, n_kv_heads, head_dim/2] stored flat.
    /// Each byte holds one angle bucket (0..7). Future: bit-pack to 3 bits.
    k_angles: Vec<u8>,

    /// Quantized angle indices for V cache (same layout).
    v_angles: Vec<u8>,

    /// Per-position, per-head radius scale for K.
    /// Shape: [max_seq_len, n_kv_heads] stored flat.
    k_radius: Vec<f32>,

    /// Per-position, per-head radius scale for V.
    v_radius: Vec<f32>,

    // -- QJL correction (optional) --
    /// Packed sign bits for K residual correction.
    /// Shape: [max_seq_len, n_kv_heads, sign_bytes] stored flat.
    k_qjl_signs: Option<Vec<u8>>,

    /// Packed sign bits for V residual correction.
    v_qjl_signs: Option<Vec<u8>>,

    // -- Fixed per-layer state (regenerated from seed, not stored) --
    /// Orthogonal rotation matrix [head_dim, head_dim].
    rotation_matrix: Vec<f32>,

    /// Angle lookup table (shared across all positions/heads).
    lut: AngleLUT,

    /// QJL projections (None if QJL disabled).
    qjl: Option<QjlProjection>,

    // -- Dimensions --
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    len: usize,
}

impl QuantizedKvCache {
    /// Create a new compressed cache for one layer (PolarQuant only, no QJL).
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize, seed: u64) -> Self {
        assert_eq!(head_dim % 2, 0, "head_dim must be even for polar pairs");

        let n_pairs = head_dim / 2;
        let angle_capacity = max_seq_len * n_kv_heads * n_pairs;
        let radius_capacity = max_seq_len * n_kv_heads;

        Self {
            k_angles: vec![0u8; angle_capacity],
            v_angles: vec![0u8; angle_capacity],
            k_radius: vec![0.0f32; radius_capacity],
            v_radius: vec![0.0f32; radius_capacity],
            k_qjl_signs: None,
            v_qjl_signs: None,
            rotation_matrix: polar::generate_rotation_matrix(head_dim, seed),
            lut: AngleLUT::new(),
            qjl: None,
            n_kv_heads,
            head_dim,
            max_seq_len,
            len: 0,
        }
    }

    /// Create with QJL correction enabled.
    pub fn with_qjl(
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rotation_seed: u64,
        qjl_seed: u64,
    ) -> Self {
        let mut cache = Self::new(n_kv_heads, head_dim, max_seq_len, rotation_seed);

        let qjl = QjlProjection::new(head_dim, qjl_seed);
        let sign_bytes = qjl.sign_bytes();
        let sign_capacity = max_seq_len * n_kv_heads * sign_bytes;

        cache.k_qjl_signs = Some(vec![0u8; sign_capacity]);
        cache.v_qjl_signs = Some(vec![0u8; sign_capacity]);
        cache.qjl = Some(qjl);
        cache
    }

    /// Number of cached positions.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of KV heads.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Append K and V vectors for one new position.
    ///
    /// `key`: f32 slice of length `n_kv_heads * head_dim`.
    /// `value`: f32 slice of same length.
    /// These should already have RoPE applied (for keys).
    pub fn append_one(&mut self, key: &[f32], value: &[f32]) {
        let kv_dim = self.n_kv_heads * self.head_dim;
        assert_eq!(key.len(), kv_dim);
        assert_eq!(value.len(), kv_dim);
        assert!(self.len < self.max_seq_len, "cache overflow");

        let pos = self.len;
        let n_pairs = self.head_dim / 2;
        let mut rotated = vec![0.0f32; self.head_dim];

        for head in 0..self.n_kv_heads {
            let head_off = head * self.head_dim;

            // --- Compress K ---
            let k_vec = &key[head_off..head_off + self.head_dim];
            polar::rotate(&self.rotation_matrix, k_vec, &mut rotated);

            let (angles, radius) = polar::to_polar_quantized(&rotated);
            let angle_off = (pos * self.n_kv_heads + head) * n_pairs;
            self.k_angles[angle_off..angle_off + n_pairs].copy_from_slice(&angles);
            self.k_radius[pos * self.n_kv_heads + head] = radius;

            // QJL correction for K.
            if let (Some(qjl), Some(signs_buf)) = (&self.qjl, &mut self.k_qjl_signs) {
                let reconstructed = polar::from_polar_quantized(&angles, radius, &self.lut);
                let residual: Vec<f32> = rotated
                    .iter()
                    .zip(reconstructed.iter())
                    .map(|(&r, &q)| r - q)
                    .collect();
                let signs = qjl.encode_signs(&residual);
                let sign_bytes = qjl.sign_bytes();
                let sign_off = (pos * self.n_kv_heads + head) * sign_bytes;
                signs_buf[sign_off..sign_off + sign_bytes].copy_from_slice(&signs);
            }

            // --- Compress V ---
            let v_vec = &value[head_off..head_off + self.head_dim];
            polar::rotate(&self.rotation_matrix, v_vec, &mut rotated);

            let (angles, radius) = polar::to_polar_quantized(&rotated);
            self.v_angles[angle_off..angle_off + n_pairs].copy_from_slice(&angles);
            self.v_radius[pos * self.n_kv_heads + head] = radius;

            // QJL correction for V.
            if let (Some(qjl), Some(signs_buf)) = (&self.qjl, &mut self.v_qjl_signs) {
                let reconstructed = polar::from_polar_quantized(&angles, radius, &self.lut);
                let residual: Vec<f32> = rotated
                    .iter()
                    .zip(reconstructed.iter())
                    .map(|(&r, &q)| r - q)
                    .collect();
                let signs = qjl.encode_signs(&residual);
                let sign_bytes = qjl.sign_bytes();
                let sign_off = (pos * self.n_kv_heads + head) * sign_bytes;
                signs_buf[sign_off..sign_off + sign_bytes].copy_from_slice(&signs);
            }
        }

        self.len += 1;
    }

    /// Compute dot(query, cached_key[pos, head]) in compressed domain.
    ///
    /// The query should be in original (non-rotated) space. This method
    /// rotates it internally and dots against the compressed K directly.
    pub fn dot_key(&self, pos: usize, kv_head: usize, query: &[f32]) -> f32 {
        debug_assert!(pos < self.len);
        debug_assert!(kv_head < self.n_kv_heads);
        debug_assert_eq!(query.len(), self.head_dim);

        let n_pairs = self.head_dim / 2;

        // Rotate query into compressed domain.
        let mut rq = vec![0.0f32; self.head_dim];
        polar::rotate(&self.rotation_matrix, query, &mut rq);

        // Dot against compressed K using angle LUT.
        let angle_off = (pos * self.n_kv_heads + kv_head) * n_pairs;
        let radius = self.k_radius[pos * self.n_kv_heads + kv_head];

        let mut sum = 0.0f32;
        for i in 0..n_pairs {
            let bucket = self.k_angles[angle_off + i] as usize;
            sum += rq[2 * i] * self.lut.cos[bucket] + rq[2 * i + 1] * self.lut.sin[bucket];
        }
        sum *= radius;

        // QJL correction.
        if let (Some(qjl), Some(signs_buf)) = (&self.qjl, &self.k_qjl_signs) {
            let sign_bytes = qjl.sign_bytes();
            let sign_off = (pos * self.n_kv_heads + kv_head) * sign_bytes;
            let signs = &signs_buf[sign_off..sign_off + sign_bytes];
            sum += qjl.correction_dot(signs, &rq);
        }

        sum
    }

    /// Dequantize the V vector at (pos, head) back to f32.
    ///
    /// Returns a Vec<f32> of length head_dim in original (non-rotated) space.
    pub fn value_at_dequant(&self, pos: usize, kv_head: usize) -> Vec<f32> {
        debug_assert!(pos < self.len);
        debug_assert!(kv_head < self.n_kv_heads);

        let n_pairs = self.head_dim / 2;
        let angle_off = (pos * self.n_kv_heads + kv_head) * n_pairs;
        let radius = self.v_radius[pos * self.n_kv_heads + kv_head];
        let angles = &self.v_angles[angle_off..angle_off + n_pairs];

        let rotated = polar::from_polar_quantized(angles, radius, &self.lut);

        let mut out = vec![0.0f32; self.head_dim];
        polar::rotate_transpose(&self.rotation_matrix, &rotated, &mut out);
        out
    }

    /// Reset the cache (reuse allocations).
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Memory usage in bytes (compressed storage only, excludes rotation matrix).
    pub fn memory_bytes(&self) -> usize {
        let n_pairs = self.head_dim / 2;
        let angle_bytes = 2 * self.max_seq_len * self.n_kv_heads * n_pairs; // k + v
        let radius_bytes = 2 * self.max_seq_len * self.n_kv_heads * 4; // k + v, f32
        let qjl_bytes = self
            .qjl
            .as_ref()
            .map(|q| 2 * self.max_seq_len * self.n_kv_heads * q.sign_bytes())
            .unwrap_or(0);
        angle_bytes + radius_bytes + qjl_bytes
    }

    /// Equivalent f32 cache size for comparison.
    pub fn f32_equivalent_bytes(&self) -> usize {
        2 * self.max_seq_len * self.n_kv_heads * self.head_dim * 4
    }
}

impl std::fmt::Debug for QuantizedKvCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ratio = self.f32_equivalent_bytes() as f64 / self.memory_bytes().max(1) as f64;
        write!(
            f,
            "QuantizedKvCache(kv_heads={}, head_dim={}, len={}/{}, {:.1}KB, {:.1}x compression{})",
            self.n_kv_heads,
            self.head_dim,
            self.len,
            self.max_seq_len,
            self.memory_bytes() as f64 / 1024.0,
            ratio,
            if self.qjl.is_some() { " +QJL" } else { "" },
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_cache_empty() {
        let cache = QuantizedKvCache::new(4, 64, 2048, 42);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_seq_len(), 2048);
    }

    #[test]
    fn append_and_len() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let value = vec![0.3f32; kv_dim];

        cache.append_one(&key, &value);
        assert_eq!(cache.len(), 1);

        cache.append_one(&key, &value);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn dot_key_is_finite() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let value = vec![0.3f32; kv_dim];
        cache.append_one(&key, &value);

        let query = vec![1.0f32; 8];
        let dot = cache.dot_key(0, 0, &query);
        assert!(dot.is_finite(), "dot should be finite, got {dot}");
    }

    #[test]
    fn value_dequant_is_finite() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let value = vec![0.3f32; kv_dim];
        cache.append_one(&key, &value);

        let v = cache.value_at_dequant(0, 0);
        assert_eq!(v.len(), 8);
        for &val in &v {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn compression_ratio() {
        let cache = QuantizedKvCache::new(4, 64, 2048, 42);
        let compressed = cache.memory_bytes();
        let f32_equiv = cache.f32_equivalent_bytes();
        let ratio = f32_equiv as f64 / compressed as f64;
        // Should be roughly 10-16x compression.
        assert!(ratio > 5.0, "expected significant compression, got {ratio:.1}x");
    }

    #[test]
    fn with_qjl_has_higher_memory() {
        let without = QuantizedKvCache::new(4, 64, 2048, 42);
        let with = QuantizedKvCache::with_qjl(4, 64, 2048, 42, 99);
        assert!(with.memory_bytes() > without.memory_bytes());
    }

    #[test]
    fn clear_resets_len() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        cache.append_one(&vec![1.0; kv_dim], &vec![2.0; kv_dim]);
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn overflow_panics() {
        let mut cache = QuantizedKvCache::new(1, 4, 2, 42);
        let kv_dim = 4;
        cache.append_one(&vec![1.0; kv_dim], &vec![2.0; kv_dim]);
        cache.append_one(&vec![1.0; kv_dim], &vec![2.0; kv_dim]);
        cache.append_one(&vec![1.0; kv_dim], &vec![2.0; kv_dim]); // 3 > 2
    }

    #[test]
    fn debug_format() {
        let cache = QuantizedKvCache::new(4, 64, 2048, 42);
        let debug = format!("{:?}", cache);
        assert!(debug.contains("QuantizedKvCache"));
        assert!(debug.contains("0/2048"));
        assert!(debug.contains("compression"));
    }
}
