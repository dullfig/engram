//! MiniProjection — minimal Q/K projection from a GGUF model.
//!
//! Loads only what engram needs for retrieval:
//! - Tokenizer (text → token IDs)
//! - Token embedding table (IDs → vectors)
//! - Input normalization (RMSNorm)
//! - Q and K projection weights (from a single transformer layer)
//! - RoPE positional encoding
//!
//! No FFN, no V projection, no output head. ~5% of the full model.

use ternary_rs::gguf::{GgufFile, GgufError};
use ternary_rs::layers::rope::{RoPE, RoPELayout};
use ternary_rs::layers::rmsnorm::RmsNorm;
use ternary_rs::tokenizer::Tokenizer;

/// Minimal projection model for engram retrieval.
///
/// Produces Q/K vectors from text, suitable for attention-based
/// retrieval against a compressed KV cache.
pub struct MiniProjection {
    /// Tokenizer from the GGUF.
    pub tokenizer: Tokenizer,
    /// Token embedding table: `[vocab_size, hidden_dim]` row-major.
    embeddings: Vec<f32>,
    /// RMSNorm weights for input normalization.
    input_norm: RmsNorm,
    /// Q projection weights: `[n_heads * head_dim, hidden_dim]` row-major.
    q_weights: Vec<f32>,
    /// K projection weights: `[n_kv_heads * head_dim, hidden_dim]` row-major.
    k_weights: Vec<f32>,
    /// Optional Q bias.
    q_bias: Option<Vec<f32>>,
    /// Optional K bias.
    k_bias: Option<Vec<f32>>,
    /// RoPE encoder.
    rope: RoPE,
    /// Model dimensions.
    pub config: ProjectionConfig,
}

/// Dimensions extracted from the model.
#[derive(Debug, Clone)]
pub struct ProjectionConfig {
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

impl MiniProjection {
    /// Load from a GGUF file, extracting only layer 0's Q/K projections.
    pub fn from_gguf(path: &str) -> Result<Self, GgufError> {
        Self::from_gguf_layer(path, 0)
    }

    /// Load from a GGUF file, extracting Q/K from a specific layer.
    pub fn from_gguf_layer(path: &str, layer: usize) -> Result<Self, GgufError> {
        let gguf = GgufFile::open(path)?;
        let model_config = gguf.model_config()?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;

        let hidden_dim = model_config.embedding_dim as usize;
        let n_heads = model_config.n_heads as usize;
        let n_kv_heads = model_config.n_kv_heads as usize;
        let head_dim = hidden_dim / n_heads;
        let vocab_size = model_config.vocab_size as usize;

        // Load token embeddings.
        let embed_tensor = gguf.load_float("token_embd.weight")?;
        let embeddings = embed_tensor.data().to_vec();

        // Load input norm for the target layer.
        let norm_tensor = gguf.load_float(&format!("blk.{layer}.attn_norm.weight"))?;
        let input_norm = RmsNorm::new(norm_tensor.data().to_vec(), model_config.rms_norm_eps);

        // Load Q projection (may be ternary or float — extract as float).
        let q_weights = load_projection_weights(&gguf, &format!("blk.{layer}.attn_q.weight"))?;
        let k_weights = load_projection_weights(&gguf, &format!("blk.{layer}.attn_k.weight"))?;

        // Optional biases (Qwen has them, LLaMA doesn't).
        let q_bias = gguf
            .load_float(&format!("blk.{layer}.attn_q.bias"))
            .ok()
            .map(|t| t.data().to_vec());
        let k_bias = gguf
            .load_float(&format!("blk.{layer}.attn_k.bias"))
            .ok()
            .map(|t| t.data().to_vec());

        // RoPE.
        let rope_base = if model_config.rope_theta > 0.0 {
            model_config.rope_theta
        } else {
            10000.0
        };
        let rope_layout = if model_config.rope_type == 2 {
            RoPELayout::Halved
        } else {
            RoPELayout::Interleaved
        };
        let rope = RoPE::with_layout(head_dim, rope_base, rope_layout);

        let config = ProjectionConfig {
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
        };

        Ok(Self {
            tokenizer,
            embeddings,
            input_norm,
            q_weights,
            k_weights,
            q_bias,
            k_bias,
            rope,
            config,
        })
    }

    /// Tokenize text and produce Q vectors for retrieval.
    ///
    /// Returns flat f32 of shape `[n_tokens, n_heads, head_dim]` with RoPE applied.
    /// `seq_offset` is the position in the sequence for RoPE (usually cache.len()).
    pub fn encode_query(&self, text: &str, seq_offset: usize) -> Vec<f32> {
        let tokens = self.tokenizer.encode(text, false);
        let c = &self.config;
        let mut all_q = Vec::with_capacity(tokens.len() * c.n_heads * c.head_dim);

        for (t, &token_id) in tokens.iter().enumerate() {
            let pos = seq_offset + t;

            // Embed.
            let embed = self.get_embedding(token_id as usize);

            // Normalize.
            let normed = self.input_norm.forward(&embed);

            // Q projection.
            let mut q = matmul(&self.q_weights, &normed, c.n_heads * c.head_dim);
            if let Some(bias) = &self.q_bias {
                for (v, b) in q.iter_mut().zip(bias.iter()) {
                    *v += b;
                }
            }

            // Apply RoPE per head.
            let q_roped = self.rope.forward_heads(&q, c.n_heads, pos);
            all_q.extend_from_slice(&q_roped);
        }

        all_q
    }

    /// Tokenize text and produce K vectors for caching.
    ///
    /// Returns flat f32 of shape `[n_tokens, n_kv_heads, head_dim]` with RoPE applied.
    pub fn encode_key(&self, text: &str, seq_offset: usize) -> Vec<f32> {
        let tokens = self.tokenizer.encode(text, false);
        let c = &self.config;
        let mut all_k = Vec::with_capacity(tokens.len() * c.n_kv_heads * c.head_dim);

        for (t, &token_id) in tokens.iter().enumerate() {
            let pos = seq_offset + t;

            let embed = self.get_embedding(token_id as usize);
            let normed = self.input_norm.forward(&embed);

            let mut k = matmul(&self.k_weights, &normed, c.n_kv_heads * c.head_dim);
            if let Some(bias) = &self.k_bias {
                for (v, b) in k.iter_mut().zip(bias.iter()) {
                    *v += b;
                }
            }

            let k_roped = self.rope.forward_heads(&k, c.n_kv_heads, pos);
            all_k.extend_from_slice(&k_roped);
        }

        all_k
    }

    /// Tokenize text and produce both K and V (raw embedding) for cache append.
    ///
    /// Returns (keys, values) where:
    /// - keys: `[n_tokens, n_kv_heads * head_dim]` per token (RoPE applied)
    /// - values: same shape but without RoPE (value is just the projection)
    ///
    /// For engram's K-only retrieval, we store K in the compressed cache
    /// and keep raw text in the PositionMap for V.
    pub fn encode_kv(&self, text: &str, seq_offset: usize) -> (Vec<Vec<f32>>, usize) {
        let tokens = self.tokenizer.encode(text, false);
        let c = &self.config;
        let n_tokens = tokens.len();
        let kv_dim = c.n_kv_heads * c.head_dim;

        let mut keys = Vec::with_capacity(n_tokens);

        for (t, &token_id) in tokens.iter().enumerate() {
            let pos = seq_offset + t;

            let embed = self.get_embedding(token_id as usize);
            let normed = self.input_norm.forward(&embed);

            let mut k = matmul(&self.k_weights, &normed, kv_dim);
            if let Some(bias) = &self.k_bias {
                for (v, b) in k.iter_mut().zip(bias.iter()) {
                    *v += b;
                }
            }
            let k_roped = self.rope.forward_heads(&k, c.n_kv_heads, pos);
            keys.push(k_roped);
        }

        (keys, n_tokens)
    }

    /// Number of tokens a text would produce.
    pub fn token_count(&self, text: &str) -> usize {
        self.tokenizer.encode(text, false).len()
    }

    fn get_embedding(&self, token_id: usize) -> Vec<f32> {
        let d = self.config.hidden_dim;
        let start = token_id * d;
        if start + d <= self.embeddings.len() {
            self.embeddings[start..start + d].to_vec()
        } else {
            vec![0.0; d] // OOV fallback
        }
    }
}

/// Simple matrix-vector multiply: `out = W @ x`.
/// W is `[out_dim, in_dim]` row-major.
fn matmul(weights: &[f32], input: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    let mut out = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let row_off = row * in_dim;
        let mut sum = 0.0f32;
        for col in 0..in_dim {
            sum += weights[row_off + col] * input[col];
        }
        out[row] = sum;
    }
    out
}

/// Load projection weights as f32, handling both float and quantized tensors.
fn load_projection_weights(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, GgufError> {
    // Try loading as float first, then dequantize if quantized.
    match gguf.load_float(name) {
        Ok(tensor) => Ok(tensor.data().to_vec()),
        Err(_) => {
            // For ternary/quantized weights, we need to dequantize.
            // ternary-rs handles this via the loader — but for a minimal
            // projection we can load the raw tensor and dequantize.
            // For now, return an error — we'll handle quantized models next.
            Err(GgufError::MissingMetadata(format!("tensor not found: {name}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration test: requires a GGUF model file.
    // Run with: cargo test -- --ignored projection
    #[test]
    #[ignore]
    fn load_qwen_projection() {
        let proj = MiniProjection::from_gguf(
            "../code-llm/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
        ).expect("failed to load model");

        let c = &proj.config;
        println!("Loaded: hidden={}, heads={}, kv_heads={}, head_dim={}, vocab={}",
            c.hidden_dim, c.n_heads, c.n_kv_heads, c.head_dim, c.vocab_size);

        // Encode a query.
        let q = proj.encode_query("What is a buffer?", 0);
        let n_tokens = proj.token_count("What is a buffer?");
        assert_eq!(q.len(), n_tokens * c.n_heads * c.head_dim);
        println!("Query: {} tokens, {} floats", n_tokens, q.len());

        // Encode keys.
        let (keys, n_key_tokens) = proj.encode_kv("Buffers spawn isolated child pipelines.", 0);
        assert_eq!(keys.len(), n_key_tokens);
        println!("Keys: {} tokens", n_key_tokens);
    }
}
