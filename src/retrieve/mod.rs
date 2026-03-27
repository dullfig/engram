//! Attention-based retrieval API.
//!
//! Provides bidirectional attention over a compressed KV cache for
//! contextual retrieval. Unlike generative attention, no causal mask
//! is applied — the query attends to ALL cached positions.

use crate::cache::quantized::QuantizedKvCache;

/// Attention scores from a retrieval pass.
///
/// Contains per-head relevance scores for each cached position,
/// aggregated across query tokens.
pub struct AttentionScores {
    /// Aggregated relevance score per cached position.
    /// Length: `cached_len`. Higher = more relevant.
    scores: Vec<f32>,
    /// Number of cached positions scored.
    cached_len: usize,
}

impl AttentionScores {
    /// Return the top-k most relevant cached positions.
    ///
    /// Returns (position_index, score) pairs sorted by descending score.
    pub fn top_k(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = self
            .scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        // Partial sort: we only need the top k.
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// All scores as a slice (position 0..cached_len).
    pub fn all_scores(&self) -> &[f32] {
        &self.scores
    }

    /// Number of cached positions that were scored.
    pub fn cached_len(&self) -> usize {
        self.cached_len
    }
}

/// Run a retrieval pass: score all cached positions against a query.
///
/// This is bidirectional attention (no causal mask). The query tokens
/// attend to every cached position. Attention scores are aggregated
/// across query tokens and heads to produce a single relevance score
/// per cached position.
///
/// `queries`: flat f32 of shape `[n_query_tokens, n_heads, head_dim]`.
///            These are the Q projections of the new user message,
///            already RoPE'd.
/// `cache`: the compressed KV cache to search.
/// `n_heads`: number of query heads (may differ from n_kv_heads for GQA).
///
/// Returns attention scores for ranking.
pub fn retrieve(
    queries: &[f32],
    n_query_tokens: usize,
    n_heads: usize,
    cache: &QuantizedKvCache,
) -> AttentionScores {
    let head_dim = cache.head_dim();
    let n_kv_heads = cache.n_kv_heads();
    let cached_len = cache.len();
    let heads_per_group = n_heads / n_kv_heads;

    assert_eq!(
        queries.len(),
        n_query_tokens * n_heads * head_dim,
        "query shape mismatch"
    );

    // Accumulate relevance scores per cached position.
    let mut position_scores = vec![0.0f32; cached_len];

    let scale = 1.0 / (head_dim as f32).sqrt();

    for qt in 0..n_query_tokens {
        for qh in 0..n_heads {
            let kv_h = qh / heads_per_group;
            let q_off = (qt * n_heads + qh) * head_dim;
            let q_vec = &queries[q_off..q_off + head_dim];

            // Score against ALL cached positions (bidirectional — no causal mask).
            let mut scores = Vec::with_capacity(cached_len);
            for pos in 0..cached_len {
                let dot = cache.dot_key(pos, kv_h, q_vec);
                scores.push(dot * scale);
            }

            // Softmax to get attention weights.
            softmax_inplace(&mut scores);

            // Accumulate into position relevance scores.
            for (pos, &weight) in scores.iter().enumerate() {
                position_scores[pos] += weight;
            }
        }
    }

    // Normalize by number of (query_token, head) pairs.
    let normalizer = (n_query_tokens * n_heads) as f32;
    for s in &mut position_scores {
        *s /= normalizer;
    }

    AttentionScores {
        scores: position_scores,
        cached_len,
    }
}

/// In-place softmax with numerical stability.
fn softmax_inplace(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    if values.len() == 1 {
        values[0] = 1.0;
        return;
    }

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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_ordering() {
        let scores = AttentionScores {
            scores: vec![0.1, 0.5, 0.3, 0.9, 0.2],
            cached_len: 5,
        };

        let top2 = scores.top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 3); // position 3 has score 0.9
        assert_eq!(top2[1].0, 1); // position 1 has score 0.5
    }

    #[test]
    fn top_k_larger_than_cache() {
        let scores = AttentionScores {
            scores: vec![0.1, 0.5],
            cached_len: 2,
        };

        let top10 = scores.top_k(10);
        assert_eq!(top10.len(), 2); // can't return more than cached_len
    }

    #[test]
    fn retrieve_basic() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;

        // Cache a few positions with distinct values.
        for i in 0..5 {
            let key: Vec<f32> = (0..kv_dim).map(|j| (i * kv_dim + j) as f32 * 0.01).collect();
            let value = vec![0.1f32; kv_dim];
            cache.append_one(&key, &value);
        }

        // Query with 1 token, 2 heads, head_dim=8.
        let n_heads = 2;
        let query = vec![0.5f32; n_heads * 8];

        let result = retrieve(&query, 1, n_heads, &cache);
        assert_eq!(result.cached_len(), 5);
        assert_eq!(result.all_scores().len(), 5);

        // All scores should sum to ~1.0 (normalized attention).
        let sum: f32 = result.all_scores().iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "scores should sum to ~1.0, got {sum}"
        );
    }

    #[test]
    fn softmax_basic() {
        let mut v = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1]);
        assert!(v[1] > v[0]);
    }
}
