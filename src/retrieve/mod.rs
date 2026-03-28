//! Attention-based retrieval API.
//!
//! Provides bidirectional attention over a compressed KV cache for
//! contextual retrieval. Unlike generative attention, no causal mask
//! is applied — the query attends to ALL cached positions.

#[cfg(feature = "gpu")]
pub mod gpu;

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
    /// Create from pre-computed scores.
    pub fn from_scores(scores: Vec<f32>, cached_len: usize) -> Self {
        Self { scores, cached_len }
    }

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

/// Compute the normalized Shannon entropy of L1 attention for a query.
///
/// Returns H / log(N) where H = -Σ p_i * log(p_i) and N = cached_len.
/// Result is in [0.0, 1.0]:
/// - 0.0 = all attention on one position (perfect focus)
/// - 1.0 = uniform attention (maximum dilution — signal is drowned)
///
/// This is the "sleep signal" — when it's high, the hippocampus needs
/// to consolidate. Like the drowsiness that precedes healthy sleep.
pub fn attention_entropy(
    queries: &[f32],
    n_query_tokens: usize,
    n_heads: usize,
    cache: &QuantizedKvCache,
) -> f32 {
    let head_dim = cache.head_dim();
    let n_kv_heads = cache.n_kv_heads();
    let cached_len = cache.len();
    let heads_per_group = n_heads / n_kv_heads;

    if cached_len <= 1 {
        return 0.0; // can't be diffuse with 0-1 entries
    }

    let scale = 1.0 / (head_dim as f32).sqrt();
    let max_entropy = (cached_len as f32).ln();
    let mut total_entropy = 0.0f32;
    let n_passes = n_query_tokens * n_heads;

    for qt in 0..n_query_tokens {
        for qh in 0..n_heads {
            let kv_h = qh / heads_per_group;
            let q_off = (qt * n_heads + qh) * head_dim;
            let q_vec = &queries[q_off..q_off + head_dim];

            let mut scores = Vec::with_capacity(cached_len);
            for pos in 0..cached_len {
                scores.push(cache.dot_key(pos, kv_h, q_vec) * scale);
            }

            softmax_inplace(&mut scores);

            // Shannon entropy: H = -Σ p * ln(p)
            let h: f32 = scores.iter()
                .filter(|&&p| p > 0.0)
                .map(|&p| -p * p.ln())
                .sum();

            total_entropy += h;
        }
    }

    // Average across all (token, head) passes, normalize to [0, 1].
    let avg_entropy = total_entropy / n_passes as f32;
    (avg_entropy / max_entropy).clamp(0.0, 1.0)
}

/// Score L1 chunks by average attention weight.
///
/// Divides the cache into chunks of `chunk_size` and returns the average
/// attention score per chunk. Lower score = more noise = evict first.
pub fn score_chunks(
    queries: &[f32],
    n_query_tokens: usize,
    n_heads: usize,
    cache: &QuantizedKvCache,
    chunk_size: usize,
) -> Vec<f32> {
    let scores = retrieve(queries, n_query_tokens, n_heads, cache);
    let all = scores.all_scores();
    let n_chunks = (all.len() + chunk_size - 1) / chunk_size;
    let mut chunk_scores = Vec::with_capacity(n_chunks);

    for c in 0..n_chunks {
        let start = c * chunk_size;
        let end = (start + chunk_size).min(all.len());
        let count = (end - start) as f32;
        let sum: f32 = all[start..end].iter().sum();
        chunk_scores.push(sum / count);
    }

    chunk_scores
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

    #[test]
    fn entropy_uniform_keys_high() {
        // All identical keys → softmax is nearly uniform → high entropy.
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let val = vec![0.0f32; kv_dim];
        for _ in 0..20 {
            cache.append_one(&key, &val);
        }

        let n_heads = 2;
        let query = vec![0.5f32; n_heads * 8];
        let e = attention_entropy(&query, 1, n_heads, &cache);
        assert!(e > 0.8, "identical keys should give high entropy, got {e}");
    }

    #[test]
    fn entropy_single_entry_zero() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        cache.append_one(&vec![0.5f32; kv_dim], &vec![0.0f32; kv_dim]);

        let n_heads = 2;
        let query = vec![0.5f32; n_heads * 8];
        let e = attention_entropy(&query, 1, n_heads, &cache);
        assert_eq!(e, 0.0, "single entry can't be diffuse");
    }

    #[test]
    fn entropy_distinct_keys_lower() {
        // Distinct keys → attention should be more peaked → lower entropy.
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        for i in 0..20 {
            let key: Vec<f32> = (0..kv_dim).map(|j| ((i * kv_dim + j) as f32) * 0.5).collect();
            cache.append_one(&key, &vec![0.0f32; kv_dim]);
        }

        let n_heads = 2;
        // Query that strongly matches one key pattern.
        let query: Vec<f32> = (0..n_heads * 8).map(|j| j as f32 * 0.5).collect();
        let e = attention_entropy(&query, 1, n_heads, &cache);

        // Should be noticeably lower than uniform case.
        assert!(e < 0.95, "distinct keys should have lower entropy than uniform, got {e}");
        assert!(e > 0.0, "should still have some entropy with 20 entries");
    }

    #[test]
    fn score_chunks_basic() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        for i in 0..10 {
            let key: Vec<f32> = (0..kv_dim).map(|j| ((i * kv_dim + j) as f32 + 1.0) * 0.01).collect();
            cache.append_one(&key, &vec![0.0f32; kv_dim]);
        }

        let n_heads = 2;
        let query = vec![0.5f32; n_heads * 8];
        let chunks = score_chunks(&query, 1, n_heads, &cache, 4);

        // 10 positions / chunk_size 4 = 3 chunks (4, 4, 2)
        assert_eq!(chunks.len(), 3);
        for &s in &chunks {
            assert!(s.is_finite());
            assert!(s >= 0.0);
        }
    }
}
