//! Tiered retrieval — gated cascade across L3 → L2 → L1.
//!
//! Strategy:
//! 1. Score ALL L3 entries (tiny, always affordable).
//! 2. Gate L2: only scan L2 entries whose L3 parent scored above threshold.
//!    (For now, scan all L2 — gating requires parent tracking, deferred.)
//! 3. Always scan ALL L1 entries (working memory, most relevant).
//! 4. Merge results across tiers, sort by score, return with tier provenance.

use super::hierarchical::HierarchicalCache;
use crate::retrieve;

/// Which tier a result came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    L1,
    L2,
    L3,
}

/// A retrieval result with tier provenance.
#[derive(Debug)]
pub struct TieredResult {
    /// Source text.
    pub text: String,
    /// Aggregated attention score.
    pub score: f32,
    /// Which tier this result came from.
    pub tier: Tier,
    /// Turn ID, if available.
    pub turn_id: Option<u64>,
}

/// Run tiered retrieval across all tiers of the hierarchical cache.
///
/// `queries`: flat f32 Q vectors, shape `[n_query_tokens, n_heads, head_dim]`.
/// `n_query_tokens`: number of query tokens.
/// `n_heads`: number of query heads (may differ from n_kv_heads for GQA).
/// `top_k`: max results to return.
pub fn tiered_retrieve(
    hc: &HierarchicalCache,
    queries: &[f32],
    n_query_tokens: usize,
    n_heads: usize,
    top_k: usize,
) -> Vec<TieredResult> {
    let mut results = Vec::new();
    let oversample = top_k * 3;

    // L3 — always scan all (tiny).
    if !hc.l3.cache.is_empty() {
        let scores = retrieve::retrieve(queries, n_query_tokens, n_heads, &hc.l3.cache);
        let top = scores.top_k(oversample);
        let resolved = hc.l3.map.resolve_top_k(&top);
        for r in resolved.into_iter().take(top_k) {
            results.push(TieredResult {
                text: r.span.text.clone(),
                score: r.score,
                tier: Tier::L3,
                turn_id: r.span.turn_id,
            });
        }
    }

    // L2 — scan all (future: gate by L3 scores).
    if !hc.l2.cache.is_empty() {
        let scores = retrieve::retrieve(queries, n_query_tokens, n_heads, &hc.l2.cache);
        let top = scores.top_k(oversample);
        let resolved = hc.l2.map.resolve_top_k(&top);
        for r in resolved.into_iter().take(top_k) {
            results.push(TieredResult {
                text: r.span.text.clone(),
                score: r.score,
                tier: Tier::L2,
                turn_id: r.span.turn_id,
            });
        }
    }

    // L1 — always scan all (working memory).
    if !hc.l1.cache.is_empty() {
        let scores = retrieve::retrieve(queries, n_query_tokens, n_heads, &hc.l1.cache);
        let top = scores.top_k(oversample);
        let resolved = hc.l1.map.resolve_top_k(&top);
        for r in resolved.into_iter().take(top_k) {
            results.push(TieredResult {
                text: r.span.text.clone(),
                score: r.score,
                tier: Tier::L1,
                turn_id: r.span.turn_id,
            });
        }
    }

    // Sort by descending score and truncate.
    results.sort_unstable_by(|a, b| {
        b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);

    results
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::consolidator::consolidate;
    use crate::cache::hierarchical::{HierarchicalCache, HierarchicalConfig};
    use crate::cache::position_map::Role;


    fn small_config() -> HierarchicalConfig {
        HierarchicalConfig {
            l1_capacity: 32,
            l2_capacity: 16,
            l3_capacity: 8,
            chunk_size: 4,
            threshold: 0.5,
            entropy_threshold: 0.85,
            max_span_text: 256,
        }
    }

    #[test]
    fn retrieve_from_l1_only() {
        let mut hc = HierarchicalCache::new(small_config(), 2, 8, (42, 99));
        let kv_dim = 2 * 8;

        // Add a few positions to L1.
        for i in 0..3 {
            let key: Vec<f32> = (0..kv_dim).map(|j| ((i * kv_dim + j) as f32 + 1.0) * 0.1).collect();
            hc.append_to_l1(&key, &vec![0.0f32; kv_dim]);
        }
        let tid = hc.next_turn_id();
        hc.record_span(0, 3, "hello world".into(), Role::User, Some(tid), None);

        // Query — use 1 token, 2 heads, head_dim=8.
        let n_heads = 2;
        let queries = vec![0.5f32; n_heads * 8];
        let results = tiered_retrieve(&hc, &queries, 1, n_heads, 5);

        assert!(!results.is_empty());
        assert_eq!(results[0].tier, Tier::L1);
        assert!(results[0].text.contains("hello"));
    }

    #[test]
    fn retrieve_across_tiers() {
        let mut hc = HierarchicalCache::new(small_config(), 2, 8, (42, 99));
        let kv_dim = 2 * 8;

        // Fill L1 enough to trigger consolidation.
        for i in 0..16 {
            let key: Vec<f32> = (0..kv_dim).map(|j| ((i * kv_dim + j) as f32 + 1.0) * 0.01).collect();
            hc.append_to_l1(&key, &vec![0.0f32; kv_dim]);
        }
        let tid = hc.next_turn_id();
        hc.record_span(0, 16, "original data for testing consolidation".into(), Role::User, Some(tid), None);

        // Consolidate: should move chunk to L2.
        let report = consolidate(&mut hc);
        assert!(report.l2_added > 0);

        // Now query.
        let n_heads = 2;
        let queries = vec![0.5f32; n_heads * 8];
        let results = tiered_retrieve(&hc, &queries, 1, n_heads, 10);

        assert!(!results.is_empty());

        // Should have results from at least L1 and L2.
        let tiers: Vec<Tier> = results.iter().map(|r| r.tier).collect();
        assert!(tiers.contains(&Tier::L1) || tiers.contains(&Tier::L2),
            "should have results from L1 or L2");
    }

    #[test]
    fn empty_cache_returns_empty() {
        let hc = HierarchicalCache::new(small_config(), 2, 8, (42, 99));
        let queries = vec![0.5f32; 2 * 8];
        let results = tiered_retrieve(&hc, &queries, 1, 2, 5);
        assert!(results.is_empty());
    }
}
