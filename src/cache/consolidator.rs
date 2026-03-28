//! Consolidator — migrates data between hierarchical cache tiers.
//!
//! Like hippocampal memory consolidation during sleep:
//! - Sleep-triggered (entropy): drain the *lowest-scoring* L1 chunk → L2
//! - Pressure-triggered (full): drain the oldest L1 chunk → L2
//! - L2 → L3 cascade when L2 fills
//! - Compaction: surviving entries copied via `append_compressed` (zero-loss)

use super::hierarchical::{ConsolidationTrigger, HierarchicalCache};
use super::position_map::Role;
use super::quantized::QuantizedKvCache;

/// Report from a consolidation pass.
#[derive(Debug, Default)]
pub struct ConsolidationReport {
    /// Positions drained from L1.
    pub l1_drained: usize,
    /// Summary entries added to L2.
    pub l2_added: usize,
    /// Positions drained from L2.
    pub l2_drained: usize,
    /// Summary entries added to L3.
    pub l3_added: usize,
    /// L1 positions after compaction.
    pub l1_remaining: usize,
    /// What triggered this consolidation.
    pub trigger: ConsolidationTrigger,
    /// Which chunk was selected for eviction (index, 0-based).
    pub evicted_chunk: usize,
    /// The attention score of the evicted chunk (lower = more noise).
    pub evicted_chunk_score: f32,
}

/// Compute a centroid (mean) of dequantized K vectors for a range of positions.
///
/// Returns one f32 vector per KV head, each of length `head_dim`.
/// The centroid smears RoPE phases, which is acceptable for coarse routing.
pub fn summarize_chunk(
    cache: &QuantizedKvCache,
    start: usize,
    end: usize,
) -> Vec<Vec<f32>> {
    let n_heads = cache.n_kv_heads();
    let head_dim = cache.head_dim();
    let count = end - start;
    assert!(count > 0, "empty chunk");

    let mut centroids = vec![vec![0.0f32; head_dim]; n_heads];

    for pos in start..end {
        for head in 0..n_heads {
            let k = cache.key_at_dequant(pos, head);
            for (i, &val) in k.iter().enumerate() {
                centroids[head][i] += val;
            }
        }
    }

    let inv = 1.0 / count as f32;
    for centroid in &mut centroids {
        for v in centroid.iter_mut() {
            *v *= inv;
        }
    }

    centroids
}

/// Run a consolidation pass on the hierarchical cache.
///
/// Pressure-triggered: drain oldest chunk (we're out of room).
/// Sleep-triggered: drain lowest-scoring chunk (it's contributing noise).
///
/// `chunk_scores`: optional per-chunk attention scores from the last query.
/// When provided and sleep-triggered, evicts the lowest-scoring chunk.
/// When None or pressure-triggered, falls back to oldest (chunk 0).
pub fn consolidate(hc: &mut HierarchicalCache) -> ConsolidationReport {
    consolidate_with_scores(hc, None)
}

/// Consolidate with optional chunk attention scores for intelligent eviction.
pub fn consolidate_with_scores(
    hc: &mut HierarchicalCache,
    chunk_scores: Option<&[f32]>,
) -> ConsolidationReport {
    let mut report = ConsolidationReport::default();
    let chunk = hc.config.chunk_size;
    let max_text = hc.config.max_span_text;
    report.trigger = hc.trigger();

    // --- L1 → L2 ---
    let l1_len = hc.l1.cache.len();
    if l1_len >= chunk {
        // Pick which chunk to evict.
        let (chunk_idx, chunk_score) = pick_eviction_chunk(
            report.trigger,
            chunk_scores,
            l1_len,
            chunk,
        );

        report.evicted_chunk = chunk_idx;
        report.evicted_chunk_score = chunk_score;

        let drain_start = chunk_idx * chunk;
        let drain_end = (drain_start + chunk).min(l1_len);

        // Summarize the chunk into a centroid.
        let centroids = summarize_chunk(&hc.l1.cache, drain_start, drain_end);

        // Gather text from spans overlapping [drain_start, drain_end).
        let mut combined_text = String::new();
        for span in hc.l1.map.spans() {
            if span.start_pos >= drain_end {
                break;
            }
            if span.end_pos <= drain_start {
                continue;
            }
            if !combined_text.is_empty() {
                combined_text.push(' ');
            }
            combined_text.push_str(&span.text);
        }
        if combined_text.len() > max_text {
            combined_text.truncate(max_text);
        }

        // Append centroid to L2 as a single position.
        let n_heads = hc.n_kv_heads();
        let head_dim = hc.head_dim();
        let kv_dim = n_heads * head_dim;
        let mut flat_key = vec![0.0f32; kv_dim];
        let flat_val = vec![0.0f32; kv_dim]; // dummy V
        for (head, centroid) in centroids.iter().enumerate() {
            flat_key[head * head_dim..(head + 1) * head_dim].copy_from_slice(centroid);
        }

        let l2_start = hc.l2.cache.len();
        hc.l2.cache.append_one(&flat_key, &flat_val);
        let l2_end = hc.l2.cache.len();
        let tid = hc.l2.map.next_turn_id();
        hc.l2.map.append(l2_start, l2_end, combined_text, Role::System, Some(tid), None);

        report.l1_drained = drain_end - drain_start;
        report.l2_added = 1;

        // Compact L1: excise the evicted range and rebuild.
        compact_l1_range(hc, drain_start, drain_end);
    }

    // --- L2 → L3 ---
    let l2_fill = hc.l2.cache.len() as f32 / hc.config.l2_capacity as f32;
    if l2_fill >= hc.config.threshold && hc.l2.cache.len() >= chunk {
        let drain_end = chunk.min(hc.l2.cache.len());

        let centroids = summarize_chunk(&hc.l2.cache, 0, drain_end);

        let mut combined_text = String::new();
        for span in hc.l2.map.spans() {
            if span.start_pos >= drain_end {
                break;
            }
            if !combined_text.is_empty() {
                combined_text.push(' ');
            }
            combined_text.push_str(&span.text);
        }
        let _drained = hc.l2.map.drain_up_to(drain_end);
        if combined_text.len() > max_text {
            combined_text.truncate(max_text);
        }

        let n_heads = hc.n_kv_heads();
        let head_dim = hc.head_dim();
        let kv_dim = n_heads * head_dim;
        let mut flat_key = vec![0.0f32; kv_dim];
        let flat_val = vec![0.0f32; kv_dim];
        for (head, centroid) in centroids.iter().enumerate() {
            flat_key[head * head_dim..(head + 1) * head_dim].copy_from_slice(centroid);
        }

        let l3_start = hc.l3.cache.len();
        hc.l3.cache.append_one(&flat_key, &flat_val);
        let l3_end = hc.l3.cache.len();
        let tid = hc.l3.map.next_turn_id();
        hc.l3.map.append(l3_start, l3_end, combined_text, Role::System, Some(tid), None);

        report.l2_drained = drain_end;
        report.l3_added = 1;

        // Compact L2.
        compact_tier(&mut hc.l2, drain_end);
    }

    report.l1_remaining = hc.l1.cache.len();
    report
}

/// Pick which chunk to evict based on trigger type and attention scores.
///
/// Sleep-triggered: evict the lowest-scoring chunk (it's noise).
/// Pressure-triggered: evict chunk 0 (oldest, simple FIFO).
///
/// Returns (chunk_index, chunk_score). Score is 0.0 when no scores available.
pub(crate) fn pick_eviction_chunk(
    trigger: ConsolidationTrigger,
    chunk_scores: Option<&[f32]>,
    l1_len: usize,
    chunk_size: usize,
) -> (usize, f32) {
    let n_chunks = l1_len / chunk_size; // only consider full chunks

    // Sleep-triggered with scores available: evict the weakest chunk.
    if matches!(trigger, ConsolidationTrigger::Sleep | ConsolidationTrigger::Both) {
        if let Some(scores) = chunk_scores {
            if let Some((min_idx, &min_score)) = scores.iter()
                .enumerate()
                .take(n_chunks) // only full chunks
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                return (min_idx, min_score);
            }
        }
    }

    // Fallback: oldest chunk.
    let score = chunk_scores.and_then(|s| s.first().copied()).unwrap_or(0.0);
    (0, score)
}

/// Compact L1: excise positions [start..end) and rebuild cache contiguously.
///
/// Handles both prefix eviction (start=0, fast path) and middle eviction
/// (sleep-triggered, requires copying before+after the hole).
fn compact_l1_range(hc: &mut HierarchicalCache, start: usize, end: usize) {
    let old_len = hc.l1.cache.len();
    let n_heads = hc.n_kv_heads();

    let mut new_cache = QuantizedKvCache::with_qjl(
        n_heads,
        hc.head_dim(),
        hc.config.l1_capacity,
        42, 99,
    );

    // Copy [0..start) then [end..old_len).
    for pos in (0..start).chain(end..old_len) {
        for head in 0..n_heads {
            let entry = hc.l1.cache.read_compressed_k(pos, head);
            new_cache.append_compressed(&entry, head);
        }
        new_cache.advance_len();
    }

    hc.l1.cache = new_cache;

    // Rebuild position map: remove spans in the evicted range, rebase survivors.
    // For prefix eviction (start=0), rebase by end.
    // For middle eviction, we need to rebuild — drain the range then compact.
    if start == 0 {
        hc.l1.map.rebase(end);
    } else {
        // Middle eviction: rebuild map by removing the gap.
        // Spans that overlap [start..end) get truncated or removed.
        // Then shift everything after the gap left by (end-start).
        rebuild_map_after_excision(&mut hc.l1.map, start, end);
    }
}

/// Rebuild position map after excising positions [start..end).
///
/// Spans fully inside the gap are removed. Spans partially overlapping
/// are truncated. Spans after the gap are shifted left by gap_size.
pub(crate) fn rebuild_map_after_excision(
    map: &mut super::position_map::PositionMap,
    start: usize,
    end: usize,
) {
    let gap = end - start;
    let old_spans: Vec<_> = map.spans().to_vec();
    map.clear();

    // We need to reconstruct with new contiguous positions.
    // This requires creating a fresh map since append() enforces contiguity.
    let mut new_spans = Vec::new();
    for span in &old_spans {
        if span.end_pos <= start {
            // Before gap — keep as-is.
            new_spans.push(span.clone());
        } else if span.start_pos >= end {
            // After gap — shift left.
            let mut s = span.clone();
            s.start_pos -= gap;
            s.end_pos -= gap;
            new_spans.push(s);
        } else if span.start_pos < start && span.end_pos > end {
            // Straddles the gap — shrink by gap size.
            let mut s = span.clone();
            s.end_pos -= gap;
            new_spans.push(s);
        } else if span.start_pos < start {
            // Overlaps start of gap — truncate end.
            let mut s = span.clone();
            s.end_pos = start;
            if s.end_pos > s.start_pos {
                new_spans.push(s);
            }
        } else if span.end_pos > end {
            // Overlaps end of gap — truncate start, shift.
            let mut s = span.clone();
            s.start_pos = start;
            s.end_pos -= gap;
            if s.end_pos > s.start_pos {
                new_spans.push(s);
            }
        }
        // else: fully inside gap — drop it
    }

    // Re-append to map (bypasses contiguity debug_assert by using internal method).
    // Since we can't bypass, we set spans directly. We'll add a set_spans method.
    // Actually, let's just use the public API — the spans should be contiguous now.
    for span in new_spans {
        map.append(
            span.start_pos,
            span.end_pos,
            span.text,
            span.role,
            span.turn_id,
            span.metadata,
        );
    }
}

/// Compact a generic tier (for L2 compaction).
fn compact_tier(tier: &mut super::hierarchical::CacheTier, drain_end: usize) {
    let surviving = tier.cache.len() - drain_end;
    if surviving == 0 {
        tier.cache.clear();
        tier.map.clear();
        return;
    }

    let n_heads = tier.cache.n_kv_heads();
    let head_dim = tier.cache.head_dim();
    let max_seq = tier.cache.max_seq_len();

    let mut new_cache = QuantizedKvCache::with_qjl(n_heads, head_dim, max_seq, 42, 99);

    for pos in drain_end..drain_end + surviving {
        for head in 0..n_heads {
            let entry = tier.cache.read_compressed_k(pos, head);
            new_cache.append_compressed(&entry, head);
        }
        new_cache.advance_len();
    }

    tier.cache = new_cache;
    tier.map.rebase(drain_end);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::hierarchical::HierarchicalConfig;
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

    fn fill_l1(hc: &mut HierarchicalCache, n: usize) {
        let kv_dim = hc.n_kv_heads() * hc.head_dim();
        for i in 0..n {
            let key: Vec<f32> = (0..kv_dim).map(|j| ((i * kv_dim + j) as f32 + 1.0) * 0.01).collect();
            let val = vec![0.0f32; kv_dim];
            hc.append_to_l1(&key, &val);
        }
        // Record one big span for all positions.
        let tid = hc.next_turn_id();
        hc.record_span(0, n, format!("test data ({n} positions)"), Role::User, Some(tid), None);
    }

    #[test]
    fn summarize_chunk_returns_centroids() {
        let mut cache = QuantizedKvCache::new(2, 8, 100, 42);
        let kv_dim = 2 * 8;
        for i in 0..4 {
            let key: Vec<f32> = (0..kv_dim).map(|j| ((i * kv_dim + j) as f32 + 1.0) * 0.1).collect();
            let val = vec![0.0f32; kv_dim];
            cache.append_one(&key, &val);
        }

        let centroids = summarize_chunk(&cache, 0, 4);
        assert_eq!(centroids.len(), 2); // 2 heads
        assert_eq!(centroids[0].len(), 8); // head_dim
        for v in &centroids[0] {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn consolidate_l1_to_l2() {
        let mut hc = HierarchicalCache::new(small_config(), 2, 8, (42, 99));
        fill_l1(&mut hc, 16); // 16 positions, threshold 0.5 of 32 = 16 → triggers

        assert!(hc.needs_consolidation());
        let report = consolidate(&mut hc);

        assert_eq!(report.l1_drained, 4); // chunk_size = 4
        assert_eq!(report.l2_added, 1);
        assert_eq!(hc.l2.cache.len(), 1);
        assert_eq!(hc.l1.cache.len(), 12); // 16 - 4 compacted
        assert_eq!(report.l1_remaining, 12);

        // L2 span should have text.
        assert_eq!(hc.l2.map.len(), 1);
        assert!(!hc.l2.map.spans()[0].text.is_empty());
    }

    #[test]
    fn l1_compaction_preserves_data() {
        let mut hc = HierarchicalCache::new(small_config(), 2, 8, (42, 99));
        fill_l1(&mut hc, 8);

        let report = consolidate(&mut hc);
        assert_eq!(report.l1_drained, 4);

        // Remaining positions should still be queryable.
        let query = vec![1.0f32; 8];
        let dot = hc.l1.cache.dot_key(0, 0, &query);
        assert!(dot.is_finite());
    }

    #[test]
    fn no_consolidation_when_below_threshold() {
        let mut hc = HierarchicalCache::new(small_config(), 2, 8, (42, 99));
        let kv_dim = 2 * 8;
        // Add just 2 positions — well below threshold.
        for _ in 0..2 {
            hc.append_to_l1(&vec![0.5f32; kv_dim], &vec![0.0f32; kv_dim]);
        }
        let tid = hc.next_turn_id();
        hc.record_span(0, 2, "small".into(), Role::User, Some(tid), None);

        assert!(!hc.needs_consolidation());
        let report = consolidate(&mut hc);
        assert_eq!(report.l1_drained, 0);
        assert_eq!(report.l2_added, 0);
    }

    #[test]
    fn sleep_triggered_evicts_weakest_chunk() {
        let config = HierarchicalConfig {
            l1_capacity: 32,
            l2_capacity: 16,
            l3_capacity: 8,
            chunk_size: 4,
            threshold: 0.99,       // pressure won't trigger
            entropy_threshold: 0.5, // low threshold so sleep triggers easily
            max_span_text: 256,
        };
        let mut hc = HierarchicalCache::new(config, 2, 8, (42, 99));
        let kv_dim = 2 * 8;

        // Fill 16 positions (4 chunks of 4).
        for i in 0..16 {
            let key: Vec<f32> = (0..kv_dim).map(|j| ((i * kv_dim + j) as f32 + 1.0) * 0.01).collect();
            hc.append_to_l1(&key, &vec![0.0f32; kv_dim]);
        }
        let tid = hc.next_turn_id();
        hc.record_span(0, 16, "test data".into(), Role::User, Some(tid), None);

        // Simulate high entropy (sleep trigger).
        hc.set_last_entropy(0.9);
        assert_eq!(hc.trigger(), ConsolidationTrigger::Sleep);

        // Provide chunk scores: chunk 2 is the weakest.
        let chunk_scores = [0.3, 0.5, 0.1, 0.4]; // chunk 2 has lowest score
        let report = consolidate_with_scores(&mut hc, Some(&chunk_scores));

        assert_eq!(report.trigger, ConsolidationTrigger::Sleep);
        assert_eq!(report.evicted_chunk, 2, "should evict weakest chunk");
        assert!((report.evicted_chunk_score - 0.1).abs() < 1e-6);
        assert_eq!(report.l1_drained, 4);
        assert_eq!(report.l2_added, 1);
        assert_eq!(hc.l1.cache.len(), 12); // 16 - 4
    }
}
