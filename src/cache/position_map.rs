//! PositionMap — maps KV cache positions back to source text and metadata.
//!
//! The compressed KV cache stores states by position index. When retrieval
//! returns "positions 147-163 are relevant," we need to know what text those
//! positions correspond to. The PositionMap is that side table.
//!
//! Each entry (a "span") covers a contiguous range of cache positions and
//! carries metadata about the source: the raw text, a timestamp, a role
//! (user/assistant/system), and an optional conversation turn ID.

use std::time::SystemTime;

/// Role of the message that produced this span of KV states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// User input.
    User,
    /// Assistant (LLM) response.
    Assistant,
    /// System prompt or injected context.
    System,
    /// Tool result.
    Tool,
}

/// A contiguous span of KV cache positions with associated metadata.
#[derive(Debug, Clone)]
pub struct Span {
    /// First cache position (inclusive).
    pub start_pos: usize,
    /// Last cache position (exclusive).
    pub end_pos: usize,
    /// The raw text that was tokenized into these positions.
    pub text: String,
    /// Who produced this text.
    pub role: Role,
    /// When this span was appended to the cache.
    pub timestamp: SystemTime,
    /// Optional conversation turn ID (for grouping user+assistant pairs).
    pub turn_id: Option<u64>,
    /// Optional free-form metadata (e.g., tool name, file path).
    pub metadata: Option<String>,
}

impl Span {
    /// Number of cache positions in this span.
    pub fn len(&self) -> usize {
        self.end_pos - self.start_pos
    }

    /// Whether the span is empty.
    pub fn is_empty(&self) -> bool {
        self.start_pos >= self.end_pos
    }

    /// Check if a cache position falls within this span.
    pub fn contains(&self, pos: usize) -> bool {
        pos >= self.start_pos && pos < self.end_pos
    }
}

/// Maps KV cache positions to their source text and metadata.
///
/// Spans are append-only and non-overlapping, covering the cache
/// from position 0 to the current length.
#[derive(Debug)]
pub struct PositionMap {
    spans: Vec<Span>,
    /// Next turn ID to assign.
    next_turn_id: u64,
}

impl PositionMap {
    /// Create an empty position map.
    pub fn new() -> Self {
        Self {
            spans: Vec::new(),
            next_turn_id: 0,
        }
    }

    /// Record a new span of positions.
    ///
    /// `start_pos` and `end_pos` must be contiguous with the previous span
    /// (no gaps, no overlaps).
    pub fn append(
        &mut self,
        start_pos: usize,
        end_pos: usize,
        text: String,
        role: Role,
        turn_id: Option<u64>,
        metadata: Option<String>,
    ) {
        // Validate contiguity.
        if let Some(last) = self.spans.last() {
            debug_assert_eq!(
                start_pos, last.end_pos,
                "position map gap: last span ends at {}, new starts at {}",
                last.end_pos, start_pos,
            );
        } else {
            debug_assert_eq!(start_pos, 0, "first span must start at position 0");
        }
        debug_assert!(end_pos > start_pos, "empty span");

        self.spans.push(Span {
            start_pos,
            end_pos,
            text,
            role,
            timestamp: SystemTime::now(),
            turn_id,
            metadata,
        });
    }

    /// Allocate a new turn ID (monotonically increasing).
    pub fn next_turn_id(&mut self) -> u64 {
        let id = self.next_turn_id;
        self.next_turn_id += 1;
        id
    }

    /// Find the span containing a given cache position.
    pub fn span_at(&self, pos: usize) -> Option<&Span> {
        // Binary search since spans are sorted by start_pos.
        self.spans
            .binary_search_by(|span| {
                if pos < span.start_pos {
                    std::cmp::Ordering::Greater
                } else if pos >= span.end_pos {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()
            .map(|idx| &self.spans[idx])
    }

    /// Given a set of top-k (position, score) pairs from retrieval,
    /// aggregate scores by span and return the most relevant spans.
    ///
    /// Multiple positions within the same span are summed. Returns
    /// spans sorted by descending aggregated score.
    pub fn resolve_top_k(&self, positions: &[(usize, f32)]) -> Vec<ResolvedSpan<'_>> {
        // Aggregate scores by span index.
        let mut span_scores: Vec<f32> = vec![0.0; self.spans.len()];

        for &(pos, score) in positions {
            if let Some(idx) = self.span_index_at(pos) {
                span_scores[idx] += score;
            }
        }

        // Collect non-zero spans.
        let mut resolved: Vec<ResolvedSpan> = Vec::new();
        for (idx, &score) in span_scores.iter().enumerate() {
            if score > 0.0 {
                resolved.push(ResolvedSpan {
                    span: &self.spans[idx],
                    score,
                });
            }
        }

        // Sort by descending score.
        resolved.sort_unstable_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });

        resolved
    }

    /// Total number of spans.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// All spans.
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Total cache positions covered.
    pub fn total_positions(&self) -> usize {
        self.spans.last().map_or(0, |s| s.end_pos)
    }

    /// Drain and return all spans whose `end_pos <= end_pos`.
    ///
    /// Spans are removed from the front (oldest first). Returns the drained spans.
    pub fn drain_up_to(&mut self, end_pos: usize) -> Vec<Span> {
        // Find the split point: first span whose end_pos > end_pos.
        let split = self.spans.iter().position(|s| s.end_pos > end_pos)
            .unwrap_or(self.spans.len());
        self.spans.drain(..split).collect()
    }

    /// Clear all spans.
    pub fn clear(&mut self) {
        self.spans.clear();
    }

    /// Shift all positions by `-offset` (for L1 compaction after drain).
    ///
    /// Spans that start before offset are truncated: their start_pos becomes 0
    /// and their text is preserved (it's the best we have). Spans fully below
    /// offset are removed.
    pub fn rebase(&mut self, offset: usize) {
        self.spans.retain_mut(|span| {
            if span.end_pos <= offset {
                // Fully below offset — remove.
                return false;
            }
            if span.start_pos < offset {
                // Partially overlapping — truncate start.
                span.start_pos = 0;
                span.end_pos -= offset;
            } else {
                span.start_pos -= offset;
                span.end_pos -= offset;
            }
            true
        });
    }

    // -- Internal --

    fn span_index_at(&self, pos: usize) -> Option<usize> {
        self.spans
            .binary_search_by(|span| {
                if pos < span.start_pos {
                    std::cmp::Ordering::Greater
                } else if pos >= span.end_pos {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()
    }
}

impl Default for PositionMap {
    fn default() -> Self {
        Self::new()
    }
}

/// A span with its aggregated retrieval score.
#[derive(Debug)]
pub struct ResolvedSpan<'a> {
    /// The source span.
    pub span: &'a Span,
    /// Aggregated attention score across all retrieved positions in this span.
    pub score: f32,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_map() -> PositionMap {
        let mut map = PositionMap::new();
        let turn = map.next_turn_id();

        map.append(0, 10, "Hello, how are you?".into(), Role::User, Some(turn), None);
        map.append(10, 25, "I'm doing well, thanks!".into(), Role::Assistant, Some(turn), None);

        let turn2 = map.next_turn_id();
        map.append(25, 35, "Tell me about triggers".into(), Role::User, Some(turn2), None);
        map.append(35, 60, "Triggers are event-driven listeners...".into(), Role::Assistant, Some(turn2), None);

        map
    }

    #[test]
    fn append_and_len() {
        let map = make_map();
        assert_eq!(map.len(), 4);
        assert_eq!(map.total_positions(), 60);
    }

    #[test]
    fn span_at_finds_correct_span() {
        let map = make_map();

        let span = map.span_at(0).unwrap();
        assert_eq!(span.role, Role::User);
        assert!(span.text.contains("Hello"));

        let span = map.span_at(15).unwrap();
        assert_eq!(span.role, Role::Assistant);
        assert!(span.text.contains("doing well"));

        let span = map.span_at(30).unwrap();
        assert_eq!(span.role, Role::User);
        assert!(span.text.contains("triggers"));

        let span = map.span_at(50).unwrap();
        assert_eq!(span.role, Role::Assistant);
        assert!(span.text.contains("event-driven"));
    }

    #[test]
    fn span_at_out_of_range() {
        let map = make_map();
        assert!(map.span_at(60).is_none());
        assert!(map.span_at(100).is_none());
    }

    #[test]
    fn span_contains() {
        let map = make_map();
        let span = &map.spans()[0];
        assert!(span.contains(0));
        assert!(span.contains(9));
        assert!(!span.contains(10));
    }

    #[test]
    fn resolve_top_k_aggregates_by_span() {
        let map = make_map();

        // Simulate retrieval: positions in the "triggers" conversation score high.
        let positions = vec![
            (27, 0.3),  // "Tell me about triggers" span
            (30, 0.4),  // same span
            (40, 0.8),  // "Triggers are event-driven..." span
            (50, 0.6),  // same span
            (5, 0.1),   // "Hello" span — low relevance
        ];

        let resolved = map.resolve_top_k(&positions);
        assert!(!resolved.is_empty());

        // Highest: the assistant response about triggers (0.8 + 0.6 = 1.4)
        assert_eq!(resolved[0].span.role, Role::Assistant);
        assert!(resolved[0].span.text.contains("event-driven"));
        assert!((resolved[0].score - 1.4).abs() < 1e-6);

        // Second: the user question about triggers (0.3 + 0.4 = 0.7)
        assert_eq!(resolved[1].span.role, Role::User);
        assert!(resolved[1].span.text.contains("triggers"));

        // Third: "Hello" (0.1)
        assert_eq!(resolved[2].span.role, Role::User);
        assert!((resolved[2].score - 0.1).abs() < 1e-6);
    }

    #[test]
    fn resolve_top_k_empty() {
        let map = make_map();
        let resolved = map.resolve_top_k(&[]);
        assert!(resolved.is_empty());
    }

    #[test]
    fn turn_ids_increment() {
        let mut map = PositionMap::new();
        assert_eq!(map.next_turn_id(), 0);
        assert_eq!(map.next_turn_id(), 1);
        assert_eq!(map.next_turn_id(), 2);
    }

    #[test]
    fn drain_up_to_removes_old_spans() {
        let mut map = make_map();
        // Spans: [0,10), [10,25), [25,35), [35,60)
        // Drain everything with end_pos <= 25.
        let drained = map.drain_up_to(25);
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].start_pos, 0);
        assert_eq!(drained[1].end_pos, 25);
        assert_eq!(map.len(), 2); // two spans remain
        assert_eq!(map.spans()[0].start_pos, 25);
    }

    #[test]
    fn drain_up_to_none() {
        let mut map = make_map();
        let drained = map.drain_up_to(0);
        assert!(drained.is_empty());
        assert_eq!(map.len(), 4);
    }

    #[test]
    fn drain_up_to_all() {
        let mut map = make_map();
        let drained = map.drain_up_to(100);
        assert_eq!(drained.len(), 4);
        assert!(map.is_empty());
    }

    #[test]
    fn rebase_shifts_positions() {
        let mut map = make_map();
        // Drain first two spans, then rebase remaining.
        let _ = map.drain_up_to(25);
        map.rebase(25);
        assert_eq!(map.spans()[0].start_pos, 0);
        assert_eq!(map.spans()[0].end_pos, 10); // was [25,35)
        assert_eq!(map.spans()[1].start_pos, 10);
        assert_eq!(map.spans()[1].end_pos, 35); // was [35,60)
    }

    #[test]
    fn rebase_truncates_partial_spans() {
        let mut map = PositionMap::new();
        let turn = map.next_turn_id();
        map.append(0, 16, "big span".into(), Role::User, Some(turn), None);

        // Rebase by 4: span [0,16) becomes [0,12) with text preserved.
        map.rebase(4);
        assert_eq!(map.len(), 1);
        assert_eq!(map.spans()[0].start_pos, 0);
        assert_eq!(map.spans()[0].end_pos, 12);
        assert_eq!(map.spans()[0].text, "big span");
    }

    #[test]
    fn clear_empties_map() {
        let mut map = make_map();
        map.clear();
        assert!(map.is_empty());
        assert_eq!(map.total_positions(), 0);
    }

    #[test]
    fn empty_map() {
        let map = PositionMap::new();
        assert!(map.is_empty());
        assert_eq!(map.total_positions(), 0);
        assert!(map.span_at(0).is_none());
    }
}
