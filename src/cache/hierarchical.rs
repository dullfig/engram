//! HierarchicalCache — three-tier compressed KV cache with consolidation.
//!
//! L1 (working memory): recent, full-resolution positions.
//! L2 (session memory): chunked summaries (centroid of K vectors).
//! L3 (archive): episode-level, tiny — coarsest summaries.
//!
//! Each tier has its own QuantizedKvCache + PositionMap, independent positions.

use super::position_map::{PositionMap, Role};
use super::quantized::QuantizedKvCache;

/// Configuration for the hierarchical cache.
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// L1 (working memory) capacity in positions.
    pub l1_capacity: usize,
    /// L2 (session memory) capacity in positions.
    pub l2_capacity: usize,
    /// L3 (archive) capacity in positions.
    pub l3_capacity: usize,
    /// Chunk size for L1→L2 consolidation (positions).
    pub chunk_size: usize,
    /// Fill threshold (0.0–1.0) — pressure trigger ("psychosis backstop").
    pub threshold: f32,
    /// Normalized entropy threshold (0.0–1.0) — sleep trigger.
    ///
    /// When L1 attention entropy exceeds this fraction of max entropy,
    /// attention is too diffuse and consolidation fires. This is the
    /// healthy sleep cycle — consolidate before signal drowns in noise.
    ///
    /// 0.0 = never trigger on entropy, 1.0 = only trigger at uniform.
    /// Default 0.85 = consolidate when attention is 85% of max diffuse.
    pub entropy_threshold: f32,
    /// Max text length to keep per L2/L3 span.
    pub max_span_text: usize,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            l1_capacity: 4096,
            l2_capacity: 2048,
            l3_capacity: 512,
            chunk_size: 512,
            threshold: 0.8,
            entropy_threshold: 0.85,
            max_span_text: 2048,
        }
    }
}

/// Why consolidation was triggered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConsolidationTrigger {
    /// Attention entropy too high — signal diluting. The healthy sleep cycle.
    Sleep,
    /// L1 physically full — must evict. The psychosis backstop.
    Pressure,
    /// Both conditions met simultaneously.
    Both,
    /// Neither — no consolidation needed.
    #[default]
    None,
}

/// A tier of the hierarchical cache.
pub struct CacheTier {
    pub cache: QuantizedKvCache,
    pub map: PositionMap,
}

impl CacheTier {
    pub(crate) fn new(n_kv_heads: usize, head_dim: usize, capacity: usize, rotation_seed: u64, qjl_seed: u64) -> Self {
        Self {
            cache: QuantizedKvCache::with_qjl(n_kv_heads, head_dim, capacity, rotation_seed, qjl_seed),
            map: PositionMap::new(),
        }
    }
}

/// Three-tier hierarchical cache.
pub struct HierarchicalCache {
    pub config: HierarchicalConfig,
    pub l1: CacheTier,
    pub l2: CacheTier,
    pub l3: CacheTier,
    pub(crate) n_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// Normalized entropy from the most recent query (0.0–1.0).
    /// 0.0 = perfect focus, 1.0 = uniform (maximum dilution).
    pub(crate) last_entropy: f32,
}

impl HierarchicalCache {
    /// Create a new hierarchical cache.
    ///
    /// `seeds`: (rotation_seed, qjl_seed) — all tiers share the same seeds
    /// so compressed entries can be copied between tiers without re-encoding.
    pub fn new(
        config: HierarchicalConfig,
        n_kv_heads: usize,
        head_dim: usize,
        seeds: (u64, u64),
    ) -> Self {
        let (rseed, qseed) = seeds;
        Self {
            l1: CacheTier::new(n_kv_heads, head_dim, config.l1_capacity, rseed, qseed),
            l2: CacheTier::new(n_kv_heads, head_dim, config.l2_capacity, rseed, qseed),
            l3: CacheTier::new(n_kv_heads, head_dim, config.l3_capacity, rseed, qseed),
            config,
            n_kv_heads,
            head_dim,
            last_entropy: 0.0,
        }
    }

    /// Append K/V vectors for one position into L1.
    pub fn append_to_l1(&mut self, key: &[f32], value: &[f32]) {
        self.l1.cache.append_one(key, value);
    }

    /// Record a span in L1's position map.
    pub fn record_span(
        &mut self,
        start_pos: usize,
        end_pos: usize,
        text: String,
        role: Role,
        turn_id: Option<u64>,
        metadata: Option<String>,
    ) {
        self.l1.map.append(start_pos, end_pos, text, role, turn_id, metadata);
    }

    /// Allocate a turn ID from L1's position map.
    pub fn next_turn_id(&mut self) -> u64 {
        self.l1.map.next_turn_id()
    }

    /// Whether L1 has reached the pressure threshold (OOM backstop).
    pub fn pressure_triggered(&self) -> bool {
        let used = self.l1.cache.len() as f32;
        let cap = self.config.l1_capacity as f32;
        used / cap >= self.config.threshold
    }

    /// Check both triggers. Uses last observed entropy if available.
    pub fn needs_consolidation(&self) -> bool {
        self.trigger() != ConsolidationTrigger::None
    }

    /// Determine which trigger condition is active.
    pub fn trigger(&self) -> ConsolidationTrigger {
        let pressure = self.pressure_triggered();
        let entropy = self.last_entropy >= self.config.entropy_threshold
            && self.l1.cache.len() >= self.config.chunk_size;
        match (entropy, pressure) {
            (true, true) => ConsolidationTrigger::Both,
            (true, false) => ConsolidationTrigger::Sleep,
            (false, true) => ConsolidationTrigger::Pressure,
            (false, false) => ConsolidationTrigger::None,
        }
    }

    /// Record the normalized entropy from the most recent query.
    ///
    /// Called by the engine after each query. The value is H/log(N)
    /// where H is Shannon entropy and N is L1 cache length.
    pub fn set_last_entropy(&mut self, normalized_entropy: f32) {
        self.last_entropy = normalized_entropy;
    }

    /// Last observed normalized entropy (0.0 = perfect focus, 1.0 = uniform).
    pub fn last_entropy(&self) -> f32 {
        self.last_entropy
    }

    /// Number of KV heads.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    /// Head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

impl std::fmt::Debug for HierarchicalCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HierarchicalCache(L1: {}/{}, L2: {}/{}, L3: {}/{})",
            self.l1.cache.len(), self.config.l1_capacity,
            self.l2.cache.len(), self.config.l2_capacity,
            self.l3.cache.len(), self.config.l3_capacity,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> HierarchicalConfig {
        HierarchicalConfig {
            l1_capacity: 64,
            l2_capacity: 32,
            l3_capacity: 16,
            chunk_size: 8,
            threshold: 0.8,
            entropy_threshold: 0.85,
            max_span_text: 256,
        }
    }

    #[test]
    fn construction() {
        let hc = HierarchicalCache::new(test_config(), 2, 8, (42, 99));
        assert_eq!(hc.l1.cache.len(), 0);
        assert_eq!(hc.l2.cache.len(), 0);
        assert_eq!(hc.l3.cache.len(), 0);
        assert_eq!(hc.n_kv_heads(), 2);
        assert_eq!(hc.head_dim(), 8);
    }

    #[test]
    fn append_to_l1() {
        let mut hc = HierarchicalCache::new(test_config(), 2, 8, (42, 99));
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let val = vec![0.3f32; kv_dim];

        hc.append_to_l1(&key, &val);
        assert_eq!(hc.l1.cache.len(), 1);
    }

    #[test]
    fn needs_consolidation_check() {
        let config = test_config(); // l1_capacity=64, threshold=0.8 → triggers at 52
        let mut hc = HierarchicalCache::new(config, 2, 8, (42, 99));
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let val = vec![0.3f32; kv_dim];

        assert!(!hc.needs_consolidation());

        // Fill to 80%.
        for _ in 0..52 {
            hc.append_to_l1(&key, &val);
        }
        assert!(hc.needs_consolidation());
    }

    #[test]
    fn record_span_and_turn_id() {
        let mut hc = HierarchicalCache::new(test_config(), 2, 8, (42, 99));
        let kv_dim = 2 * 8;
        let key = vec![0.5f32; kv_dim];
        let val = vec![0.3f32; kv_dim];

        for _ in 0..5 {
            hc.append_to_l1(&key, &val);
        }

        let tid = hc.next_turn_id();
        hc.record_span(0, 5, "hello world".into(), Role::User, Some(tid), None);
        assert_eq!(hc.l1.map.len(), 1);
        assert_eq!(hc.l1.map.spans()[0].text, "hello world");
    }

    #[test]
    fn entropy_trigger() {
        let config = HierarchicalConfig {
            l1_capacity: 64,
            l2_capacity: 32,
            l3_capacity: 16,
            chunk_size: 8,
            threshold: 0.99, // pressure won't trigger
            entropy_threshold: 0.5,
            max_span_text: 256,
        };
        let mut hc = HierarchicalCache::new(config, 2, 8, (42, 99));
        let kv_dim = 2 * 8;

        // Fill 16 positions (well below pressure threshold of 99%).
        for _ in 0..16 {
            hc.append_to_l1(&vec![0.5f32; kv_dim], &vec![0.0f32; kv_dim]);
        }

        // No trigger yet — entropy hasn't been observed.
        assert_eq!(hc.trigger(), ConsolidationTrigger::None);

        // Simulate high entropy from a query.
        hc.set_last_entropy(0.9);
        assert_eq!(hc.trigger(), ConsolidationTrigger::Sleep);
    }

    #[test]
    fn both_triggers() {
        let config = HierarchicalConfig {
            l1_capacity: 20,
            l2_capacity: 16,
            l3_capacity: 8,
            chunk_size: 4,
            threshold: 0.8,
            entropy_threshold: 0.5,
            max_span_text: 256,
        };
        let mut hc = HierarchicalCache::new(config, 2, 8, (42, 99));
        let kv_dim = 2 * 8;

        // Fill to 80% (16/20).
        for _ in 0..16 {
            hc.append_to_l1(&vec![0.5f32; kv_dim], &vec![0.0f32; kv_dim]);
        }

        hc.set_last_entropy(0.9);
        assert_eq!(hc.trigger(), ConsolidationTrigger::Both);
    }

    #[test]
    fn debug_display() {
        let hc = HierarchicalCache::new(test_config(), 2, 8, (42, 99));
        let dbg = format!("{:?}", hc);
        assert!(dbg.contains("HierarchicalCache"));
        assert!(dbg.contains("L1: 0/64"));
    }
}
