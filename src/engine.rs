//! Engram Engine — end-to-end retrieval over persistent compressed memory.
//!
//! Wires together: MiniProjection → QuantizedKvCache → PositionMap → Retrieval.
//! Feed text in, query later, get ranked relevant spans back.

use crate::cache::consolidator::{self, ConsolidationReport};
use crate::cache::hierarchical::{ConsolidationTrigger, HierarchicalCache, HierarchicalConfig};
use crate::cache::position_map::Role;
use crate::cache::tiered_retrieve::{self, Tier};
use crate::projection::{MiniProjection, ProjectionConfig};
use crate::retrieve;

/// The engram engine — a persistent associative memory.
pub struct Engine {
    /// Q/K projection from a real model.
    proj: MiniProjection,
    /// Hierarchical three-tier cache.
    cache: HierarchicalCache,
    /// Chunk scores from the last query (for sleep-triggered eviction).
    last_chunk_scores: Vec<f32>,
}

/// A retrieved memory with its source text and relevance score.
#[derive(Debug)]
pub struct Memory {
    pub text: String,
    pub role: Role,
    pub turn_id: Option<u64>,
    pub score: f32,
    /// Which tier this result came from.
    pub tier: Tier,
}

/// Cache statistics across all tiers.
#[derive(Debug)]
pub struct CacheStats {
    pub l1_positions: usize,
    pub l1_capacity: usize,
    pub l1_spans: usize,
    pub l2_positions: usize,
    pub l2_capacity: usize,
    pub l2_spans: usize,
    pub l3_positions: usize,
    pub l3_capacity: usize,
    pub l3_spans: usize,
    /// Normalized attention entropy from last query (0.0 = focused, 1.0 = uniform).
    pub l1_entropy: f32,
    /// Current consolidation trigger state.
    pub trigger: ConsolidationTrigger,
}

impl Engine {
    /// Create an engine from a GGUF model file.
    ///
    /// `max_positions` is the L1 capacity. L2 and L3 are sized by the default config.
    pub fn from_gguf(model_path: &str, max_positions: usize) -> Result<Self, String> {
        Self::from_gguf_with_config(model_path, max_positions, 0, HierarchicalConfig {
            l1_capacity: max_positions,
            ..Default::default()
        })
    }

    /// Create an engine from a specific layer of a GGUF model.
    pub fn from_gguf_layer(model_path: &str, max_positions: usize, layer: usize) -> Result<Self, String> {
        Self::from_gguf_with_config(model_path, max_positions, layer, HierarchicalConfig {
            l1_capacity: max_positions,
            ..Default::default()
        })
    }

    /// Create with explicit hierarchical config.
    pub fn from_gguf_with_config(
        model_path: &str,
        _max_positions: usize,
        layer: usize,
        config: HierarchicalConfig,
    ) -> Result<Self, String> {
        let proj = MiniProjection::from_gguf_layer(model_path, layer)
            .map_err(|e| format!("failed to load model: {e}"))?;

        let c = &proj.config;
        let cache = HierarchicalCache::new(config, c.n_kv_heads, c.head_dim, (42, 99));

        Ok(Self { proj, cache, last_chunk_scores: Vec::new() })
    }

    /// Ingest text into the memory. Returns the number of tokens consumed.
    pub fn ingest(&mut self, text: &str, role: Role) -> usize {
        self.ingest_with_metadata(text, role, None)
    }

    /// Ingest text with optional metadata.
    pub fn ingest_with_metadata(
        &mut self,
        text: &str,
        role: Role,
        metadata: Option<String>,
    ) -> usize {
        let start_pos = self.cache.l1.cache.len();
        let seq_offset = start_pos;

        // Project text to K vectors.
        let (keys, n_tokens) = self.proj.encode_kv(text, seq_offset);

        if n_tokens == 0 {
            return 0;
        }

        // Append each token's K vector to the compressed cache.
        // For V, we use a dummy — retrieval only needs K.
        // The actual text is in the PositionMap.
        let kv_dim = self.proj.config.n_kv_heads * self.proj.config.head_dim;
        let dummy_v = vec![0.0f32; kv_dim];

        for key in &keys {
            self.cache.append_to_l1(key, &dummy_v);
        }

        // Record the span.
        let end_pos = self.cache.l1.cache.len();
        let turn_id = Some(self.cache.next_turn_id());
        self.cache.record_span(start_pos, end_pos, text.to_string(), role, turn_id, metadata);

        n_tokens
    }

    /// Ingest a conversation turn (user + assistant pair).
    pub fn ingest_turn(&mut self, user_text: &str, assistant_text: &str) -> usize {
        let turn_id = self.cache.next_turn_id();
        let mut total = 0;

        let kv_dim = self.proj.config.n_kv_heads * self.proj.config.head_dim;
        let dummy_v = vec![0.0f32; kv_dim];

        // User message.
        let start = self.cache.l1.cache.len();
        let (keys, n) = self.proj.encode_kv(user_text, start);
        for key in &keys {
            self.cache.append_to_l1(key, &dummy_v);
        }
        self.cache.record_span(start, self.cache.l1.cache.len(), user_text.to_string(), Role::User, Some(turn_id), None);
        total += n;

        // Assistant response.
        let start = self.cache.l1.cache.len();
        let (keys, n) = self.proj.encode_kv(assistant_text, start);
        for key in &keys {
            self.cache.append_to_l1(key, &dummy_v);
        }
        self.cache.record_span(start, self.cache.l1.cache.len(), assistant_text.to_string(), Role::Assistant, Some(turn_id), None);
        total += n;

        total
    }

    /// Query the memory using tiered retrieval across all tiers.
    ///
    /// As a side effect, computes and stores L1 attention entropy and chunk
    /// scores. These are used by `consolidate()` to decide *when* and *what*
    /// to evict.
    pub fn query(&mut self, text: &str, top_k: usize) -> Vec<Memory> {
        if self.cache.l1.cache.is_empty() && self.cache.l2.cache.is_empty() && self.cache.l3.cache.is_empty() {
            return vec![];
        }

        // Project query to Q vectors.
        let seq_len = self.cache.l1.cache.len();
        let q = self.proj.encode_query(text, seq_len);
        let n_tokens = self.proj.token_count(text);
        let n_heads = self.proj.config.n_heads;

        // Compute L1 attention entropy (the "drowsiness signal").
        if self.cache.l1.cache.len() > 1 {
            let entropy = retrieve::attention_entropy(&q, n_tokens, n_heads, &self.cache.l1.cache);
            self.cache.set_last_entropy(entropy);

            // Score chunks for intelligent eviction.
            let chunk_size = self.cache.config.chunk_size;
            self.last_chunk_scores = retrieve::score_chunks(
                &q, n_tokens, n_heads, &self.cache.l1.cache, chunk_size,
            );
        }

        // Run tiered retrieval.
        let results = tiered_retrieve::tiered_retrieve(&self.cache, &q, n_tokens, n_heads, top_k);

        results
            .into_iter()
            .map(|r| Memory {
                text: r.text,
                role: Role::System, // tier results don't carry original role yet
                turn_id: r.turn_id,
                score: r.score,
                tier: r.tier,
            })
            .collect()
    }

    /// Run consolidation if needed. Returns a report of what was migrated.
    ///
    /// Uses chunk scores from the last query to pick the eviction target:
    /// - Sleep-triggered: evict the lowest-scoring chunk (most noise).
    /// - Pressure-triggered: evict the oldest chunk (FIFO).
    pub fn consolidate(&mut self) -> ConsolidationReport {
        let scores = if self.last_chunk_scores.is_empty() {
            None
        } else {
            Some(self.last_chunk_scores.as_slice())
        };
        consolidator::consolidate_with_scores(&mut self.cache, scores)
    }

    /// Cache statistics across all tiers.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            l1_positions: self.cache.l1.cache.len(),
            l1_capacity: self.cache.config.l1_capacity,
            l1_spans: self.cache.l1.map.len(),
            l2_positions: self.cache.l2.cache.len(),
            l2_capacity: self.cache.config.l2_capacity,
            l2_spans: self.cache.l2.map.len(),
            l3_positions: self.cache.l3.cache.len(),
            l3_capacity: self.cache.config.l3_capacity,
            l3_spans: self.cache.l3.map.len(),
            l1_entropy: self.cache.last_entropy(),
            trigger: self.cache.trigger(),
        }
    }

    /// Total tokens across all tiers.
    pub fn cached_tokens(&self) -> usize {
        self.cache.l1.cache.len() + self.cache.l2.cache.len() + self.cache.l3.cache.len()
    }

    /// Number of spans (messages) stored across all tiers.
    pub fn span_count(&self) -> usize {
        self.cache.l1.map.len() + self.cache.l2.map.len() + self.cache.l3.map.len()
    }

    /// Projection config (model dimensions).
    pub fn config(&self) -> &ProjectionConfig {
        &self.proj.config
    }

    /// Whether L1 needs consolidation.
    pub fn needs_consolidation(&self) -> bool {
        self.cache.needs_consolidation()
    }
}

// ---------------------------------------------------------------------------
// ConcurrentEngine — thread-safe engine with background consolidation
// ---------------------------------------------------------------------------

use crate::cache::shared::{ConsolidatorHandle, SharedCache};
use std::sync::Arc;

/// Thread-safe engine with background consolidation.
///
/// All methods take `&self` — mutation happens behind per-tier RwLocks.
/// Queries coexist with consolidation. The consolidator thread runs the
/// three-phase sleep cycle (Drowsy → REM → Wake) in the background.
pub struct ConcurrentEngine {
    /// Q/K projection (immutable, used by caller thread only).
    proj: MiniProjection,
    /// Shared cache with per-tier locking.
    cache: Arc<SharedCache>,
    /// Background consolidator thread handle.
    consolidator: std::sync::Mutex<Option<ConsolidatorHandle>>,
}

impl ConcurrentEngine {
    /// Create from a GGUF model file.
    pub fn from_gguf(model_path: &str, max_positions: usize) -> Result<Self, String> {
        Self::from_gguf_with_config(
            model_path,
            0,
            HierarchicalConfig {
                l1_capacity: max_positions,
                ..Default::default()
            },
        )
    }

    /// Create with explicit config.
    pub fn from_gguf_with_config(
        model_path: &str,
        layer: usize,
        config: HierarchicalConfig,
    ) -> Result<Self, String> {
        let proj = MiniProjection::from_gguf_layer(model_path, layer)
            .map_err(|e| format!("failed to load model: {e}"))?;

        let c = &proj.config;
        let cache = Arc::new(SharedCache::new(config, c.n_kv_heads, c.head_dim, (42, 99)));

        Ok(Self {
            proj,
            cache,
            consolidator: std::sync::Mutex::new(None),
        })
    }

    /// Start the background consolidator thread.
    pub fn start_consolidator(&self) {
        let mut guard = self.consolidator.lock().unwrap();
        if guard.is_none() {
            *guard = Some(ConsolidatorHandle::start(self.cache.clone()));
        }
    }

    /// Stop the background consolidator thread.
    pub fn stop_consolidator(&self) {
        let mut guard = self.consolidator.lock().unwrap();
        if let Some(mut handle) = guard.take() {
            handle.stop();
        }
    }

    /// Ingest text. Write-locks L1 briefly.
    pub fn ingest(&self, text: &str, role: Role) -> usize {
        self.ingest_with_metadata(text, role, None)
    }

    /// Ingest text with metadata. Write-locks L1.
    pub fn ingest_with_metadata(
        &self,
        text: &str,
        role: Role,
        metadata: Option<String>,
    ) -> usize {
        let mut l1 = self.cache.l1.write().unwrap();
        let start_pos = l1.cache.len();

        let (keys, n_tokens) = self.proj.encode_kv(text, start_pos);
        if n_tokens == 0 {
            return 0;
        }

        let kv_dim = self.proj.config.n_kv_heads * self.proj.config.head_dim;
        let dummy_v = vec![0.0f32; kv_dim];
        for key in &keys {
            l1.cache.append_one(key, &dummy_v);
        }

        let end_pos = l1.cache.len();
        let turn_id = Some(l1.map.next_turn_id());
        l1.map.append(start_pos, end_pos, text.to_string(), role, turn_id, metadata);

        n_tokens
    }

    /// Query the memory. Read-locks all tiers.
    ///
    /// Computes entropy + chunk scores and signals the consolidator if needed.
    pub fn query(&self, text: &str, top_k: usize) -> Vec<Memory> {
        let l1_empty;
        let l2_empty;
        let l3_empty;
        {
            l1_empty = self.cache.l1.read().unwrap().cache.is_empty();
            l2_empty = self.cache.l2.read().unwrap().cache.is_empty();
            l3_empty = self.cache.l3.read().unwrap().cache.is_empty();
        }
        if l1_empty && l2_empty && l3_empty {
            return vec![];
        }

        let n_heads = self.proj.config.n_heads;
        let chunk_size = self.cache.config.chunk_size;

        // Project query to Q vectors.
        let seq_len = self.cache.l1.read().unwrap().cache.len();
        let q = self.proj.encode_query(text, seq_len);
        let n_tokens = self.proj.token_count(text);

        // Compute entropy + chunk scores under L1 read lock.
        {
            let l1 = self.cache.l1.read().unwrap();
            if l1.cache.len() > 1 {
                let entropy =
                    retrieve::attention_entropy(&q, n_tokens, n_heads, &l1.cache);
                let chunk_scores =
                    retrieve::score_chunks(&q, n_tokens, n_heads, &l1.cache, chunk_size);
                drop(l1); // release before updating sleep state
                self.cache.update_sleep_state(entropy, chunk_scores);

                // Signal consolidator to check.
                if let Some(handle) = self.consolidator.lock().unwrap().as_ref() {
                    handle.signal_wake();
                }
            }
        }

        // Tiered retrieval — takes read locks on each tier independently.
        let mut results = Vec::new();
        let oversample = top_k * 3;

        // L3.
        {
            let l3 = self.cache.l3.read().unwrap();
            if !l3.cache.is_empty() {
                let scores = retrieve::retrieve(&q, n_tokens, n_heads, &l3.cache);
                let top = scores.top_k(oversample);
                let resolved = l3.map.resolve_top_k(&top);
                for r in resolved.into_iter().take(top_k) {
                    results.push(Memory {
                        text: r.span.text.clone(),
                        score: r.score,
                        tier: Tier::L3,
                        role: Role::System,
                        turn_id: r.span.turn_id,
                    });
                }
            }
        }

        // L2.
        {
            let l2 = self.cache.l2.read().unwrap();
            if !l2.cache.is_empty() {
                let scores = retrieve::retrieve(&q, n_tokens, n_heads, &l2.cache);
                let top = scores.top_k(oversample);
                let resolved = l2.map.resolve_top_k(&top);
                for r in resolved.into_iter().take(top_k) {
                    results.push(Memory {
                        text: r.span.text.clone(),
                        score: r.score,
                        tier: Tier::L2,
                        role: Role::System,
                        turn_id: r.span.turn_id,
                    });
                }
            }
        }

        // L1.
        {
            let l1 = self.cache.l1.read().unwrap();
            if !l1.cache.is_empty() {
                let scores = retrieve::retrieve(&q, n_tokens, n_heads, &l1.cache);
                let top = scores.top_k(oversample);
                let resolved = l1.map.resolve_top_k(&top);
                for r in resolved.into_iter().take(top_k) {
                    results.push(Memory {
                        text: r.span.text.clone(),
                        score: r.score,
                        tier: Tier::L1,
                        role: Role::System,
                        turn_id: r.span.turn_id,
                    });
                }
            }
        }

        results.sort_unstable_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(top_k);
        results
    }

    /// Cache statistics.
    pub fn stats(&self) -> CacheStats {
        let l1 = self.cache.l1.read().unwrap();
        let l2 = self.cache.l2.read().unwrap();
        let l3 = self.cache.l3.read().unwrap();
        CacheStats {
            l1_positions: l1.cache.len(),
            l1_capacity: self.cache.config.l1_capacity,
            l1_spans: l1.map.len(),
            l2_positions: l2.cache.len(),
            l2_capacity: self.cache.config.l2_capacity,
            l2_spans: l2.map.len(),
            l3_positions: l3.cache.len(),
            l3_capacity: self.cache.config.l3_capacity,
            l3_spans: l3.map.len(),
            l1_entropy: self.cache.last_entropy(),
            trigger: self.cache.trigger(),
        }
    }

    /// Projection config.
    pub fn config(&self) -> &ProjectionConfig {
        &self.proj.config
    }

    /// Access the shared cache (for advanced use).
    pub fn shared_cache(&self) -> &Arc<SharedCache> {
        &self.cache
    }
}

impl Drop for ConcurrentEngine {
    fn drop(&mut self) {
        self.stop_consolidator();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN_PATH: &str = "../code-llm/models/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    #[test]
    #[ignore] // Requires GGUF model file
    fn end_to_end_retrieval() {
        let mut engine = Engine::from_gguf(QWEN_PATH, 100_000).unwrap();
        let c = engine.config().clone();
        println!("Model: hidden={}, heads={}, kv_heads={}, head_dim={}",
            c.hidden_dim, c.n_heads, c.n_kv_heads, c.head_dim);

        let docs = &[
            (Role::System, "A buffer makes another agent callable as a tool. The calling agent sends parameters; the system spawns an isolated child pipeline to execute. Buffers enable delegation without direct agent-to-agent communication."),
            (Role::System, "Security is structural, not behavioral. You cannot prompt-inject your way past a dispatch table that does not contain the route you are trying to reach."),
            (Role::System, "The librarian is a Haiku-class model that curates context the way kswapd manages pages — proactively, by relevance, before pressure forces eviction."),
            (Role::System, "Triggers are listener nodes that fire messages rather than handle them. Types: file_watch, timer, cron, event, webhook, custom."),
            (Role::System, "An organism is a YAML file that declares everything an agent needs to run: listeners, prompts, security profiles, tools, and buffer interfaces."),
            (Role::System, "The VDrive is a QCOW2 sandbox. All file operations are jailed to the mounted workspace. Path traversal is validated at the filesystem layer."),
            (Role::System, "Bob is the concierge agent. He reads the user's task and delegates to the right specialist. Bob uses tools: auto to discover all available tools and agents."),
        ];

        let mut total_tokens = 0;
        for (role, text) in docs {
            let n = engine.ingest(text, *role);
            total_tokens += n;
            println!("  Ingested {} tokens: {}...", n, &text[..40.min(text.len())]);
        }

        let stats = engine.stats();
        println!("Total: {} tokens, {} spans", total_tokens, engine.span_count());
        println!("Stats: L1={}/{}, L2={}/{}, L3={}/{}",
            stats.l1_positions, stats.l1_capacity,
            stats.l2_positions, stats.l2_capacity,
            stats.l3_positions, stats.l3_capacity);

        // Query: "What is a buffer?"
        println!("\n--- Query: 'What is a buffer?' ---");
        let results = engine.query("What is a buffer?", 3);
        for (i, m) in results.iter().enumerate() {
            println!("  #{}: [{:?} score={:.4}] {}...", i + 1, m.tier, m.score, &m.text[..60.min(m.text.len())]);
        }
        assert!(!results.is_empty(), "retrieval should return results");
        assert!(results.len() <= 3, "should return at most 3");
        assert!(results[0].score > 0.0, "top result should have positive score");

        // Query: "How does security work?"
        println!("\n--- Query: 'How does security work?' ---");
        let results = engine.query("How does security work?", 3);
        for (i, m) in results.iter().enumerate() {
            println!("  #{}: [{:?} score={:.4}] {}...", i + 1, m.tier, m.score, &m.text[..60.min(m.text.len())]);
        }
        assert!(!results.is_empty());

        // Query: "Tell me about triggers"
        println!("\n--- Query: 'Tell me about triggers' ---");
        let results = engine.query("Tell me about triggers", 3);
        for (i, m) in results.iter().enumerate() {
            println!("  #{}: [{:?} score={:.4}] {}...", i + 1, m.tier, m.score, &m.text[..60.min(m.text.len())]);
        }
        assert!(!results.is_empty());
    }

    #[test]
    #[ignore] // Requires GGUF model file
    fn consolidation_integration() {
        use crate::cache::hierarchical::HierarchicalConfig;

        let config = HierarchicalConfig {
            l1_capacity: 2048,
            l2_capacity: 512,
            l3_capacity: 128,
            chunk_size: 256,
            threshold: 0.5,
            entropy_threshold: 0.85,
            max_span_text: 1024,
        };

        let mut engine = Engine::from_gguf_with_config(QWEN_PATH, 2048, 0, config).unwrap();

        // Ingest enough text to trigger consolidation.
        let paragraphs = &[
            "Buffers enable delegation. The calling agent sends parameters and the system spawns an isolated child pipeline.",
            "Security is structural. Dispatch tables are the enforcement layer, not behavioral rules or prompt engineering.",
            "The librarian curates context proactively, like kswapd manages pages, by relevance before pressure forces eviction.",
            "Triggers are listener nodes. Types include file_watch, timer, cron, event, webhook, and custom triggers.",
            "An organism YAML declares listeners, prompts, security profiles, tools, and buffer interfaces for an agent.",
            "The VDrive is a QCOW2 sandbox with path traversal validated at the filesystem layer for security.",
            "Bob is the concierge agent who delegates to specialists using tools auto for discovery.",
            "The kernel provides WAL-backed durability with thread table, context store, and journal subsystems.",
            "Pipeline orchestration wires listeners to handlers through the XML message bus architecture.",
            "Ghost text completion uses LSP integration for intelligent code suggestions in the editor.",
        ];

        for text in paragraphs {
            engine.ingest(text, Role::System);
            // Consolidate after each ingest if needed.
            if engine.needs_consolidation() {
                let report = engine.consolidate();
                println!("Consolidated: {:?}", report);
            }
        }

        let stats = engine.stats();
        println!("Final stats: L1={}/{} spans={}, L2={}/{} spans={}, L3={}/{} spans={}",
            stats.l1_positions, stats.l1_capacity, stats.l1_spans,
            stats.l2_positions, stats.l2_capacity, stats.l2_spans,
            stats.l3_positions, stats.l3_capacity, stats.l3_spans);

        // L2 should have entries from consolidation.
        assert!(stats.l2_positions > 0 || stats.l1_positions > 0,
            "should have data in L1 or L2");

        // Query should still return results.
        let results = engine.query("What is a buffer?", 5);
        assert!(!results.is_empty(), "should return results after consolidation");
        println!("Query results after consolidation:");
        for (i, m) in results.iter().enumerate() {
            println!("  #{}: [{:?} score={:.4}] {}...",
                i + 1, m.tier, m.score, &m.text[..60.min(m.text.len())]);
        }
    }

    #[test]
    #[ignore] // Requires GGUF model file
    fn latency_benchmark() {
        use std::time::Instant;
        use crate::cache::hierarchical::HierarchicalConfig;

        let config = HierarchicalConfig {
            l1_capacity: 100_000,
            l2_capacity: 10_000,
            l3_capacity: 1_000,
            chunk_size: 512,
            threshold: 0.8,
            entropy_threshold: 0.85,
            max_span_text: 2048,
        };

        let mut engine = Engine::from_gguf_with_config(QWEN_PATH, 100_000, 0, config).unwrap();
        let c = engine.config().clone();
        println!("\n=== Engram Latency Benchmark ===");
        println!("Model: heads={}, kv_heads={}, head_dim={}", c.n_heads, c.n_kv_heads, c.head_dim);

        // --- Corpus: realistic conversation turns ---
        let corpus = &[
            "A buffer makes another agent callable as a tool. The calling agent sends parameters; the system spawns an isolated child pipeline to execute.",
            "Security is structural, not behavioral. You cannot prompt-inject your way past a dispatch table that does not contain the route.",
            "The librarian is a Haiku-class model that curates context proactively, by relevance, before pressure forces eviction.",
            "Triggers are listener nodes that fire messages rather than handle them. Types: file_watch, timer, cron, event, webhook, custom.",
            "An organism is a YAML file that declares everything an agent needs to run: listeners, prompts, security profiles, tools.",
            "The VDrive is a QCOW2 sandbox. All file operations are jailed to the mounted workspace. Path traversal validated at filesystem layer.",
            "Bob is the concierge agent. He reads the user's task and delegates to the right specialist via tools: auto.",
            "The kernel provides WAL-backed durability with thread table, context store, and journal subsystems for crash recovery.",
            "Pipeline orchestration wires listeners to handlers through the XML message bus. Every app on the pipeline is automatically agentic.",
            "Ghost text completion uses LSP integration for intelligent code suggestions. The editor runs in a TUI buffer with focus management.",
            "The relay computer uses TTL-compatible logic levels with mechanical relays. Propagation delay is the main constraint on clock speed.",
            "CNC programming via AgentOS uses WASM sandboxed CAD/CAM tools. The neural KV database navigates huge API surfaces via learned embeddings.",
            "Code-Savant uses AlphaEdit to inject codebase knowledge directly into model weights. Module-level topology, not function-level details.",
            "The three-tier VMM context store uses Active/Shelved/Folded tiers with fold_store eviction. Bounded event bus with lagged subscribers skipped.",
            "TurboQuant achieves 12x compression over f32 KV caches via 3-bit PolarQuant angles plus per-head radius and optional QJL correction.",
            "Memory consolidation runs in three phases: Drowsy (snapshot under read lock), REM (compute centroid, no locks), Wake (fast atomic swap).",
        ];

        // --- Ingest benchmark ---
        let t0 = Instant::now();
        let mut total_tokens = 0usize;
        let n_repeats = 10; // repeat corpus to build up cache
        for _ in 0..n_repeats {
            for text in corpus {
                total_tokens += engine.ingest(text, Role::System);
            }
        }
        let ingest_time = t0.elapsed();
        let ingest_per_token = ingest_time.as_nanos() as f64 / total_tokens as f64;
        println!("\n--- Ingest ---");
        println!("  {} tokens in {:.1}ms ({:.0}ns/token, {:.0} tok/s)",
            total_tokens,
            ingest_time.as_secs_f64() * 1000.0,
            ingest_per_token,
            1e9 / ingest_per_token,
        );
        println!("  L1 positions: {}", engine.stats().l1_positions);

        // --- Query benchmark at various cache sizes ---
        let queries = &[
            "What is a buffer?",
            "How does security work in the pipeline?",
            "Tell me about triggers and event handling",
            "How does TurboQuant compress the KV cache?",
            "What is the relay computer project?",
        ];

        println!("\n--- Query (top_k=3, cache={} positions) ---", engine.stats().l1_positions);
        for query_text in queries {
            let t0 = Instant::now();
            let n_runs = 10;
            let mut results = vec![];
            for _ in 0..n_runs {
                results = engine.query(query_text, 3);
            }
            let avg_us = t0.elapsed().as_micros() as f64 / n_runs as f64;
            let top_text = results.first().map(|r| {
                let t = &r.text;
                &t[..50.min(t.len())]
            }).unwrap_or("(none)");
            println!("  {:.1}ms  \"{}\" → {}...",
                avg_us / 1000.0,
                query_text,
                top_text,
            );
        }

        // --- Entropy ---
        let stats = engine.stats();
        println!("\n--- Sleep Signal ---");
        println!("  L1 entropy: {:.4} (threshold: {})", stats.l1_entropy, 0.85);
        println!("  Trigger: {:?}", stats.trigger);

        // --- Consolidation benchmark ---
        if engine.needs_consolidation() {
            let t0 = Instant::now();
            let report = engine.consolidate();
            let consolidate_time = t0.elapsed();
            println!("\n--- Consolidation ---");
            println!("  {:?} in {:.1}ms",
                report.trigger,
                consolidate_time.as_secs_f64() * 1000.0,
            );
            println!("  L1 drained: {}, L2 added: {}, L1 remaining: {}",
                report.l1_drained, report.l2_added, report.l1_remaining);
        } else {
            // Force a consolidation to measure it anyway.
            println!("\n--- Consolidation (forced) ---");
            let t0 = Instant::now();
            let report = engine.consolidate();
            let consolidate_time = t0.elapsed();
            println!("  {:.1}ms (drained: {}, added: {})",
                consolidate_time.as_secs_f64() * 1000.0,
                report.l1_drained, report.l2_added,
            );
        }

        // --- Post-consolidation query ---
        let stats = engine.stats();
        println!("\n--- Post-consolidation stats ---");
        println!("  L1: {}/{} ({} spans)", stats.l1_positions, stats.l1_capacity, stats.l1_spans);
        println!("  L2: {}/{} ({} spans)", stats.l2_positions, stats.l2_capacity, stats.l2_spans);
        println!("  L3: {}/{} ({} spans)", stats.l3_positions, stats.l3_capacity, stats.l3_spans);

        // --- Memory footprint ---
        println!("\n--- Memory ---");
        let l1_kb = stats.l1_positions as f64 * c.n_kv_heads as f64 * (c.head_dim as f64 / 2.0 + 4.0) / 1024.0;
        let f32_kb = stats.l1_positions as f64 * c.n_kv_heads as f64 * c.head_dim as f64 * 4.0 / 1024.0;
        println!("  L1 compressed: ~{:.1} KB (f32 equivalent: ~{:.1} KB, ~{:.1}x)",
            l1_kb, f32_kb, f32_kb / l1_kb.max(0.001));
    }

    #[test]
    #[ignore] // Requires GGUF model file
    fn rope_vs_no_rope_retrieval() {
        use std::time::Instant;
        use crate::cache::quantized::QuantizedKvCache;

        println!("\n=== Path B: RoPE vs No-RoPE Retrieval ===\n");

        let proj = crate::projection::MiniProjection::from_gguf(QWEN_PATH).unwrap();
        let c = proj.config.clone();
        println!("Model: heads={}, kv_heads={}, head_dim={}\n", c.n_heads, c.n_kv_heads, c.head_dim);

        let corpus: &[(&str, &str)] = &[
            ("buffer", "A buffer makes another agent callable as a tool. The calling agent sends parameters; the system spawns an isolated child pipeline."),
            ("security", "Security is structural, not behavioral. You cannot prompt-inject your way past a dispatch table."),
            ("librarian", "The librarian is a Haiku-class model that curates context proactively, by relevance, before pressure forces eviction."),
            ("triggers", "Triggers are listener nodes that fire messages rather than handle them. Types: file_watch, timer, cron, event, webhook, custom."),
            ("organism", "An organism is a YAML file that declares everything an agent needs to run: listeners, prompts, security profiles, tools."),
            ("vdrive", "The VDrive is a QCOW2 sandbox. All file operations are jailed to the mounted workspace."),
            ("bob", "Bob is the concierge agent. He reads the user's task and delegates to the right specialist via tools: auto."),
            ("relay", "The relay computer uses TTL-compatible logic with mechanical relays. Propagation delay constrains clock speed."),
            ("turboquant", "TurboQuant achieves 12x compression via 3-bit PolarQuant angles plus per-head radius and optional QJL correction."),
            ("consolidation", "Memory consolidation runs in three phases: Drowsy (snapshot), REM (compute centroid), Wake (fast swap)."),
        ];

        let queries: &[(&str, &str)] = &[
            ("What is a buffer?", "buffer"),
            ("How does security work?", "security"),
            ("Tell me about triggers", "triggers"),
            ("What is TurboQuant compression?", "turboquant"),
            ("What is the relay computer?", "relay"),
        ];

        // --- Build WITH RoPE ---
        let mut cache_rope = QuantizedKvCache::with_qjl(c.n_kv_heads, c.head_dim, 100_000, 42, 99);
        let mut map_rope = crate::cache::position_map::PositionMap::new();
        let kv_dim = c.n_kv_heads * c.head_dim;
        let dummy_v = vec![0.0f32; kv_dim];

        for (label, text) in corpus {
            let start = cache_rope.len();
            let (keys, _n) = proj.encode_kv(text, start);
            for key in &keys {
                cache_rope.append_one(key, &dummy_v);
            }
            let end = cache_rope.len();
            let tid = map_rope.next_turn_id();
            map_rope.append(start, end, format!("[{label}] {text}"), Role::System, Some(tid), None);
        }

        // --- Build WITHOUT RoPE ---
        let mut cache_norope = QuantizedKvCache::with_qjl(c.n_kv_heads, c.head_dim, 100_000, 42, 99);
        let mut map_norope = crate::cache::position_map::PositionMap::new();

        for (label, text) in corpus {
            let start = cache_norope.len();
            let (keys, _n) = proj.encode_kv_no_rope(text);
            for key in &keys {
                cache_norope.append_one(key, &dummy_v);
            }
            let end = cache_norope.len();
            let tid = map_norope.next_turn_id();
            map_norope.append(start, end, format!("[{label}] {text}"), Role::System, Some(tid), None);
        }

        println!("Cache: {} positions ({} docs)\n", cache_rope.len(), corpus.len());

        // --- Compare retrieval ---
        println!("{:<40} {:>8} {:>8}  {}", "Query", "RoPE", "NoRoPE", "Winner");
        println!("{}", "-".repeat(90));

        let mut rope_wins = 0;
        let mut norope_wins = 0;

        for (query_text, expected_label) in queries {
            // With RoPE.
            let t0 = Instant::now();
            let q_rope = proj.encode_query(query_text, cache_rope.len());
            let n_tokens = proj.token_count(query_text);
            let scores_rope = crate::retrieve::retrieve(&q_rope, n_tokens, c.n_heads, &cache_rope);
            let time_rope = t0.elapsed();
            let top_rope = scores_rope.top_k(3);
            let resolved_rope = map_rope.resolve_top_k(&top_rope);

            // Without RoPE.
            let t0 = Instant::now();
            let q_norope = proj.encode_query_no_rope(query_text);
            let scores_norope = crate::retrieve::retrieve(&q_norope, n_tokens, c.n_heads, &cache_norope);
            let time_norope = t0.elapsed();
            let top_norope = scores_norope.top_k(3);
            let resolved_norope = map_norope.resolve_top_k(&top_norope);

            // Check which one ranked the expected doc #1.
            let rope_rank = resolved_rope.iter()
                .position(|r| r.span.text.starts_with(&format!("[{expected_label}]")))
                .map(|p| p + 1);
            let norope_rank = resolved_norope.iter()
                .position(|r| r.span.text.starts_with(&format!("[{expected_label}]")))
                .map(|p| p + 1);

            let rope_str = rope_rank.map(|r| format!("#{r}")).unwrap_or("miss".into());
            let norope_str = norope_rank.map(|r| format!("#{r}")).unwrap_or("miss".into());

            let winner = match (rope_rank, norope_rank) {
                (Some(a), Some(b)) if a < b => { rope_wins += 1; "RoPE" },
                (Some(a), Some(b)) if b < a => { norope_wins += 1; "NoRoPE" },
                (Some(_), None) => { rope_wins += 1; "RoPE" },
                (None, Some(_)) => { norope_wins += 1; "NoRoPE" },
                _ => "tie",
            };

            println!("{:<40} {:>8} {:>8}  {}",
                format!("\"{}\" → [{}]", query_text, expected_label),
                rope_str, norope_str, winner);

            // Show top-3 for each.
            println!("  RoPE  top-3: {} ({:.1}ms)",
                resolved_rope.iter().take(3)
                    .map(|r| r.span.text.split(']').next().unwrap_or("?").trim_start_matches('[').to_string())
                    .collect::<Vec<_>>().join(", "),
                time_rope.as_secs_f64() * 1000.0);
            println!("  NoRoPE top-3: {} ({:.1}ms)",
                resolved_norope.iter().take(3)
                    .map(|r| r.span.text.split(']').next().unwrap_or("?").trim_start_matches('[').to_string())
                    .collect::<Vec<_>>().join(", "),
                time_norope.as_secs_f64() * 1000.0);
        }

        println!("\n--- Result: RoPE wins {rope_wins}, NoRoPE wins {norope_wins} ---");

        // --- Entropy comparison ---
        let q_rope = proj.encode_query("random test query", cache_rope.len());
        let q_norope = proj.encode_query_no_rope("random test query");
        let n_tok = proj.token_count("random test query");
        let h_rope = crate::retrieve::attention_entropy(&q_rope, n_tok, c.n_heads, &cache_rope);
        let h_norope = crate::retrieve::attention_entropy(&q_norope, n_tok, c.n_heads, &cache_norope);
        println!("\n--- Entropy (lower = more focused) ---");
        println!("  RoPE:   {:.4}", h_rope);
        println!("  NoRoPE: {:.4}", h_norope);
    }

    #[test]
    #[ignore] // Requires GGUF model file + GPU
    #[cfg(feature = "gpu")]
    fn gpu_vs_cpu_retrieval() {
        use std::time::Instant;
        use crate::retrieve::gpu::GpuRetriever;

        println!("\n=== GPU vs CPU Retrieval Benchmark ===\n");

        // Init GPU.
        let gpu = match GpuRetriever::new() {
            Some(g) => {
                println!("GPU: {}", g.adapter_name);
                g
            }
            None => {
                println!("No GPU available — skipping");
                return;
            }
        };

        let proj = crate::projection::MiniProjection::from_gguf(QWEN_PATH).unwrap();
        let c = proj.config.clone();
        println!("Model: heads={}, kv_heads={}, head_dim={}\n", c.n_heads, c.n_kv_heads, c.head_dim);

        let corpus = &[
            "A buffer makes another agent callable as a tool. The calling agent sends parameters.",
            "Security is structural, not behavioral. Dispatch tables are the enforcement layer.",
            "The librarian curates context proactively, by relevance, before pressure forces eviction.",
            "Triggers are listener nodes that fire messages rather than handle them.",
            "An organism is a YAML declaring listeners, prompts, security profiles, tools.",
            "The VDrive is a QCOW2 sandbox. Path traversal validated at filesystem layer.",
            "Bob is the concierge agent who delegates to specialists.",
            "The relay computer uses TTL-compatible logic with mechanical relays.",
            "TurboQuant achieves 12x compression via 3-bit PolarQuant angles.",
            "Memory consolidation: Drowsy (snapshot), REM (centroid), Wake (swap).",
        ];

        // Build cache with varying sizes.
        for &n_repeats in &[1, 5, 10, 50] {
            let mut cache = crate::cache::quantized::QuantizedKvCache::with_qjl(
                c.n_kv_heads, c.head_dim, 200_000, 42, 99,
            );
            let kv_dim = c.n_kv_heads * c.head_dim;
            let dummy_v = vec![0.0f32; kv_dim];

            for _ in 0..n_repeats {
                for text in corpus {
                    let (keys, _) = proj.encode_kv(text, cache.len());
                    for key in &keys {
                        cache.append_one(key, &dummy_v);
                    }
                }
            }

            let n_pos = cache.len();
            let query_text = "What is a buffer?";
            let n_tokens = proj.token_count(query_text);

            // CPU retrieval.
            let q = proj.encode_query(query_text, cache.len());
            let t0 = Instant::now();
            let n_runs = 3;
            for _ in 0..n_runs {
                let _ = crate::retrieve::retrieve(&q, n_tokens, c.n_heads, &cache);
            }
            let cpu_avg = t0.elapsed().as_secs_f64() * 1000.0 / n_runs as f64;

            // GPU retrieval.
            let t0 = Instant::now();
            for _ in 0..n_runs {
                let _ = crate::retrieve::gpu::gpu_retrieve(&gpu, &q, n_tokens, c.n_heads, &cache);
            }
            let gpu_avg = t0.elapsed().as_secs_f64() * 1000.0 / n_runs as f64;

            let speedup = cpu_avg / gpu_avg;
            println!("{:>6} positions: CPU {:.1}ms, GPU {:.1}ms, speedup {:.1}x",
                n_pos, cpu_avg, gpu_avg, speedup);

            // Verify results match (approximately).
            let cpu_scores = crate::retrieve::retrieve(&q, n_tokens, c.n_heads, &cache);
            let gpu_scores = crate::retrieve::gpu::gpu_retrieve(&gpu, &q, n_tokens, c.n_heads, &cache);
            let cpu_top = cpu_scores.top_k(3);
            let gpu_top = gpu_scores.top_k(3);
            if !cpu_top.is_empty() && !gpu_top.is_empty() {
                // Top position should match (or be close).
                if cpu_top[0].0 != gpu_top[0].0 {
                    println!("  NOTE: top-1 differs: CPU pos={}, GPU pos={} (no QJL on GPU)",
                        cpu_top[0].0, gpu_top[0].0);
                }
            }
        }
    }
}
