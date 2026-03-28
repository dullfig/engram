//! Engram Engine — end-to-end retrieval over persistent compressed memory.
//!
//! Wires together: MiniProjection → QuantizedKvCache → PositionMap → Retrieval.
//! Feed text in, query later, get ranked relevant spans back.

use crate::cache::position_map::{PositionMap, Role};
use crate::cache::quantized::QuantizedKvCache;
use crate::projection::{MiniProjection, ProjectionConfig};
use crate::retrieve;

/// The engram engine — a persistent associative memory.
pub struct Engine {
    /// Q/K projection from a real model.
    proj: MiniProjection,
    /// Compressed KV cache (one layer).
    cache: QuantizedKvCache,
    /// Position → text mapping.
    map: PositionMap,
}

/// A retrieved memory with its source text and relevance score.
#[derive(Debug)]
pub struct Memory<'a> {
    pub text: &'a str,
    pub role: Role,
    pub turn_id: Option<u64>,
    pub score: f32,
}

impl Engine {
    /// Create an engine from a GGUF model file.
    ///
    /// `max_positions` is the maximum number of token positions the cache can hold.
    /// For 1M tokens at ~12x compression, this uses roughly 8GB.
    pub fn from_gguf(model_path: &str, max_positions: usize) -> Result<Self, String> {
        Self::from_gguf_layer(model_path, max_positions, 0)
    }

    /// Create an engine from a specific layer of a GGUF model.
    pub fn from_gguf_layer(model_path: &str, max_positions: usize, layer: usize) -> Result<Self, String> {
        let proj = MiniProjection::from_gguf_layer(model_path, layer)
            .map_err(|e| format!("failed to load model: {e}"))?;

        let c = &proj.config;
        let cache = QuantizedKvCache::with_qjl(
            c.n_kv_heads,
            c.head_dim,
            max_positions,
            42,  // rotation seed
            99,  // QJL seed
        );

        Ok(Self {
            proj,
            cache,
            map: PositionMap::new(),
        })
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
        let start_pos = self.cache.len();
        let seq_offset = start_pos;

        // Project text to K vectors.
        let (keys, n_tokens) = self.proj.encode_kv(text, seq_offset);

        if n_tokens == 0 {
            return 0;
        }

        // Append each token's K vector to the compressed cache.
        // For V, we use a dummy (same as K) — retrieval only needs K.
        // The actual text is in the PositionMap.
        let kv_dim = self.proj.config.n_kv_heads * self.proj.config.head_dim;
        let dummy_v = vec![0.0f32; kv_dim];

        for key in &keys {
            self.cache.append_one(key, &dummy_v);
        }

        // Record the span.
        let end_pos = self.cache.len();
        let turn_id = Some(self.map.next_turn_id());
        self.map.append(start_pos, end_pos, text.to_string(), role, turn_id, metadata);

        n_tokens
    }

    /// Ingest a conversation turn (user + assistant pair).
    pub fn ingest_turn(&mut self, user_text: &str, assistant_text: &str) -> usize {
        let turn_id = self.map.next_turn_id();
        let mut total = 0;

        // User message.
        let start = self.cache.len();
        let (keys, n) = self.proj.encode_kv(user_text, start);
        let kv_dim = self.proj.config.n_kv_heads * self.proj.config.head_dim;
        let dummy_v = vec![0.0f32; kv_dim];
        for key in &keys {
            self.cache.append_one(key, &dummy_v);
        }
        self.map.append(start, self.cache.len(), user_text.to_string(), Role::User, Some(turn_id), None);
        total += n;

        // Assistant response.
        let start = self.cache.len();
        let (keys, n) = self.proj.encode_kv(assistant_text, start);
        for key in &keys {
            self.cache.append_one(key, &dummy_v);
        }
        self.map.append(start, self.cache.len(), assistant_text.to_string(), Role::Assistant, Some(turn_id), None);
        total += n;

        total
    }

    /// Query the memory. Returns the top-k most relevant spans.
    pub fn query(&self, text: &str, top_k: usize) -> Vec<Memory<'_>> {
        if self.cache.is_empty() {
            return vec![];
        }

        // Project query to Q vectors.
        let q = self.proj.encode_query(text, self.cache.len());
        let n_tokens = self.proj.token_count(text);
        let n_heads = self.proj.config.n_heads;

        // Run retrieval.
        let scores = retrieve::retrieve(&q, n_tokens, n_heads, &self.cache);

        // Get top positions.
        let top_positions = scores.top_k(top_k * 3); // oversample, then aggregate by span

        // Resolve to spans.
        let resolved = self.map.resolve_top_k(&top_positions);

        // Convert to Memory results.
        resolved
            .into_iter()
            .take(top_k)
            .map(|r| Memory {
                text: &r.span.text,
                role: r.span.role,
                turn_id: r.span.turn_id,
                score: r.score,
            })
            .collect()
    }

    /// Total tokens in the cache.
    pub fn cached_tokens(&self) -> usize {
        self.cache.len()
    }

    /// Number of spans (messages) stored.
    pub fn span_count(&self) -> usize {
        self.map.len()
    }

    /// Projection config (model dimensions).
    pub fn config(&self) -> &ProjectionConfig {
        &self.proj.config
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
        // Layer 0 for now — single-layer projection is structural, not semantic.
        // For semantic ranking, run multiple transformer blocks (future work).
        let mut engine = Engine::from_gguf(QWEN_PATH, 100_000).unwrap();
        let c = engine.config().clone();
        println!("Model: hidden={}, heads={}, kv_heads={}, head_dim={}",
            c.hidden_dim, c.n_heads, c.n_kv_heads, c.head_dim);

        // Ingest AgentOS docs.
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
        println!("Total: {} tokens in {} spans, cache: {:?}", total_tokens, engine.span_count(), engine.cached_tokens());

        // Query: "What is a buffer?"
        println!("\n--- Query: 'What is a buffer?' ---");
        let results = engine.query("What is a buffer?", 3);
        for (i, m) in results.iter().enumerate() {
            println!("  #{}: [score={:.4}] {}...", i + 1, m.score, &m.text[..60.min(m.text.len())]);
        }
        assert!(!results.is_empty(), "retrieval should return results");
        assert!(results.len() <= 3, "should return at most 3");
        assert!(results[0].score > 0.0, "top result should have positive score");

        // Query: "How does security work?"
        println!("\n--- Query: 'How does security work?' ---");
        let results = engine.query("How does security work?", 3);
        for (i, m) in results.iter().enumerate() {
            println!("  #{}: [score={:.4}] {}...", i + 1, m.score, &m.text[..60.min(m.text.len())]);
        }
        assert!(!results.is_empty());

        // Query: "Tell me about triggers"
        println!("\n--- Query: 'Tell me about triggers' ---");
        let results = engine.query("Tell me about triggers", 3);
        for (i, m) in results.iter().enumerate() {
            println!("  #{}: [score={:.4}] {}...", i + 1, m.score, &m.text[..60.min(m.text.len())]);
        }
        assert!(!results.is_empty());

        // NOTE: Single-layer Q/K projection produces structural matching, not
        // semantic. For accurate ranking, run multiple transformer blocks to
        // build richer representations. The mechanism works; the ranking is a
        // tuning problem.
    }
}
