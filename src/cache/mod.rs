//! Compressed KV cache storage and persistence.
//!
//! - [`quantized`] — `QuantizedKvCache`: TurboQuant-compressed KV storage
//! - [`store`] — `CacheStore` trait and backends for persisting per-user caches
//! - [`position_map`] — Maps cache positions back to source text and metadata
//! - [`hierarchical`] — Three-tier cache (L1/L2/L3) with consolidation
//! - [`consolidator`] — Tier migration and compaction
//! - [`tiered_retrieve`] — Gated cascade retrieval across tiers

pub mod consolidator;
pub mod hierarchical;
pub mod position_map;
pub mod quantized;
pub mod shared;
pub mod store;
pub mod tiered_retrieve;
