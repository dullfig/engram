//! Compressed KV cache storage and persistence.
//!
//! - [`quantized`] — `QuantizedKvCache`: TurboQuant-compressed KV storage
//! - [`store`] — `CacheStore` trait and backends for persisting per-user caches
//! - [`position_map`] — Maps cache positions back to source text and metadata

pub mod position_map;
pub mod quantized;
pub mod store;
