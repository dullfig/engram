//! Compressed KV cache storage and persistence.
//!
//! - [`quantized`] — `QuantizedKvCache`: TurboQuant-compressed KV storage
//! - [`store`] — `CacheStore` trait and backends for persisting per-user caches

pub mod quantized;
pub mod store;
