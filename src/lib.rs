//! # Engram — Neural Associative Memory Engine
//!
//! Attention-driven retrieval over compressed KV stores. Uses a small ternary
//! transformer as a **hippocampus** — it doesn't generate text, it attends over
//! a persistent, TurboQuant-compressed KV cache to retrieve contextually
//! relevant history for a primary LLM.
//!
//! ## Architecture
//!
//! ```text
//! User message
//!     ├──► Primary LLM (Anthropic API / Candle+Qwen)
//!     │        awaits retrieved context
//!     │
//!     ├──► Engram (ternary model, single forward pass)
//!     │        │ tokenize → Q projection → bidirectional attention
//!     │        │ over compressed KV cache → top-k relevant positions
//!     │        ▼
//!     │    Retrieved context (relevant prior turns)
//!     │        │
//!     └────────┴──► Primary LLM generates response with
//!                   surgically relevant history
//! ```
//!
//! ## Modules
//!
//! - [`ops`] — Core compression algorithms (PolarQuant, QJL)
//! - [`cache`] — Compressed KV cache storage and persistence
//! - [`retrieve`] — Attention-based retrieval API

pub mod cache;
pub mod ops;
pub mod retrieve;
