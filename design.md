# Engram — Neural Associative Memory Engine

*TurboRecall: attention-driven retrieval over compressed KV stores*

**Date:** 2026-03-26
**Source:** [turboquant.net](https://turboquant.net) (Google Research)
**Repository:** `C:\src\engram`
**Upstream inference crate:** `agentos/crates/bitnet`

---

## Problem

The bitnet crate has 1.58-bit ternary weights (16x smaller than f32), but the
KV cache is **full f32**. During autoregressive generation, the cache becomes
the dominant memory consumer — often larger than the model weights themselves.

Current KV cache (`layers/kv_cache.rs`):
```rust
pub struct KvCache {
    k_cache: Vec<f32>,  // [max_seq_len, n_kv_heads * head_dim] as flat f32
    v_cache: Vec<f32>,  // same
    // ...
}
```

**Example: 22-layer model, 4 KV heads, head_dim=64, max_seq=2048**
- KV cache: 22 layers x 2 buffers x 2048 x 256 x 4 bytes = **92 MB**
- Model weights (ternary): roughly **~10-20 MB** for a 2B parameter model

The cache is 5-10x larger than the weights. This defeats the purpose of running
a 1.58-bit model on edge hardware.

---

## What TurboQuant Does

Two-stage KV cache compression — no training, no calibration, fully online:

### Stage 1: PolarQuant

1. **Random rotation**: multiply K/V vectors by a fixed random orthogonal
   matrix R (head_dim x head_dim). This spreads information uniformly across
   dimensions, eliminating outlier channels that break naive quantization.

2. **Polar coordinate transform**: convert each pair of rotated dimensions
   (x, y) into polar form (r, theta). The radius r concentrates tightly
   (low variance), so only the angle theta needs fine-grained quantization.

3. **Quantize angles to 3-bit** (8 buckets around the unit circle). The radius
   gets a coarse 1-2 bit encoding or is approximated by a per-head constant.

Net effect: 32-bit floats per dimension -> ~3 bits per dimension, with the
distortion approaching the information-theoretic limit.

### Stage 2: QJL (Quantized Johnson-Lindenstrauss)

A 1-bit residual correction that provides **unbiased inner-product estimation**.
After PolarQuant compresses the cache, QJL stores sign bits of a random
projection of the quantization residual. When computing Q*K dot products, the
QJL correction term is added back, keeping attention scores accurate without
full dequantization.

### Combined Result

| Metric           | Value                         |
|------------------|-------------------------------|
| Bits per element | ~3 (down from 32)             |
| Memory reduction | ~10.7x per cache element      |
| Quality          | Matches f32 on LongBench      |
| Overhead         | One rotation matvec per append |
| Requirements     | No training, no calibration   |

---

## Current Architecture (What Exists)

Files involved:

```
crates/bitnet/src/
  layers/
    kv_cache.rs      -- KvCache + ModelKvCache (f32 storage)
    attention.rs     -- MultiHeadAttention::forward_cached() uses KvCache
  ops/
    quantize.rs      -- absmax 8-bit activation quantization (reusable patterns)
```

### KvCache API surface (kv_cache.rs)

```rust
KvCache::new(n_kv_heads, head_dim, max_seq_len) -> Self
KvCache::append(&mut self, keys: &[f32], values: &[f32])
KvCache::key_at(&self, pos: usize, kv_head: usize) -> &[f32]  // returns [head_dim]
KvCache::value_at(&self, pos: usize, kv_head: usize) -> &[f32] // returns [head_dim]
KvCache::keys(&self) -> &[f32]      // flat slice, all cached positions
KvCache::values(&self) -> &[f32]    // flat slice, all cached positions
KvCache::clear(&mut self)
KvCache::memory_bytes(&self) -> usize
```

### How attention.rs uses the cache (forward_cached, ~line 262-359)

1. Project new tokens through Q/K/V BitLinear projections
2. Apply RoPE to Q and new K
3. **`cache.append(&k_new, &v_new)`** — store new K/V as f32
4. For each Q head, for each Q position:
   - Loop over all cached positions s=0..attend_len:
     - **`cache.key_at(s, kv_h)`** — read K vector, compute dot product with Q
   - Softmax over scores
   - Weighted sum: loop over s again:
     - **`cache.value_at(s, kv_h)`** — read V vector, accumulate
5. Output projection

The hot path is steps 4's inner loops — every cached position is read per query
token. This is where compressed storage + fast dot-product estimation pays off.

---

## Proposed Design: QuantizedKvCache

### Storage Layout

```rust
/// 3-bit quantized KV cache using TurboQuant (PolarQuant + QJL).
pub struct QuantizedKvCache {
    // -- PolarQuant compressed storage --
    /// Quantized angle indices for K cache.
    /// Each element is 3 bits (stored packed: 8 angles per 3 bytes, or
    /// simpler: Vec<u8> with values 0..7, optimize packing later).
    /// Shape: [max_seq_len, n_kv_heads, head_dim/2] (pairs of dims)
    k_angles: Vec<u8>,

    /// Quantized angle indices for V cache (same layout).
    v_angles: Vec<u8>,

    /// Per-position, per-head radius scale for K.
    /// Shape: [max_seq_len, n_kv_heads]
    k_radius: Vec<f32>,

    /// Per-position, per-head radius scale for V.
    v_radius: Vec<f32>,

    // -- QJL correction bits --
    /// 1-bit sign of random projection of quantization residual (K).
    /// Packed as bits: ceil(head_dim / 8) bytes per (position, head).
    k_qjl_signs: Vec<u8>,

    /// Same for V.
    v_qjl_signs: Vec<u8>,

    // -- Fixed random matrices (generated from seed) --
    /// Orthogonal rotation matrix for PolarQuant.
    /// Shape: [head_dim, head_dim]. One per cache instance (shared across positions).
    rotation_matrix: Vec<f32>,

    /// Random projection vectors for QJL correction.
    /// Shape: [n_qjl_projections, head_dim].
    qjl_projection: Vec<f32>,

    // -- Dimensions --
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    len: usize,

    // -- Precomputed angle->unit-vector lookup (8 entries for 3-bit) --
    angle_cos: [f32; 8],  // cos(2*pi*i/8) for i=0..7
    angle_sin: [f32; 8],  // sin(2*pi*i/8) for i=0..7
}
```

### Key Operations

#### Append (write path)

Called once per new token during decode. Cost is bounded and small.

```
fn append(&mut self, keys: &[f32], values: &[f32]):
    for each new position, for each kv_head:
        kv_vec = keys[pos, head]           // f32 [head_dim]

        // 1. Rotate: spread information uniformly
        rotated = rotation_matrix @ kv_vec  // matvec: head_dim x head_dim

        // 2. Pair dimensions and convert to polar coordinates
        for i in 0..head_dim/2:
            x = rotated[2*i]
            y = rotated[2*i + 1]
            r = sqrt(x*x + y*y)
            theta = atan2(y, x)            // range [-pi, pi]

            // 3. Quantize angle to 3-bit bucket (0..7)
            bucket = round((theta + pi) / (2*pi) * 8) % 8
            store bucket in k_angles

            // Accumulate radius for per-head scale
            radius_sum += r

        // 4. Store per-head average radius
        k_radius[pos, head] = radius_sum / (head_dim / 2)

        // 5. QJL correction: compute residual, project, store signs
        reconstructed = dequantize_polar(angles, radius)
        residual = rotated - reconstructed
        for each qjl projection vector p:
            sign_bit = (dot(residual, p) >= 0) as u1
            pack into k_qjl_signs
```

#### Dot product (read path — replaces key_at + manual dot)

Instead of dequantizing K back to f32 and then dotting with Q, we can compute
the dot product **directly** in compressed domain. But the simpler v1 approach
is dequantize-on-read:

```
fn key_at_dequant(&self, pos: usize, kv_head: usize) -> Vec<f32>:
    radius = k_radius[pos, kv_head]
    for i in 0..head_dim/2:
        bucket = k_angles[pos, head, i]
        x = radius * angle_cos[bucket]
        y = radius * angle_sin[bucket]
        rotated[2*i] = x
        rotated[2*i+1] = y

    // Inverse rotation to get back to original space
    original = rotation_matrix^T @ rotated

    // QJL correction (optional, improves accuracy)
    correction = qjl_decode(k_qjl_signs[pos, head], qjl_projection)
    original += rotation_matrix^T @ correction

    return original
```

**V1 (simple):** Dequantize on read, drop into existing attention loop unchanged.
**V2 (fast):** Compute Q*K dot product in rotated domain without materializing
the full f32 vector. Since rotation is orthogonal, `dot(Q, K) = dot(R@Q, R@K)`,
so rotate Q once and dot against compressed K directly.

### Integration into attention.rs

#### V1 — Drop-in replacement (minimal changes)

Make `QuantizedKvCache` implement the same API as `KvCache`, but `key_at` /
`value_at` return owned `Vec<f32>` instead of `&[f32]`. This requires changing
the attention loop to not borrow the cache across iterations:

```rust
// attention.rs — forward_cached, inner loop change

// BEFORE (borrows cache):
let k_vec = cache.key_at(s, kv_h);

// AFTER (owned, dequantized on the fly):
let k_vec = cache.key_at_dequant(s, kv_h);
```

Or better, use a trait:

```rust
pub trait KvStore {
    fn append(&mut self, keys: &[f32], values: &[f32]);
    fn dot_key(&self, pos: usize, kv_head: usize, query: &[f32]) -> f32;
    fn weighted_value_sum(&self, kv_head: usize, weights: &[f32], out: &mut [f32]);
    fn len(&self) -> usize;
    fn clear(&mut self);
    fn memory_bytes(&self) -> usize;
}
```

This lets attention code work with either `KvCache` (f32) or
`QuantizedKvCache` (3-bit) through the same interface, and lets the quantized
version compute dot products without full dequantization.

#### V2 — Fused attention kernel

For maximum performance, fuse the dequantize + dot product:

```rust
impl QuantizedKvCache {
    /// Compute dot(query, cached_key[pos, head]) without full dequantization.
    ///
    /// Since R is orthogonal: dot(q, k) = dot(Rq, Rk).
    /// Rk is stored as polar angles + radius. Rq is computed once per query.
    fn dot_key_fast(&self, pos: usize, kv_head: usize, rotated_query: &[f32]) -> f32 {
        let radius = self.k_radius[pos * self.n_kv_heads + kv_head];
        let mut sum = 0.0f32;

        for i in 0..self.head_dim / 2 {
            let bucket = self.k_angles[/* index */] as usize;
            let rq_x = rotated_query[2 * i];
            let rq_y = rotated_query[2 * i + 1];
            // dot contribution = radius * (rq_x * cos(theta) + rq_y * sin(theta))
            sum += rq_x * self.angle_cos[bucket] + rq_y * self.angle_sin[bucket];
        }
        sum *= radius;

        // Add QJL correction term
        sum += self.qjl_correction(pos, kv_head, rotated_query);
        sum
    }
}
```

This avoids materializing the full f32 K vector entirely. The inner loop is
just a table lookup + 2 multiplies + 1 add per dimension pair.

---

## Generating the Rotation Matrix

The rotation matrix R must be orthogonal (preserves dot products). Standard
approach: generate a random matrix from a seeded RNG, then QR-decompose it.

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn generate_rotation_matrix(head_dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = head_dim;

    // Generate random Gaussian matrix
    let mut mat = vec![0.0f32; n * n];
    for v in mat.iter_mut() {
        // Box-Muller or use rand_distr::Normal
        *v = sample_normal(&mut rng);
    }

    // QR decomposition (Gram-Schmidt) to get orthogonal Q
    gram_schmidt_inplace(&mut mat, n);
    mat
}
```

This is computed once at cache creation time. For head_dim=64, it's a 64x64
matrix = 16 KB — trivial. The seed should be fixed (e.g., derived from
layer index) so results are reproducible.

**Note:** We could also use a structured random rotation (Hadamard + random
sign flips) which is O(n log n) to apply instead of O(n^2) for the dense
matvec. For head_dim=64 the dense matvec is only 4096 FMAs — probably not
worth the complexity for v1, but worth considering if profiling shows the
rotation is a bottleneck during prefill of long prompts.

---

## QJL Correction Detail

QJL stores the sign of random projections of the quantization error:

```
residual = R @ original - dequantize(quantized)    // quantization error in rotated space
for j in 0..n_projections:
    bit_j = sign(dot(residual, projection[j]))     // 1 bit per projection
```

To correct a dot product `dot(q, k)`:

```
correction = (1 / n_projections) * sum_j(
    sign_bit_j * |dot(residual_estimate, projection[j])| * dot(rotated_q, projection[j])
)
```

The paper shows that even a small number of projection vectors (16-32) gives
meaningful correction. Each adds only 1 bit per position per head, so 32
projections = 4 bytes per (position, head, K-or-V).

**For v1:** Skip QJL entirely. PolarQuant alone at 3-bit gives most of the
benefit. Add QJL as a refinement in v2 if quality isn't sufficient.

---

## Memory Savings

### Per-position, per-head storage

| Component       | f32 baseline | PolarQuant 3-bit | PolarQuant + QJL |
|-----------------|-------------|------------------|------------------|
| K or V vector   | 64 x 4B = 256B | 32 angles x 3bit + 4B radius = 16B | 16B + 4B QJL = 20B |
| Compression     | 1x          | **16x**          | **12.8x**        |

### Full model (22 layers, 4 KV heads, head_dim=64, max_seq=2048)

| Config           | KV cache size | Notes |
|------------------|--------------|-------|
| f32 (current)    | 92 MB        | 22 x 2 x 2048 x 256 x 4 |
| PolarQuant 3-bit | ~5.8 MB      | angles + radius per head |
| PolarQuant + QJL | ~7.2 MB      | + 32 projection sign bits |
| Rotation matrices| ~0.35 MB     | 22 layers x 16KB each (one-time) |

**Total with TurboQuant: ~7.5 MB vs 92 MB current = 12x reduction.**

The KV cache drops from being the dominant allocation to being smaller than
the model weights. This is exactly what's needed for edge/mobile deployment
of ternary models.

---

## Implementation Plan

### Phase 1 — PolarQuant only (no QJL)

1. **`ops/polar.rs`** (new file)
   - `generate_rotation_matrix(head_dim, seed) -> Vec<f32>`
   - `rotate(matrix: &[f32], vec: &[f32], out: &mut [f32])` (matvec)
   - `rotate_transpose(matrix: &[f32], vec: &[f32], out: &mut [f32])` (R^T @ v)
   - `to_polar_quantized(rotated: &[f32]) -> (Vec<u8>, f32)` (angles + radius)
   - `from_polar_quantized(angles: &[u8], radius: f32, head_dim: usize) -> Vec<f32>`
   - Gram-Schmidt orthogonalization (or use Householder for numerical stability)

2. **`layers/quantized_kv_cache.rs`** (new file)
   - `QuantizedKvCache` struct as described above (skip QJL fields for now)
   - Implement same logical API as `KvCache`
   - `append()`: rotate + polar quantize + store
   - `key_at_dequant()` / `value_at_dequant()`: reconstruct f32 on the fly
   - `dot_key_fast()`: fused dot product in rotated domain
   - `ModelQuantizedKvCache` wrapper (Vec of per-layer caches)

3. **`layers/attention.rs` changes**
   - Add `forward_cached_quantized()` method or make `forward_cached()` generic
     over a `KvStore` trait
   - Wire `dot_key_fast()` into the attention score loop
   - Wire `weighted_value_sum()` for the value aggregation

4. **Tests**
   - Roundtrip: quantize -> dequantize, check error is small relative to magnitude
   - Dot product preservation: `|dot(q, k_original) - dot(q, k_dequantized)| < epsilon`
   - Rotation orthogonality: `R^T @ R ≈ I`
   - Full attention output: compare `forward_cached` (f32) vs quantized version,
     measure max/mean error across a batch of random inputs
   - Memory reporting: verify `memory_bytes()` is ~12x smaller

### Phase 2 — QJL correction

5. **`ops/qjl.rs`** (new file)
   - `generate_projections(n_proj, head_dim, seed) -> Vec<f32>`
   - `encode_signs(residual: &[f32], projections: &[f32]) -> Vec<u8>` (packed bits)
   - `correction_dot(signs: &[u8], projections: &[f32], query: &[f32]) -> f32`

6. Integrate into `QuantizedKvCache::append()` and `dot_key_fast()`

7. **Benchmark:** measure perplexity / LongBench quality with and without QJL
   to see if the extra bits are worth it for ternary-sourced KV values

### Phase 3 — Optimizations

8. **Bit-pack angles**: 3 bits x 32 pairs = 96 bits = 12 bytes per (pos, head)
   instead of 32 bytes with u8-per-angle. Saves another ~40% on angle storage.

9. **SIMD dot_key_fast**: the inner loop (lookup cos/sin, 2 FMAs per pair) is
   very SIMD-friendly. AVX2 version: process 4 angle pairs per 256-bit register.

10. **Structured rotation** (Hadamard + random signs): O(n log n) apply instead
    of O(n^2). Matters for prefill where we rotate many tokens at once.

---

## Persistent Session Cache (Single-User Mode)

The ternary engine serves one user at a time — no batched requests, no
concurrent sessions. This means the KV cache should **not** be cleared between
turns. It represents the full conversation state.

### Current behavior

`KvCache::clear()` is all-or-nothing. After clearing, the next turn must
re-process the entire conversation history through all layers to rebuild K/V
projections — wasting all prior compute.

### Desired behavior

Keep the cache hot across turns. User sends a new message → tokenize → feed
only the new tokens through `forward_cached()` → cache already holds all prior
context. Only clear on explicit new-conversation.

### When you hit max_seq_len

Instead of clearing the entire cache, use VMM-style relevance eviction:

```rust
impl QuantizedKvCache {
    /// Evict a contiguous range of positions and compact the cache.
    /// Use for removing old, low-relevance turns.
    fn evict_range(&mut self, start: usize, end: usize);

    /// Compress a position's K/V to a shelved entry and remove from active cache.
    /// The shelved entry can be stored in a retrieval index for later recall.
    fn shelve(&mut self, pos: usize) -> ShelvedEntry;

    /// Decompress and re-insert a previously shelved entry.
    /// Used when the retrieval index scores a shelved entry as relevant
    /// to the current query.
    fn recall(&mut self, entry: ShelvedEntry, insert_pos: usize);
}
```

This turns the KV cache into the **active tier** of a two-tier memory system:

- **Active tier** (QuantizedKvCache): positions the model attends to, 3-bit
  compressed, bounded by max_seq_len
- **Shelved tier** (retrieval index): evicted positions, more aggressively
  compressed or stored as embedding vectors for similarity lookup

Eviction policy options:
- **Attention-weighted**: track cumulative attention scores per position (H2O
  style). Evict positions that consistently receive low attention.
- **Recency + pinning**: keep the first N tokens (system prompt / original task)
  pinned, sliding window for the rest, but with promotion for high-attention
  positions.
- **Hybrid**: pin first message, keep recent window, evict middle positions
  with lowest cumulative attention scores.

The shelved tier's retrieval check runs once per new query — score shelved
entries against the new Q vectors, promote any that score higher than the
lowest-attention active entry. This is the "associative recall" mechanism:
new input triggers retrieval of relevant older context, just like human
working memory.

### Impact on RoPE

Evicting positions from the middle of the cache creates gaps in the position
sequence. RoPE encodes absolute position, so compacting the cache would shift
position indices. Options:
- **Re-encode**: after eviction, recompute RoPE for remaining positions (expensive)
- **NTK-aware RoPE**: use relative position encoding that's robust to gaps
- **Accept gaps**: some models tolerate small position discontinuities without
  significant quality loss — worth benchmarking

---

## Ternary Model as Hippocampus (Associative Memory Engine)

The ternary model's 3 tok/s generation speed makes it unsuitable as the
conversational interface — that's real Bob's job (Anthropic API). But the
ternary model doesn't need to *talk*. It needs to *attend*.

### Core Insight

Use the ternary transformer not for text generation but as a **dedicated
attention machine** — a learned associative retrieval index that tells real
Bob which parts of conversation history are relevant to the current input.

### How It Works

```
┌──────────────────────────────────────────────────────┐
│  User sends message                                  │
│       │                                              │
│       ├──► Real Bob (Anthropic API)                  │
│       │       │                                      │
│       │       │  awaits retrieved context            │
│       │       ▼                                      │
│       ├──► Ternary Model (local, single forward pass)│
│       │       │                                      │
│       │       │  1. Tokenize new message             │
│       │       │  2. Forward through transformer      │
│       │       │  3. DON'T generate — read attention  │
│       │       │     scores against KV cache          │
│       │       │  4. Return top-k cached positions    │
│       │       │     ranked by attention weight       │
│       │       │                                      │
│       │       ▼                                      │
│       │  Retrieved context (relevant prior turns)    │
│       │       │                                      │
│       │       ▼                                      │
│       └──► Real Bob generates response with          │
│            surgically relevant history               │
└──────────────────────────────────────────────────────┘
```

### Why This Is Better Than Embedding Search

| Approach           | What it captures                              |
|--------------------|-----------------------------------------------|
| TF-IDF             | Lexical overlap (keyword matching)            |
| Embedding model    | Semantic similarity (per-chunk, independent)  |
| **Ternary attention** | **Contextual relevance** (given the full conversation so far, which past positions matter for this new input) |

An embedding model encodes each chunk independently — it can't know that
turn 12 is relevant to turn 847 because of something established in turn 5.
Attention can. The KV cache holds the accumulated context of the entire
conversation, and the attention scores reflect how the *whole history*
interacts with the new query.

### Bidirectional Attention for Retrieval Mode

When the ternary model acts as a hippocampus, **causal masking is wrong**.

In generative mode, causal masking prevents token t from attending to future
positions — necessary for autoregressive generation. But in retrieval mode
we're not generating. We're asking: "given this new input, which cached
positions are relevant?" The query should attend to ALL cached positions
regardless of ordering.

```rust
impl MultiHeadAttention {
    /// Retrieval-mode forward pass: bidirectional attention over KV cache.
    ///
    /// Unlike forward_cached(), NO causal mask is applied. The query tokens
    /// attend to every cached position. Returns attention scores per
    /// (query_head, cached_position) for relevance ranking.
    ///
    /// Does NOT append to cache — the query is ephemeral, only the
    /// attention scores matter.
    pub fn retrieve(
        &self,
        input: &[f32],
        seq_len: usize,
        cache: &KvCache,  // or &QuantizedKvCache
    ) -> AttentionScores {
        // 1. Project query tokens through Q (skip K/V — we're not caching)
        // 2. Apply RoPE to Q
        // 3. Score Q against ALL cached K (no causal mask)
        // 4. Return raw attention weights (pre-softmax or post-softmax)
        //    for external ranking
    }
}

/// Attention scores from a retrieval pass.
pub struct AttentionScores {
    /// Per-head attention weights: [n_heads, query_len, cached_len]
    scores: Vec<f32>,
    n_heads: usize,
    query_len: usize,
    cached_len: usize,
}

impl AttentionScores {
    /// Aggregate across heads and query positions to get a single
    /// relevance score per cached position.
    pub fn top_k(&self, k: usize) -> Vec<(usize, f32)>;

    /// Return scores for a specific layer (for multi-layer aggregation).
    pub fn per_head(&self) -> Vec<&[f32]>;
}
```

This is a critical distinction:
- **Generative mode** (`forward_cached`): causal mask ON, appends to cache,
  produces logits for next-token prediction
- **Retrieval mode** (`retrieve`): causal mask OFF, read-only on cache,
  produces attention scores for relevance ranking

The retrieval pass can also aggregate attention across multiple layers.
Deeper layers tend to capture more semantic relationships, while early layers
capture syntactic patterns. A weighted combination (or just using the last
layer) gives the best relevance signal.

### Performance Characteristics

- **Not generating tokens**: a single forward pass through the transformer
  computes attention maps. This is a fixed cost (not per-output-token), and
  it's fast — the expensive part of generation is the autoregressive loop.
  A single pass for a short input (one user message) is milliseconds to low
  single-digit seconds even at 3 tok/s generation speed.

- **KV cache is persistent**: conversation turns accumulate in the cache
  across the full session. With TurboQuant at 3-bit, thousands of turns fit
  in ~8 MB. The cache is never cleared, only evicted by relevance.

- **Runs in parallel**: the ternary forward pass can execute concurrently
  with the API request to real Bob. Start both when the user message arrives.
  If the attention results come back before Bob's response, inject them as
  additional context. If Bob is faster, send without — the retrieval is
  best-effort enrichment, not a hard dependency.

### Integration with VMM Tiers

The three-tier VMM from agentos-kernel maps directly:

| VMM Tier   | Ternary Memory Role | Storage | Retrieval |
|------------|-------------------|---------|-----------|
| **Active** | Recent turns | Full f32 in KV cache | Attended every forward pass |
| **Shelved**| Older turns | TurboQuant 3-bit in KV cache | Attended but cheaper (compressed dot products) |
| **Folded** | Very old turns | Evicted from cache; only their final attention score fingerprint survives | Recalled into shelved tier if fingerprint matches new query pattern |

The "attention score fingerprint" for a folded entry is the vector of attention
weights it received on its last forward pass before eviction. This acts as a
compact descriptor of *what kind of queries found this position relevant*. When
a new query's attention pattern is similar, the folded entry is a candidate for
recall.

### What the Ternary Model Becomes

It's not the AI. It's the **hippocampus** — the brain region that doesn't
think or speak, it encodes experiences and retrieves them associatively when
triggered by current context. Real Bob (Anthropic API) is the prefrontal
cortex — the part that reasons, plans, and produces language.

This reframes the entire bitnet crate's purpose:
- Old role: budget local LLM for when you have no API key
- New role: **persistent associative memory engine** running alongside the
  primary LLM, compressing and indexing conversation history, retrieving
  contextually relevant fragments on demand

The 3 tok/s speed is irrelevant because it never generates. The ternary
weight compression is a feature because it means the model itself is tiny
(~10-20 MB), leaving almost all device memory for the KV cache — i.e., for
*memories*.

---

## Multi-Tenant Deployment (BHS Concierge Use Case)

The hippocampus architecture generalizes from single-user (AgentOS on-device)
to multi-tenant SaaS. Target example: AI concierge for the Barbershop Harmony
Society (~20-30K members), where every member experiences an AI that "knows"
them personally.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Member sends message                                        │
│       │                                                      │
│       ├──► Load personal KV cache from storage (by user_id)  │
│       │       ~8 MB from SSD/S3, sub-second                  │
│       │                                                      │
│       ├──► Ternary model forward pass (shared, stateless)    │
│       │       Model weights: ~10-20 MB (loaded once)         │
│       │       KV cache: user's personal cache (swapped in)   │
│       │       Output: attention scores → top-k relevant facts│
│       │                                                      │
│       ├──► Real LLM (API) generates response                 │
│       │       System prompt + retrieved personal facts       │
│       │       Stateless — all personalization from retrieval │
│       │                                                      │
│       ├──► Append new turn to user's KV cache                │
│       │                                                      │
│       └──► Persist updated cache back to storage             │
└──────────────────────────────────────────────────────────────┘
```

### Capacity Planning

| Resource | Calculation | Total |
|----------|------------|-------|
| Model weights (shared) | 1 ternary model, ~10-20 MB | 20 MB |
| Per-user KV cache | TurboQuant 3-bit, ~8 MB per user | — |
| 30K users cold storage | 30,000 × 8 MB | **240 GB** (fits one SSD) |
| Hot cache (concurrent) | ~100 concurrent users × 8 MB | **800 MB RAM** |
| Forward pass cost | 1 pass per message, no generation | ~50-200ms |

A single server with 32 GB RAM and a 1 TB SSD could serve the entire society.
The ternary model weights stay resident. User caches page in from SSD on
demand, LRU-evict from RAM when not in use.

### Cache Serialization

The `QuantizedKvCache` needs to be serializable for persistence:

```rust
impl QuantizedKvCache {
    /// Serialize to a compact binary format for storage.
    /// Header: version(u8), n_kv_heads(u16), head_dim(u16), len(u32)
    /// Body: angle data, radius data, QJL signs (all packed)
    /// Does NOT include rotation matrix (regenerated from seed).
    fn serialize(&self, writer: &mut impl Write) -> io::Result<()>;

    /// Deserialize from storage. Rotation matrix regenerated from seed.
    fn deserialize(reader: &mut impl Read, seed: u64) -> io::Result<Self>;

    /// Byte size of the serialized form (for storage budget tracking).
    fn serialized_size(&self) -> usize;
}
```

The rotation matrix and QJL projections are deterministic from the seed —
don't store them, regenerate on load. Only the compressed angles, radii,
and sign bits need to persist. This keeps the per-user storage minimal.

### Storage Backend Trait

```rust
/// Pluggable storage for user KV caches.
pub trait CacheStore {
    /// Load a user's KV cache. Returns None if user has no history.
    fn load(&self, user_id: &str) -> Result<Option<QuantizedKvCache>>;

    /// Persist a user's KV cache.
    fn save(&self, user_id: &str, cache: &QuantizedKvCache) -> Result<()>;

    /// Delete a user's cache (account deletion / GDPR).
    fn delete(&self, user_id: &str) -> Result<()>;

    /// List all user IDs with stored caches.
    fn list_users(&self) -> Result<Vec<String>>;
}
```

Implementations:
- **`FileCacheStore`**: one file per user in a directory, good for dev/small deployments
- **`S3CacheStore`**: object storage, good for production multi-region
- **`SqliteCacheStore`**: single-file database, good for moderate scale (30K users fits easily)

### Cache Lifecycle

```
New member joins
    → Empty cache created on first message
    → Cache grows as conversations accumulate

Active member
    → Cache loaded on message, persisted after response
    → Stays in RAM LRU for repeat visitors (hot path)

Inactive member returns after months
    → Cache loaded from cold storage (SSD/S3)
    → Attention mechanism finds old facts relevant to new query
    → "Welcome back, how did the Cedar Rapids contest go?"

Account deletion
    → cache_store.delete(user_id) — no residual personal data
    → GDPR-friendly: all personal memory in one deletable blob
```

### Why Not a Traditional User Profile Database?

| Approach | Store | Retrieve | Limitation |
|----------|-------|----------|------------|
| SQL profile table | Manually extract facts, choose schema | Query by field | Must anticipate what matters. "Sings baritone" is a column, but "mentioned knee surgery before contest" isn't |
| Vector DB (RAG) | Chunk + embed conversations | Cosine similarity | Each chunk encoded independently. Can't capture cross-turn relevance |
| **Neural KV cache** | Store every turn, compressed | **Attention scores** | Context-dependent retrieval. Same cache, different facts surface depending on *what the user is asking right now* |

The neural KV store doesn't require a schema. You don't decide what facts
to extract about a member. You store everything. When they ask about contest
dates, attention surfaces their chapter and competition history. When they
ask about learning a new part, attention surfaces their voice range and
repertoire discussions. Same cache, different retrieval — driven by the
query, not by a predetermined schema.

### Multi-Model Evolution Path

The ternary hippocampus isn't locked to a specific conversational LLM:

- **Phase 1 (now):** Anthropic API as the conversational LLM, ternary model
  as hippocampus
- **Phase 2:** Swap Anthropic for a self-hosted Qwen via Candle on GPU —
  reduces per-message cost, keeps the same hippocampus
- **Phase 3:** Fine-tune the conversational model on BHS domain knowledge
  (contest rules, music theory, chapter management). The hippocampus layer
  is unaffected — it's retrieving personal facts, not domain knowledge

The hippocampus and the conversational model are fully decoupled. Upgrade
either independently.

---

## Open Questions

1. **Ternary-sourced KV distributions**: K and V come from BitLinear (ternary
   weights, 8-bit absmax activations). The resulting distributions may have
   lower dynamic range than standard models. This could mean:
   - PolarQuant works even better (less outlier mass to rotate away)
   - Or: values cluster near zero, making polar coordinates less efficient
   - **Action:** Histogram K/V values from a real bitnet model run, check distribution shape

2. **Radius encoding**: The paper uses per-head average radius. For ternary-
   sourced KV, the per-pair radius variance might be low enough that a single
   scalar suffices. If not, consider 4-bit per-pair radius (still much cheaper
   than f32).

3. **Value cache strategy**: TurboQuant's dot-product-in-compressed-domain trick
   works cleanly for K (we compute Q*K dot products). For V, we need weighted
   sums (`sum(weight_s * V_s)`), which is harder to do without dequantization.
   Options:
   - Dequantize V on the fly (still saves memory, just not compute)
   - Keep V at 8-bit instead of 3-bit (simpler, still 4x savings over f32)
   - Batch-dequantize only the top-k positions by attention weight

4. **Dependency situation**: Gram-Schmidt / QR needs no external crate for
   head_dim=64. We can write a simple in-crate version. RNG: `rand` +
   `rand_chacha` are likely already in the dependency tree; if not, a simple
   xoshiro from a seed would work.

5. **Interaction with GQA**: With n_kv_heads < n_heads, the cache is already
   smaller. TurboQuant still helps proportionally — 4 KV heads at 3-bit is
   still 12x smaller than 4 KV heads at f32.

6. **KV cache versioning / model drift**: The K/V projections in the cache
   are a function of the model weights. If you update or swap the ternary
   model, every persisted cache becomes invalid — the stored K/V vectors
   were computed by different weight matrices, so attention scores against
   them are meaningless.

   Options:
   - **Version tag**: store a model hash in the cache header. On load, if
     the hash doesn't match the current model, invalidate and rebuild.
     Simple, correct, but loses all history on model update.
   - **Migration pass**: on model update, iterate all cached positions
     through the new model's K/V projections to recompute. Expensive
     (full re-encoding of every user's cache) but preserves history.
     Could be done lazily — re-encode a user's cache on their next visit.
   - **Stable retriever**: commit to a specific ternary model as the
     hippocampus and never change it. The conversational LLM can be
     swapped freely (it's decoupled), but the retriever's weights are
     frozen. This is probably the right answer — the retriever doesn't
     need to be SOTA, it needs to be *stable*.
   - **Distilled retriever**: if the retriever is distilled from the
     conversational model, changing the teacher means the student drifts.
     A generic small ternary model (not distilled from anything specific)
     avoids this coupling entirely.

7. **Memory-mapped cache persistence**: For the single-server BHS deployment
   (or AgentOS on-device), mmap is attractive:

   ```rust
   /// Memory-mapped cache store. Each user's cache is a file that can be
   /// mapped directly into the process address space — zero-copy load,
   /// OS handles paging, dirty pages flushed on unmap.
   pub struct MmapCacheStore {
       base_dir: PathBuf,
   }
   ```

   Benefits:
   - **Zero-copy load**: no deserialization, the file IS the data structure
   - **OS-managed paging**: hot users stay in RAM, cold users page out to SSD
     automatically — the OS LRU replaces our manual LRU
   - **Crash safety**: `msync` or `madvise` for durability guarantees

   Constraints:
   - Requires a fixed, stable binary layout (no pointers, no Vec — flat
     arrays only). The `QuantizedKvCache` angle/radius storage is already
     flat, so this maps well.
   - Platform-specific (Unix mmap vs Windows MapViewOfFile). Rust `memmap2`
     crate abstracts this.
   - File size must be pre-allocated to max capacity, or grown with
     `ftruncate` + remap on append. For a fixed max_seq_len this is fine.

   This also naturally solves the "100 concurrent users in 800 MB RAM"
   problem — you mmap all 30K files and let the OS figure out which pages
   are hot. No manual cache management needed.

8. **RoPE in retrieval mode**: When using bidirectional attention (no causal
   mask), RoPE still encodes position. The query tokens get positions
   following the cache, which means the model sees a distance between the
   query and old cached positions. This might bias attention toward recent
   positions even in retrieval mode.

   Options:
   - **Position zero for query**: assign the query tokens position 0 so
     they have equal RoPE distance to all cached positions. Breaks the
     temporal signal but maximizes retrieval fairness.
   - **Relative position reset**: assign query tokens the same position as
     each cached position they attend to (effectively computing attention
     without positional bias). Requires modifying the attention loop.
   - **Keep it**: the positional bias toward recency might actually be
     a useful inductive bias — more recent turns *should* be slightly
     preferred, all else being equal. Benchmark both ways.

---

## Candidate Ternary Models for Prototyping

All native 1.58-bit (trained ternary from scratch, not post-quantized). All
load through the existing GGUF/TQ1_0/TQ2_0 path in the bitnet crate.

### 1. Microsoft BitNet b1.58-2B-4T (flagship reference)

- **Size:** ~2.4B params, ~1.2 GB GGUF
- **Trained on:** 4T tokens
- **Format:** `ggml-model-i2_s.gguf` (I2S compatible)
- **Quality:** Competitive with FP16 2B baselines on commonsense, math
- **HF:** `microsoft/bitnet-b1.58-2B-4T-gguf`
- **Best for:** Quality baseline, already tested with the crate

### 2. TII Falcon-Edge Series (recommended for hippocampus)

- **Sizes:** 1B and 3B variants, base + instruct-tuned
- **Architecture:** Native BitNet, designed for edge fine-tuning
- **Format:** Ternary GGUF (TQ-compatible)
- **Released:** May 2025
- **HF:** `tiiuae/falcon-edge-series` collection
- **Best for:** The hippocampus use case. The 1B instruct variant is ideal:
  - Small = fast forward pass for retrieval (we don't need generation quality)
  - Instruction-tuned = richer semantic space in K/V projections (learned
    to distinguish intent, not just predict next token)
  - Edge-optimized = designed for exactly the constrained environment we're
    targeting

### 3. Other options

- **1bitLLM/bitnet_b1_58-3B** — community 3B ternary, GGUF available
- **qvac/fabric-llm-finetune-bitnet XL** — larger TQ2_0 variant

### Which model for which role?

| Role | Recommended | Why |
|------|------------|-----|
| Hippocampus (retrieval) | Falcon-Edge 1B instruct | Smallest, fastest forward pass, instruction tuning gives better K/V semantics |
| Conversational (AgentOS Bob) | Anthropic API (real Bob) | Quality ceiling for conversation, no local compute |
| Conversational (BHS, cost-sensitive) | Candle + Qwen 7B on GPU | Self-hosted, good quality, lower per-message cost |
| Prototyping / benchmarking | Microsoft 2B | Reference model, known-good, already validated |

### Note on instruction tuning and retrieval quality

An instruction-tuned model's K/V projections encode richer semantic
distinctions than a base model's. The base model learned "what word comes
next"; the instruct model learned "what is the user asking for." This means
attention scores in retrieval mode will better reflect *intent-level*
similarity, not just lexical similarity.

This is why Falcon-Edge 1B instruct is preferred over the larger Microsoft
2B base for the hippocampus — it's not about parameter count, it's about
what the attention heads learned to attend *to*.

---

## References

- [TurboQuant](https://turboquant.net) — Google Research, 2024
- [Falcon-Edge Series](https://huggingface.co/collections/tiiuae/falcon-edge-series-6804fd13344d6d8a8fa71130) — TII, May 2025
- [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) — Microsoft, April 2025
- Current KV cache: `crates/bitnet/src/layers/kv_cache.rs`
- Current attention: `crates/bitnet/src/layers/attention.rs`
- Existing quantization patterns: `crates/bitnet/src/ops/quantize.rs`
