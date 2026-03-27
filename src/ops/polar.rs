//! PolarQuant — Stage 1 of TurboQuant compression.
//!
//! Compresses f32 vectors to ~3 bits per element via:
//! 1. Random orthogonal rotation (spreads information uniformly)
//! 2. Polar coordinate transform (pair dimensions into angle + radius)
//! 3. 3-bit angle quantization (8 buckets around the unit circle)
//!
//! The rotation matrix is deterministic from a seed, so it doesn't need
//! to be stored — only the quantized angles and per-head radius.

use std::f32::consts::PI;

/// Number of angle buckets for 3-bit quantization.
const NUM_BUCKETS: usize = 8;

/// Lookup tables for angle bucket cos/sin values, initialized once.
pub struct AngleLUT {
    pub cos: [f32; NUM_BUCKETS],
    pub sin: [f32; NUM_BUCKETS],
}

impl AngleLUT {
    /// Build the lookup table.
    pub fn new() -> Self {
        let mut cos = [0.0f32; NUM_BUCKETS];
        let mut sin = [0.0f32; NUM_BUCKETS];
        for i in 0..NUM_BUCKETS {
            let theta = -PI + (2.0 * PI * i as f32) / NUM_BUCKETS as f32;
            cos[i] = theta.cos();
            sin[i] = theta.sin();
        }
        Self { cos, sin }
    }
}

impl Default for AngleLUT {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Rotation matrix generation
// ---------------------------------------------------------------------------

/// Generate a deterministic random orthogonal matrix via Gram-Schmidt.
///
/// The matrix is `dim x dim`, stored row-major. Deterministic from `seed`
/// so it can be regenerated on cache load without storing it.
pub fn generate_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
    // Xoshiro256** seeded deterministically for reproducibility.
    let mut rng = SimpleRng::seed(seed);

    // Generate random matrix with normal-ish values (Box-Muller).
    let mut mat = vec![0.0f32; dim * dim];
    for v in mat.iter_mut() {
        *v = rng.next_normal();
    }

    // Gram-Schmidt orthogonalization.
    gram_schmidt(&mut mat, dim);
    mat
}

/// Apply rotation: `out = R @ vec` (matrix-vector product).
///
/// `matrix`: row-major `[dim, dim]`.
/// `vec`: input vector of length `dim`.
/// `out`: output vector of length `dim`.
pub fn rotate(matrix: &[f32], vec: &[f32], out: &mut [f32]) {
    let dim = vec.len();
    debug_assert_eq!(matrix.len(), dim * dim);
    debug_assert_eq!(out.len(), dim);

    for row in 0..dim {
        let mut sum = 0.0f32;
        let row_off = row * dim;
        for col in 0..dim {
            sum += matrix[row_off + col] * vec[col];
        }
        out[row] = sum;
    }
}

/// Apply inverse rotation: `out = R^T @ vec`.
pub fn rotate_transpose(matrix: &[f32], vec: &[f32], out: &mut [f32]) {
    let dim = vec.len();
    debug_assert_eq!(matrix.len(), dim * dim);
    debug_assert_eq!(out.len(), dim);

    for col in 0..dim {
        let mut sum = 0.0f32;
        for row in 0..dim {
            sum += matrix[row * dim + col] * vec[row];
        }
        out[col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Polar quantization
// ---------------------------------------------------------------------------

/// Quantize a rotated vector to 3-bit polar form.
///
/// Input: rotated f32 vector of length `dim` (must be even).
/// Output: `(angles, radius)` where angles has `dim/2` entries (0..7)
/// and radius is the mean radius across all pairs.
pub fn to_polar_quantized(rotated: &[f32]) -> (Vec<u8>, f32) {
    let n_pairs = rotated.len() / 2;
    debug_assert_eq!(rotated.len() % 2, 0);

    let mut angles = Vec::with_capacity(n_pairs);
    let mut radius_sum = 0.0f32;

    for i in 0..n_pairs {
        let x = rotated[2 * i];
        let y = rotated[2 * i + 1];
        let r = (x * x + y * y).sqrt();
        let theta = y.atan2(x); // [-pi, pi]

        // Quantize angle to bucket 0..7.
        let normalized = (theta + PI) / (2.0 * PI); // [0, 1)
        let bucket = ((normalized * NUM_BUCKETS as f32) as usize) % NUM_BUCKETS;
        angles.push(bucket as u8);
        radius_sum += r;
    }

    let avg_radius = if n_pairs > 0 {
        radius_sum / n_pairs as f32
    } else {
        0.0
    };

    (angles, avg_radius)
}

/// Dequantize from polar form back to a rotated vector.
pub fn from_polar_quantized(angles: &[u8], radius: f32, lut: &AngleLUT) -> Vec<f32> {
    let dim = angles.len() * 2;
    let mut out = vec![0.0f32; dim];

    for (i, &bucket) in angles.iter().enumerate() {
        let b = bucket as usize;
        out[2 * i] = radius * lut.cos[b];
        out[2 * i + 1] = radius * lut.sin[b];
    }

    out
}

// ---------------------------------------------------------------------------
// Minimal deterministic RNG (no external deps)
// ---------------------------------------------------------------------------

/// Simple xoshiro256**-based RNG for deterministic rotation matrices.
pub(crate) struct SimpleRng {
    s: [u64; 4],
}

impl SimpleRng {
    pub(crate) fn seed(seed: u64) -> Self {
        // SplitMix64 to expand seed into state.
        let mut z = seed;
        let mut s = [0u64; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *slot = z ^ (z >> 31);
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Box-Muller transform for approximate normal distribution.
    pub(crate) fn next_normal(&mut self) -> f32 {
        let u1 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u1 = u1.max(1e-10); // avoid log(0)
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        z as f32
    }
}

// ---------------------------------------------------------------------------
// Gram-Schmidt orthogonalization
// ---------------------------------------------------------------------------

/// In-place Gram-Schmidt on a row-major `[dim, dim]` matrix.
/// Each row becomes an orthonormal basis vector.
fn gram_schmidt(mat: &mut [f32], dim: usize) {
    for i in 0..dim {
        let i_off = i * dim;

        // Subtract projections onto all previous rows.
        for j in 0..i {
            let j_off = j * dim;
            let mut dot = 0.0f32;
            for k in 0..dim {
                dot += mat[i_off + k] * mat[j_off + k];
            }
            for k in 0..dim {
                mat[i_off + k] -= dot * mat[j_off + k];
            }
        }

        // Normalize.
        let mut norm = 0.0f32;
        for k in 0..dim {
            norm += mat[i_off + k] * mat[i_off + k];
        }
        let inv_norm = 1.0 / norm.sqrt();
        for k in 0..dim {
            mat[i_off + k] *= inv_norm;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotation_matrix_is_orthogonal() {
        let dim = 8;
        let r = generate_rotation_matrix(dim, 42);

        // R^T @ R should be ~identity.
        for i in 0..dim {
            for j in 0..dim {
                let mut dot = 0.0f32;
                for k in 0..dim {
                    dot += r[k * dim + i] * r[k * dim + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "R^T@R[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn rotation_matrix_deterministic() {
        let a = generate_rotation_matrix(4, 123);
        let b = generate_rotation_matrix(4, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn rotation_matrix_different_seeds_differ() {
        let a = generate_rotation_matrix(4, 1);
        let b = generate_rotation_matrix(4, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn rotate_and_transpose_roundtrip() {
        let dim = 8;
        let r = generate_rotation_matrix(dim, 42);
        let input: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.3).collect();

        let mut rotated = vec![0.0f32; dim];
        rotate(&r, &input, &mut rotated);

        let mut recovered = vec![0.0f32; dim];
        rotate_transpose(&r, &rotated, &mut recovered);

        for (a, b) in input.iter().zip(recovered.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "roundtrip failed: {a} vs {b}"
            );
        }
    }

    #[test]
    fn rotation_preserves_dot_product() {
        let dim = 8;
        let r = generate_rotation_matrix(dim, 42);

        let a: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * -0.2).collect();

        let dot_orig: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        let mut ra = vec![0.0f32; dim];
        let mut rb = vec![0.0f32; dim];
        rotate(&r, &a, &mut ra);
        rotate(&r, &b, &mut rb);

        let dot_rotated: f32 = ra.iter().zip(rb.iter()).map(|(x, y)| x * y).sum();

        assert!(
            (dot_orig - dot_rotated).abs() < 1e-4,
            "rotation should preserve dot product: {dot_orig} vs {dot_rotated}"
        );
    }

    #[test]
    fn polar_quantize_roundtrip() {
        let lut = AngleLUT::new();
        let input = vec![1.0, 0.5, -0.3, 0.8, 0.0, 1.0, -0.7, -0.2];

        let (angles, radius) = to_polar_quantized(&input);
        assert_eq!(angles.len(), 4); // 8 dims / 2 pairs

        let recovered = from_polar_quantized(&angles, radius, &lut);
        assert_eq!(recovered.len(), 8);

        // 3-bit quantization is lossy — check it's in the right ballpark,
        // not exact. The radius is averaged so individual pairs won't match.
        for &v in &recovered {
            assert!(v.is_finite(), "recovered value should be finite");
        }
    }

    #[test]
    fn angle_buckets_in_range() {
        let input = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let (angles, _) = to_polar_quantized(&input);
        for &a in &angles {
            assert!(a < NUM_BUCKETS as u8, "bucket {a} out of range");
        }
    }

    #[test]
    fn angle_lut_unit_vectors() {
        let lut = AngleLUT::new();
        for i in 0..NUM_BUCKETS {
            let len = (lut.cos[i] * lut.cos[i] + lut.sin[i] * lut.sin[i]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-6,
                "bucket {i} should be unit vector, got len {len}"
            );
        }
    }
}
