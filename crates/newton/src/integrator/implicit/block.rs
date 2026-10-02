// SPDX-License-Identifier: MIT

//! The 6×6 block the Newton system matrix is made of, and the one arithmetic
//! primitive over it.
//!
//! Blocking is an identity, not an approximation: a `6m × 6m` matrix is an
//! `m × m` matrix over the ring of 6×6 matrices, and
//! `C[i][j] = Σ_k A[i][k]·B[k][j]` is ordinary matrix multiplication in that
//! ring. The ring is NOT commutative — the factor order above is fixed.

use peano::prelude::*;

/// One 6×6 block, row-major: `block[row][col]`. Six is the number of degrees of
/// freedom of a rigid body, so one block couples one body's equation to one
/// body's unknown.
pub(crate) type Block<T> = [[T; 6]; 6];

/// The zero block.
pub(crate) const fn zero_block<T: Scalar>() -> Block<T> {
    [[T::ZERO; 6]; 6]
}

/// `acc += a · b`, the inner step of a block matrix product. Non-commutative:
/// `a` is the left factor. Reference implementation — the shipping path builds
/// the product on the GPU, so this only ever backs the oracles in tests.
#[cfg(test)]
pub(crate) fn block_mul_acc<T: Scalar>(acc: &mut Block<T>, a: &Block<T>, b: &Block<T>) {
    for row in 0..6 {
        for col in 0..6 {
            let mut s = acc[row][col];
            for k in 0..6 {
                s += a[row][k] * b[k][col];
            }
            acc[row][col] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flatten an `m × m` table of blocks into a plain `6m × 6m` matrix.
    fn flatten(blocks: &[Block<f32>], m: usize) -> Vec<Vec<f32>> {
        let mut out = vec![vec![0.0; 6 * m]; 6 * m];
        for i in 0..m {
            for j in 0..m {
                let b = &blocks[i * m + j];
                for r in 0..6 {
                    for c in 0..6 {
                        out[i * 6 + r][j * 6 + c] = b[r][c];
                    }
                }
            }
        }
        out
    }

    /// Deterministic pseudo-random fill — no rand dependency, and a fixed seed
    /// means a failure reproduces exactly.
    fn fill(seed: u64, m: usize) -> Vec<Block<f32>> {
        let mut s = seed;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        };
        (0..m * m)
            .map(|_| {
                let mut b = zero_block::<f32>();
                for row in b.iter_mut() {
                    for cell in row.iter_mut() {
                        *cell = next();
                    }
                }
                b
            })
            .collect()
    }

    /// The load-bearing test of this module. A flipped factor order
    /// (`b·a` instead of `a·b`) has the right dimensions and typechecks; only
    /// the value is wrong. So does a transposed index. Both die here.
    #[test]
    fn blocked_product_equals_flat_product() {
        let m = 3;
        let a = fill(1, m);
        let b = fill(2, m);

        let mut c = vec![zero_block::<f32>(); m * m];
        for i in 0..m {
            for j in 0..m {
                let mut acc = zero_block::<f32>();
                for k in 0..m {
                    block_mul_acc(&mut acc, &a[i * m + k], &b[k * m + j]);
                }
                c[i * m + j] = acc;
            }
        }

        let (fa, fb, fc) = (flatten(&a, m), flatten(&b, m), flatten(&c, m));
        let n = 6 * m;
        for i in 0..n {
            for j in 0..n {
                let mut want = 0.0;
                for k in 0..n {
                    want += fa[i][k] * fb[k][j];
                }
                assert!(
                    (fc[i][j] - want).abs() < 1e-12,
                    "blocked != flat at [{i}][{j}]: {} vs {want}",
                    fc[i][j]
                );
            }
        }
    }

    /// Guards the factor order explicitly, in case the test above is ever
    /// weakened: for non-commuting blocks, `a·b` and `b·a` must differ.
    #[test]
    fn block_multiply_is_not_commutative() {
        let a = fill(3, 1)[0];
        let b = fill(4, 1)[0];
        let mut ab = zero_block::<f32>();
        let mut ba = zero_block::<f32>();
        block_mul_acc(&mut ab, &a, &b);
        block_mul_acc(&mut ba, &b, &a);
        let diff: f32 = (0..6)
            .flat_map(|r| (0..6).map(move |c| (r, c)))
            .map(|(r, c)| (ab[r][c] - ba[r][c]).abs())
            .fold(0.0, f32::max);
        assert!(diff > 1e-6, "test blocks commute; pick different ones");
    }
}
