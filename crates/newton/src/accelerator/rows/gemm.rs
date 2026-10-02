// SPDX-License-Identifier: MIT

//! Baked rows for one block matrix product `out = a · b`.
//!
//! | word | meaning                               |
//! |------|---------------------------------------|
//! | 0    | `out` — `Block<T>` slot               |
//! | 1    | `accumulate`                          |
//! | 2    | `n_terms`                             |
//! | 3    | `sign` — bits of `+1.0` or `-1.0`     |
//! | 4..  | pairs `(a_block, b_block * 2 + diag)` |
//!
//! `two_minus` folds the `2I − b` of the Newton–Schulz step into the read: the
//! right factor becomes `sign · b + diag · 2 · I`, so no pass materialises that
//! matrix and the kernel carries no branch.
//!
//! `rows_of[i]` is the sparsity of the LEFT factor — the block columns summed for
//! output row `i`. Pass all of `0..m` for a dense product.
//!
//! The result is indexed by ROUND: `out[k]` holds round `k` of every output cell,
//! cells in row-major order (`i * m + j`). That is the order the dispatch wants —
//! the rounds are sequential (each seeds from the previous one's output) while the
//! cells within a round are independent. Cells can differ in depth when the left
//! factor is sparse, so later rounds are simply shorter.

use crate::accelerator::row::{ROW, Row};
use crate::integrator::implicit::block::Block;
use aristotle::{World, WorldKey};
use bytemuck::Pod;
use peano::prelude::*;
use std::sync::Arc;

const HEAD: usize = 4;
pub(crate) const MAX_TERMS: usize = (ROW - HEAD) / 2;

#[allow(clippy::too_many_arguments)]
pub(crate) fn bake<T: Scalar + Pod>(
    world: &Arc<World>,
    a: &[WorldKey<Block<T>>],
    b: &[WorldKey<Block<T>>],
    out: &[WorldKey<Block<T>>],
    m: usize,
    rows_of: &[Arc<[usize]>],
    two_minus: bool,
) -> Vec<Arc<[WorldKey<Row>]>> {
    let sign = if two_minus { -1.0f32 } else { 1.0f32 };
    let mut map = world.write::<Row>();
    let mut rounds: Vec<Vec<WorldKey<Row>>> = Vec::new();
    for i in 0..m {
        for j in 0..m {
            let ks = &rows_of[i];
            // An empty column list would leave the cell holding the previous
            // product; one zeroing row keeps every cell defined.
            let chunks: Vec<&[usize]> = if ks.is_empty() {
                vec![&[]]
            } else {
                ks.chunks(MAX_TERMS).collect()
            };
            for (round, chunk) in chunks.into_iter().enumerate() {
                let mut r: Row = [0; ROW];
                r[0] = out[i * m + j].raw_index() as u32;
                r[1] = u32::from(round > 0);
                r[2] = chunk.len() as u32;
                r[3] = f32::to_bits(sign);
                for (t, &k) in chunk.iter().enumerate() {
                    r[HEAD + 2 * t] = a[i * m + k].raw_index() as u32;
                    // The `2I` of the fold belongs to the k == j term only.
                    let diag = u32::from(two_minus && k == j);
                    r[HEAD + 2 * t + 1] = (b[k * m + j].raw_index() as u32) * 2 + diag;
                }
                if rounds.len() == round {
                    rounds.push(Vec::new());
                }
                rounds[round].push(map.add(r));
            }
        }
    }
    rounds.into_iter().map(Arc::from).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrator::implicit::block::zero_block;
    use aristotle::World;

    #[test]
    fn the_diagonal_bit_and_sign_encode_the_two_minus_fold() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = 2;
        let blocks: Vec<_> = {
            let mut map = world.write::<Block<f32>>();
            (0..m * m * 3)
                .map(|_| map.add(zero_block::<f32>()))
                .collect()
        };
        let a = blocks[0..4].to_vec();
        let b = blocks[4..8].to_vec();
        let out = blocks[8..12].to_vec();
        let all: Arc<[usize]> = Arc::from(vec![0usize, 1]);
        let rows_of = vec![all.clone(), all.clone()];

        let baked = bake(&world, &a, &b, &out, m, &rows_of, true);

        // Round 0 holds every cell's first round, cells row-major.
        assert_eq!(baked.len(), 1, "two columns fit in one round");
        // Cell (0, 0): term k = 0 == j = 0 sets the diagonal bit.
        let r00 = baked[0][0].read();
        assert_eq!(f32::from_bits(r00[3]), -1.0, "two_minus negates b");
        assert_eq!(r00[5] & 1, 1, "k == j sets the diagonal bit");
        // Cell (0, 1): the k = 0 term has k != j, so the bit is clear.
        let r01 = baked[0][1].read();
        assert_eq!(r01[5] & 1, 0);
    }
}
