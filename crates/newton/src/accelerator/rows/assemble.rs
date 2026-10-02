// SPDX-License-Identifier: MIT

//! Baked rows for the system-matrix assembly: per destination block, the sum of
//! the connection Jacobians landing in `(i, j)`, with the midpoint factors applied
//! as the columns are read, plus the body's mass block on the diagonal.
//!
//! | word | meaning                                                 |
//! |------|---------------------------------------------------------|
//! | 0    | `out` — `Block<T>` slot                                 |
//! | 1    | `mass` — `Block<T>` slot (any live slot when unused)     |
//! | 2    | `mass_scale` — bits of `1.0` on the diagonal, else `0.0` |
//! | 3    | `pose_factor` — `T` slot holding `−half²`               |
//! | 4    | `vel_factor` — `T` slot holding `−half`                 |
//! | 5    | `accumulate`                                            |
//! | 6    | `n_terms`                                               |
//! | 7..  | `jac_slot * 4 + row_end * 2 + col_end`                  |
//!
//! `if i == j` becomes the `mass_scale` multiplier, and rounds past the first
//! carry `0.0` there, so the mass block is added exactly once however many rounds
//! a cell takes.
//!
//! The two midpoint factors are SLOTS, not values. They are the only per-sub-step
//! numbers the assembly needs, and holding them inline made the row depend on the
//! sub-step: every change of `half` reallocated all `m²` rows and dropped the old
//! ones, and `RawMap::remove` walks the free list, so a bulk drop was quadratic —
//! measured at 66 unknowns as a third of the step's entire CPU time. Through slots
//! the row depends on the topology alone and is never rebaked.

use crate::accelerator::row::{ROW, Row};
use crate::integrator::implicit::block::Block;
use crate::integrator::implicit::cache::BlockTerm;
use aristotle::{World, WorldKey};
use bytemuck::Pod;
use peano::prelude::*;
use std::sync::Arc;

const HEAD: usize = 7;
pub(crate) const MAX_TERMS: usize = ROW - HEAD;

/// Rounds for ONE destination block.
pub(crate) fn bake_cell<T: Scalar + Pod>(
    world: &Arc<World>,
    out: &WorldKey<Block<T>>,
    mass: &WorldKey<Block<T>>,
    diagonal: bool,
    pose_factor: &WorldKey<T>,
    vel_factor: &WorldKey<T>,
    terms: &[BlockTerm<T>],
) -> Vec<WorldKey<Row>> {
    let mut map = world.write::<Row>();
    // A cell with no connection terms still needs a row: it must be zeroed (and,
    // on the diagonal, hold the mass block) rather than keep the last step's
    // value.
    let chunks: Vec<&[BlockTerm<T>]> = if terms.is_empty() {
        vec![&[]]
    } else {
        terms.chunks(MAX_TERMS).collect()
    };
    chunks
        .into_iter()
        .enumerate()
        .map(|(round, chunk)| {
            let mut r: Row = [0; ROW];
            r[0] = out.raw_index() as u32;
            r[1] = mass.raw_index() as u32;
            r[2] = f32::to_bits(if diagonal && round == 0 { 1.0 } else { 0.0 });
            r[3] = pose_factor.raw_index() as u32;
            r[4] = vel_factor.raw_index() as u32;
            r[5] = u32::from(round > 0);
            r[6] = chunk.len() as u32;
            for (i, t) in chunk.iter().enumerate() {
                r[HEAD + i] =
                    (t.key.raw_index() as u32) * 4 + (t.row_end as u32) * 2 + t.col_end as u32;
            }
            map.add(r)
        })
        .collect()
}

/// The whole matrix, indexed by ROUND: `out[k]` is round `k` of every destination
/// block, blocks row-major (`i * m + j`). That is the order the dispatch consumes
/// — rounds are sequential, the blocks within one are independent.
pub(crate) fn bake<T: Scalar + Pod>(
    world: &Arc<World>,
    terms: &[Arc<[BlockTerm<T>]>],
    mass: &[WorldKey<Block<T>>],
    out: &[WorldKey<Block<T>>],
    m: usize,
    pose_factor: &WorldKey<T>,
    vel_factor: &WorldKey<T>,
) -> Vec<Arc<[WorldKey<Row>]>> {
    let mut rounds: Vec<Vec<WorldKey<Row>>> = Vec::new();
    for i in 0..m {
        for j in 0..m {
            let cell = bake_cell(
                world,
                &out[i * m + j],
                &mass[i],
                i == j,
                pose_factor,
                vel_factor,
                &terms[i * m + j],
            );
            for (round, key) in cell.into_iter().enumerate() {
                if rounds.len() == round {
                    rounds.push(Vec::new());
                }
                rounds[round].push(key);
            }
        }
    }
    rounds.into_iter().map(Arc::from).collect()
}
