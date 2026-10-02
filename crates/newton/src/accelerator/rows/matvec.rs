// SPDX-License-Identifier: MIT

//! Baked rows for the Newton correction `dv_i = Σ_j x[i*m + j] · rhs[j]`.
//!
//! | word | meaning                       |
//! |------|-------------------------------|
//! | 0    | `out` — `Twist<T>` slot       |
//! | 1    | `accumulate`                  |
//! | 2    | `n_terms`                     |
//! | 3..  | pairs `(x_block, rhs_wrench)` |
//!
//! `bake` returns one entry per ROUND: `out[k]` is round `k` of every body, bodies
//! in `order` index order. Rounds must go out in order (each seeds from the
//! previous one's output); rows within a round are independent.

use crate::accelerator::row::{ROW, Row};
use crate::integrator::implicit::block::Block;
use aristotle::{World, WorldKey};
use bytemuck::Pod;
use clifford::pga3::{Twist, Wrench};
use peano::prelude::*;
use std::sync::Arc;

const HEAD: usize = 3;
/// Two words per term.
pub(crate) const MAX_TERMS: usize = (ROW - HEAD) / 2;

pub(crate) fn bake<T: Scalar + Pod>(
    world: &Arc<World>,
    x: &[WorldKey<Block<T>>],
    rhs: &[WorldKey<Wrench<T>>],
    out: &[WorldKey<Twist<T>>],
    m: usize,
) -> Vec<Arc<[WorldKey<Row>]>> {
    let mut map = world.write::<Row>();
    let columns: Vec<usize> = (0..m).collect();
    let mut rounds: Vec<Vec<WorldKey<Row>>> = Vec::new();
    for i in 0..m {
        for (round, chunk) in columns.chunks(MAX_TERMS).enumerate() {
            let mut r: Row = [0; ROW];
            r[0] = out[i].raw_index() as u32;
            r[1] = u32::from(round > 0);
            r[2] = chunk.len() as u32;
            for (k, &j) in chunk.iter().enumerate() {
                r[HEAD + 2 * k] = x[i * m + j].raw_index() as u32;
                r[HEAD + 2 * k + 1] = rhs[j].raw_index() as u32;
            }
            if rounds.len() == round {
                rounds.push(Vec::new());
            }
            rounds[round].push(map.add(r));
        }
    }
    rounds.into_iter().map(Arc::from).collect()
}
