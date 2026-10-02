// SPDX-License-Identifier: MIT

//! Baked rows for the block reduction `Σ ‖diag·I − b‖²_F`.
//!
//! | word | meaning                          |
//! |------|----------------------------------|
//! | 0    | `out` — `T` slot for the partial |
//! | 1    | `n_terms`                        |
//! | 2..  | `block_slot * 2 + diag`          |
//!
//! One row folds up to `MAX_TERMS` blocks into one scalar, and `bake` returns a
//! SINGLE round: the rows are independent, and what is left for the host is
//! `⌈blocks/MAX_TERMS⌉` numbers to add — 17 at 45 unknowns, against the `m²`
//! guarded slot reads this replaces. Folding those on the GPU too would cost a
//! dispatch to save a handful of additions.
//!
//! `diag` marks the cells whose target is the identity rather than zero. The
//! caller knows which those are (`i == j`) and the kernel never learns about `m`.

use crate::accelerator::row::{ROW, Row};
use crate::integrator::implicit::block::Block;
use aristotle::{World, WorldKey};
use bytemuck::Pod;
use peano::prelude::*;
use std::sync::Arc;

const HEAD: usize = 2;
pub(crate) const MAX_TERMS: usize = ROW - HEAD;

/// Partial slots a reduction over `n` blocks needs.
pub(crate) fn partials_for(n: usize) -> usize {
    n.div_ceil(MAX_TERMS).max(1)
}

/// `blocks[i]` is folded with `diag(i)` deciding whether the identity is
/// subtracted from it. `partials` must hold `partials_for(blocks.len())` slots.
pub(crate) fn bake<T: Scalar + Pod>(
    world: &Arc<World>,
    blocks: &[WorldKey<Block<T>>],
    diag: impl Fn(usize) -> bool,
    partials: &[WorldKey<T>],
) -> Vec<Arc<[WorldKey<Row>]>> {
    let mut map = world.write::<Row>();
    let mut round = Vec::with_capacity(partials.len());
    for (chunk_idx, chunk) in blocks.chunks(MAX_TERMS).enumerate() {
        let mut r: Row = [0; ROW];
        r[0] = partials[chunk_idx].raw_index() as u32;
        r[1] = chunk.len() as u32;
        for (i, key) in chunk.iter().enumerate() {
            let at = chunk_idx * MAX_TERMS + i;
            r[HEAD + i] = (key.raw_index() as u32) * 2 + u32::from(diag(at));
        }
        round.push(map.add(r));
    }
    // An empty block list still needs a row: the partial must be overwritten with
    // zero rather than keep the previous round's sum.
    if round.is_empty() {
        let mut r: Row = [0; ROW];
        r[0] = partials[0].raw_index() as u32;
        round.push(map.add(r));
    }
    vec![Arc::from(round)]
}
