// SPDX-License-Identifier: MIT

//! Baked rows for the block copy `dst = src`.
//!
//! | word | meaning                  |
//! |------|--------------------------|
//! | 0    | `dst` — `Block<T>` slot  |
//! | 1    | `src` — `Block<T>` slot  |
//!
//! One row per block, and `bake` returns a single round: there is no reduction,
//! so nothing is carried between blocks and every copy is its own invocation.

use crate::accelerator::row::{ROW, Row};
use crate::integrator::implicit::block::Block;
use aristotle::{World, WorldKey};
use bytemuck::Pod;
use peano::prelude::*;
use std::sync::Arc;

/// `dst[i] ← src[i]`, pairwise. The two slices must be the same length.
pub(crate) fn bake<T: Scalar + Pod>(
    world: &Arc<World>,
    dst: &[WorldKey<Block<T>>],
    src: &[WorldKey<Block<T>>],
) -> Vec<Arc<[WorldKey<Row>]>> {
    assert_eq!(dst.len(), src.len(), "block copy over mismatched arrays");
    let mut map = world.write::<Row>();
    let round: Vec<WorldKey<Row>> = dst
        .iter()
        .zip(src)
        .map(|(d, s)| {
            let mut r: Row = [0; ROW];
            r[0] = d.raw_index() as u32;
            r[1] = s.raw_index() as u32;
            map.add(r)
        })
        .collect();
    vec![Arc::from(round)]
}
