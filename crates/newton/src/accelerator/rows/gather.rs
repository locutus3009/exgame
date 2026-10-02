// SPDX-License-Identifier: MIT

//! Baked rows for the wrench gather.
//!
//! | word | meaning                                     |
//! |------|---------------------------------------------|
//! | 0    | `out` — `Wrench<T>` slot                    |
//! | 1    | `external` — `Wrench<T>` slot               |
//! | 2    | `accumulate` (0 on the first round, else 1) |
//! | 3    | `n_terms`                                   |
//! | 4..  | `pair_slot * 2 + end`                       |
//!
//! A body with more incident connections than fit in one row is baked as several
//! rows; the caller dispatches them in order, awaiting between rounds, and every
//! row past the first seeds from `out` instead of `external`.

use crate::GatherTerm;
use crate::accelerator::row::{ROW, Row};
use aristotle::{World, WorldKey};
use bytemuck::Pod;
use clifford::pga3::Wrench;
use peano::prelude::*;
use std::sync::Arc;

/// Header words before the first term.
const HEAD: usize = 4;
/// Terms per row.
pub(crate) const MAX_TERMS: usize = ROW - HEAD;

pub(crate) fn bake<T: Scalar + Pod>(
    world: &Arc<World>,
    out: &WorldKey<Wrench<T>>,
    external: &WorldKey<Wrench<T>>,
    terms: &[GatherTerm<T>],
) -> Vec<WorldKey<Row>> {
    let mut map = world.write::<Row>();
    // A body with no incident connections still needs one row: `total_wrench`
    // must be overwritten with `external`, not left holding the previous step's
    // sum.
    let chunks: Vec<&[GatherTerm<T>]> = if terms.is_empty() {
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
            r[1] = external.raw_index() as u32;
            r[2] = u32::from(round > 0);
            r[3] = chunk.len() as u32;
            for (i, t) in chunk.iter().enumerate() {
                r[HEAD + i] = (t.key.raw_index() as u32) * 2 + t.slot as u32;
            }
            map.add(r)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use aristotle::World;

    #[test]
    fn a_long_gather_is_baked_as_several_rounds() {
        let world = Arc::new(World::builder().usual::<f32>());
        let (out, ext) = {
            let mut map = world.write::<Wrench<f32>>();
            (map.add(Wrench::zero()), map.add(Wrench::zero()))
        };
        let pairs: Vec<_> = {
            let mut map = world.write::<[Wrench<f32>; 2]>();
            (0..MAX_TERMS + 3)
                .map(|_| map.add([Wrench::zero(), Wrench::zero()]))
                .collect()
        };
        let terms: Vec<GatherTerm<f32>> = pairs
            .iter()
            .map(|k| GatherTerm {
                key: k.clone(),
                slot: 0,
            })
            .collect();

        let rows = bake(&world, &out, &ext, &terms);

        assert_eq!(rows.len(), 2, "MAX_TERMS + 3 terms is two rounds");
        let first = rows[0].read();
        assert_eq!(first[2], 0, "the first round seeds from `external`");
        assert_eq!(first[3] as usize, MAX_TERMS);
        let second = rows[1].read();
        assert_eq!(second[2], 1, "later rounds accumulate");
        assert_eq!(second[3], 3);
        assert_eq!(second[4] >> 1, pairs[MAX_TERMS].raw_index() as u32);
    }

    /// A body with no connections still gets a row: without it `total_wrench`
    /// would keep the previous step's sum instead of falling back to `external`.
    #[test]
    fn a_body_with_no_connections_still_gets_one_row() {
        let world = Arc::new(World::builder().usual::<f32>());
        let (out, ext) = {
            let mut map = world.write::<Wrench<f32>>();
            (map.add(Wrench::zero()), map.add(Wrench::zero()))
        };
        let rows = bake(&world, &out, &ext, &[]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].read()[3], 0, "no terms");
        assert_eq!(rows[0].read()[2], 0, "seeds from external");
    }
}
