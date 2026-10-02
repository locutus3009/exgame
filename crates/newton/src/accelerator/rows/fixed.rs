// SPDX-License-Identifier: MIT

//! Baked rows for the FIXED-ARITY kernels — `Pre` and the connection kernels.
//!
//! Unlike the reduction stages these have no term list: their row is a short,
//! constant sequence of slots. They were the last stages still carrying their
//! indices in the batch table, which meant one message per body and per
//! connection; with eight line-search lanes that was the whole of the message
//! traffic once the block stages went to one message per round.
//!
//! The word order IS `build.rs`'s `body_pre`: twists, motors, outputs, scalars,
//! params. Nothing else may assume it, and the two must be changed together.
//!
//! **`Pre`** — one row per body:
//!
//! | word | meaning                              |
//! |------|--------------------------------------|
//! | 0    | `vel` — `Twist<T>` slot (the iterate) |
//! | 1    | `pose` — `Motor<T>` slot             |
//! | 2    | `midpoint` — `Motor<T>` out          |
//! | 3    | `solve_vel` — `Twist<T>` out         |
//! | 4    | `retraction` — `T` slot              |
//!
//! **Connection** — one row per connection. The Jacobian row is the plain row
//! plus the block slot, which is why the two are baked by one function:
//!
//! | word  | meaning                                     |
//! |-------|---------------------------------------------|
//! | 0, 1  | `solve_vel` of ends a, b — `Twist<T>` slots |
//! | 2, 3  | `midpoint` of ends a, b — `Motor<T>` slots  |
//! | 4     | Jacobian block slot (JACOBIAN ONLY)         |
//! | 4/5   | `conn` — the `[Wrench<T>; 2]` value pair    |
//! | ...   | `dt`, `warp` — `T` slots                    |
//! | ...   | one `T` slot per joint param                |

use crate::accelerator::row::{ROW, Row};
use aristotle::{World, WorldKey};
use bytemuck::Pod;
use clifford::pga3::{Motor, Twist, Wrench};
use peano::prelude::*;
use std::sync::Arc;

/// One body's `Pre` inputs and outputs.
pub(crate) struct PreRow<T: Scalar + Pod> {
    pub vel: WorldKey<Twist<T>>,
    pub pose: WorldKey<Motor<T>>,
    pub midpoint: WorldKey<Motor<T>>,
    pub solve_vel: WorldKey<Twist<T>>,
}

/// One round of `Pre` rows — they are independent, so there is exactly one.
pub(crate) fn bake_pre<T: Scalar + Pod>(
    world: &Arc<World>,
    bodies: &[PreRow<T>],
    retraction: &WorldKey<T>,
) -> Vec<Arc<[WorldKey<Row>]>> {
    let mut map = world.write::<Row>();
    let round: Vec<WorldKey<Row>> = bodies
        .iter()
        .map(|b| {
            let mut r: Row = [0; ROW];
            r[0] = b.vel.raw_index() as u32;
            r[1] = b.pose.raw_index() as u32;
            r[2] = b.midpoint.raw_index() as u32;
            r[3] = b.solve_vel.raw_index() as u32;
            r[4] = retraction.raw_index() as u32;
            map.add(r)
        })
        .collect();
    vec![Arc::from(round)]
}

/// One connection's kernel inputs and outputs.
pub(crate) struct JointRow<T: Scalar + Pod> {
    /// Both ends' solve velocity and midpoint pose, `a` then `b`.
    pub vels: [WorldKey<Twist<T>>; 2],
    pub poses: [WorldKey<Motor<T>>; 2],
    pub conn: WorldKey<[Wrench<T>; 2]>,
    pub block: WorldKey<[[Wrench<T>; 24]; 2]>,
    pub params: Vec<WorldKey<T>>,
}

/// One round of connection rows. `jacobian` picks the layout AND the kernel: the
/// two differ by the block slot, which the value-only kernel has no output for.
pub(crate) fn bake_joints<T: Scalar + Pod>(
    world: &Arc<World>,
    edges: &[JointRow<T>],
    dt: &WorldKey<T>,
    warp: &WorldKey<T>,
    jacobian: bool,
) -> Vec<Arc<[WorldKey<Row>]>> {
    let mut map = world.write::<Row>();
    let round: Vec<WorldKey<Row>> = edges
        .iter()
        .map(|e| {
            let mut words: Vec<u32> = vec![
                e.vels[0].raw_index() as u32,
                e.vels[1].raw_index() as u32,
                e.poses[0].raw_index() as u32,
                e.poses[1].raw_index() as u32,
            ];
            if jacobian {
                words.push(e.block.raw_index() as u32);
            }
            words.push(e.conn.raw_index() as u32);
            words.push(dt.raw_index() as u32);
            words.push(warp.raw_index() as u32);
            words.extend(e.params.iter().map(|p| p.raw_index() as u32));
            assert!(words.len() <= ROW, "connection row does not fit ROW");
            let mut r: Row = [0; ROW];
            r[..words.len()].copy_from_slice(&words);
            map.add(r)
        })
        .collect();
    vec![Arc::from(round)]
}
