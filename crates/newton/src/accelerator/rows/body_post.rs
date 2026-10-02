// SPDX-License-Identifier: MIT

//! Baked rows for the per-body POST stage. No reduction: several bodies per row,
//! nine slots each.
//!
//! | word          | meaning                                  |
//! |---------------|------------------------------------------|
//! | 0             | `n_bodies`                               |
//! | 1             | `half` — `T` slot                        |
//! | 2             | `floor2` — `T` slot                      |
//! | 3 + 9b + 0..9 | the body's nine slots, in the order below |
//!
//! Per body: `midpoint_pose`, `solve_vel`, `mass`, `angular`, `snap_mom`,
//! `total_wrench`, `mass_out`, `rhs_out`, `scale_out`.
//!
//! Bodies with a DIAGONAL angular inertia and bodies with a FULL one land in
//! different rows: the two bind different storages, hence different kernels.
//! `Kinematic` bodies never reach here — they are not unknowns, so
//! `NewtonCache::order` excludes them.

use crate::GatherTerm;
use crate::accelerator::row::{ROW, Row};
use crate::integrator::implicit::block::Block;
use crate::{AngularKeys, InertiaKeys};
use aristotle::{World, WorldKey};
use bytemuck::Pod;
use clifford::pga3::{Motor, Twist, Wrench};
use peano::prelude::*;
use std::sync::Arc;

const HEAD: usize = 3;
const WORDS: usize = 9;
pub(crate) const MAX_BODIES: usize = (ROW - HEAD) / WORDS;

/// One body's slots for the POST stage.
#[derive(Clone)]
pub(crate) struct BodyRow<T: Scalar + Pod> {
    pub midpoint_pose: WorldKey<Motor<T>>,
    pub solve_vel: WorldKey<Twist<T>>,
    pub inertia: InertiaKeys<T>,
    /// `P_s^n` — world momentum at the start of the span.
    pub snap_mom: WorldKey<Wrench<T>>,
    /// External + Σ connection wrenches, the gather's output.
    pub total_wrench: WorldKey<Wrench<T>>,
    pub mass_out: WorldKey<Block<T>>,
    pub rhs_out: WorldKey<Wrench<T>>,
    pub scale_out: WorldKey<Wrench<T>>,
}

/// Which kernel a body needs — the angular inertia's storage differs, so the
/// binding does, so the kernel does.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Variant {
    Diagonal,
    Full,
}

pub(crate) fn bake<T: Scalar + Pod>(
    world: &Arc<World>,
    bodies: &[BodyRow<T>],
    half: &WorldKey<T>,
    floor2: &WorldKey<T>,
) -> Vec<(Variant, WorldKey<Row>)> {
    let mut map = world.write::<Row>();
    let mut out = Vec::new();
    for v in [Variant::Diagonal, Variant::Full] {
        let group: Vec<&BodyRow<T>> = bodies
            .iter()
            .filter(|b| variant_of(&b.inertia) == v)
            .collect();
        for chunk in group.chunks(MAX_BODIES) {
            let mut r: Row = [0; ROW];
            r[0] = chunk.len() as u32;
            r[1] = half.raw_index() as u32;
            r[2] = floor2.raw_index() as u32;
            for (i, b) in chunk.iter().enumerate() {
                let o = HEAD + WORDS * i;
                let (mass_k, ang_k) = inertia_slots(&b.inertia);
                r[o] = b.midpoint_pose.raw_index() as u32;
                r[o + 1] = b.solve_vel.raw_index() as u32;
                r[o + 2] = mass_k;
                r[o + 3] = ang_k;
                r[o + 4] = b.snap_mom.raw_index() as u32;
                r[o + 5] = b.total_wrench.raw_index() as u32;
                r[o + 6] = b.mass_out.raw_index() as u32;
                r[o + 7] = b.rhs_out.raw_index() as u32;
                r[o + 8] = b.scale_out.raw_index() as u32;
            }
            out.push((v, map.add(r)));
        }
    }
    out
}

fn variant_of<T: Scalar + Pod>(k: &InertiaKeys<T>) -> Variant {
    match k {
        InertiaKeys::Rigid {
            angular: AngularKeys::Diagonal(_),
            ..
        } => Variant::Diagonal,
        InertiaKeys::Rigid {
            angular: AngularKeys::Full(_),
            ..
        } => Variant::Full,
        InertiaKeys::Kinematic => unreachable!("kinematic body in the POST stage"),
    }
}

fn inertia_slots<T: Scalar + Pod>(k: &InertiaKeys<T>) -> (u32, u32) {
    match k {
        InertiaKeys::Rigid { mass, angular } => (
            mass.raw_index() as u32,
            match angular {
                AngularKeys::Diagonal(key) => key.raw_index() as u32,
                AngularKeys::Full(key) => key.raw_index() as u32,
            },
        ),
        InertiaKeys::Kinematic => unreachable!("kinematic body in the POST stage"),
    }
}

// ── Gathered variant ────────────────────────────────────────────────────────
// The wrench gather folded into POST. One body per row, because the term list is
// per body; see `build.rs::generate_body_post_gathered` for the kernel and the
// word order, which this must mirror.

/// Fixed words before the term list.
const G_HEAD: usize = 12;
/// Incident connections one row can carry. A body with more keeps the two-stage
/// path — `bake_gathered` says so by returning `None`.
pub(crate) const G_MAX_TERMS: usize = ROW - G_HEAD;

/// One round of fused rows, or `None` if any body has more incident connections
/// than a row holds. All-or-nothing on purpose: a mixed dispatch would need the
/// unfused stages for some bodies and the fused one for others, which is two
/// waves again — exactly what the fusion exists to avoid.
///
/// All three slices are indexed by the SOLVER's body order and must therefore be
/// the same length. `external` and `terms` are per-body maps in the caller, and
/// the island they come from also holds kinematic bodies, which are not unknowns
/// and have no row here — so ordering them by the island rather than by the solver
/// silently hands a body its neighbour's external wrench and its neighbour's
/// connections. That is not a shape error a `zip` would catch; it just truncates.
/// Hence the assert.
/// Baked rows for the fused GATHER+POST stage, grouped by kernel: bodies with a
/// diagonal angular inertia and bodies with a full one bind different storages,
/// so they cannot share a row.
pub(crate) type GatheredRows = Vec<(Variant, Arc<[WorldKey<Row>]>)>;

pub(crate) fn bake_gathered<T: Scalar + Pod>(
    world: &Arc<World>,
    bodies: &[BodyRow<T>],
    external: &[WorldKey<Wrench<T>>],
    terms: &[Vec<GatherTerm<T>>],
    half: &WorldKey<T>,
    floor2: &WorldKey<T>,
) -> Option<GatheredRows> {
    assert!(
        bodies.len() == external.len() && bodies.len() == terms.len(),
        "fused POST inputs are not in solver order: {} rows, {} external, {} term lists",
        bodies.len(),
        external.len(),
        terms.len(),
    );
    if terms.iter().any(|t| t.len() > G_MAX_TERMS) {
        return None;
    }
    let mut map = world.write::<Row>();
    let mut out: Vec<(Variant, Vec<WorldKey<Row>>)> = Vec::new();
    for ((b, ext), ts) in bodies.iter().zip(external).zip(terms) {
        let v = variant_of(&b.inertia);
        let (mass_k, ang_k) = inertia_slots(&b.inertia);
        let mut r: Row = [0; ROW];
        r[0] = half.raw_index() as u32;
        r[1] = floor2.raw_index() as u32;
        r[2] = b.midpoint_pose.raw_index() as u32;
        r[3] = b.solve_vel.raw_index() as u32;
        r[4] = mass_k;
        r[5] = ang_k;
        r[6] = b.snap_mom.raw_index() as u32;
        r[7] = ext.raw_index() as u32;
        r[8] = b.mass_out.raw_index() as u32;
        r[9] = b.rhs_out.raw_index() as u32;
        r[10] = b.scale_out.raw_index() as u32;
        r[11] = ts.len() as u32;
        for (i, t) in ts.iter().enumerate() {
            r[G_HEAD + i] = (t.key.raw_index() as u32) * 2 + t.slot as u32;
        }
        let key = map.add(r);
        match out.iter_mut().find(|(k, _)| *k == v) {
            Some((_, rows)) => rows.push(key),
            None => out.push((v, vec![key])),
        }
    }
    Some(out.into_iter().map(|(v, r)| (v, Arc::from(r))).collect())
}
