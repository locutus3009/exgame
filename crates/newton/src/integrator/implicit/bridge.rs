// SPDX-License-Identifier: MIT

use crate::Component;
use crate::integrator::explicit::Integrator;
use crate::{Accelerator, EvalError};
use aristotle::{Epoch, WorldId, WorldKey};
use bytemuck::Pod;
use clifford::{Lift, pga3::Wrench};
use indexmap::IndexMap;
use joints::JointEdge;
use peano::prelude::*;

/// Bridge helper: applies a per-body `Integrator::step` to every body in
/// the slice, doing the world → body pullback via `cotransform` first. Used
/// by the explicit variants of `ImplicitIntegrator::step_all`. Kept as a
/// free helper (not folded into the `enum` match) so the per-body scheme is
/// the only thing that varies — the coupling/pullback loop is written once.
#[inline]
pub(super) async fn explicit_step_all<I, T, S>(
    accelerator: &Accelerator<T>,
    integ: &I,
    bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
    joints: &IndexMap<WorldId, JointEdge<T>>,
    wrenches: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
    cache: &mut super::cache::NewtonCache<T>,
    epoch: &Epoch<T>,
) -> Result<(), EvalError>
where
    I: Integrator<T>,
    T: Scalar + StandardPart + PartialOrd + Lift<T> + Pod,
    S: Ring,
{
    // Explicit family: the kernel evaluates at the CURRENT pose / world velocity,
    // so the solve velocity is the world velocity and the retraction time is 0
    // (`midpoint = pose ∘ exp(0) = pose`). Value only — no Jacobian.
    //
    // ONE lane: the explicit family has a single trial per step, so the eight the
    // line search needs would be pure waste here.
    let Some(world) = bodies.values().next().map(|e| e.body().pose.world()) else {
        return Ok(());
    };
    cache.bake_lanes(&world, bodies, joints, wrenches, 1);
    cache.publish_dispatch_scalars(T::ZERO, *epoch.dt(), *epoch.warp());
    // Lane 0 aliases the body-owned slots, so recomputing each body's world
    // velocity IS seeding the lane's iterate.
    for e in bodies.values() {
        e.body().world_velocity();
    }
    // The explicit family gathers: it reads `total_wrench` directly and never
    // runs POST, fused or otherwise.
    accelerator.dispatch(&cache.lanes()[0], false, true).await?;

    // Integrate each body from its gathered world wrench (external + joints).
    for (_id, entity) in bodies.iter() {
        let w_body = entity
            .body()
            .total_wrench
            .read()
            .transport(&entity.body().pose.read().inverse());
        <I as Integrator<T>>::step(integ, entity.body(), w_body, epoch);
    }
    Ok(())
}
