// SPDX-License-Identifier: MIT

use super::{Joint, JointImpl};
use crate::JointFromParams;
use aristotle::{Epoch, World, WorldKey};
use bytemuck::Pod;
use clifford::{
    Lift,
    pga3::{Motor, Twist, Wrench},
};
use peano::prelude::*;
use std::sync::Arc;

/// A torsional (angular-velocity) PAIRWISE damper: a pure couple that damps the
/// RELATIVE angular velocity of two bodies.
///
/// PGA formulation. The relative angular velocity is the Euclidean
/// (non-ideal) part of the relative SPATIAL twist: `ω_rel =
/// angular(Ad_{M_a} V_a − Ad_{M_b} V_b)`. The damping torque is a "pure couple":
/// a wrench with ZERO force part and torque `−b·ω_rel`; intrinsically it is the wrench
/// dual to a LINE AT INFINITY (an ideal line). A couple is a free vector
/// (the same at every point of the body), so the point of application is irrelevant —
/// no anchors are needed here, and the damper acts exactly "at the CoM".
///
/// Dissipative without caveats: `power(W, ΔV) = τ·ω_rel = −b·|ω_rel|² ≤ 0`.
///
/// ABSOLUTE damping. Between two dynamic bodies the couple conserves the total
/// angular momentum (it damps only the relative spin). But if the far end is a
/// KINEMATIC body (the camera's `target_tracker` with forced `ω=0`,
/// whose reaction is overwritten every pre-step), the couple becomes a ONE-SIDED
/// external rate damper and removes the ABSOLUTE rotation of the frame. A closed
/// (conservative) damper cannot do this — an internal torque does not change
/// the total `L`.
#[derive(Debug)]
pub struct TorsionalDamperWarped<T: Scalar + Pod> {
    damping: WorldKey<T>,
}

pub struct TorsionalDamperWarpedBuilder;

impl<T> JointFromParams<T> for TorsionalDamperWarpedBuilder
where
    T: Scalar + StandardPart + Pod,
{
    fn shader_name(&self) -> &'static str {
        "TorsionalDamperWarped"
    }
    fn n_params(&self) -> usize {
        1
    }
    fn build_from_params(&self, world: Arc<World>, params: &[T]) -> Joint<T> {
        TorsionalDamperWarped::<T>::new_joint(world.clone(), params[0])
    }
}

impl<T: Scalar + Pod> TorsionalDamperWarped<T> {
    pub fn new_joint(world: Arc<World>, damping: T) -> Joint<T> {
        let binding = world.clone();
        let mut map = binding.write();
        Joint::TorsionalDamperWarped(Self {
            damping: map.add(damping),
        })
    }

    pub fn new(world: Arc<World>, damping: T) -> Self {
        let binding = world.clone();
        let mut map = binding.write();
        Self {
            damping: map.add(damping),
        }
    }

    /// Stored scalar parameters, flattened in canonical order: `[damping]`.
    pub fn params(&self) -> Vec<T> {
        vec![self.damping.read()]
    }

    pub fn params_keys(&self) -> Vec<WorldKey<T>> {
        vec![self.damping.clone()]
    }

    pub fn shader_name(&self) -> &'static str {
        <TorsionalDamperWarpedBuilder as JointFromParams<T>>::shader_name(
            &TorsionalDamperWarpedBuilder,
        )
    }
}

impl<T> JointImpl<T> for TorsionalDamperWarped<T>
where
    T: Scalar + StandardPart + Pod,
{
    fn wrench<S>(
        &self,
        _poses: &Vector<N2, Motor<S>>,
        vels: &Vector<N2, Twist<S>>,
        epoch: &Epoch<T>,
    ) -> Vector<N2, Wrench<S>>
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        // `a_vel`/`b_vel` are already WORLD (spatial) twists — the Ad conversion
        // body→world is done at the mechanism boundary (`world_velocity`). The couple lives
        // in the world, so we take the relative angular velocity directly in world axes;
        // poses are not needed here (a pure couple without anchors).
        let w_rel = (vels[0] - vels[1]).angular();

        // warp: PLANNED (not yet tested). The sibling critically-damped dampers
        // go through `c0 = k/warp²`; a pure torsional damper has no
        // companion stiffness, so we scale the coefficient consistently —
        // divide by `warp²`. At `warp = 1` this is the identity.
        // TODO: optimization, don't convert here, store values somewhere
        let warp: S = (*epoch.warp()).lift();
        let damping: S = self.damping.read().lift();
        let b_eff = damping / warp.powi_explicit(2);

        // A pure couple (no force) against the relative spin — torque `−b·ω_rel`.
        // Built through the typed API so as not to depend on the sign
        // conventions of the multivector components.
        let couple = Wrench::new(&Vector3::ZERO, &w_rel.scale(-b_eff));
        vector![couple, -couple]
    }
}
