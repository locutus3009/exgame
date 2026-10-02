// SPDX-License-Identifier: MIT

use super::{Joint, JointImpl, anchor_point};
use crate::JointFromParams;
use aristotle::{Epoch, World, WorldKey};
use bytemuck::Pod;
use clifford::{
    Lift,
    pga3::{Motor, Point, Twist, Wrench},
};
use peano::prelude::*;
use std::sync::Arc;

/// A transverse (perpendicular to the line of sight) LINEAR pairwise damper.
///
/// Damps the ORBITAL swing — the transverse component of the relative velocity of the
/// attachment points (for the camera — the bodies' CoMs). It does NOT touch the radial component:
/// that is handled by the axial `AxialSpringDamper` (with its own critical-damping
/// calibration), and the spin by `TorsionalDamperWarped`. Together the three cover
/// the relative twist by non-overlapping projections.
///
/// FRAME CORRECTNESS. `a_vel`/`b_vel` arrive already as WORLD twists (the
/// mechanism boundary, `world_velocity`); the velocity of a point is taken as
/// `twist.velocity_at(point)`. Taking the body-frame velocity directly is wrong for
/// an offset and rotating body (this was a bug of the old `Simple…`, removed).
/// The force is applied AT the point `pa`: we build it as the line `pa→(pa+F)` and take
/// `Wrench::from_line` — the Plücker moment about the origin (`pa×F`) comes for free, with no
/// manual cross product.
///
/// THIRD LAW AND DISSIPATIVITY. The reaction is identically `−wa`, i.e. the
/// total wrench of the pair is strictly zero: BOTH linear momentum AND angular momentum
/// about the origin are conserved. This is NOT the same as "`−F` at the point `pb`": the force here is transverse to the line
/// `pa−pb` by construction, so a pair of forces at two different points would give
/// an uncancelled torque `(pa−pb)×F`. Closing through `−wa` is equivalent to that pair
/// plus a compensating force couple, attributed entirely to body B.
///
/// The price of this closure is the choice of the velocity sampling point: the power of the pair is
/// `power(wa, ξa − ξb)`, and it is negative ONLY if the force is built from the
/// relative twist sampled at the same `pa`. Hence the single sampling point below.
#[derive(Debug)]
pub struct PerpendicularDamperWarped<T: Scalar + Pod> {
    damping: WorldKey<T>,
    /// Softening length ε for the line distance: `√(d²+ε²)` keeps `1/d` and its AD
    /// gradient finite through anchor coincidence. Scene-scaled (e.g. a small
    /// fraction of the body radius); see [`super::default_softening`].
    softening: WorldKey<T>,
}

pub struct PerpendicularDamperWarpedBuilder;

impl<T> JointFromParams<T> for PerpendicularDamperWarpedBuilder
where
    T: Scalar + StandardPart + Pod,
{
    fn shader_name(&self) -> &'static str {
        "PerpendicularDamperWarped"
    }
    fn n_params(&self) -> usize {
        2
    }
    fn build_from_params(&self, world: Arc<World>, params: &[T]) -> Joint<T> {
        PerpendicularDamperWarped::<T>::new_joint(world.clone(), params[0], params[1])
    }
}

impl<T: Scalar + Pod> PerpendicularDamperWarped<T> {
    /// Builds the joint enum directly. Named `new_joint`, not `new`, because it
    /// does not return `Self`; `new_raw` below is the `Self` constructor.
    pub fn new_joint(world: Arc<World>, damping: T, softening: T) -> Joint<T> {
        let binding = world.clone();
        let mut map = binding.write();
        Joint::PerpendicularDamperWarped(Self {
            damping: map.add(damping),
            softening: map.add(softening),
        })
    }

    pub fn new_raw(world: Arc<World>, damping: T, softening: T) -> Self {
        let binding = world.clone();
        let mut map = binding.write();
        Self {
            damping: map.add(damping),
            softening: map.add(softening),
        }
    }

    /// Stored scalar parameters, flattened in canonical order: `[damping, softening]`.
    pub fn params(&self) -> Vec<T> {
        vec![self.damping.read(), self.softening.read()]
    }

    pub fn params_keys(&self) -> Vec<WorldKey<T>> {
        vec![self.damping.clone(), self.softening.clone()]
    }

    pub fn shader_name(&self) -> &'static str {
        <PerpendicularDamperWarpedBuilder as JointFromParams<T>>::shader_name(
            &PerpendicularDamperWarpedBuilder,
        )
    }
}

impl<T> JointImpl<T> for PerpendicularDamperWarped<T>
where
    T: Scalar + StandardPart + Pod,
{
    fn wrench<S>(
        &self,
        poses: &Vector<N2, Motor<S>>,
        vels: &Vector<N2, Twist<S>>,
        epoch: &Epoch<T>,
    ) -> Vector<N2, Wrench<S>>
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        // Anchors at the CoM.
        // TODO: optimization, don't convert here, store values somewhere
        let pa = anchor_point(&poses[0], &Vector3::ZERO);
        let pb = anchor_point(&poses[1], &Vector3::ZERO);

        let line = pa.join(&pb);

        // Regularized length √(d²+ε²): smooth and finite THROUGH coincidence
        // (the ε floor), so `1/d` and its AD derivative are bounded, without the singularity
        // of the bare norm `x/‖x‖`. At coincidence the line direction → 0, so the force
        // → 0 by itself — no separate guard is needed (see soft_weight_norm).
        let d = line.soft_weight_norm(self.softening.read().lift());

        // TODO: optimization, don't convert here, store values somewhere
        let warp: S = (*epoch.warp()).lift();
        let damping: S = self.damping.read().lift();
        let b_eff = damping / warp.powi_explicit(2);

        // Unit direction of the line of sight (world); the sign does not matter for the projection.
        let dir = line.direction();
        let inv_d = S::ONE / d;
        let n = dir.scale(inv_d);

        // RELATIVE velocity — the relative twist sampled at ONE point,
        // and specifically at `pa`. The point is not arbitrary: the reaction below is identically
        // `−wa`, so the total power of the pair equals `power(wa, ξa − ξb)`, and
        // `wa` is a pure force at `pa`, so this power is exactly `F·(ξ_rel at pa)`.
        // Sampling the velocities at DIFFERENT points (`ξa` at `pa`, `ξb` at `pb`) means
        // building the force from one quantity and computing the power from another: they
        // differ by `ω_b × (pa − pb)`, and with a rotating far end the
        // damper starts PUMPING energy. The axial sibling pairs the same way —
        // `unit.power(&(vels[0] - vels[1]))`.
        let v_rel = (vels[0] - vels[1]).velocity_at(&pa).coords();

        // Discard the radial component — keep the transverse one.
        let v_dot_n = v_rel.dot(n);
        let v_perp = v_rel - n.scale(v_dot_n);

        // Force on A at the point pa: F = −b·v_perp. Built as the line pa→(pa+F) and
        // we take from_line — the moment about the origin (pa×F) comes for free.
        let f = v_perp.scale(-b_eff);
        let ca = pa.coords();
        let tip = Point::new(ca + f);
        let wa = Wrench::from_line(&tip.join(&pa));
        vector![wa, -wa]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aristotle::World;

    const B: f32 = 1.0;

    /// Pose of a body whose centre of mass sits at `p` (no rotation — the anchors
    /// are at the centres of mass, so orientation does not enter the geometry).
    fn pose_at(p: Vector3<f32>) -> Motor<f32> {
        Twist::new(&p, &Vector3::ZERO).exp(1.0)
    }

    /// The two wrenches for one configuration, through the public joint impl.
    fn wrenches(poses: [Motor<f32>; 2], vels: [Twist<f32>; 2]) -> (Wrench<f32>, Wrench<f32>) {
        let world = Arc::new(World::builder().usual::<f32>());
        let j = PerpendicularDamperWarped::new_raw(world, B, 1.0e-3);
        let w = JointImpl::<f32>::wrench(
            &j,
            &vector![poses[0], poses[1]],
            &vector![vels[0], vels[1]],
            &Epoch::standalone(1.0 / 60.0, 1.0),
        );
        (w[0], w[1])
    }

    fn total_power(vels: [Twist<f32>; 2], w: (Wrench<f32>, Wrench<f32>)) -> f32 {
        w.0.power(&vels[0]) + w.1.power(&vels[1])
    }

    /// The pair's total wrench is zero, moment included. The force is transverse
    /// to `pa − pb` by construction, so `−F` applied at `pb` instead would leave
    /// an uncancelled `(pa − pb) × F` and spin the pair up out of nothing.
    #[test]
    fn reaction_cancels_force_and_moment() {
        let poses = [
            pose_at(Vector3::from([0.0, 1.0, 0.0])),
            pose_at(Vector3::from([2.0, 0.0, 0.0])),
        ];
        // A slides transversally; B is still.
        let vels = [
            Twist::new(&Vector3::from([0.0, 0.0, 1.0]), &Vector3::ZERO),
            Twist::new(&Vector3::ZERO, &Vector3::ZERO),
        ];
        let (wa, wb) = wrenches(poses, vels);

        assert!(
            wa.force().dot(wa.force()) > 1.0e-6,
            "the configuration must actually excite the damper: {:?}",
            wa.force()
        );
        for k in 0..3 {
            assert!(
                (wa.force()[k] + wb.force()[k]).abs() < 1.0e-6,
                "force axis {k} does not cancel: {:?} vs {:?}",
                wa.force(),
                wb.force()
            );
            assert!(
                (wa.torque()[k] + wb.torque()[k]).abs() < 1.0e-6,
                "moment axis {k} does not cancel: {:?} vs {:?}",
                wa.torque(),
                wb.torque()
            );
        }
    }

    /// A damper may only remove energy. The load-bearing case is a SPINNING far
    /// end: reading `ξa` at `pa` but `ξb` at `pb` builds the force from one
    /// velocity while the pair's power is paid on another, the two differing by
    /// `ω_b × (pa − pb)`. Here that made the total power `+1` — the damper drove
    /// the pair instead of damping it.
    #[test]
    fn dissipates_when_the_far_end_spins() {
        let pb = Vector3::from([2.0, 0.0, 0.0]);
        let poses = [pose_at(Vector3::ZERO), pose_at(pb)];

        // B spins about its own centre of mass at ω = (0,0,−1) while that centre
        // moves at (0,−1,0): as a SPATIAL twist that is linear = v_com − ω × p_com.
        let omega = Vector3::from([0.0, 0.0, -1.0]);
        let v_com = Vector3::from([0.0, -1.0, 0.0]);
        let vels = [
            Twist::new(&Vector3::ZERO, &Vector3::ZERO),
            Twist::new(&(v_com - omega.cross(pb)), &omega),
        ];

        let w = wrenches(poses, vels);
        let p = total_power(vels, w);
        assert!(p < 0.0, "damper injects energy: total power = {p}");
    }

    /// The plain case still damps, so the fix above did not simply zero the joint.
    #[test]
    fn dissipates_in_pure_translation() {
        let poses = [
            pose_at(Vector3::ZERO),
            pose_at(Vector3::from([2.0, 0.0, 0.0])),
        ];
        let vels = [
            Twist::new(&Vector3::from([0.0, 1.0, 0.0]), &Vector3::ZERO),
            Twist::new(&Vector3::ZERO, &Vector3::ZERO),
        ];
        let w = wrenches(poses, vels);
        let p = total_power(vels, w);
        assert!(
            p < 0.0,
            "transverse motion must dissipate: total power = {p}"
        );
    }

    /// Radial motion belongs to the axial spring, not here.
    #[test]
    fn ignores_motion_along_the_line() {
        let poses = [
            pose_at(Vector3::ZERO),
            pose_at(Vector3::from([2.0, 0.0, 0.0])),
        ];
        let vels = [
            Twist::new(&Vector3::from([1.0, 0.0, 0.0]), &Vector3::ZERO),
            Twist::new(&Vector3::ZERO, &Vector3::ZERO),
        ];
        let (wa, _) = wrenches(poses, vels);
        assert!(
            wa.force().dot(wa.force()) < 1.0e-6,
            "radial motion must not be damped here: {:?}",
            wa.force()
        );
    }
}
