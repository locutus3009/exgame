// SPDX-License-Identifier: MIT

use super::Integrator;
use crate::RigidBody;
use aristotle::Epoch;
use bytemuck::Pod;
use clifford::pga3::Wrench;
use peano::prelude::*;

/// Lie integrator (coadjoint-preserving, Lie–Poisson). Instead of the additive
/// gyroscopic term `P += [P,V]dt` it updates the momentum by CONJUGATION
/// `P ← δM̃·P·δM` with the same increment δM that goes into the pose. This is an exact (to
/// machine precision) discrete precession: the world momentum `M·P·M̃` is conserved
/// STRUCTURALLY for any dt, not merely in the limit dt→0. The gyroscopic term
/// arises from the conjugation by itself (its linearization = `[P,V]dt`), so it is
/// NOT added separately.
///
/// Preserves the COADJOINT ORBIT (|P| in the Casimir norm → the shape of the precession), but NOT
/// the energy: this is an explicit first-order scheme, δM is built from the velocity at the start
/// of the step, so the energy drifts (energy preservation requires an implicit
/// midpoint/RATTLE). The value is exact conservation of angular momentum at a coarse
/// step, which for long simulations usually matters more than trajectory accuracy.
#[derive(Debug, Clone, Copy, Default)]
pub struct LieEuler;

impl<T> Integrator<T> for LieEuler
where
    T: Scalar + StandardPart + Pod,
{
    #[inline]
    fn step(&self, body: &RigidBody<T>, wrench: Wrench<T>, epoch: &Epoch<T>) {
        if body.inertia.is_kinematic() {
            return;
        }

        let dt = *epoch.dt();

        // wrench arrives in the body frame (trait contract; the world→body pullback lives
        // in Mechanism::step). We simply pour it into the body-frame momentum.
        body.momentum.write(body.momentum.read() + wrench * dt);

        // Precession of the body-frame momentum under the body increment: coadjoint by δR.
        // δ = velocity().exp(dt) — the body-frame velocity is taken AFTER the momentum
        // update (symplectic-like ordering: momentum first, pose second).
        let delta = body.velocity().exp(dt);
        let drot = delta.rotation_part();
        body.momentum
            .write(body.momentum.read().transport(&drot.inverse()));

        // body-frame velocity → right composition: `M_new = M · δ` (δ acts
        // in the body's local axes; in pipeline notation — δ.compose(&M)). The left
        // `δ · M` would mean a spatial-frame increment and would turn a body spin
        // into an orbit around the world origin.
        body.pose.write(delta.compose(&body.pose.read()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Inertia;
    use aristotle::World;
    use clifford::pga3::{Point, Twist};
    use std::sync::Arc;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // Drift of the world angular momentum over t = dt·steps for free precession.
    fn world_momentum_drift<I: Integrator<f32>>(integ: I, dt: f32, steps: usize) -> f32 {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::diagonal(world.clone(), 1.0, [2.0, 3.0, 4.0]);
        let body = RigidBody::new(world.clone(), inertia.clone());
        body.momentum
            .write(inertia.apply(&Twist::new(&Vector3::ZERO, &Vector3::from([1.0, 1.5, 0.7]))));

        let l0 = body.world_momentum().read().torque();
        for _ in 0..steps {
            integ.step(&body, Wrench::zero(), &Epoch::standalone(dt, 1.0));
        }
        let l1 = body.world_momentum().read().torque();

        let dx = l1[0] - l0[0];
        let dy = l1[1] - l0[1];
        let dz = l1[2] - l0[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// STRUCTURAL property of the Lie integrator: the world angular momentum is conserved
    /// to MACHINE precision — at a COARSE step (dt=0.01, t=20), where the Euler schemes drift
    /// by tens of percent. The conjugation `δM̃·P·δM` cancels against the pose rotation
    /// identically, so the drift does NOT depend on dt (it is already at round-off level).
    #[test]
    fn lie_conserves_world_momentum_to_machine_precision() {
        let drift = world_momentum_drift(LieEuler, 0.01, 2000); // t = 20, coarse
        assert!(
            drift < 1e-3,
            "world momentum must hold to machine precision: {drift}"
        );
    }

    /// STRUCTURAL property of the Lie integrator: conservation of the world momentum does NOT
    /// depend on dt. At both a small and a coarse step the drift stays at machine level —
    /// unlike the Euler schemes, where drift ∝ dt. This is exactly "structure preservation"
    /// (the conjugation cancels against the pose identically for any δM).
    #[test]
    fn lie_world_momentum_independent_of_dt() {
        let coarse = world_momentum_drift(LieEuler, 0.01, 200); // t = 2, coarse
        let fine = world_momentum_drift(LieEuler, 0.0005, 4000); // t = 2, fine
        assert!(coarse < 1e-4, "coarse step: {coarse}");
        assert!(fine < 1e-2, "fine step: {fine}");
    }

    /// BUG B (compose order). The body-frame velocity must be composed
    /// on the RIGHT (`pose * δ`): then δ acts in the body's local axes. A free
    /// spinning off-axis body then keeps its CoM in place and spins around
    /// its OWN CoM. If instead `δ * pose` (as now), the body-frame spin
    /// is interpreted as a rotation of the world — the CoM flies off along a circle around
    /// the world origin at radius |r_pose|.
    #[test]
    fn off_axis_spinning_body_keeps_com_in_place() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::isotropic(world.clone(), 1.0, 0.4);
        let body = RigidBody::new(world.clone(), inertia.clone());
        body.pose
            .write(Twist::new(&Vector3::from([10.0, 0.0, 0.0]), &Vector3::ZERO).exp(1.0));
        // pure body-frame spin ω_z = 1 rad/s
        body.momentum
            .write(inertia.apply(&Twist::new(&Vector3::ZERO, &Vector3::from([0.0, 0.0, 1.0]))));

        let p0 = body
            .pose
            .read()
            .conjugate(&Point::new(Vector3::ZERO))
            .coords();

        let dt = 0.01;
        for _ in 0..100 {
            LieEuler.step(&body, Wrench::zero(), &Epoch::standalone(dt, 1.0)); // no force
        }

        let p1 = body
            .pose
            .read()
            .conjugate(&Point::new(Vector3::ZERO))
            .coords();

        for k in 0..3 {
            assert!(
                approx(p1[k], p0[k], 1e-4),
                "CoM of the spinning off-axis body moved: {p0:?} → {p1:?} (component {k})"
            );
        }
    }

    /// Basic correctness of the Lie integrator: free rectilinear motion
    /// stays uniform (the translational conjugation is trivial).
    #[test]
    fn lie_free_linear_motion_is_uniform() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::diagonal(world.clone(), 2.0, [1.0, 1.0, 1.0]);
        let body = RigidBody::new(world.clone(), inertia.clone());
        body.momentum
            .write(inertia.apply(&Twist::new(&Vector3::from([3.0, 0.0, 0.0]), &Vector3::ZERO)));

        let dt = 0.01;
        for _ in 0..100 {
            LieEuler.step(&body, Wrench::zero(), &Epoch::standalone(dt, 1.0));
        }

        let v = body.velocity();
        assert!(approx(v.linear()[0], 3.0, 1e-4));

        let moved = body.pose.read().conjugate(&Point::new(Vector3::ZERO));
        assert!(approx(moved.coords()[0], 3.0, 1e-4), "x = v·t = 3·1");
    }
}
