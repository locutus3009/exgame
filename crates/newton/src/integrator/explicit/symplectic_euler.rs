// SPDX-License-Identifier: MIT

use super::Integrator;
use crate::RigidBody;
use aristotle::Epoch;
use bytemuck::Pod;
use clifford::pga3::Wrench;
use peano::prelude::*;

/// Semi-implicit ("symplectic") Euler: momentum first, then the pose with the NEW
/// velocity. First order.
///
/// WARNING: free rotation of a body in body coordinates is a Lie–Poisson system,
/// NOT a canonical one, so the standard bounded-energy guarantee of
/// symplectic Euler does NOT hold here. The energy (and the world momentum)
/// drift SECULARLY, ∝ dt·T. True structure preservation requires a
/// Lie–Poisson / variational integrator (momentum update by conjugation).
#[derive(Debug, Clone, Copy, Default)]
pub struct SymplecticEuler;

impl<T> Integrator<T> for SymplecticEuler
where
    T: Scalar + StandardPart + Pod,
{
    #[inline]
    fn step(&self, body: &RigidBody<T>, wrench: Wrench<T>, epoch: &Epoch<T>) {
        if body.inertia.is_kinematic() {
            return;
        }

        let dt = *epoch.dt();
        let gyro = body.gyroscopic();

        body.momentum
            .write(body.momentum.read() + (wrench + gyro) * dt);

        let v_new = body.velocity();
        body.pose.write(v_new.exp(dt).compose(&body.pose.read()));
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

    /// Free rectilinear motion: the velocity is constant, the body travels v·t.
    #[test]
    fn free_linear_motion_is_uniform() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::diagonal(world.clone(), 2.0, [1.0, 1.0, 1.0]);
        let body = RigidBody::new(world.clone(), inertia.clone());
        body.momentum
            .write(inertia.apply(&Twist::new(&Vector3::from([3.0, 0.0, 0.0]), &Vector3::ZERO)));

        let dt = 0.01;
        for _ in 0..100 {
            SymplecticEuler.step(&body, Wrench::zero(), &Epoch::standalone(dt, 1.0));
        }

        let v = body.velocity();
        assert!(approx(v.linear()[0], 3.0, 1e-9));
        assert!(approx(v.linear()[1], 0.0, 1e-9));
        assert!(approx(v.angular()[2], 0.0, 1e-9));

        let moved = body.pose.read().conjugate(&Point::new(Vector3::ZERO));
        assert!(approx(moved.coords()[0], 3.0, 1e-4), "x = v·t = 3·1");
    }

    /// A constant wrench accelerates the body: p_y ≈ F·t.
    #[test]
    fn constant_force_accelerates() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::diagonal(world.clone(), 1.0, [1.0, 1.0, 1.0]);
        let body = RigidBody::new(world.clone(), inertia.clone());
        let push = Wrench::new(&Vector3::from([0.0, 5.0, 0.0]), &Vector3::ZERO);

        let dt = 0.01;
        for _ in 0..50 {
            SymplecticEuler.step(&body, push, &Epoch::standalone(dt, 1.0));
        }

        assert!(approx(body.momentum.read().force()[1], 2.5, 1e-5)); // 5·0.5
        assert!(body.velocity().linear()[1] > 0.0);
    }

    /// On a SHORT run with a small step the energy stays within ~1%.
    /// This is NOT conservation: the drift is secular (∝ dt·T) — see the first-order test.
    #[test]
    fn free_precession_energy_bounded_short_run() {
        let world = Arc::new(World::builder().usual::<f32>());

        let inertia = Inertia::diagonal(world.clone(), 1.0, [2.0, 3.0, 4.0]);
        let body = RigidBody::new(world.clone(), inertia.clone());
        body.momentum.write(inertia.apply(&Twist::new(
            &Vector3::from([0.5, -0.3, 0.2]),
            &Vector3::from([1.0, 1.5, 0.7]),
        )));

        let e0: f32 = body.kinetic_energy();
        let dt = 0.001;
        for _ in 0..2000 {
            SymplecticEuler.step(&body, Wrench::zero(), &Epoch::standalone(dt, 1.0));
        }
        assert!((body.kinetic_energy() - e0).abs() / e0 < 1e-2);
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

    /// The world angular momentum is conserved in the limit dt → 0: drift ∝ dt (first
    /// order). Checks both the correctness of the integrator and the SIGN of gyro — a wrong
    /// sign would give a convergent but incorrect precession (ratio → 1, not 2).
    #[test]
    fn free_precession_world_momentum_converges() {
        let coarse = world_momentum_drift(SymplecticEuler, 0.002, 1000); // t = 2.0
        let fine = world_momentum_drift(SymplecticEuler, 0.001, 2000); // t = 2.0

        assert!(fine < 1e-2, "drift is too large: {fine}");
        let ratio = coarse / fine;
        assert!(
            (1.6..=2.4).contains(&ratio),
            "ratio {ratio}: expected ≈2 (first order); ≈1 → wrong gyro sign"
        );
    }
}
