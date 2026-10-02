// SPDX-License-Identifier: MIT

use super::Integrator;
use crate::RigidBody;
use aristotle::Epoch;
use bytemuck::Pod;
use clifford::pga3::Wrench;
use peano::prelude::*;

/// Explicit (forward) Euler: the pose is updated with the OLD velocity. Also first
/// order, but NOT symplectic — the energy drifts secularly (for rotation it
/// usually grows). A reference for comparison, not for long simulations.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExplicitEuler;

impl<T> Integrator<T> for ExplicitEuler
where
    T: Scalar + StandardPart + Pod,
{
    #[inline]
    fn step(&self, body: &RigidBody<T>, wrench: Wrench<T>, epoch: &Epoch<T>) {
        if body.inertia.is_kinematic() {
            return;
        }

        let dt = *epoch.dt();
        let v = body.velocity();
        let gyro = body.gyroscopic();

        body.pose.write(v.exp(dt).compose(&body.pose.read())); // OLD velocity → explicit
        body.momentum
            .write(body.momentum.read() + (wrench + gyro) * dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Inertia;
    use aristotle::World;
    use clifford::pga3::Twist;
    use std::sync::Arc;

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

    /// Explicit Euler is first order too: its drift likewise decreases ∝ dt. This
    /// exercises the second implementation of the trait. Claiming "explicit is worse than
    /// symplectic in energy" on this Lie–Poisson system is not allowed — there is no
    /// guarantee; both schemes are first order and both drift secularly.
    #[test]
    fn explicit_euler_is_first_order_too() {
        let coarse = world_momentum_drift(ExplicitEuler, 0.002, 1000);
        let fine = world_momentum_drift(ExplicitEuler, 0.001, 2000);

        assert!(fine < 1e-2, "drift is finite: {fine}");
        let ratio = coarse / fine;
        assert!(
            (1.6..=2.4).contains(&ratio),
            "ratio {ratio} ≈ 2 (first order)"
        );
    }
}
