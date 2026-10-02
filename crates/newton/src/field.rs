// SPDX-License-Identifier: MIT

use crate::{Component, ForceField};
use aristotle::{Epoch, WorldId, WorldKey};
use async_trait::async_trait;
use bytemuck::Pod;
use clifford::pga3::Wrench;
use indexmap::IndexMap;
use peano::prelude::*;

/// Uniform force field: a constant world acceleration `accel`, the same for
/// all bodies. Each body gets an additive WORLD wrench `F = m · accel`, torque 0.
/// Surface gravity is a special case (`[0, 0, −g]`); a crosswind is
/// `[0, a, 0]` etc. (unlike the N-body `GravityPropagator`). Mass
/// weighting as with gravity: kinematic bodies (`mass() == 0`) get
/// zero force automatically. Static field → `publish` stays the default no-op.
pub struct UniformField<T: Scalar + Pod> {
    accel: Vector3<T>,
}

impl<T: Scalar + Pod> UniformField<T> {
    /// `accel` — constant world acceleration (e.g. `[0, 0, −g]` for weight).
    pub fn new(accel: Vector3<T>) -> Self {
        Self { accel }
    }
}

#[async_trait]
impl<T: Scalar + Pod, S: Scalar + From<T>> ForceField<T, S> for UniformField<T> {
    async fn accumulate(
        &self,
        bodies: &IndexMap<WorldId, Box<dyn Component<T, S>>>,
        out: &IndexMap<WorldId, WorldKey<Wrench<T>>>,
        _epoch: &Epoch<T>,
        _origin: &Vector3<S>,
    ) {
        for (id, e) in bodies.iter() {
            let m = e.body().inertia.mass();
            let force = self.accel.scale(m); // F = m·a, world frame
            let w = Wrench::new(&force, &Vector3::ZERO);
            let o = &out[id];
            o.write(o.read() + w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Inert, Inertia, RigidBody};
    use aristotle::World;
    use std::sync::Arc;

    #[tokio::test]
    async fn applies_m_times_accel_and_ignores_kinematic() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = 2.0_f32;
        // Arbitrary world acceleration: gravity along −Z plus a sideways +Y "wind".
        let accel = Vector3::from([0.0, 1.5, -9.81]);

        // Keys are arbitrary WorldIds — accumulate indexes `out` by the map key,
        // not by the body's internal id (mirrors the direct-dispatch unit tests).
        let dyn_id = WorldId::get();
        let kin_id = WorldId::get();

        let dyn_body: Box<dyn Component<f32, f32>> =
            Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::ZERO,
                &Vector3::ZERO,
                m,
            ));
        let kin_body: Box<dyn Component<f32, f32>> =
            Inert::new(RigidBody::new(world.clone(), Inertia::Kinematic));

        let mut bodies: IndexMap<WorldId, Box<dyn Component<f32, f32>>> = IndexMap::new();
        bodies.insert(dyn_id, dyn_body);
        bodies.insert(kin_id, kin_body);

        let mut out: IndexMap<WorldId, WorldKey<Wrench<f32>>> = IndexMap::new();
        {
            let mut map = world.write();
            out.insert(dyn_id, map.add(Wrench::zero()));
            out.insert(kin_id, map.add(Wrench::zero()));
        }

        let field = UniformField::new(accel);
        field
            .accumulate(&bodies, &out, &Epoch::standalone(0.01, 1.0), &Vector3::ZERO)
            .await;

        // Dynamic body: F = m·a per axis. Kinematic: zero (mass 0).
        assert_eq!(
            out[&dyn_id].read().force().split(),
            [0.0, m * 1.5, m * -9.81]
        );
        assert_eq!(out[&kin_id].read().force().split(), [0.0, 0.0, 0.0]);
    }
}
