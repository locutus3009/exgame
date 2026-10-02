// SPDX-License-Identifier: MIT

use aristotle::{Epoch, World, WorldId, WorldKey};
use bytemuck::Pod;
use clifford::{
    Lift,
    pga3::{Dynamics, Motor, Point, Twist, Wrench},
};
use peano::prelude::*;
use std::sync::Arc;

mod axial_spring_damper;
mod perpendicular_damper_warped;
mod torsional_damper_warped;

/// Default softening length ε for the distance-using joints, applied by builders /
/// constructors when the caller doesn't supply one. A tiny regularization floor
/// (~1e-6) so `soft_weight_norm`'s `√(d²+ε²)` keeps `1/d` and its AD gradient finite
/// through anchor coincidence. Real consumers (e.g. the camera rig) should pass a
/// scene-scaled value (a small fraction of the body radius) instead of this default.
///
/// ε is a GEOMETRIC length — the smallest physically-meaningful anchor separation —
/// NOT a solver tolerance. Keep it well above the Newton residual band: `soft_norm`
/// plateaus at ε, so any constraint that enters the residual through a distance and
/// is driven to zero cannot converge below the ε scale. With a residual ~1e-13, an
/// ε of ~1e-6·(geometry) sits safely above the band.
pub fn default_softening<T: Scalar>() -> T {
    T::from_rational(1, 1_000_000)
}

pub use axial_spring_damper::{
    AxialSpringDamper, AxialSpringDamperBuilder, AxialSpringDamperType, AxialSpringForce,
    CriticallyDampedWarped, SimpleSpringDamper,
};
pub use perpendicular_damper_warped::{
    PerpendicularDamperWarped, PerpendicularDamperWarpedBuilder,
};
pub use torsional_damper_warped::{TorsionalDamperWarped, TorsionalDamperWarpedBuilder};

// ============================================================================
// JOINT EDGE — a joint registered in the graph
// ============================================================================

/// A joint bound to a pair of bodies (by internal keys — direct access in the hot
/// loop without a hash lookup). Parallel joints = several JointEdges with the same pair.
#[derive(Debug)]
pub struct JointEdge<T: Scalar + Pod> {
    id: WorldId,
    a: WorldId,
    b: WorldId,
    joint: Joint<T>,
    /// Per-connection VALUE output (accelerator gather input): the two wrenches
    /// `[on body a, on body b]`. Written by `enqueue` (its OWN slot — disjoint per
    /// connection, no contention), summed into per-body `total_wrench` by the
    /// gather stage. Lives in World storage, freed with the edge.
    wrenches: WorldKey<[Wrench<T>; 2]>,
    /// Per-connection Jacobian block (accelerator output): `[[Wrench; 24]; 2]` —
    /// 24 columns × 2 body-ends. Written by `Accelerator::enqueue`, read by the
    /// implicit matrix assembler. Lives in World storage, freed with the edge.
    jacobian: WorldKey<[[Wrench<T>; 24]; 2]>,
}

impl<T: Scalar + Pod> JointEdge<T> {
    pub fn new(a: WorldId, b: WorldId, joint: Joint<T>) -> Self {
        let world = joint.world();
        let wrenches = world.write().add([Wrench::zero(); 2]);
        let jacobian = world.write().add(core::array::from_fn(|_| {
            core::array::from_fn(|_| Wrench::zero())
        }));
        Self {
            id: WorldId::get(),
            a,
            b,
            joint,
            wrenches,
            jacobian,
        }
    }

    /// Handle to this edge's per-connection VALUE slot `[wrench on a, on b]`,
    /// for the integrator to hand to `enqueue` as the value write destination.
    pub fn wrench_key(&self) -> WorldKey<[Wrench<T>; 2]> {
        self.wrenches.clone()
    }

    /// Handle to this edge's per-connection Jacobian slot, for the integrator to
    /// hand to `Accelerator::enqueue` as the block's write destination.
    pub fn jacobian_key(&self) -> WorldKey<[[Wrench<T>; 24]; 2]> {
        self.jacobian.clone()
    }

    pub fn id(&self) -> WorldId {
        self.id
    }

    pub fn a(&self) -> WorldId {
        self.a
    }

    pub fn b(&self) -> WorldId {
        self.b
    }

    pub fn split(self) -> (WorldId, WorldId, Joint<T>) {
        (self.a, self.b, self.joint)
    }

    /// Scalar parameters of this edge's joint, flattened in canonical order.
    /// Feeds the viete `run_lua` param slice (its length must equal the trace's
    /// PARAMS).
    pub fn params(&self) -> Vec<T> {
        self.joint.params()
    }

    pub fn params_keys(&self) -> Vec<WorldKey<T>> {
        self.joint.params_keys()
    }

    pub fn shader_name(&self) -> &'static str {
        self.joint.shader_name()
    }
}

impl<T: Scalar + StandardPart + Lift<T> + Pod> JointEdge<T> {
    pub fn potential_energy(&self, a_pose: &Motor<T>, b_pose: &Motor<T>) -> T {
        self.joint.potential_energy(a_pose, b_pose)
    }
}

impl<T: Scalar + StandardPart + Pod> Dynamics<N2, T> for JointEdge<T> {
    type Context = Epoch<T>;
    fn eval<S>(
        &self,
        poses: &Vector<N2, Motor<S>>,
        vels: &Vector<N2, Twist<S>>,
        epoch: &Epoch<T>,
    ) -> Vector<N2, Wrench<S>>
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        self.joint.wrench(poses, vels, epoch)
    }
}

#[derive(Debug)]
pub enum Joint<T: Scalar + Pod> {
    PerpendicularDamperWarped(PerpendicularDamperWarped<T>),
    TorsionalDamperWarped(TorsionalDamperWarped<T>),
    AxialSpringDamper(AxialSpringDamper<T>),
}

impl<T: Scalar + Pod> Joint<T> {
    /// Stored scalar parameters of the wrapped joint, in its canonical order.
    pub fn params(&self) -> Vec<T> {
        match self {
            Joint::PerpendicularDamperWarped(i) => i.params(),
            Joint::TorsionalDamperWarped(i) => i.params(),
            Joint::AxialSpringDamper(i) => i.params(),
        }
    }

    pub fn params_keys(&self) -> Vec<WorldKey<T>> {
        match self {
            Joint::PerpendicularDamperWarped(i) => i.params_keys(),
            Joint::TorsionalDamperWarped(i) => i.params_keys(),
            Joint::AxialSpringDamper(i) => i.params_keys(),
        }
    }

    /// The `World` this joint's parameters live in — used by `JointEdge::new` to
    /// allocate the per-connection Jacobian slot in the same world without a
    /// separate handle. Every joint carries at least one parameter key.
    pub fn world(&self) -> Arc<World> {
        self.params_keys()[0].world()
    }

    pub fn shader_name(&self) -> &'static str {
        match self {
            Joint::PerpendicularDamperWarped(i) => i.shader_name(),
            Joint::TorsionalDamperWarped(i) => i.shader_name(),
            Joint::AxialSpringDamper(i) => i.shader_name(),
        }
    }
}

impl<T: Scalar + StandardPart + Pod> JointImpl<T> for Joint<T> {
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
        match self {
            Joint::PerpendicularDamperWarped(i) => i.wrench(poses, vels, epoch),
            Joint::TorsionalDamperWarped(i) => i.wrench(poses, vels, epoch),
            Joint::AxialSpringDamper(i) => i.wrench(poses, vels, epoch),
        }
    }
    fn potential_energy<S>(&self, a_pose: &Motor<S>, b_pose: &Motor<S>) -> S
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        match self {
            Joint::PerpendicularDamperWarped(i) => i.potential_energy(a_pose, b_pose),
            Joint::TorsionalDamperWarped(i) => i.potential_energy(a_pose, b_pose),
            Joint::AxialSpringDamper(i) => i.potential_energy(a_pose, b_pose),
        }
    }
}

pub trait JointImpl<T>: std::fmt::Debug
where
    T: Scalar + StandardPart,
{
    /// FRAME CONTRACT. A joint is a pure WORLD function: `a_vel`/`b_vel` arrive
    /// already as WORLD (spatial) twists of the bodies (`RigidBody::world_velocity`,
    /// the Ad conversion body→world is done at the mechanism boundary), and the returned wrench
    /// is WORLD as well. Poses are needed only to place the anchors (world points), not
    /// to convert velocities. Bodies/integrator stay in the body frame; the transport
    /// world→body lives in `Mechanism::step` (`transport(pose⁻¹)` (Co law)).
    fn wrench<S>(
        &self,
        poses: &Vector<N2, Motor<S>>,
        vels: &Vector<N2, Twist<S>>,
        epoch: &Epoch<T>,
    ) -> Vector<N2, Wrench<S>>
    where
        S: Scalar + StandardPart,
        T: Lift<S>;

    fn potential_energy<S>(&self, _a_pose: &Motor<S>, _b_pose: &Motor<S>) -> S
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        S::ZERO
    }
}

pub trait JointFromParams<T>
where
    T: Scalar + StandardPart + Pod,
{
    fn shader_name(&self) -> &'static str;
    fn n_params(&self) -> usize;
    fn build_from_params(&self, world: Arc<World>, params: &[T]) -> Joint<T>;
}

/// World position of a local anchor as a POINT (motor sandwich). We return a Point
/// rather than coordinates: the point goes straight into a join without unpacking.
#[inline]
pub fn anchor_point<T>(pose: &Motor<T>, local: &Vector3<T>) -> Point<T>
where
    T: Ring,
{
    pose.conjugate(&Point::new(*local))
}

#[cfg(test)]
mod params_tests {
    use super::*;
    use crate::{AxialSpringDamper, CriticallyDampedWarped};
    use aristotle::World;
    use clifford::pga3::Wrench;
    use std::sync::Arc;

    #[test]
    fn axial_params_flatten_in_canonical_order() {
        let j = AxialSpringDamper::<f64>::builder(
            Arc::new(
                World::builder()
                    .with_storage::<f64>()
                    .with_storage::<Vector3<f64>>()
                    .with_storage::<[Wrench<f64>; 2]>()
                    .with_storage::<[[Wrench<f64>; 24]; 2]>()
                    .build(),
            ),
            CriticallyDampedWarped,
        )
        .a(Vector3::from([0.3, -0.2, 0.15]))
        .b(Vector3::from([-0.1, 0.25, -0.05]))
        .rest(1.3)
        .stiffness(7.0)
        .damping(2.0)
        .softening(1e-6)
        .build_raw();
        let edge = JointEdge::new(WorldId::get(), WorldId::get(), Joint::AxialSpringDamper(j));
        assert_eq!(
            edge.params(),
            vec![0.3, -0.2, 0.15, -0.1, 0.25, -0.05, 1.3, 7.0, 2.0, 1e-6]
        );
    }
}
