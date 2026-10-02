// SPDX-License-Identifier: MIT

use super::{Joint, JointImpl, anchor_point};
use crate::JointFromParams;
use aristotle::{Epoch, World, WorldKey};
use bytemuck::Pod;
use clifford::{
    Lift,
    pga3::{Motor, Twist, Wrench},
};
use peano::prelude::*;
use std::sync::Arc;

mod critically_damped_warped;
mod simple_spring_damper;

pub use critically_damped_warped::CriticallyDampedWarped;
pub use simple_spring_damper::SimpleSpringDamper;

// ============================================================================
// JOINT — spring + damper (penalty method)
// ============================================================================
//
// A joint is a FORCE SOURCE, not a rigid constraint (penalty method): any
// joint is modelled as a very stiff spring, without solving a DAE. Anchors are stored in
// the bodies' LOCAL coordinates so that they automatically rotate with the body.
//
// COORDINATE-FREE — both the force and the rate. The force acts along the LINE OF ACTION through the two
// anchors, `L = anchor_a ∨ anchor_b` (join). The line already carries the direction AND
// the Plücker moment about the origin, so the wrench is a single dual
// `Wrench::from_line(&L)`, and the moment about the origin comes for free (no cross). Length =
// `L.weight_norm()`. The closing rate `ḋ` is geometric too: it is the `power` of the line
// wrench on the bodies' RELATIVE world twist, `power(from_line(L), ξ_a − ξ_b)/d`
// — `power` pairs the force-along-the-line with the velocity, which is exactly the rate of displacement
// along the line (the lever arm `ω×p` comes from the moment part of the line on its own). The same
// line↔forque duality as in the energy.
//
// Exactly one metric remains — the division by `d` (which is also the `1/d` inside `power`).
// Coordinates [T;3] appear only when reading the input fields anchor_a/anchor_b.
//
// Both forces are ±(one scalar)·dual(one line), so the third law is
// the bivector identity `(wa, -wa)`: conservation of linear momentum AND angular momentum about
// the origin is structural. Body momentum is stored in the world frame, so the pullback (coadjoint)
// lives on the velocity side, once per body per step inside the integrator.

pub trait AxialSpringForce<T: Scalar>: std::fmt::Debug {
    fn force<S>(
        &self,
        stiffness: S,
        distance: S,
        rest_length: S,
        damping: S,
        velocity: S,
        _epoch: &Epoch<T>,
    ) -> S
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        stiffness * (distance - rest_length) + damping * velocity
    }

    fn potential_energy<S>(&self, stiffness: S, distance: S, rest_length: S) -> S
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        let dl = distance - rest_length;
        let half = S::ONE / (S::ONE + S::ONE);
        half * stiffness * dl * dl
    }

    fn into_enum(self) -> AxialSpringDamperType;
}

#[derive(Debug)]
pub enum AxialSpringDamperType {
    CriticallyDampedWarped(CriticallyDampedWarped),
    SimpleSpringDamper(SimpleSpringDamper),
}

impl<T: Scalar> AxialSpringForce<T> for AxialSpringDamperType {
    fn force<S>(
        &self,
        stiffness: S,
        distance: S,
        rest_length: S,
        damping: S,
        velocity: S,
        epoch: &Epoch<T>,
    ) -> S
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        match self {
            AxialSpringDamperType::CriticallyDampedWarped(inner) => {
                inner.force(stiffness, distance, rest_length, damping, velocity, epoch)
            }
            AxialSpringDamperType::SimpleSpringDamper(inner) => {
                inner.force(stiffness, distance, rest_length, damping, velocity, epoch)
            }
        }
    }

    fn potential_energy<S>(&self, stiffness: S, distance: S, rest_length: S) -> S
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        match self {
            AxialSpringDamperType::CriticallyDampedWarped(inner) => {
                <CriticallyDampedWarped as AxialSpringForce<T>>::potential_energy::<S>(
                    inner,
                    stiffness,
                    distance,
                    rest_length,
                )
            }
            AxialSpringDamperType::SimpleSpringDamper(inner) => {
                <SimpleSpringDamper as AxialSpringForce<T>>::potential_energy::<S>(
                    inner,
                    stiffness,
                    distance,
                    rest_length,
                )
            }
        }
    }

    fn into_enum(self) -> AxialSpringDamperType {
        panic!("Cannot call into_enum() as I am the enum");
    }
}

#[derive(Debug)]
pub struct AxialSpringDamper<T: Scalar + Pod> {
    /// Attachment point on body A (local coordinates).
    anchor_a: [WorldKey<T>; 3],
    /// Attachment point on body B (local coordinates).
    anchor_b: [WorldKey<T>; 3],
    /// Spring rest length.
    rest_length: WorldKey<T>,
    /// Stiffness k.
    stiffness: WorldKey<T>,
    /// Viscous damping c.
    damping: WorldKey<T>,
    /// Softening length ε for the distance between anchors: `√(d²+ε²)` keeps `1/d` and
    /// its AD derivative finite THROUGH anchor coincidence (see
    /// [`super::default_softening`]).
    softening: WorldKey<T>,
    /// Force expression
    f: AxialSpringDamperType,
}

pub struct AxialSpringDamperBuilder<T: Scalar + Pod> {
    world: Arc<World>,
    anchor_a: Option<Vector3<T>>,
    anchor_b: Option<Vector3<T>>,
    rest_length: Option<T>,
    stiffness: Option<T>,
    damping: Option<T>,
    softening: Option<T>,
    f: AxialSpringDamperType,
}

impl<T: Scalar + Pod> AxialSpringDamperBuilder<T> {
    pub fn a(mut self, a: Vector3<T>) -> Self {
        self.anchor_a = Some(a);
        self
    }

    pub fn b(mut self, b: Vector3<T>) -> Self {
        self.anchor_b = Some(b);
        self
    }

    pub fn rest(mut self, rest_length: T) -> Self {
        self.rest_length = Some(rest_length);
        self
    }

    pub fn stiffness(mut self, stiffness: T) -> Self {
        self.stiffness = Some(stiffness);
        self
    }

    pub fn damping(mut self, damping: T) -> Self {
        self.damping = Some(damping);
        self
    }

    /// Softening length ε for the anchor distance. Defaults to
    /// [`super::default_softening`] when not set.
    pub fn softening(mut self, softening: T) -> Self {
        self.softening = Some(softening);
        self
    }

    pub fn build(self) -> Joint<T> {
        Joint::AxialSpringDamper(self.build_raw())
    }

    pub fn build_raw(self) -> AxialSpringDamper<T> {
        let binding = self.world.clone();
        let mut map_plain = binding.write();
        let a = self.anchor_a.unwrap_or(Vector3::ZERO);
        let b = self.anchor_b.unwrap_or(Vector3::ZERO);
        AxialSpringDamper {
            f: self.f,
            anchor_a: [
                map_plain.add(a[0]),
                map_plain.add(a[1]),
                map_plain.add(a[2]),
            ],
            anchor_b: [
                map_plain.add(b[0]),
                map_plain.add(b[1]),
                map_plain.add(b[2]),
            ],
            rest_length: map_plain.add(self.rest_length.unwrap_or(T::ZERO)),
            stiffness: map_plain.add(self.stiffness.unwrap_or(T::ZERO)),
            damping: map_plain.add(self.damping.unwrap_or(T::ZERO)),
            softening: map_plain.add(self.softening.unwrap_or(super::default_softening())),
        }
    }
}

impl<T: Scalar + Pod> AxialSpringDamper<T> {
    /// Stored scalar parameters, flattened in canonical order:
    /// `[a0,a1,a2, b0,b1,b2, rest_length, stiffness, damping, softening]`.
    /// This order is the contract mirrored by the Sym-side `Sym::param(i)` build.
    pub fn params(&self) -> Vec<T> {
        vec![
            self.anchor_a[0].read(),
            self.anchor_a[1].read(),
            self.anchor_a[2].read(),
            self.anchor_b[0].read(),
            self.anchor_b[1].read(),
            self.anchor_b[2].read(),
            self.rest_length.read(),
            self.stiffness.read(),
            self.damping.read(),
            self.softening.read(),
        ]
    }

    pub fn params_keys(&self) -> Vec<WorldKey<T>> {
        vec![
            self.anchor_a[0].clone(),
            self.anchor_a[1].clone(),
            self.anchor_a[2].clone(),
            self.anchor_b[0].clone(),
            self.anchor_b[1].clone(),
            self.anchor_b[2].clone(),
            self.rest_length.clone(),
            self.stiffness.clone(),
            self.damping.clone(),
            self.softening.clone(),
        ]
    }

    pub fn a(&self) -> Vector3<T> {
        vector![
            self.anchor_a[0].read(),
            self.anchor_a[1].read(),
            self.anchor_a[2].read()
        ]
    }

    pub fn b(&self) -> Vector3<T> {
        vector![
            self.anchor_b[0].read(),
            self.anchor_b[1].read(),
            self.anchor_b[2].read()
        ]
    }

    pub fn builder<F>(world: Arc<World>, f: F) -> AxialSpringDamperBuilder<T>
    where
        F: AxialSpringForce<T>,
    {
        AxialSpringDamperBuilder {
            world,
            anchor_a: None,
            anchor_b: None,
            rest_length: None,
            stiffness: None,
            damping: None,
            softening: None,
            f: f.into_enum(),
        }
    }

    pub fn shader_name(&self) -> &'static str {
        match &self.f {
            AxialSpringDamperType::CriticallyDampedWarped(i) => {
                <CriticallyDampedWarped as JointFromParams<T>>::shader_name(i)
            }
            AxialSpringDamperType::SimpleSpringDamper(i) => {
                <SimpleSpringDamper as JointFromParams<T>>::shader_name(i)
            }
        }
    }
}

impl<T> JointImpl<T> for AxialSpringDamper<T>
where
    T: Scalar + StandardPart + Pod,
{
    /// WORLD wrenches the joint applies to bodies A and B. The force is central
    /// (along the line of action), so wb ≡ −wa exactly: equal and opposite
    /// forces collinear with the line give zero total moment about the origin, and the moments
    /// of the two bodies differ by (pa−pb)×F = 0. The third law is an identity.
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
        // TODO: optimization, don't convert here, store values somewhere
        let a_tmp: [_; 3] = [
            self.anchor_a[0].read(),
            self.anchor_a[1].read(),
            self.anchor_a[2].read(),
        ];
        let b_tmp: [_; 3] = [
            self.anchor_b[0].read(),
            self.anchor_b[1].read(),
            self.anchor_b[2].read(),
        ];

        let a_uplift = Vector3::from(a_tmp.map(Lift::lift));
        let b_uplift = Vector3::from(b_tmp.map(Lift::lift));

        let pa = anchor_point(&poses[0], &a_uplift);
        let pb = anchor_point(&poses[1], &b_uplift);

        // Line of action and distance — all of the force geometry is here.
        let line = pa.join(&pb);

        // Regularized length √(d²+ε²): smooth THROUGH anchor coincidence.
        // At coincidence the line pa∨pb = 0, so `raw = from_line(line) = 0` and
        // the resulting wrench `unit·f = 0` by itself — the same "no force" that the
        // former guard gave, but without a discontinuity in the derivative.
        let d = line.soft_weight_norm(self.softening.read().lift());

        // The "raw" line wrench: |force| = d (line weight = distance), direction
        // force = B→A, plus the moment about the origin for free. Normalize to the UNIT
        // direction wrench — this removes the spurious length d (like n = e/‖e‖ in
        // coordinate form). Division by d is normalization, not physics.
        let raw = Wrench::from_line(&line);
        let unit = raw * (S::ONE / d);

        // Scalar force along the line (>0 = attraction, pulls A toward B):
        //   spring k·(d−L₀) + damper c·ḋ.    Here and ONLY here is the physics.
        // ḋ = power(unit, ξ_a − ξ_b): power pairs the unit force-along-the-line
        // with the relative velocity of the bodies = the separation rate along the line. The lever arm
        // ω×p comes from the moment part of the line inside power, without a velocity field.
        //
        // FRAME. `unit` lives in the WORLD (a line through the world anchors), and `power` must
        // pair with the bodies' WORLD (spatial) twists. `a_vel`/`b_vel`
        // arrive already in world frame (the mechanism boundary, `world_velocity`) — we pair
        // them directly, with no Ad here.
        let d_dot = unit.power(&(vels[0] - vels[1]));

        // TODO: optimization, don't convert here, store values somewhere
        let f = self.f.force(
            self.stiffness.read().lift(),
            d,
            self.rest_length.read().lift(),
            self.damping.read().lift(),
            d_dot,
            epoch,
        );

        // Wrench on A = (unit direction toward B)·f = unit·(−f). The − sign turns
        // the B→A direction of unit into attraction. The wrench on B is identically −wa.
        let w = unit * (-f);
        vector![w, -w]
    }

    /// Potential energy of the spring: `½·k·(d − L₀)²`. The damper is dissipative and
    /// has no potential, so only this term enters the total energy
    /// (plus the kinetic energy of the bodies).
    fn potential_energy<S>(&self, a_pose: &Motor<S>, b_pose: &Motor<S>) -> S
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        let a_tmp: [_; 3] = [
            self.anchor_a[0].read(),
            self.anchor_a[1].read(),
            self.anchor_a[2].read(),
        ];
        let b_tmp: [_; 3] = [
            self.anchor_b[0].read(),
            self.anchor_b[1].read(),
            self.anchor_b[2].read(),
        ];
        let a_uplift = Vector3::from(a_tmp.map(Lift::lift));
        let b_uplift = Vector3::from(b_tmp.map(Lift::lift));
        let pa = anchor_point(a_pose, &a_uplift);
        let pb = anchor_point(b_pose, &b_uplift);
        let d = pa.join(&pb).soft_weight_norm(self.softening.read().lift());
        <AxialSpringDamperType as AxialSpringForce<T>>::potential_energy::<S>(
            &self.f,
            self.stiffness.read().lift(),
            d,
            self.rest_length.read().lift(),
        )
    }
}
