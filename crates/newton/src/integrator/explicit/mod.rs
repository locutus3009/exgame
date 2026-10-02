// SPDX-License-Identifier: MIT

use crate::RigidBody;
use aristotle::Epoch;
use bytemuck::Pod;
use clifford::pga3::Wrench;
use peano::prelude::*;

mod explicit_euler;
mod lie_euler;
mod symplectic_euler;

pub use explicit_euler::ExplicitEuler;
pub use lie_euler::LieEuler;
pub use symplectic_euler::SymplecticEuler;

// ============================================================================
// INTEGRATOR — the time-step strategy
// ============================================================================
//
// The body provides the instantaneous dynamics (velocity, gyroscopic); the integrator
// decides HOW to advance the state. An open set of schemes → a trait (unlike the closed
// InertiaTensor, which is an enum). The external wrench for the step (gravity + couplings)
// is accumulated by the caller beforehand.

pub trait Integrator<T: Scalar + StandardPart + Pod>: std::fmt::Debug {
    /// Advance `body` by `dt` under the total external wrench `wrench` (body frame).
    fn step(&self, body: &RigidBody<T>, wrench: Wrench<T>, epoch: &Epoch<T>);
}
