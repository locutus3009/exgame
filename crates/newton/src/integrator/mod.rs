// SPDX-License-Identifier: MIT

//! Time-step integrators.
//!
//! `explicit/` — explicit per-body schemes (`Integrator`): symplectic/explicit Euler
//! and the Lie integrator. `implicit/` — the coupled implicit step (`ImplicitIntegrator`):
//! `Newton` (the midpoint rule on stiff couplings) and `bridge` — a wrapper that
//! lifts the explicit integrators under the same `ImplicitIntegrator`.

mod explicit;
pub(crate) mod implicit;

pub use explicit::{ExplicitEuler, Integrator, LieEuler, SymplecticEuler};
pub use implicit::{ImplicitIntegrator, Newton};
