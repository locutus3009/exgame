// SPDX-License-Identifier: MIT

#![cfg_attr(not(test), no_std)]

pub mod algebra;
mod base;
pub mod pga3;
mod tangent;

pub use algebra::mv::Mv;
pub use algebra::{Algebra, Cl30, Complex, Dual, Gen, Nil, Pga3, Quaternion};
pub use base::Lift;
pub use tangent::{GradVec, Jet1, Jet6, Jet12, Jet24, Jet24Tower, Tangent};
