// SPDX-License-Identifier: MIT

#![cfg_attr(not(test), no_std)]

mod abelian;
mod invertible;
mod mul_monoid;
mod nat;
mod ring;
mod scalar;
mod storage;
mod support;
mod vec;

pub mod prelude {
    /// Subtype chain:
    /// AbelianGroup ──┐
    ///                ├─→ Ring ─→ Commutative ─┐
    /// MulMonoid ─────┘     │                  ├─→ Scalar
    ///                      └─→ Invertible ────┘
    pub use crate::abelian::AbelianGroup;
    pub use crate::invertible::Invertible;
    pub use crate::mul_monoid::MulMonoid;
    pub use crate::nat::{
        Add, Mul, N0, N1, N2, N3, N4, N5, N6, N10, N12, N24, Nat, PeanoAdd, PeanoMul, Succ, Z,
    };
    pub use crate::ring::{Commutative, Ring};
    pub use crate::scalar::Scalar;
    pub use crate::storage::{Cons, Nil, Storage, StorageLen, assert_packed};
    pub use crate::support::{
        EffectiveZero, FromRational, FromRationalConstant, RationalConstant, StandardPart,
    };
    pub use crate::vec::{
        Access, Cross, Dot, ScalarMul, SplitVector, Vector, Vector2, Vector3, Vector6,
    };
    pub use crate::vector;
}

#[cfg(feature = "fixed")]
pub mod fixed {
    pub use ::fixed::*;
    pub use fix::Fix;
    mod fix;
}
