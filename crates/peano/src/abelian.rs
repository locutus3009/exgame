// SPDX-License-Identifier: MIT

use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};
use num_traits::ConstZero;

// ── ADDITION ──────────────────────────────────────────────────────────────
/// An abelian group under addition. The bottom of EVERYTHING. We go no lower: Neg is mandatory
/// (subtraction of signed coefficients, reverse), and below an abelian group there is no
/// operation that any function uses.
///
/// Send + Sync sit here, at the bottom, and not on `Scalar`: the carrier moves into a
/// rayon thread as a whole (`Vector<N, T>`, a multivector, `Tangent`), and the requirement
/// on the element is needed exactly where the carrier is assembled — in the `Nat::Repr` bound.
/// Put it higher up the tower and every function over `Ring` will drag `+ Send +
/// Sync` into its signature. The restriction is honest: all carriers here are plain data
/// (`f32`, `Fix`, `Sym`, `Jet`), and none of them holds an `Rc`/`Cell`.
pub trait AbelianGroup:
    Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + Clone
    + Copy
    + Debug
    + Send
    + Sync
{
    const ZERO: Self;
}

impl<T> AbelianGroup for T
where
    T: Add<Output = Self>
        + Sub<Output = Self>
        + Neg<Output = Self>
        + AddAssign
        + SubAssign
        + Clone
        + Copy
        + ConstZero
        + Debug
        + Send
        + Sync,
{
    const ZERO: Self = <T as ConstZero>::ZERO;
}
