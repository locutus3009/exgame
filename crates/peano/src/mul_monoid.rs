// SPDX-License-Identifier: MIT

use core::{fmt::Debug, ops::Mul};
use num_traits::ConstOne;

// ── MULTIPLICATION ────────────────────────────────────────────────────────
/// A monoid under multiplication: associativity + identity, WITHOUT inverses.
/// A monoid specifically, not a group — elements have no guarantee of invertibility
/// (0 never has one; in Z/n only elements coprime to n do).
pub trait MulMonoid: Mul<Output = Self> + Copy + Debug {
    const ONE: Self;
}

impl<T> MulMonoid for T
where
    T: Mul<Output = Self> + Copy + ConstOne + Debug,
{
    const ONE: Self = <T as ConstOne>::ONE;
}
