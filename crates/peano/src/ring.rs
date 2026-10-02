// SPDX-License-Identifier: MIT

use crate::prelude::{AbelianGroup, MulMonoid};
use num_traits::Float;

// ── RING = two operations on one carrier, linked by distributivity ─────────
/// The additive side is a group, the multiplicative side a monoid.
/// Distributivity is a promise (Rust does not check axioms), as is everything below.
/// Does NOT promise commutativity of multiplication.
pub trait Ring: AbelianGroup + MulMonoid {}

// ── COMMUTATIVITY — a marker, not an operation ──────────────────────────────
/// Empty. Its only payload: the promise a*b == b*a. Serves as a bound where
/// an algorithm reorders factors (series, Neumann).
pub trait Commutative: Ring {}

impl<T> Ring for T where T: AbelianGroup + MulMonoid + Float {}
impl<T> Commutative for T where T: Ring + Float {}
