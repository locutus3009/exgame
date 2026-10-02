// SPDX-License-Identifier: MIT

//! Sections as predicates on blades (without a single blade literal).
//!
//! «Take grade-2» / «the force part» is not a list of indices but a predicate
//! «does blade b belong to this subspace», derived from the grade and the METRIC
//! (`Algebra::ZERO_MASK`). Predicates compose (a Boolean algebra),
//! cardinality falls out by counting. The only primitive operation is
//! `contains`; everything else (cardinality, projections, physical quantities)
//! is derived from it.

use super::{Algebra, blade_grade};
use core::marker::PhantomData;
use peano::prelude::*;

pub trait Section<A: Algebra> {
    fn contains(blade: usize) -> bool;
}

// ── Atoms ───────────────────────────────────────────────────────────────────
/// Grade⟨k⟩: rank(b) = k (the number of generators in the blade). k is a peano Nat (LEN).
pub struct Grade<K>(PhantomData<K>);
impl<A: Algebra, K: Nat> Section<A> for Grade<K> {
    fn contains(b: usize) -> bool {
        blade_grade(b) == K::LEN
    }
}
/// Contains⟨g⟩: the blade touches generator g.
pub struct Contains<G>(PhantomData<G>);
impl<A: Algebra, G: Nat> Section<A> for Contains<G> {
    fn contains(b: usize) -> bool {
        (b >> G::LEN) & 1 == 1
    }
}
/// Metric⟨p⟩: the signature of the blade (∏ of the squares of its generators) equals p ∈ {0,+1,−1}.
/// The signature is taken from the algebra's masks: a degenerate shared bit ⇒ 0, otherwise the sign
/// is determined by the parity of the number of −1 generators.
pub trait Signature {
    fn matches(b: usize, zero_mask: usize, neg_mask: usize) -> bool;
}
pub struct Degen; // degenerate (p = 0)
pub struct Plus; // p = +1
pub struct Minus; // p = −1
impl Signature for Degen {
    fn matches(b: usize, zero_mask: usize, _neg: usize) -> bool {
        b & zero_mask != 0
    }
}
impl Signature for Plus {
    fn matches(b: usize, zero_mask: usize, neg: usize) -> bool {
        b & zero_mask == 0 && (b & neg).count_ones() & 1 == 0
    }
}
impl Signature for Minus {
    fn matches(b: usize, zero_mask: usize, neg: usize) -> bool {
        b & zero_mask == 0 && (b & neg).count_ones() & 1 == 1
    }
}
pub struct Metric<P>(PhantomData<P>);
impl<A: Algebra, P: Signature> Section<A> for Metric<P> {
    fn contains(b: usize) -> bool {
        P::matches(b, A::ZERO_MASK, A::NEG_MASK)
    }
}

// ── Boolean combinators (sections form a Boolean algebra) ────────────────────
pub struct And<P, Q>(PhantomData<(P, Q)>);
pub struct Or<P, Q>(PhantomData<(P, Q)>);
pub struct Not<P>(PhantomData<P>);
pub struct Empty;
pub struct Full;
impl<A: Algebra, P: Section<A>, Q: Section<A>> Section<A> for And<P, Q> {
    fn contains(b: usize) -> bool {
        P::contains(b) && Q::contains(b)
    }
}
impl<A: Algebra, P: Section<A>, Q: Section<A>> Section<A> for Or<P, Q> {
    fn contains(b: usize) -> bool {
        P::contains(b) || Q::contains(b)
    }
}
impl<A: Algebra, P: Section<A>> Section<A> for Not<P> {
    fn contains(b: usize) -> bool {
        !P::contains(b)
    }
}
impl<A: Algebra> Section<A> for Empty {
    fn contains(_b: usize) -> bool {
        false
    }
}
impl<A: Algebra> Section<A> for Full {
    fn contains(_b: usize) -> bool {
        true
    }
}

/// Cardinality of a section = |{b : contains(b)}| — a fold over the predicate.
pub fn section_dim<A: Algebra, P: Section<A>>() -> usize {
    let mut n = 0;
    let mut b = 0;
    while b < A::DIM {
        if P::contains(b) {
            n += 1;
        }
        b += 1;
    }
    n
}

// ── Grade sections as marker types (strata carriers will arrive in A6) ──────────────
/// The whole algebra (all grades).
pub struct AllGrades;
/// The even subalgebra (grades 0,2,4,…) — the home of versors/motors.
pub struct EvenGrades;
/// Scalars (grade 0 only).
pub struct ScalarGrade;
/// Grade 2 only (bivectors) — the carrier of twist/wrench (se(3) and se(3)*).
pub struct Grade2;

/// Marker: the section is CLOSED under gp (a subring). Closed: scalars {0},
/// the even subalgebra {0,2,4,…}, the whole algebra. NOT closed: Odd (odd·odd=even),
/// a single Grade-k>0 (grade2·grade2 gives grades 0,2,4).
pub trait ClosedGp {}
impl ClosedGp for AllGrades {}
impl ClosedGp for EvenGrades {}
impl ClosedGp for ScalarGrade {}

/// Marker: the section is CLOSED under the commutator `[a,b]`=ab−ba (a Lie subalgebra), even
/// if it is NOT closed under gp itself. A DIFFERENT closedness, do not merge with ClosedGp:
/// Grade2 is ClosedBracket but NOT ClosedGp: for two bivectors gp gives grades
/// 0,2,4, but the symmetric part (0 and 4) cancels in ab−ba — what remains is pure
/// grade-2. The even subalgebra is closed under both gp and the bracket.
pub trait ClosedBracket {}
impl ClosedBracket for AllGrades {}
impl ClosedBracket for EvenGrades {}
impl ClosedBracket for ScalarGrade {}
impl ClosedBracket for Grade2 {}

// ── Physical quantities as Boolean expressions over atoms (without blade indices) ──────
/// Radical = degenerate metric; Euclidean = its complement.
pub type Radical = Metric<Degen>;
pub type Euclidean = Not<Metric<Degen>>;
/// Force ≡ the degenerate part of a bivector; moment ≡ the Euclidean part of a bivector.
pub type Force = And<Grade<N2>, Radical>;
pub type Moment = And<Grade<N2>, Euclidean>;

#[cfg(test)]
mod tests {
    use super::super::Pga3;
    use super::*;

    #[test]
    fn sections_have_no_blade_literals() {
        // Cardinalities fall out by counting over the predicate, not written out:
        assert_eq!(section_dim::<Pga3, Grade<N2>>(), 6); // bivector
        assert_eq!(section_dim::<Pga3, Force>(), 3); // grade-2 ∩ radical
        assert_eq!(section_dim::<Pga3, Moment>(), 3); // grade-2 ∩ Euclidean

        // Atoms: Contains⟨e0⟩ = half of the blades (8/16); the Pga3 metric = (0,+,+,+),
        // so there is no p=−1 at all, and p=+1 is all 8 non-degenerate blades.
        assert_eq!(section_dim::<Pga3, Contains<N3>>(), 8); // e0 = bit 3
        assert_eq!(section_dim::<Pga3, Metric<Minus>>(), 0);
        assert_eq!(section_dim::<Pga3, Metric<Plus>>(), 8);

        // Boolean algebra: Full=2⁴, Empty=0, P ∪ ¬P = the whole, the complement adds up.
        assert_eq!(section_dim::<Pga3, Full>(), 16);
        assert_eq!(section_dim::<Pga3, Empty>(), 0);
        assert_eq!(section_dim::<Pga3, Or<Radical, Not<Radical>>>(), 16);
        assert_eq!(
            section_dim::<Pga3, Radical>() + section_dim::<Pga3, Not<Radical>>(),
            16
        );
    }
}
