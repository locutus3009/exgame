// SPDX-License-Identifier: MIT

//! Geometric algebra descriptor: a generator chain + a sign law.
//!
//! THE ONLY FREE CHOICE OF THE ENTIRE SIGN SYSTEM IS ONE CONSTANT:
//!
//! b ∧ dual(b) = DUAL_ORIENTATION · I      (currently +I)
//!
//! It is expressed by exactly one value — `DUAL_ORIENTATION` (pinned by the
//! compile-time anchor below); `complement_sign` carries only the parity
//! of the permutation (not free) × this constant. Everything else — permutation
//! signs (`blade_prod`), the dual axis markup (⋆₃), cross,
//! the sign of the screw's angular embedding (`angular_embed_sign`), the wrench's force/moment,
//! the translator's direction, the even/Euclidean subalgebras — IS DERIVED from it
//! plus two structural conventions that carry no freedom: the generator chain
//! `Gen<SQ, Rest>` and the linear order of blades (increasing bitmask).
//! Flip the constant and the whole sign system flips consistently,
//! and not a single test can be «fixed up by hand».
//!
//! Bit convention: a gen gets bit = its position from `Nil` (the innermost
//! gen of the chain is bit 0, the outermost is the highest).

#[cfg(test)]
mod invariant_tests;
pub mod markup;
pub mod mv;
pub mod section;
pub mod store;
pub mod strata;

use core::marker::PhantomData;
use peano::prelude::*;

/// Field: 0 generators.
#[derive(Copy, Clone, Debug)]
pub struct Nil;
/// Add a generator e with e² = SQ (it gets the HIGHEST blade bit).
#[derive(Copy, Clone, Debug)]
pub struct Gen<const SQ: i8, Rest>(PhantomData<Rest>);

/// Properties of the algebra: the number of generators and the signature as associated
/// CONSTANTS (the sign law becomes a pure const fn; monomorphization
/// substitutes the masks as literals). There are no methods — const-traitness is not needed.
pub trait Algebra: Clone {
    /// The number of generators as a Peano type (the length of the strata list).
    type NgenP;
    /// 2^NGEN as a Peano type — structural doubling for each gen; the
    /// constant 2^n itself is written out nowhere. Needed by the boundary linear algebra
    /// (the DimP×DimP matrix of the regular representation).
    type DimP: Nat;
    const NGEN: usize;
    /// 2^NGEN — the number of blades (boundary traversals over blades).
    const DIM: usize;
    /// Bits of the degenerate generators (SQ = 0).
    const ZERO_MASK: usize;
    /// Bits of the generators with SQ = −1.
    const NEG_MASK: usize;
}
impl Algebra for Nil {
    type NgenP = Z;
    type DimP = Succ<Z>;
    const NGEN: usize = 0;
    const DIM: usize = 1;
    const ZERO_MASK: usize = 0;
    const NEG_MASK: usize = 0;
}
impl<const SQ: i8, Rest: Algebra> Algebra for Gen<SQ, Rest>
where
    Rest::DimP: PeanoAdd<Rest::DimP>,
    <Rest::DimP as PeanoAdd<Rest::DimP>>::Sum: Nat,
{
    type NgenP = Succ<Rest::NgenP>;
    type DimP = <Rest::DimP as PeanoAdd<Rest::DimP>>::Sum;
    const NGEN: usize = Rest::NGEN + 1;
    const DIM: usize = Rest::DIM * 2;
    const ZERO_MASK: usize = Rest::ZERO_MASK | if SQ == 0 { 1 << Rest::NGEN } else { 0 };
    const NEG_MASK: usize = Rest::NEG_MASK | if SQ == -1 { 1 << Rest::NGEN } else { 0 };
}

// ── Named algebras ────────────────────────────────────────────────────
// They live HERE and are not re-exported from the crate root: the names Gen/Dual/Cl30
// would collide with the legacy re-exports of blade.rs. The root re-export is at the
// cutover (A7), when blade.rs dies.
pub type Pga3 = Gen<0, Gen<1, Gen<1, Gen<1, Nil>>>>; // e0²=0, e1²=e2²=e3²=1
pub type Complex = Gen<-1, Nil>; //                      ≅ ℂ  (e²=−1)
pub type Quaternion = Gen<-1, Gen<-1, Nil>>; //          ≅ ℍ
pub type Dual = Gen<0, Nil>; //                          dual numbers (ε²=0)
pub type Cl30 = Gen<1, Gen<1, Gen<1, Nil>>>; //          Euclidean Cl(3,0)

// ── Sign law on bitmasks (compile-time) ─────────────────────────────
// A blade is a bitmask of the generators involved. grade = popcount.
// Product: the result is a^b (symmetric difference); sign = parity of the permutation
// when merging into increasing order × the metric (from the masks):
//   a degenerate shared generator (∈ zero) → sign 0 (annihilation);
//   a shared generator with SQ=−1 (∈ neg) adds a minus.
// Returns sign=0 as «the blade annihilated».
pub const fn blade_grade(b: usize) -> usize {
    b.count_ones() as usize
}
pub const fn blade_prod(a: usize, b: usize, zero_mask: usize, neg_mask: usize) -> (i8, usize) {
    let shared = a & b;
    if shared & zero_mask != 0 {
        return (0, 0); // degenerate generator squared → annihilation
    }
    // metric: (−1)^(number of shared generators with SQ=−1)
    let neg = (shared & neg_mask).count_ones() & 1;
    // permutation sign when merging into increasing order
    let mut swaps = 0u32;
    let mut aa = a >> 1;
    while aa != 0 {
        swaps += (aa & b).count_ones();
        aa >>= 1;
    }
    let sign = if (swaps + neg) & 1 == 0 { 1i8 } else { -1i8 };
    (sign, a ^ b)
}

/// ★ THE SYSTEM ANCHOR — the only free sign, as ONE constant:
/// b ∧ dual(b) = DUAL_ORIENTATION · I. Everything else is math without freedom.
pub const DUAL_ORIENTATION: i8 = 1;

/// Dual sign: the parity of the permutation gluing the disjoint blades a and
/// b into I (NOT free), × DUAL_ORIENTATION (all the free choice is in it).
pub const fn complement_sign(a: usize, b: usize) -> i8 {
    let mut inversions = 0u32;
    let mut bi = 0;
    while bi < usize::BITS as usize {
        if (a >> bi) & 1 == 1 {
            let lower = (1usize << bi) - 1;
            inversions += (b & lower).count_ones();
        }
        bi += 1;
    }
    if inversions.is_multiple_of(2) {
        DUAL_ORIENTATION
    } else {
        -DUAL_ORIENTATION
    }
}

// ── Compile-time orientation anchor ──────────────────────────────────────────
// b ∧ dual(b) = DUAL_ORIENTATION·I for EVERY blade: dual(b) carries
// complement_sign(b, comp), and the wedge of disjoint blades is their product
// (the metric plays no part: there are no shared generators). Checked against THE SAME
// anchor constant. Verified AT COMPILE TIME — const fn all the way down.
const fn orientation_anchor_holds(dim: usize, zero_mask: usize, neg_mask: usize) -> bool {
    let top = dim - 1;
    let mut b = 0;
    while b < dim {
        let comp = top ^ b;
        let (wedge_sign, c) = blade_prod(b, comp, zero_mask, neg_mask);
        if c != top
            || wedge_sign as i32 * complement_sign(b, comp) as i32 != DUAL_ORIENTATION as i32
        {
            return false;
        }
        b += 1;
    }
    true
}
const _: () = {
    assert!(orientation_anchor_holds(
        <Cl30 as Algebra>::DIM,
        <Cl30 as Algebra>::ZERO_MASK,
        <Cl30 as Algebra>::NEG_MASK,
    ));
    assert!(orientation_anchor_holds(
        <Pga3 as Algebra>::DIM,
        <Pga3 as Algebra>::ZERO_MASK,
        <Pga3 as Algebra>::NEG_MASK,
    ));
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_and_dims_derive_from_the_chain() {
        assert_eq!(<Pga3 as Algebra>::NGEN, 4);
        assert_eq!(<Pga3 as Algebra>::DIM, 16);
        // e0 is the outermost gen of the chain ⇒ the highest bit.
        assert_eq!(<Pga3 as Algebra>::ZERO_MASK, 0b1000);
        assert_eq!(<Pga3 as Algebra>::NEG_MASK, 0);
        assert_eq!(<Complex as Algebra>::NEG_MASK, 1);
        // The type-level dimension matches the constant.
        assert_eq!(<<Pga3 as Algebra>::DimP as Nat>::LEN, 16);
        assert_eq!(<<Complex as Algebra>::DimP as Nat>::LEN, 2);
        assert_eq!(<<Nil as Algebra>::DimP as Nat>::LEN, 1);
    }

    #[test]
    fn quaternion_signs_from_masks() {
        // ℍ = Cl(0,2): i = bit0, j = bit1, k = i·j = mask 3.
        let (zm, nm) = (
            <Quaternion as Algebra>::ZERO_MASK,
            <Quaternion as Algebra>::NEG_MASK,
        );
        assert_eq!(blade_prod(1, 2, zm, nm), (1, 3)); // i·j = +k
        assert_eq!(blade_prod(2, 1, zm, nm), (-1, 3)); // j·i = −k (anticommutation)
        assert_eq!(blade_prod(1, 1, zm, nm), (-1, 0)); // i² = −1
        assert_eq!(blade_prod(2, 2, zm, nm), (-1, 0)); // j² = −1
        assert_eq!(blade_prod(3, 3, zm, nm), (-1, 0)); // k² = −1
    }

    #[test]
    fn degenerate_generator_annihilates() {
        let (zm, nm) = (<Pga3 as Algebra>::ZERO_MASK, <Pga3 as Algebra>::NEG_MASK);
        assert_eq!(blade_prod(0b1000, 0b1000, zm, nm), (0, 0)); // e0² = 0
        // but e0 in a product with a disjoint blade survives
        let (s, c) = blade_prod(0b1000, 0b0001, zm, nm);
        assert_eq!(c, 0b1001);
        assert!(s != 0);
    }

    #[test]
    fn orientation_anchor_runtime_mirror() {
        // A duplicate of the compile-time anchor — so that a breakage is also visible in the prove-run
        // of the tests, with the coordinates of the failure.
        for b in 0..<Pga3 as Algebra>::DIM {
            let top = <Pga3 as Algebra>::DIM - 1;
            let comp = top ^ b;
            let (s, c) = blade_prod(
                b,
                comp,
                <Pga3 as Algebra>::ZERO_MASK,
                <Pga3 as Algebra>::NEG_MASK,
            );
            assert_eq!(c, top, "b={b}: complement is not top");
            assert_eq!(
                s as i32 * complement_sign(b, comp) as i32,
                DUAL_ORIENTATION as i32,
                "b={b}: b∧dual(b) ≠ DUAL_ORIENTATION·I"
            );
        }
    }
}
