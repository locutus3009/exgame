// SPDX-License-Identifier: MIT

//! Dual axis markup (⋆₃) — COMPUTED, not a table; generic over the algebra.
//!
//! Axis i ↦ eᵢ (the i-th non-degenerate generator in increasing chain-bit order).
//! Physical quantities are marked up through ⋆₃ — the Euclidean sub-dual (complement in I₃,
//! the pseudoscalar of the Euclidean subspace, WITHOUT e0) — the same one that defines
//! cross. All signs come from complement_sign/blade_prod, i.e. from the +I anchor.
//! Axes and masks are derived from the metric (Algebra::ZERO_MASK), not written out.

use super::{Algebra, blade_prod, complement_sign};

/// The i-th Euclidean generator (in increasing bit order) — the basis vector of axis i.
pub const fn axis_vec<A: Algebra>(i: usize) -> usize {
    let euclid = (A::DIM - 1) & !A::ZERO_MASK; // non-degenerate bits = I₃
    let mut e = euclid;
    let mut k = i;
    loop {
        let low = e.isolate_lowest_one(); // lowest set bit
        if k == 0 {
            return low;
        }
        e ^= low;
        k -= 1;
    }
}

/// Euclidean blade of axis i: ⋆₃eᵢ = the complement in I₃ with the orientation sign.
/// This is the direction/moment (the same ⋆₃ that defines cross).
pub const fn euclid_blade<A: Algebra>(i: usize) -> (usize, i8) {
    let i3 = (A::DIM - 1) & !A::ZERO_MASK;
    let v = axis_vec::<A>(i);
    let comp = i3 ^ v;
    (comp, complement_sign(v, comp))
}

/// Radical blade of axis i: e0 ∧ eᵢ (e0 first) — sign from blade_prod.
/// This is the e0 part (force/linear part) and the translation generator.
pub const fn radical_blade<A: Algebra>(i: usize) -> (usize, i8) {
    let v = axis_vec::<A>(i);
    let (sign, mask) = blade_prod(A::ZERO_MASK, v, A::ZERO_MASK, A::NEG_MASK);
    (mask, sign)
}

/// Blade of a point's position along axis i: e0 ∧ ⋆₃eᵢ (grade-3), sign computed.
pub const fn point_blade<A: Algebra>(i: usize) -> (usize, i8) {
    let (ec_mask, ec_sign) = euclid_blade::<A>(i);
    let (s2, mask) = blade_prod(A::ZERO_MASK, ec_mask, A::ZERO_MASK, A::NEG_MASK);
    (mask, ec_sign * s2)
}

/// The Euclidean pseudoscalar I₃ (point weight / origin).
pub const fn i3_blade<A: Algebra>() -> usize {
    (A::DIM - 1) & !A::ZERO_MASK
}

/// Sign of the embedding of the angular generator (ω of a twist / τ of a wrench) into the Euclidean bivector.
///
/// It is NOT free and therefore NOT written out: it is forced by the consistency of two already
/// derived channels — the action of a twist on a point (velocity = ½`[T,P]`, read
/// with the point_blade markup) must agree with the ⋆₃-cross (the euclid_blade markup
/// of the wedge). Both markups are derived from complement_sign, so flipping the anchor
/// +I → −I coherently flips both cross and σ — there is no hand-written sign.
///
/// Computation — the defining basis case ω = e_z, r = e_x ⇒ v = e_z×e_x:
///   cross channel:    c_y = s_e1 · s_w,  (s_w, ·) = blade_prod(e_z, e_x)
///   velocity channel: v_y = s_d · (σ·s_b) · s_p · ½(s₁ − s₂)
/// where s_b/s_p/s_d are the markup signs of slot ω_z, slot r_x and the v_y read, and
/// ½(s₁ − s₂) is the coefficient of the commutator of a bivector with a trivector. σ solves
/// v_y = c_y; all factors are ±1 — self-inverse.
pub const fn angular_embed_sign<A: Algebra>() -> i8 {
    // cross channel
    let (vz, vx) = (axis_vec::<A>(2), axis_vec::<A>(0));
    let (s_w, m_w) = blade_prod(vz, vx, A::ZERO_MASK, A::NEG_MASK);
    let (m_e1, s_e1) = euclid_blade::<A>(1);
    assert!(m_w == m_e1); // e_z∧e_x — bivector of axis y
    // velocity channel
    let (m_b, s_b) = euclid_blade::<A>(2); // slot ω_z
    let (m_p, s_p) = point_blade::<A>(0); // slot r_x
    let (m_d, s_d) = point_blade::<A>(1); // read of v_y (grade-3, weight 0)
    let (s1, c1) = blade_prod(m_b, m_p, A::ZERO_MASK, A::NEG_MASK);
    let (s2, c2) = blade_prod(m_p, m_b, A::ZERO_MASK, A::NEG_MASK);
    assert!(c1 == m_d && c2 == m_d);
    let hs = (s1 as i32 - s2 as i32) / 2; // commutator: ±1, 0 — degeneracy
    assert!(hs == 1 || hs == -1);
    s_e1 * s_w * s_d * (hs as i8) * s_b * s_p
}

/// Markup of the angular slot of a screw: the Euclidean blade of axis i × the embedding sign σ.
/// Screw's ω/τ constructors write through it and angular()/torque() read through it — one σ for
/// both ends; the embed for twist and wrench is SHARED (they differ only in variance).
pub const fn angular_blade<A: Algebra>(i: usize) -> (usize, i8) {
    let (m, s) = euclid_blade::<A>(i);
    (m, s * angular_embed_sign::<A>())
}

/// The degenerate generator e0 (plane offset).
pub const fn e0_blade<A: Algebra>() -> usize {
    A::ZERO_MASK
}
