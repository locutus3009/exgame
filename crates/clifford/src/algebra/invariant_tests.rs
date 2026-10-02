// SPDX-License-Identifier: MIT

//! Invariant tests of the algebraic core (strata, CD gp, dual, sections) —
//! the specification that moved over from experiments/peano when it was dissolved (A6/A7).

use peano::prelude::*;

#[test]
fn pascal_triangle() {
    use crate::algebra::store::Count;
    use crate::algebra::strata::{Bivec, GradeForm};
    use crate::algebra::{Gen, Nil};
    type Z0 = Z;
    type Z1 = Succ<Z>;
    type Z2 = Succ<Succ<Z>>;
    type Z3 = Succ<Succ<Succ<Z>>>;
    type Z4 = Succ<Succ<Succ<Succ<Z>>>>;

    // The grade-2 stratum of PGA3 (4 generators) — a form of C(4,2)=6 leaves.
    assert_eq!(<Bivec<f64> as Count>::N, 6);

    // The full Pascal row for n=4: 1,4,6,4,1  (sum = 2⁴ = 16).
    type Pga3 = Gen<0, Gen<1, Gen<1, Gen<1, Nil>>>>;
    assert_eq!(<<Pga3 as GradeForm<Z0>>::Store<f64> as Count>::N, 1);
    assert_eq!(<<Pga3 as GradeForm<Z1>>::Store<f64> as Count>::N, 4);
    assert_eq!(<<Pga3 as GradeForm<Z2>>::Store<f64> as Count>::N, 6);
    assert_eq!(<<Pga3 as GradeForm<Z3>>::Store<f64> as Count>::N, 4);
    assert_eq!(<<Pga3 as GradeForm<Z4>>::Store<f64> as Count>::N, 1);
}

#[test]
fn geometric_product() {
    use crate::algebra::mv::Mv;
    use crate::algebra::{Cl30, Complex, Dual, Pga3, Quaternion};

    // ── ℂ: e² = −1, full complex multiplication ───────────────────
    type C = Mv<Complex, f64>;
    let mk_c = |a: f64, b: f64| {
        let mut m = C::zero();
        m.set(0, a);
        m.set(1, b);
        m
    };
    assert_eq!(C::basis(1).gp(&C::basis(1)).get(0), -1.0);
    let z = mk_c(2.0, 3.0).gp(&mk_c(4.0, 5.0));
    assert_eq!(z.get(0), 2.0 * 4.0 - 3.0 * 5.0); // −7
    assert_eq!(z.get(1), 2.0 * 5.0 + 3.0 * 4.0); // 22

    // ── ℍ: Hamilton's relations ──────────────────────────────────
    type H = Mv<Quaternion, f64>;
    let one = H::scalar(1.0);
    let i = H::basis(1);
    let j = H::basis(2);
    let k = i.gp(&j); // i·j = e₁₂
    assert_eq!(i.gp(&i), -one); // i² = −1
    assert_eq!(j.gp(&j), -one); // j² = −1
    assert_eq!(k.gp(&k), -one); // k² = −1
    assert_eq!(j.gp(&k), i); // jk = i
    assert_eq!(k.gp(&i), j); // ki = j
    assert_eq!(j.gp(&i), -k); // ji = −k (anticommutation)

    // ── dual numbers: ε² = 0 ─────────────────────────────────────
    type D = Mv<Dual, f64>;
    assert_eq!(D::basis(1).gp(&D::basis(1)), D::zero());

    // ── Cl(3,0): pseudoscalar I = e₁₂₃, I² = −1 ────────────────────
    type G = Mv<Cl30, f64>;
    assert_eq!(G::basis(7).gp(&G::basis(7)).get(0), -1.0);

    // ── PGA3: degenerate generator e₀ (highest bit = 8), e₀² = 0 ──
    type P = Mv<Pga3, f64>;
    assert_eq!(P::basis(8).gp(&P::basis(8)), P::zero());
    // but the Euclidean e₁ (bit 4) gives e₁² = +1
    assert_eq!(P::basis(4).gp(&P::basis(4)).get(0), 1.0);
}

#[test]
fn gp_is_associative_and_distributive() {
    use crate::algebra::Cl30;
    use crate::algebra::mv::Mv;
    type G = Mv<Cl30, f64>;
    let mk = |c: [f64; 8]| {
        let mut m = G::zero();
        for (s, &v) in c.iter().enumerate() {
            m.set(s, v);
        }
        m
    };
    let a = mk([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = mk([2.0, 0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 1.0]);
    let c = mk([1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0]);
    assert_eq!(a.gp(&b).gp(&c), a.gp(&b.gp(&c))); // associativity
    assert_eq!(a.gp(&(b + c)), a.gp(&b) + a.gp(&c)); // distributivity L
    assert_eq!((a + b).gp(&c), a.gp(&c) + b.gp(&c)); // distributivity R
}

#[test]
fn autodiff_through_geometric_product() {
    // We differentiate the GEOMETRIC PRODUCT without a single special case:
    // the multivector's coefficients are Jets, and the derivative flows through gp.
    // In ℂ: z(x) = x + i, z·z = (x²−1) + 2x·i.  d/dx: scalar 2x, at i → 2.
    use crate::Jet1 as Jet;
    use crate::algebra::Complex;
    use crate::algebra::mv::Mv;
    type J = Jet<f64>;
    type Z2 = Mv<Complex, J>;

    let x = 3.0;
    let mut z = Z2::zero();
    z.set(0, J::from_grad(x, vector![1.0])); // real part = the variable x
    z.set(1, J::embed(1.0)); // imaginary part = the constant 1

    let zz = z.gp(&z);
    // scalar: x²−1 = 8, ∂/∂x = 2x = 6
    assert_eq!(zz.get(0).base(), x * x - 1.0);
    assert_eq!(*zz.get(0).grad().get::<Z>(), 2.0 * x);
    // at i: 2x = 6, ∂/∂x = 2
    assert_eq!(zz.get(1).base(), 2.0 * x);
    assert_eq!(*zz.get(1).grad().get::<Z>(), 2.0);
}

#[test]
fn products_as_grade_sections_of_gp() {
    use crate::algebra::Cl30;
    use crate::algebra::mv::Mv;
    type G = Mv<Cl30, f64>;
    let e1 = G::basis(1);
    let e2 = G::basis(2);

    // wedge = the grade-⟨sum⟩ section of gp: grades add, ∧ with itself = 0.
    assert_eq!(e1.wedge(&e2).get(3), 1.0); // e1∧e2 = e12 (mask 3)
    assert_eq!(e1.wedge(&e1), G::zero());
    // dot = grade-⟨difference⟩: e1·e1 = 1 (scalar), orthogonal → 0.
    assert_eq!(e1.dot(&e1).get(0), 1.0);
    assert_eq!(e1.dot(&e2), G::zero());
    // For orthogonal vectors gp = wedge + dot.
    assert_eq!(e1.gp(&e2), e1.wedge(&e2) + e1.dot(&e2));

    // The dual maps grade k to n−k: dual(scalar) = pseudoscalar (mask 7).
    assert_eq!(G::scalar(1.0).dual().get(7), 1.0);
    assert_eq!(G::basis(7).dual().get(0).abs(), 1.0); // dual(pseudoscalar) = ±scalar
}

#[test]
fn cross_is_hodge_dual_of_wedge() {
    // Principle: the axis markup is not a rank in the section but the Hodge dual. Axis i ↦ ⋆eᵢ,
    // and one and the same ⋆ (multiplication by the pseudoscalar, signs from blade_prod)
    // defines cross. We check that the HAND-WRITTEN cross is exactly ⋆(a∧b): all signs
    // (including the «cyclic» e₃₁ = ⋆e₂) fall out of the dual rather than being chosen by hand.
    use crate::algebra::Cl30;
    use crate::algebra::mv::{Mv, inject, project};
    use crate::algebra::section::Grade;
    type One = Succ<Z>;
    type G = Mv<Cl30, f64>;

    // Vec3 ↔ grade-1 in Cl(3,0): axes = basis vectors (increasing order).
    let lift = |v: &Vector<N3, f64>| inject::<Cl30, Grade<One>, f64, N3>(v);
    let hodge_cross = |a: &Vector<N3, f64>, b: &Vector<N3, f64>| -> Vector<N3, f64> {
        // ⋆(a∧b): wedge, then dual, then read grade-1 back into a Vec3.
        let ab: G = lift(a).wedge(&lift(b));
        project::<Cl30, Grade<One>, f64, N3>(&ab.dual())
    };

    // The hand-written cross from the vector part == the Hodge dual of the wedge, componentwise.
    let a: Vector<N3, f64> = vector![1.0, 2.0, 3.0];
    let b: Vector<N3, f64> = vector![-4.0, 5.0, 6.0];
    assert_eq!(hodge_cross(&a, &b), a.cross(b));

    // The collinear case is STRUCTURALLY zero: a∧a = 0 ⇒ ⋆(a∧a) = 0, with no magnitudes.
    assert_eq!(lift(&a).wedge(&lift(&a)), G::zero());
    assert_eq!(hodge_cross(&a, &a), <Vector<N3, f64> as AbelianGroup>::ZERO);
}

#[test]
fn radical_and_euclid_labels_are_one_anchor() {
    // Symmetric to euclid_cross, but for the moment branch and ALGEBRAICALLY (without
    // a sandwich): the full dual maps the euclid markup (⋆₃eᵢ, force) to the
    // radical markup (e₀∧eᵢ, moment). So both tables are one and the same thing,
    // hung on the dual anchor. euclid_cross has already tied the euclid markup to
    // cross ⇒ the radical markup is tied to cross transitively, directly, not only
    // through the aggregate parallel-axis.
    use crate::algebra::Pga3;
    use crate::algebra::mv::Mv;
    use crate::pga3::Twist;
    type P = Mv<Pga3, f64>;
    let zero3 = <Vector<N3, f64> as AbelianGroup>::ZERO;
    let v: Vector<N3, f64> = vector![1.0, -2.0, 3.0]; // asymmetric

    // Both embeddings go through the PUBLIC Twist bridge (as_mv = widen of the narrow carrier),
    // without the flat contour. Twist puts the linear part into radical (+), the angular part
    // into euclid (−): euclid embedding v = −Twist(0,v), radical embedding v = Twist(v,0).
    let euclid_embed: P = -Twist::new(&zero3, &v).as_mv(); // ⋆₃eᵢ (force)
    let radical_embed: P = Twist::new(&v, &zero3).as_mv(); // e₀∧eᵢ (moment)

    // dual maps a euclid-marked bivector to a radical-marked one: ONE
    // global sign, not a permutation/motley signs. Were there a cross-axis knip
    // between the markups, we would get a scramble here, not −v.
    let dualized: P = euclid_embed.dual();
    // We read the radical part with a narrow twist: linear() = line_moment_store (radical).
    assert_eq!(Twist::from_mv(&dualized).linear(), zero3 - v); // == −v, all axes
    // At the multivector level: dual(euclid embedding) = −(radical embedding).
    assert_eq!(dualized, -radical_embed);
}

#[test]
fn mv_inverse_norm_deepen() {
    use crate::Jet1 as Jet;
    use crate::algebra::mv::Mv;
    use crate::algebra::{Cl30, Complex};
    // Inversion via Bareiss: (2+3i)⁻¹ = (2−3i)/13.
    type C = Mv<Complex, f64>;
    let mut z = C::zero();
    z.set(0, 2.0);
    z.set(1, 3.0);
    let inv = z.inverse().unwrap();
    let prod = z.gp(&inv);
    assert!((prod.get(0) - 1.0).abs() < 1e-9 && prod.get(1).abs() < 1e-9);
    // 0 is not invertible.
    assert!(C::zero().inverse().is_none());

    // Norm: the vector (3,4) in Cl(3,0) ⇒ ‖·‖ = 5.
    type G = Mv<Cl30, f64>;
    let mut v = G::zero();
    v.set(1, 3.0);
    v.set(2, 4.0);
    assert!((v.norm() - 5.0).abs() < 1e-9);

    // deepen: lift an f64 multivector into a Jet (constant ⇒ zero derivative).
    let d: Mv<Cl30, Jet<f64>> = v.deepen();
    assert_eq!(d.get(1).base(), 3.0);
    assert_eq!(*d.get(1).grad().get::<Z>(), 0.0);
}

#[test]
fn general_grade_sections_and_closure() {
    use crate::algebra::mv::closed_gp;
    use crate::algebra::section::{AllGrades, EvenGrades, ScalarGrade};
    use crate::algebra::store::{store_basis, store_get};
    use crate::algebra::strata::SectionStore;
    use crate::algebra::{Algebra, Pga3};
    use core::mem::size_of;

    // The structural sizes of the PGA3 sections fall out of grade membership:
    // scalar {0} = 1, even {0,2,4} = 8, the whole algebra = 16.
    assert_eq!(
        size_of::<SectionStore<Pga3, ScalarGrade, f64>>(),
        size_of::<f64>()
    );
    assert_eq!(
        size_of::<SectionStore<Pga3, EvenGrades, f64>>(),
        8 * size_of::<f64>()
    );
    assert_eq!(
        size_of::<SectionStore<Pga3, AllGrades, f64>>(),
        16 * size_of::<f64>()
    );

    // Closedness: closed_gp requires Spec: ClosedGp and stays in the carrier.
    // Scalars are a subring: grade0·grade0 = grade0.
    type Sc = SectionStore<Pga3, ScalarGrade, f64>;
    let two: Sc = store_basis::<Pga3, f64, Sc>(0, 2.0);
    let three: Sc = store_basis::<Pga3, f64, Sc>(0, 3.0);
    let six = closed_gp::<Pga3, ScalarGrade, f64>(&two, &three);
    assert_eq!(store_get(&six, 0, Pga3::NGEN), 6.0);

    // (Odd and a single Grade-k are NOT closed ⇒ closed_gp does not typecheck for them —
    // this is checked by there being no impl ClosedGp for them.)
}

#[test]
fn orientation_anchor() {
    // ★ The ONLY free choice of the whole system: b ∧ dual(b) = +I for
    // EVERY basis blade. This is the definition of the dual's orientation; from it
    // are derived ⋆₃, cross, the wrench's force/moment, the translator's sign, the whole sign
    // system. It is pinned here: flip it to −I and this test fails, rather than
    // having to «fix up the signs by hand» somewhere further down.
    use crate::algebra::Cl30;
    use crate::algebra::mv::Mv;
    type G = Mv<Cl30, f64>;
    const TOP: usize = 7; // pseudoscalar e₁₂₃ (all generators)
    for b in 0..8 {
        let bb = G::basis(b);
        let w = bb.wedge(&bb.dual()); // b ∧ dual(b)
        assert_eq!(w.get(TOP), 1.0, "b={b}: b∧dual(b) must be +I");
        for s in 0..8 {
            if s != TOP {
                assert_eq!(w.get(s), 0.0, "b={b}, s={s}: pseudoscalar only");
            }
        }
    }
}

#[test]
fn euclidean_blade_section() {
    use crate::algebra::store::{gp_store, store_basis, store_get};
    use crate::algebra::strata::EuclideanStore;
    use crate::algebra::{Algebra, Pga3};
    use core::mem::size_of;
    // BLADE section: a filter WITHIN a grade. The Euclidean subalgebra of PGA3 stores
    // grade-2 = 3 blades (e12,e13,e23), NOT 6; in total 1+3+3+1 = 8 = dim Cl(3,0).
    type Eucl = EuclideanStore<Pga3, f64>;
    assert_eq!(size_of::<Eucl>(), 8 * size_of::<f64>());

    // Closed: e1·e2 = e12, both Euclidean (bit e₀=0), the result is in the carrier.
    // e1=mask4, e2=mask2, e12=mask6 (e₀=8 — not involved).
    let e1: Eucl = store_basis::<Pga3, f64, Eucl>(4, 1.0);
    let e2: Eucl = store_basis::<Pga3, f64, Eucl>(2, 1.0);
    let e12 = gp_store::<Pga3, f64, Eucl>(&e1, &e2);
    assert_eq!(store_get(&e12, 6, Pga3::NGEN).abs(), 1.0);
    // e1² = +1 (a scalar is Euclidean too).
    let sq = gp_store::<Pga3, f64, Eucl>(&e1, &e1);
    assert_eq!(store_get(&sq, 0, Pga3::NGEN), 1.0);
}

#[test]
fn grade_algebra_involutions() {
    use crate::algebra::Cl30;
    use crate::algebra::mv::Mv;
    type G = Mv<Cl30, f64>;
    // An element with all grades 0..3.
    let mut a = G::zero();
    for s in 0..8 {
        a.set(s, (s as f64) + 1.0);
    }
    // even + odd = the whole; grade projections sum to the whole.
    assert_eq!(a.even() + a.odd(), a);
    assert_eq!(a.grade(0) + a.grade(1) + a.grade(2) + a.grade(3), a);
    // Involutions: signs by grade (−1)^g / (−1)^{g(g−1)/2} / (−1)^{g(g+1)/2}.
    // grade involution: even +, odd −.
    assert_eq!(a.grade_involution(), a.even() - a.odd());
    // Clifford = reverse∘involution.
    assert_eq!(a.clifford_conjugate(), a.reverse().grade_involution());
}

#[test]
fn closed_form_exp_and_ad_through_it() {
    use crate::Jet1 as Jet;
    use crate::algebra::Cl30;
    use crate::algebra::mv::Mv;
    type G = Mv<Cl30, f64>;
    const E12: usize = 6; // e₁∧e₂ in Cl(3,0), e₁₂² = −1
    const E1: usize = 4;
    const E2: usize = 2;

    // CLOSED form (not a series): exp(θ e₁₂) = cos θ + sin θ e₁₂, exactly.
    let th = 0.7_f64;
    let r = G::basis(E12).scale(th).exp();
    assert!((r.get(0) - th.cos()).abs() < 1e-12);
    assert!((r.get(E12) - th.sin()).abs() < 1e-12);
    // The seam at zero is smooth: exp(0) = 1 exactly (the Taylor branch of sinc/cos).
    assert_eq!(G::zero().exp().get(0), 1.0);

    // The rotor R = exp(½θ e₁₂) rotates e₁ in the 12 plane (norm preserved).
    let out = G::basis(E12).scale(0.5 * th).exp().conjugate(&G::basis(E1));
    assert!((out.get(E1).powi(2) + out.get(E2).powi(2) - 1.0).abs() < 1e-12);

    // AD THROUGH the closed-form exp: θ is a Jet variable, exp differentiates itself.
    type J = Jet<f64>;
    type GJ = Mv<Cl30, J>;
    let bj = GJ::basis(E12).scale(J::from_grad(th, vector![1.0]));
    let rj = bj.exp();
    assert!((rj.get(0).base() - th.cos()).abs() < 1e-12);
    assert!((*rj.get(0).grad().get::<Z>() + th.sin()).abs() < 1e-12); // d(cos θ)=−sin θ
    assert!((*rj.get(E12).grad().get::<Z>() - th.cos()).abs() < 1e-12); // d(sin θ)=cos θ
}

#[test]
fn se3_dof_via_peano() {
    // The 6 DOF of se(3) = the dimension of the PGA3 bivector = C(4,2) — and this FALLS OUT as
    // the Peano length of the twist, not written out.
    use crate::algebra::Pga3;
    use crate::algebra::mv::{Mv, inject};
    use crate::algebra::section::{Grade, section_dim};
    type Six = Succ<Succ<Succ<Succ<Succ<Succ<Z>>>>>>;
    type Two = Succ<Succ<Z>>;
    type P = Mv<Pga3, f64>;
    assert_eq!(section_dim::<Pga3, Grade<Two>>(), 6);
    assert_eq!(<Vector<Six, f64> as StorageLen>::LEN, 6); // twist length = number of DOFs

    // A twist as a Peano vector of 6 DOF → a PGA3 bivector (embedding by section).
    // A pure rotation about the z axis (only the Rz component).
    let twist: Vector<Six, f64> = vector![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let biv: P = inject::<Pga3, Grade<Two>, f64, Six>(&twist);
    // exp of a simple (Euclidean) bivector is a rotor; norm ⟨R R̃⟩₀ = 1.
    let rotor = biv.scale(0.5).exp();
    let nn = rotor.gp(&rotor.reverse()).get(0);
    assert!((nn - 1.0).abs() < 1e-12);

    // so(3) structure constants via the gp commutator: [e₂₃, e₁₃] ∝ e₁₂.
    const E23: usize = 3;
    const E13: usize = 5;
    const E12: usize = 6;
    let (rx, ry) = (P::basis(E23), P::basis(E13));
    let comm = rx.commutator(&ry).scale(0.5); // = ½(rx·ry − ry·rx)
    assert!(comm.get(E12).abs() > 0.5); // closes onto the third rotation axis
}

// ── se(3) as a Lie algebra: bracket, ad, ad*, velocity_at ────────────────

#[test]
fn mv_commutator_is_antisymmetric() {
    use crate::algebra::Cl30;
    use crate::algebra::mv::Mv;
    type G = Mv<Cl30, f64>;
    // Asymmetric elements (all grades).
    let mut a = G::zero();
    let mut b = G::zero();
    for s in 0..8 {
        a.set(s, (s as f64) + 1.0);
        b.set(s, (7 - s) as f64 * 0.5 + 0.3);
    }
    // commutator(a,b) = −commutator(b,a), componentwise, WITHOUT ½.
    let ab = a.commutator(&b);
    let ba = b.commutator(&a);
    for s in 0..8 {
        assert!((ab.get(s) + ba.get(s)).abs() < 1e-12, "slot {s}");
    }
    // And this is exactly ab − ba (raw, without a factor).
    assert_eq!(ab, a.gp(&b) - b.gp(&a));
}

#[test]
fn bracket_closed_where_gp_is_not() {
    // TWO DIFFERENT closednesses. Grade2 is closed under the BRACKET (closed_bracket
    // typechecks), but NOT under gp (closed_gp::<Grade2> does not typecheck — there is no impl
    // ClosedGp). Here we use the closed bracket on two bivectors and
    // check that the result is purely grade-2 (structural closedness).
    use crate::algebra::Pga3;
    use crate::algebra::mv::{Mv, closed_bracket};
    use crate::algebra::section::Grade2;
    type P = Mv<Pga3, f64>;
    const E23: usize = 3;
    const E13: usize = 5;
    let comm = closed_bracket::<Pga3, Grade2, f64>(&P::basis(E23), &P::basis(E13));
    // Purely grade-2: the odd/scalar/grade-4 components are zero.
    for s in 0..16usize {
        if s.count_ones() != 2 {
            assert_eq!(comm.get(s), 0.0, "non-grade-2 slot {s}");
        }
    }
    assert!(comm.get(6).abs() > 0.5); // e12 — the third axis
    // (closed_gp::<Pga3, Grade2, f64> would NOT compile: Grade2 ∉ ClosedGp.)
}

// ── Restored invariants of the legacy carrier (A7): inverse (Bareiss with
// pivoting, degenerate metric), norms by signature, soft_norm,
// undual/wedge, the f32 normalization guard. Slots go through the derived markup.

#[test]
fn pga_inverse_roundtrip_and_degenerate() {
    use crate::algebra::Pga3;
    use crate::algebra::markup::{e0_blade, radical_blade};
    use crate::algebra::mv::Mv;
    type P = Mv<Pga3, f64>;
    // 2 + 3·e01 is invertible in PGA3 (despite the degenerate metric).
    let mut a = P::zero();
    a.set(0, 2.0);
    a.set(radical_blade::<Pga3>(0).0, 3.0);
    let inv = a.inverse().expect("invertible");
    let prod = a.gp(&inv);
    assert!((prod.get(0) - 1.0).abs() < 1e-9);
    for s in 1..16 {
        assert!(prod.get(s).abs() < 1e-9, "slot {s}");
    }
    // e0 is nilpotent — there is no inverse.
    let mut e0 = P::zero();
    e0.set(e0_blade::<Pga3>(), 1.0);
    assert!(e0.inverse().is_none());
}

#[test]
fn invert_bivector_needs_pivoting() {
    use crate::algebra::Pga3;
    use crate::algebra::markup::axis_vec;
    use crate::algebra::mv::Mv;
    // e12 is invertible (e12² = −1 ⇒ e12⁻¹ = −e12), but its scalar part is zero ⇒
    // mat[0][0] = 0 — the inversion must go through a row swap (pivot).
    let e12 = axis_vec::<Pga3>(0) | axis_vec::<Pga3>(1);
    let mut m = Mv::<Pga3, f64>::zero();
    m.set(e12, 1.0);
    let inv = m.inverse().expect("e12 is invertible");
    assert!(
        (inv.get(e12) - (-1.0)).abs() < 1e-9,
        "e12⁻¹ = −e12: {}",
        inv.get(e12)
    );
    let prod = m.gp(&inv);
    assert!((prod.get(0) - 1.0).abs() < 1e-9);
    for s in 1..16 {
        assert!(prod.get(s).abs() < 1e-9, "slot {s}");
    }
}

#[test]
fn inverse_identity_and_zero() {
    use crate::algebra::Cl30;
    use crate::algebra::mv::Mv;
    type G = Mv<Cl30, f64>;
    assert!(G::zero().inverse().is_none()); // zero has no inverse
    let one = G::scalar(1.0);
    let inv = one.inverse().unwrap();
    for s in 0..8 {
        assert!((inv.get(s) - one.get(s)).abs() < 1e-12, "slot {s}");
    }
}

#[test]
fn quaternion_inverse_is_conjugate_over_norm() {
    use crate::algebra::Quaternion;
    use crate::algebra::mv::Mv;
    // (1 + 2i + 3j + 4k)⁻¹ = (1 − 2i − 3j − 4k)/30.
    type Q = Mv<Quaternion, f64>;
    let mut q = Q::zero();
    q.set(0, 1.0);
    q.set(1, 2.0);
    q.set(2, 3.0);
    q.set(3, 4.0);
    let inv = q.inverse().unwrap();
    let prod = q.gp(&inv);
    assert!((prod.get(0) - 1.0).abs() < 1e-12);
    for s in 1..4 {
        assert!(prod.get(s).abs() < 1e-12, "slot {s}");
    }
    assert!((inv.get(1) - (-2.0 / 30.0)).abs() < 1e-12);
}

#[test]
fn norm_squared_respects_signature() {
    use crate::algebra::mv::Mv;
    use crate::algebra::{Gen, Nil};
    // Euclidean Cl(2,0): ⟨M M̃⟩₀ sums the squares of ALL slots (including e12).
    type Euc2 = Gen<1, Gen<1, Nil>>;
    let mut m = Mv::<Euc2, f64>::zero();
    m.set(0, 1.0);
    m.set(1, 2.0);
    m.set(2, 3.0);
    m.set(3, 4.0);
    assert!((m.norm_squared() - 30.0).abs() < 1e-12);

    // Cl(1,0,1): the degenerate gen (bit 0) contributes nothing.
    type Sig101 = Gen<1, Gen<0, Nil>>;
    let mut s = Mv::<Sig101, f64>::zero();
    s.set(2, 3.0); // e1, ²=+1
    s.set(1, 4.0); // degenerate, ²=0
    assert!((s.norm_squared() - 9.0).abs() < 1e-12);
}

#[test]
fn minkowski_norm_signs() {
    use crate::algebra::mv::Mv;
    use crate::algebra::{Gen, Nil};
    // Cl(3,1): three spatial (+1, high bits) + one temporal (−1, bit 0).
    type Cl31 = Gen<1, Gen<1, Gen<1, Gen<-1, Nil>>>>;
    type M = Mv<Cl31, f64>;
    let mut spacelike = M::zero();
    spacelike.set(0b1000, 1.0);
    let mut timelike = M::zero();
    timelike.set(0b0001, 1.0);
    assert_eq!(spacelike.norm_squared(), 1.0);
    assert_eq!(timelike.norm_squared(), -1.0);
    let mut x = M::zero();
    x.set(0b1000, 3.0);
    x.set(0b0001, 5.0);
    assert_eq!(x.norm_squared(), 9.0 - 25.0); // −16
    let mut null = M::zero();
    null.set(0b1000, 1.0);
    null.set(0b0001, 1.0);
    assert_eq!(null.norm_squared(), 0.0);
}

#[test]
fn soft_norm_value_and_floor() {
    use crate::algebra::Pga3;
    use crate::algebra::markup::axis_vec;
    use crate::algebra::mv::Mv;
    type P = Mv<Pga3, f64>;
    // away from zero ≈ the bare norm; at zero — a floor of ε, not 0.
    let mut m = P::zero();
    m.set(axis_vec::<Pga3>(0), 3.0);
    m.set(axis_vec::<Pga3>(1), 4.0);
    assert!((m.soft_norm(1e-6) - 5.0).abs() < 1e-6); // √(25+ε²) ≈ 5
    assert!((P::zero().soft_norm(1e-3) - 1e-3).abs() < 1e-12); // floor = ε
}

#[test]
fn soft_norm_finite_gradient_through_origin() {
    // A PGA vector that is zero at the evaluation point but moving (∂/∂x = 1). The gradient
    // of the bare norm here is x/‖x‖ = 0/0 (NaN); soft_norm regularizes it to a finite value
    // (exactly 0 at zero) and does not panic. This is the case of coinciding camera anchors.
    use crate::Tangent;
    use crate::algebra::Pga3;
    use crate::algebra::markup::axis_vec;
    use crate::algebra::mv::Mv;
    type J = Tangent<N1, f64>;
    let mut m = Mv::<Pga3, J>::zero();
    m.set(axis_vec::<Pga3>(0), J::from_grad(0.0, vector![1.0]));
    let d = m.soft_norm(J::from_grad(1e-6, vector![0.0]));
    assert!(d.base().is_finite() && d.component(0).is_finite());
    assert!((d.base() - 1e-6).abs() < 1e-9);
}

#[test]
fn undual_inverts_dual() {
    use crate::algebra::Pga3;
    use crate::algebra::mv::Mv;
    // dual/undual are mutual inverses on an arbitrary multivector.
    let mut m = Mv::<Pga3, f64>::zero();
    for s in 0..16 {
        m.set(s, (s as f64) - 7.5);
    }
    let back = m.dual().undual();
    for s in 0..16 {
        assert_eq!(back.get(s), m.get(s), "slot {s}");
    }
}

#[test]
fn outer_is_antisymmetric_on_vectors() {
    use crate::algebra::Pga3;
    use crate::algebra::markup::axis_vec;
    use crate::algebra::mv::Mv;
    type P = Mv<Pga3, f64>;
    let mut a = P::zero();
    let mut b = P::zero();
    a.set(axis_vec::<Pga3>(0), 2.0);
    a.set(axis_vec::<Pga3>(2), -1.0);
    b.set(axis_vec::<Pga3>(1), 3.0);
    b.set(axis_vec::<Pga3>(2), 0.5);
    let ab = a.wedge(&b);
    let ba = b.wedge(&a);
    for s in 0..16 {
        assert_eq!(ab.get(s), -ba.get(s), "slot {s}");
    }
}

#[test]
fn normalize_f32_degeneracy_guard() {
    use crate::algebra::Pga3;
    use crate::algebra::markup::axis_vec;
    use crate::algebra::mv::Mv;
    // Category-(2) EffectiveZero on f32: norm² below the threshold (1e-5) ⇒ leave it alone;
    // a normal multivector ⇒ scaled to unit Euclidean norm.
    let e1 = axis_vec::<Pga3>(0);
    let e2 = axis_vec::<Pga3>(1);
    let mut tiny = Mv::<Pga3, f32>::zero();
    tiny.set(e1, 1e-3); // norm² = 1e-6 < 1e-5 → degeneracy
    let before = tiny;
    tiny.normalize();
    assert_eq!(tiny.get(e1), before.get(e1));

    let mut ok = Mv::<Pga3, f32>::zero();
    ok.set(e1, 3.0);
    ok.set(e2, 4.0);
    ok.normalize();
    assert!((ok.get(e1) - 0.6).abs() < 1e-6);
    assert!((ok.get(e2) - 0.8).abs() < 1e-6);
}
