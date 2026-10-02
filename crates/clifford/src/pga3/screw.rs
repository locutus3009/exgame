// SPDX-License-Identifier: MIT

//! Screw: one narrow Grade2 carrier (structurally 6 components), two transport
//! laws — variance in the type (Contra = Ad, Co = coAd). The Lie bracket exists only
//! for the twist; the wrench's cobracket is the dual form, the same as the Co transport.

use super::{Conjugatable, Direction, Dof, Line, Motor, Point};
use crate::algebra::mv::Mv;
use crate::algebra::mv::{
    angular_from_store, angular_store, closed_bracket, line_moment_store, translation_store,
};
use crate::algebra::section::Grade2;
use crate::algebra::store::scale_store;
use crate::algebra::store::{GList, GradeMap};
use crate::algebra::strata::{MaskStrataG, SectionStore};
use crate::algebra::{Algebra, Pga3};
use bytemuck::{Pod, Zeroable};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use peano::prelude::*;

type M<S> = Mv<Pga3, S>;
type V3<S> = Vector<Succ<Succ<Succ<Z>>>, S>;

// The grade-2 carrier (6 components), like EvenCarrier but for Grade2: the scalar is a
// parameter of the GAT, the bound bundle and the Ctx aggregator are not needed (Pga3: Grade2Carrier
// + Carrier + EvenCarrier hold statically, with no per-S mentions).
pub trait Grade2Carrier: Algebra {
    type St<S: Ring>: GList<S> + AbelianGroup + ScalarMul<S> + GradeMap;
}
impl<A> Grade2Carrier for A
where
    A: Algebra + MaskStrataG<Grade2, A::NgenP>,
{
    type St<S: Ring> = SectionStore<A, Grade2, S>;
}
type St<S> = <Pga3 as Grade2Carrier>::St<S>;

// widen/narrow: grade-2 carrier ↔ full Mv (for the sandwich — the honest lift A).
fn widen<S: Ring>(c: &St<S>) -> M<S> {
    let (n, dim) = (Pga3::NGEN, Pga3::DIM);
    let mut m = M::zero();
    let mut b = 0;
    while b < dim {
        if b.count_ones() == 2 {
            m.add_at(b, c.get(b, n, n));
        }
        b += 1;
    }
    m
}
fn narrow<S: Ring>(m: &M<S>) -> St<S> {
    let (n, dim) = (Pga3::NGEN, Pga3::DIM);
    let mut c = <St<S> as AbelianGroup>::ZERO;
    let mut b = 0;
    while b < dim {
        if b.count_ones() == 2 {
            c.add_at(b, n, n, m.get(b));
        }
        b += 1;
    }
    c
}

// The bracket of two bivectors is PURELY grade-2 in exact arithmetic (the symmetric
// grades 0 and 4 cancel in ab−ba). In float the summation order leaves
// ~ε in those grades, so we check «closedness» as an EFFECTIVE zero
// of the part dropped by narrow, not as bitwise equality. A guard for ClosedBracket.
fn bracket_pure_grade2<S: Scalar + StandardPart>(comm: &M<S>) -> bool {
    let resid = *comm - widen(&narrow(comm));
    let mut b = 0;
    while b < Pga3::DIM {
        if !resid.get(b).standard_part().is_effective_zero() {
            return false;
        }
        b += 1;
    }
    true
}

// ── Variance — the transformation law under a motor ──────────────────
#[derive(Clone, Copy)]
pub struct Contra; // twist: Adjoint
#[derive(Clone, Copy)]
pub struct Co; // wrench: coAdjoint
pub trait Variance {
    fn transport<S: Scalar + StandardPart>(c: &St<S>, m: &Motor<S>) -> St<S>;
}
impl Variance for Contra {
    // Adjoint: M · t · M̃.
    fn transport<S: Scalar + StandardPart>(c: &St<S>, m: &Motor<S>) -> St<S> {
        narrow(&m.conjugate_mv(&widen(c)))
    }
}
impl Variance for Co {
    // coAdjoint: undual(M · dual(w) · M̃) — the dual of Adjoint (polarity I).
    fn transport<S: Scalar + StandardPart>(c: &St<S>, m: &Motor<S>) -> St<S> {
        narrow(&m.conjugate_mv(&widen(c).dual()).undual())
    }
}

// ── Screw: a shared carrier + phantom variance ────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Screw<V, S: Ring>(St<S>, PhantomData<V>);
pub type Twist<S> = Screw<Contra, S>;
pub type Wrench<S> = Screw<Co, S>;

// Grade2 store is Pod ⇒ Screw is Pod: the carrier first, the variance `PhantomData<V>`
// is an align-1 ZST at the tail. Manual impl (not derive): derive would attach a spurious
// `V: Pod` to the variance marker. `'static` is a Pod requirement.
unsafe impl<V, S: Ring> Zeroable for Screw<V, S> where St<S>: Zeroable {}
unsafe impl<V: Copy + 'static, S: Ring + 'static> Pod for Screw<V, S> where St<S>: Pod {}

// Twist/wrench over f32 — the grade-2 section, 6 packed slots (ω, v).
const _: () = peano::prelude::assert_packed::<Twist<f32>, f32>(6);

impl<V, S: Ring> PartialEq for Screw<V, S>
where
    St<S>: PartialEq,
{
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}
impl<V, S: Ring> core::fmt::Debug for Screw<V, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Screw({:?})", self.0)
    }
}

// Group and scaling — on the narrow carrier (the strata are an abelian group).
impl<V, S: Ring> Add for Screw<V, S>
where
    Pga3: Grade2Carrier,
{
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Screw(self.0 + o.0, PhantomData)
    }
}
impl<V, S: Ring> Sub for Screw<V, S>
where
    Pga3: Grade2Carrier,
{
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Screw(self.0 - o.0, PhantomData)
    }
}
impl<V, S: Ring> Neg for Screw<V, S>
where
    Pga3: Grade2Carrier,
{
    type Output = Self;
    fn neg(self) -> Self {
        Screw(-self.0, PhantomData)
    }
}
impl<V, S: Ring> AddAssign for Screw<V, S>
where
    Pga3: Grade2Carrier,
{
    fn add_assign(&mut self, o: Self) {
        self.0 += o.0;
    }
}
impl<V, S: Ring> SubAssign for Screw<V, S>
where
    Pga3: Grade2Carrier,
{
    fn sub_assign(&mut self, o: Self) {
        self.0 -= o.0;
    }
}
impl<V, S: Ring> Mul<S> for Screw<V, S> {
    type Output = Self;
    fn mul(self, s: S) -> Self {
        self.scale(s)
    }
}

impl<V, S: Ring> Screw<V, S> {
    /// One DOF: a unit component in the canonical order Dof::ALL.
    pub fn basis(dof: Dof) -> Self
    where
        S: Scalar + StandardPart,
    {
        let mut lin = V3::<S>::ZERO;
        let mut ang = V3::<S>::ZERO;
        let i = dof as usize;
        if i < 3 {
            lin[i] = S::ONE;
        } else {
            ang[i - 3] = S::ONE;
        }
        // assembly goes through the same markup as new (angular carries the derived σ)
        let r = translation_store::<S, St<S>>(&lin);
        let e = angular_store::<S, St<S>>(&ang);
        Screw(r + e, PhantomData)
    }
    /// The narrow carrier directly (for the closed paths of the facade, e.g. Motor::exp).
    pub(crate) fn store(&self) -> &St<S> {
        &self.0
    }
    /// Lift the coefficients via the canonical injection (through the full Mv).
    pub fn deepen<T: Ring>(&self) -> Screw<V, T>
    where
        S: crate::Lift<T>,
    {
        Screw::from_mv(&self.as_mv().deepen())
    }
    pub fn zero() -> Self {
        Screw(<St<S> as AbelianGroup>::ZERO, PhantomData)
    }
    pub fn as_mv(&self) -> M<S> {
        widen(&self.0)
    }
    /// Narrow a full Mv into the grade-2 carrier (for the bridge; narrow∘widen = id).
    pub fn from_mv(m: &M<S>) -> Self {
        Screw(narrow(m), PhantomData)
    }
    /// AbelianGroup: screws ADD (grade-2 is closed under +) and scale.
    pub fn plus(&self, o: &Self) -> Self {
        Screw(self.0 + o.0, PhantomData)
    }
    pub fn scale(&self, s: S) -> Self {
        Screw(scale_store::<Pga3, S, St<S>>(&self.0, s), PhantomData)
    }
}

impl<V: Variance, S: Scalar + StandardPart> Screw<V, S> {
    /// Transport under a motor BY ITS OWN VARIANCE: twist — Adjoint, wrench —
    /// coAdjoint. The law is chosen by the TYPE V, not by hand.
    pub fn transport(&self, m: &Motor<S>) -> Self {
        Screw(V::transport(&self.0, m), PhantomData)
    }
}

// Twist (Contra): a constructor from (linear, angular), projections, and exp —
// ONLY for the twist (a force does not exponentiate into a motor).
impl<S: Scalar + StandardPart> Twist<S> {
    pub fn new(linear: &V3<S>, angular: &V3<S>) -> Self {
        // We build DIRECTLY in the narrow grade-2 carrier (closed-form write), without
        // an intermediate 16-component Mv: linear→radical, angular→angular
        // markup (Euclidean × the COMPUTED embedding sign σ — see
        // markup::angular_embed_sign; there is no hand-written minus). The stores of the halves
        // are disjoint, adding the slots generates no arithmetic on zeros.
        // The inverse of linear()/angular().
        let r = translation_store::<S, St<S>>(linear);
        let e = angular_store::<S, St<S>>(angular);
        Screw(r + e, PhantomData)
    }
    pub fn linear(&self) -> V3<S> {
        line_moment_store(&self.0)
    }
    pub fn angular(&self) -> V3<S> {
        angular_from_store(&self.0)
    }
    /// exp: Lie algebra → group. The ½ lives here (Motor::exp is pure).
    /// Narrow path: scaling of the grade-2 carrier and a direct embedding into the even one —
    /// without a detour through the full 16-slot Mv.
    pub fn exp(&self, dt: S) -> Motor<S> {
        Motor::exp(&self.scale(dt * S::get_constant(RationalConstant::HALF)))
    }

    /// The se(3) Lie bracket: `[a,b]` = ½·(ab−ba). The ½ is the same convention as in exp.
    /// The type requires Grade2: ClosedBracket (the bracket is closed in grade-2, though gp
    /// is not): the commutator of two bivectors is purely grade-2, so narrow loses nothing —
    /// the debug_assert guards the structural closedness. ONLY for the twist: the wrench
    /// (a covector) has no Lie bracket of its own — the method is not on the generic `Screw<V>`.
    pub fn bracket(&self, other: &Self) -> Self {
        let comm = closed_bracket::<Pga3, Grade2, S>(&widen(&self.0), &widen(&other.0));
        debug_assert!(
            bracket_pure_grade2(&comm),
            "[twist,twist] is not pure grade-2"
        );
        let half = S::get_constant(RationalConstant::HALF);
        Self::from_mv(&comm).scale(half)
    }

    /// The action of a twist on a point: the linear velocity v = ½·[T, P] as a grade-3
    /// weight-0 ⇒ a free Direction vector. The sign comes from the same anchor
    /// velocity=ω×r as the angular markup in new() (it is not fitted here).
    pub fn velocity_at(&self, p: &Point<S>) -> Direction<S> {
        let comm = widen(&self.0).commutator(&p.as_mv());
        let half = S::get_constant(RationalConstant::HALF);
        Direction::from_mv(comm.scale(half))
    }
}

// Wrench (Co): force/moment projections; NO exp; pairing with a twist.
impl<S: Scalar + StandardPart> Wrench<S> {
    pub fn new(force: &V3<S>, torque: &V3<S>) -> Self {
        // THE SAME embed as the twist's (the same 6 numbers → THE SAME bivector):
        // force→radical, torque→angular markup (shared σ). The difference
        // twist↔wrench is NOT in the components and NOT in the basis, but in the
        // transformation LAW (Adjoint/coAdjoint). Directly into the narrow carrier, without
        // an intermediate Mv.
        let r = translation_store::<S, St<S>>(force);
        let e = angular_store::<S, St<S>>(torque);
        Screw(r + e, PhantomData)
    }
    pub fn force(&self) -> V3<S> {
        line_moment_store(&self.0) // radical (invariant under the coAdjoint transport)
    }
    pub fn torque(&self) -> V3<S> {
        angular_from_store(&self.0)
    }

    /// The coadjoint action (infinitesimal ad*) of a twist on a wrench —
    /// the GYROSCOPIC term. ½·undual([widen(t), dual(widen(w))]): the generator t
    /// acts on the DUAL of the wrench, the result is undualed back — the INFINITESIMAL
    /// twin of the Co transport (undual∘Ad∘dual), of the same dual form.
    /// A flat `commutator([w,t])` is wrong here — just as a flat sandwich is for the
    /// Co transport: it would violate the duality ⟨ad*_t w, s⟩ = −⟨w,`[t,s]`⟩ (not in
    /// sign — in magnitude). The form is FORCED by the polar pairing, the arbiter is
    /// the coad_is_dual_to_ad test; not a single fitted sign.
    /// SEPARATE from transport: bracket is INFINITESIMAL (the algebra, ad*), transport
    /// is FINITE (the group action). Closed in grade-2 (Grade2: ClosedBracket).
    /// Power ⟨W ∧ dual(V)⟩_I = f·v + τ·ω (a wrapper over pairing).
    pub fn power(&self, t: &Twist<S>) -> S {
        pairing(self, t)
    }

    /// A wrench along a line of action: the dual of the line is a co-screw.
    pub fn from_line(line: &Line<S>) -> Self {
        Self::from_mv(&line.as_mv().dual())
    }

    pub fn coad_bracket(&self, t: &Twist<S>) -> Self {
        let raw = widen(&t.0).commutator(&widen(&self.0).dual()).undual();
        debug_assert!(
            bracket_pure_grade2(&raw),
            "[wrench,twist] is not pure grade-2"
        );
        let half = S::get_constant(RationalConstant::HALF);
        Screw(
            scale_store::<Pga3, S, St<S>>(&narrow(&raw), half),
            PhantomData,
        )
    }
}

impl<S: Scalar + StandardPart> Conjugatable<S> for Twist<S> {
    fn to_mv(&self) -> M<S> {
        self.as_mv()
    }
    fn from_mv(mv: M<S>) -> Self {
        Screw::from_mv(&mv)
    }
}

/// The power pairing ⟨w,t⟩ = ⟨w ∧ dual(t)⟩_pseudoscalar. INVARIANT under a
/// motor: coAdjoint(w)·Adjoint(t) preserves it — this is exactly what forces
/// the opposite transformation laws (see the test).
pub fn pairing<S: Scalar + StandardPart>(w: &Wrench<S>, t: &Twist<S>) -> S {
    widen(&w.0).wedge(&widen(&t.0).dual()).get(Pga3::DIM - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::Float;

    #[test]
    fn wrench_markup_roundtrip_on_narrow_carrier() {
        // Round-trip of the dual markup on crate::pga3::Wrench (the narrow carrier):
        // assemble a wrench from force and moment and read them back — read∘write = id.
        // (The predicate part moved to clifford::algebra::section, A4.)
        let f: Vector<N3, f64> = vector![1.0, 2.0, 3.0];
        let m: Vector<N3, f64> = vector![4.0, 5.0, 6.0];
        let w = crate::pga3::Wrench::new(&f, &m);
        assert_eq!(w.force(), f);
        assert_eq!(w.torque(), m);
    }

    #[test]
    fn twist_wrench_variance_pins_power() {
        use crate::pga3::{Twist, Wrench, pairing};
        // Structurally: both the twist and the wrench are 6 components (the shared Grade<2> carrier),
        // the Variance phantom has zero size. Not 16 with zeros.
        assert_eq!(
            core::mem::size_of::<Wrench<f64>>(),
            6 * core::mem::size_of::<f64>()
        );
        assert_eq!(
            core::mem::size_of::<Twist<f64>>(),
            6 * core::mem::size_of::<f64>()
        );

        fn close3(a: &Vector<N3, f64>, b: &Vector<N3, f64>) -> bool {
            (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9)
        }

        // Round-trip: NARROW construction (without an intermediate Mv) ↔ projections.
        let (l, a) = (vector![1.0, 2.0, 3.0], vector![0.5, -1.0, 0.7]);
        let t = Twist::new(&l, &a);
        assert!(close3(&t.linear(), &l) && close3(&t.angular(), &a));
        let (f, tq) = (vector![2.0, -1.0, 0.5], vector![0.3, 0.8, -0.4]);
        let w = Wrench::new(&f, &tq);
        assert!(close3(&w.force(), &f) && close3(&w.torque(), &tq));

        // POWER IS INVARIANT: ⟨coAdjoint(w), Adjoint(t)⟩ == ⟨w, t⟩. This is exactly what
        // forces twist and wrench to transform OPPOSITELY.
        let m = Twist::new(&vector![0.3, -0.2, 0.5], &vector![0.1, 0.4, -0.2]).exp(1.0);
        let before = pairing(&w, &t);
        let after = pairing(&w.transport(&m), &t.transport(&m));
        assert!((before - after).abs() < 1e-9, "power: {before} vs {after}");

        // The same 6 numbers → THE SAME bivector: twist and wrench are INDISTINGUISHABLE as
        // multivectors. The difference is not in the components and not in the basis, but in the
        // transformation LAW (Adjoint vs coAdjoint), and at the TYPE level they cannot be mixed up
        // (Wrench has no exp — the compiler catches `w.exp(1.0)`).
        assert_eq!(Twist::new(&l, &a).as_mv(), Wrench::new(&l, &a).as_mv());
        // But under a motor they diverge: Adjoint ≠ coAdjoint.
        assert_ne!(
            Twist::new(&l, &a).transport(&m).as_mv(),
            Wrench::new(&l, &a).transport(&m).as_mv()
        );
    }

    #[test]
    fn wrench_parallel_axis_on_narrow_carrier() {
        // Parallel-axis is PROVEN ON crate::pga3::Wrench (narrow, with variance), not on the old
        // flat contour. Plus two bridge oracles: facade == flat, narrow∘widen=id.
        use crate::pga3::{Twist, Wrench};
        fn n2(v: &Vector<N3, f64>) -> f64 {
            (0..3).map(|i| v[i] * v[i]).sum()
        }
        fn close3(a: &Vector<N3, f64>, b: &Vector<N3, f64>) -> bool {
            (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9)
        }
        let f: Vector<N3, f64> = vector![2.0, -1.0, 3.0];
        let m0: Vector<N3, f64> = vector![0.5, 1.0, -0.5];
        let r: Vector<N3, f64> = vector![1.0, 2.0, 3.0];

        // A pure translation by r as a motor; the wrench transport is coAdjoint (by type).
        let mt = Twist::new(&r, &vector![0.0, 0.0, 0.0]).exp(1.0);
        let w = Wrench::new(&f, &m0);
        let tw = w.transport(&mt);

        // Under coAdjoint the force (radical) is INVARIANT to translation; the moment (Euclidean)
        // shifted by the amount |r×F| — parallel-axis, on the narrow carrier.
        assert!(close3(&tw.force(), &f), "force invariant: {:?}", tw.force());
        let dtau = tw.torque() - m0;
        assert!((n2(&dtau) - n2(&r.cross(f))).abs() < 1e-9, "|Δτ| = |r×F|");

        // narrow∘widen = id on grade-2 (the bridge does not lose a component).
        assert_eq!(Wrench::from_mv(&w.as_mv()), w);
    }

    #[test]
    fn se3_structure_constants() {
        use crate::pga3::Twist;
        let z3: Vector<N3, f64> = vector![0.0, 0.0, 0.0];
        let ex: Vector<N3, f64> = vector![1.0, 0.0, 0.0];
        let ey: Vector<N3, f64> = vector![0.0, 1.0, 0.0];
        // se(3) bases via the PUBLIC bridge Twist::new (the same ⋆₃ markup).
        let rx = Twist::new(&z3, &ex); // rotation about x
        let ry = Twist::new(&z3, &ey);
        let tx = Twist::new(&ex, &z3); // translation along x
        let ty = Twist::new(&ey, &z3);
        let big = |v: &Vector<N3, f64>, k: usize| v[k].abs() > 0.5;
        let sm = |v: &Vector<N3, f64>| (0..3).all(|i| v[i].abs() < 1e-12);

        // [Rx,Ry] → ±Rz: angular part along z, linear part zero.
        let rr = rx.bracket(&ry);
        assert!(big(&rr.angular(), 2) && sm(&rr.linear()), "[Rx,Ry]=±Rz");
        // [Tx,Ty] → 0: translations commute (e₀²=0).
        let tt = tx.bracket(&ty);
        assert!(sm(&tt.angular()) && sm(&tt.linear()), "[Tx,Ty]=0");
        // [Rx,Ty] → ±Tz: linear part along z, angular part zero.
        let rt = rx.bracket(&ty);
        assert!(big(&rt.linear(), 2) && sm(&rt.angular()), "[Rx,Ty]=±Tz");
    }

    #[test]
    fn twist_bracket_satisfies_jacobi() {
        use crate::pga3::Twist;
        // Asymmetric twists (general screws).
        let a = Twist::new(&vector![1.0, 2.0, 3.0], &vector![0.5, -1.0, 0.7]);
        let b = Twist::new(&vector![-2.0, 1.0, 0.5], &vector![0.3, 0.8, -0.4]);
        let c = Twist::new(&vector![0.2, -0.6, 1.1], &vector![-0.5, 0.9, 0.2]);
        // [a,[b,c]] + [b,[c,a]] + [c,[a,b]] = 0 — a genuine Lie bracket, not just
        // an antisymmetric operator. If it does not hold ⇒ commutator is wrong.
        let j = a
            .bracket(&b.bracket(&c))
            .plus(&b.bracket(&c.bracket(&a)))
            .plus(&c.bracket(&a.bracket(&b)));
        let z = j.as_mv();
        for s in 0..16 {
            assert!(z.get(s).abs() < 1e-12, "Jacobi slot {s}: {}", z.get(s));
        }
    }

    #[test]
    fn ad_is_a_bracket_automorphism() {
        use crate::pga3::Twist;
        let a = Twist::new(&vector![1.0, 2.0, 3.0], &vector![0.5, -1.0, 0.7]);
        let b = Twist::new(&vector![-2.0, 1.0, 0.5], &vector![0.3, 0.8, -0.4]);
        let m = Twist::new(&vector![0.3, -0.2, 0.5], &vector![0.1, 0.4, -0.2]).exp(1.0);
        // Ad_M is an automorphism: [a,b] transported as a single twist equals
        // the bracket of the transported ones. The adjoint commutes with the bracket.
        let lhs = a.bracket(&b).transport(&m).as_mv();
        let rhs = a.transport(&m).bracket(&b.transport(&m)).as_mv();
        for s in 0..16 {
            assert!((lhs.get(s) - rhs.get(s)).abs() < 1e-9, "slot {s}");
        }
    }

    #[test]
    fn velocity_at_is_omega_cross_r() {
        use crate::pga3::Point;
        use crate::pga3::Twist;
        let z3: Vector<N3, f64> = vector![0.0, 0.0, 0.0];
        // A pure rotation ω about a general direction; v at point r = ω×r (the anchor).
        let w: Vector<N3, f64> = vector![0.2, -0.5, 1.3];
        let r: Vector<N3, f64> = vector![1.0, 2.0, -0.5];
        let t = Twist::new(&z3, &w);
        let v = t.velocity_at(&Point::new(r)).coords();
        let wr = w.cross(r);
        for i in 0..3 {
            assert!((v[i] - wr[i]).abs() < 1e-9, "axis {i}: {v:?} vs {wr:?}");
        }
    }

    #[test]
    fn coad_is_dual_to_ad_through_pairing() {
        use crate::pga3::{Twist, Wrench, pairing};
        // ad* is dual to ad with respect to the pairing: ⟨coad_bracket(w,t), s⟩ = −⟨w,[t,s]⟩.
        // The sign (−) is FORCED by orientation/polarity, not fitted — it ties together
        // with a single invariant the twist bracket, the wrench cobracket and the pairing.
        let w = Wrench::new(&vector![2.0, -1.0, 0.5], &vector![0.3, 0.8, -0.4]);
        let t = Twist::new(&vector![1.0, 2.0, 3.0], &vector![0.5, -1.0, 0.7]);
        let s = Twist::new(&vector![-0.5, 0.9, 0.2], &vector![0.2, -0.6, 1.1]);
        let lhs = pairing(&w.coad_bracket(&t), &s);
        let rhs = -pairing(&w, &t.bracket(&s));
        assert!((lhs - rhs).abs() < 1e-9, "ad*↔ad: {lhs} vs {rhs}");
    }

    #[test]
    fn jacobian_of_rigid_motion_is_ad_over_peano_dof() {
        // The Jacobian is NOT a separate construction — it falls out of AD with the number of axes =
        // the number of DOFs (Peano-6). Coefficient = Tangent<Six>: one forward pass
        // through twist_embed → Study-form exp → sandwich → coordinates gives the whole
        // 3×6 matrix ∂(point coordinates)/∂(twist) at twist = 0.
        use crate::Tangent;
        use crate::pga3::Point;
        use crate::pga3::Twist;
        type Six = Succ<Succ<Succ<Succ<Succ<Succ<Z>>>>>>;
        type J6 = Tangent<Six, f64>;

        // Seed: DOF k is a variable with value 0 and a unit gradient along axis k.
        fn seed(axis: usize) -> J6 {
            let mut g = Vector::<Six, f64>::ZERO;
            g[axis] = 1.0;
            Tangent::from_grad(0.0, g)
        }
        let p_vec: Vector<N3, f64> = vector![1.0, 2.0, 3.0];

        let lin = vector![seed(0), seed(1), seed(2)]; // Tx,Ty,Tz
        let ang = vector![seed(3), seed(4), seed(5)]; // Rx,Ry,Rz
        let g = Twist::new(&lin, &ang);
        let m = g.exp(J6::embed(1.0));
        let p = Point::new(vector![J6::embed(1.0), J6::embed(2.0), J6::embed(3.0)]);
        let out = Point::from_mv(m.conjugate_mv(&p.as_mv())).coords();

        // Jacobian[i][k] = ∂(coords_i)/∂(twist_k) = the grad along axis k of the i-th coordinate.
        let row = |i: usize| -> [f64; 6] {
            let ci = match i {
                0 => *out.get::<Z>(),
                1 => *out.get::<Succ<Z>>(),
                _ => *out.get::<Succ<Succ<Z>>>(),
            };
            core::array::from_fn(|k| ci.grad()[k])
        };
        let jac = [row(0), row(1), row(2)];

        // The linear block = I₃ (translation moves the point 1:1).
        for (i, jrow) in jac.iter().enumerate() {
            // Only the 3x3 linear block: `jrow` is 6 wide and columns 3..6 are the
            // angular block, checked separately below.
            for (k, &v) in jrow.iter().take(3).enumerate() {
                let want = if i == k { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-9, "lin[{i}][{k}]={v}");
            }
        }
        // The angular block: column j = eⱼ × p (with our own cross — self-consistently,
        // because the sign of the twist's angular part is exactly what we pinned to velocity=ω×r).
        for j in 0..3 {
            let mut ej: Vector<N3, f64> = vector![0.0, 0.0, 0.0];
            ej[j] = 1.0;
            let cj = ej.cross(p_vec);
            let col = [cj[0], cj[1], cj[2]];
            for i in 0..3 {
                assert!(
                    (jac[i][3 + j] - col[i]).abs() < 1e-9,
                    "ang[{i}][{j}]={} want {}",
                    jac[i][3 + j],
                    col[i]
                );
            }
        }

        // The directional derivative is a special case: a single Jet axis.
        // ∂/∂Rz of the y coordinate when rotating (1,0,0): column Rz, component y.
        // (Here p=(1,2,3): ∂y/∂Rz = (e_z×p)_y = 1.)
        assert!((jac[1][5] - 1.0).abs() < 1e-9);
    }

    // ── Restored invariants of the legacy screws (A7): readers, basis ─────
    #[test]
    fn twist_roundtrip() {
        let t = Twist::new(&vector![1.0, 2.0, 3.0], &vector![4.0, 5.0, 6.0]);
        assert_eq!(t.linear(), vector![1.0, 2.0, 3.0]);
        assert_eq!(t.angular(), vector![4.0, 5.0, 6.0]);
    }

    #[test]
    fn wrench_roundtrip() {
        let w = Wrench::new(&vector![1.0, 2.0, 3.0], &vector![0.5, 1.5, 2.5]);
        assert_eq!(w.force(), vector![1.0, 2.0, 3.0]);
        assert_eq!(w.torque(), vector![0.5, 1.5, 2.5]);
    }

    #[test]
    fn dof_all_is_ordered() {
        assert_eq!(
            Dof::ALL,
            [Dof::Tx, Dof::Ty, Dof::Tz, Dof::Rx, Dof::Ry, Dof::Rz]
        );
    }

    #[test]
    fn basis_lights_one_slot() {
        // Through the readers (the strata carrier has no component array):
        // exactly one of the six components is lit, and it is the component of its own Dof.
        for d in Dof::ALL {
            let b = Twist::<f64>::basis(d);
            let (l, a) = (b.linear(), b.angular());
            let comps = [l[0], l[1], l[2], a[0], a[1], a[2]];
            let lit = comps.iter().filter(|&&c| c != 0.0).count();
            assert_eq!(lit, 1, "basis({d:?}) lights exactly one component");
            assert_eq!(comps[d as usize], 1.0, "basis({d:?})");
        }
    }

    #[test]
    fn twist_decomposes_into_basis() {
        let t = Twist::new(&vector![1.0, 2.0, 3.0], &vector![4.0, 5.0, 6.0]);
        let comps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut sum = Twist::<f64>::zero();
        for (i, d) in Dof::ALL.iter().enumerate() {
            sum += Twist::basis(*d) * comps[i];
        }
        assert_eq!(sum.linear(), t.linear());
        assert_eq!(sum.angular(), t.angular());
    }

    #[test]
    fn power_is_linear() {
        let w = Wrench::new(&vector![1.0, -2.0, 0.5], &vector![0.3, 0.7, -0.1]);
        let a = Twist::new(&vector![1.0, 0.0, 0.0], &vector![0.0, 1.0, 0.0]);
        let b = Twist::new(&vector![0.0, 2.0, 0.0], &vector![1.0, 0.0, 3.0]);
        let sum = w.power(&(a + b));
        assert!((sum - (w.power(&a) + w.power(&b))).abs() < 1e-12);
    }

    #[test]
    fn coadjoint_preserves_power_under_rotation() {
        // ⟨Ad*_M w, Ad_M t⟩ = ⟨w, t⟩ — power invariance for a ROTATION
        // (legacy pinned only the translator; the rotational arm is held by newton's
        // integrators — now the pin is here).
        let m = Motor::exp(&Twist::new(
            &vector![0.0, 0.0, 0.0],
            &vector![0.4, -0.7, 0.3],
        ));
        let t = Twist::new(&vector![1.0, -2.0, 0.5], &vector![0.3, 0.7, -0.1]);
        let w = Wrench::new(&vector![2.0, 1.0, -1.0], &vector![-0.5, 0.2, 0.8]);
        let before = w.power(&t);
        let after = w.transport(&m).power(&t.transport(&m));
        assert!((before - after).abs() < 1e-9, "{before} vs {after}");
    }
}
