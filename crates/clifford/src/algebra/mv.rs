// SPDX-License-Identifier: MIT

//! Strata-based multivector: the Carrier GAT binds an algebra to its triangular
//! carrier; gp is the Cayley–Dickson recursion, involutions are a per-stratum GradeMap,
//! inverse is a peano matrix of the regular representation. Plus project/inject over
//! predicate sections and dual readers of physical quantities.

use super::markup::{angular_blade, axis_vec, euclid_blade, point_blade, radical_blade};
use super::section::{ClosedBracket, ClosedGp, Section};
use super::store::{GList, GpRec, GradeMap, StratumSign, gp_store};
use super::strata::{MaskStrataG, SectionStore, StrataUpTo};
use super::{Algebra, Pga3, blade_grade, blade_prod, complement_sign};
use crate::Lift;
use bytemuck::{Pod, Zeroable};
use core::fmt::Debug;
use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};
use peano::prelude::*;

/// gp closed within a section: by type it requires Spec: ClosedGp and works DIRECTLY on
/// the narrow carrier of the section. For a non-closed Spec (Odd, a single Grade-k) this
/// call will not compile — and if it did compile, add_at would land in a
/// Void stratum and silently drop the result. Closedness is a precondition of the type.
pub fn closed_gp<A, Spec, S>(
    a: &SectionStore<A, Spec, S>,
    b: &SectionStore<A, Spec, S>,
) -> SectionStore<A, Spec, S>
where
    A: Algebra + MaskStrataG<Spec, <A as Algebra>::NgenP>,
    Spec: ClosedGp,
    S: Ring,
{
    gp_store::<A, S, SectionStore<A, Spec, S>>(a, b)
}

/// Commutator `[a,b]`=ab−ba, requiring by type Spec: ClosedBracket — the section is
/// closed under the LIE BRACKET (even if NOT under gp). This is a DIFFERENT closedness than
/// closed_gp's: Grade2 passes here (bivector·bivector in the bracket stays a
/// bivector), but not in closed_gp. Computed in the FULL carrier A (gp of the section
/// is not closed, hence no store version), but for a closed section the result
/// lies in it — projecting back loses nothing (see the debug_assert at the call site).
pub fn closed_bracket<A: Carrier, Spec: ClosedBracket, S: Ring>(
    a: &Mv<A, S>,
    b: &Mv<A, S>,
) -> Mv<A, S>
where
    // the structural gp needs the carrier unfolded by the top gen
    <A as Carrier>::St<S>: GpRec<A>,
{
    a.commutator(b)
}

// Storage of the even subalgebra (motor) — a special case of the general section.

// ── Multivector carrier: binds an algebra to its triangular store ──
// The scalar is a parameter of the GAT: one bound `A: Carrier` gives the carrier for ALL
// scalars at once, and the former bound bundle on the projection is not needed — the promises
// are already declared on the StrataUpTo GAT, here they are merely re-promised.
pub trait Carrier: Algebra {
    type St<S: Ring>: GList<S> + AbelianGroup + ScalarMul<S> + GradeMap;
}
impl<A> Carrier for A
where
    A: Algebra + StrataUpTo<<A as Algebra>::NgenP>,
{
    type St<S: Ring> = <A as StrataUpTo<<A as Algebra>::NgenP>>::Store<S>;
}

// ── Multivector ─────────────────────────────────────────────────────
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mv<A: Carrier, S: Ring>(<A as Carrier>::St<S>, PhantomData<A>);

// Carrier is Pod ⇒ Mv is Pod. `#[repr(C)]` lays out `St<S>` first; `PhantomData<A>` is an
// align-1 ZST at the tail and adds no padding. Manual impls (not derive): derive
// would attach a spurious `A: Pod` to the algebra marker. `'static` is a Pod requirement.
unsafe impl<A: Carrier, S: Ring> Zeroable for Mv<A, S> where <A as Carrier>::St<S>: Zeroable {}
unsafe impl<A: Carrier + Copy + 'static, S: Ring + 'static> Pod for Mv<A, S> where
    <A as Carrier>::St<S>: Pod
{
}

// The full Pga3 multivector over f32 — 16 packed blades (64 bytes), with no
// holes: the Void strata and PhantomData are align-1 ZSTs. A regression in the carrier layout
// or a break in the Pod chain of strata will not compile.
const _: () = peano::prelude::assert_packed::<Mv<Pga3, f32>, f32>(16);

impl<A: Carrier, S: Ring> Debug for Mv<A, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Mv({:?})", self.0)
    }
}
impl<A: Carrier, S: Ring> PartialEq for Mv<A, S>
where
    <A as Carrier>::St<S>: PartialEq,
{
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}

impl<A: Carrier, S: Ring> Mv<A, S> {
    /// The zero multivector.
    pub fn zero() -> Self {
        Mv(<A as Carrier>::St::<S>::ZERO, PhantomData)
    }
    /// The coefficient of blade `blade` (a bitmask of generators).
    pub fn get(&self, blade: usize) -> S {
        self.0.get(blade, A::NGEN, A::NGEN)
    }
    /// Add `v` to the coefficient of a blade.
    pub fn add_at(&mut self, blade: usize, v: S) {
        self.0.add_at(blade, A::NGEN, A::NGEN, v);
    }
    /// Set the coefficient of a blade (overwrite).
    pub fn set(&mut self, blade: usize, v: S) {
        let cur = self.get(blade);
        self.add_at(blade, v - cur);
    }
    /// Scalar (grade-0).
    pub fn scalar(s: S) -> Self {
        let mut m = Self::zero();
        m.add_at(0, s);
        m
    }
    /// A unit basis blade with coefficient 1.
    pub fn basis(blade: usize) -> Self {
        let mut m = Self::zero();
        m.add_at(blade, S::ONE);
        m
    }
    // Involutions — a componentwise sign by grade g (a single pattern):
    //   reverse           (−1)^{g(g−1)/2}   (Ã)
    //   grade_involution  (−1)^g            (Â, even +, odd −)
    //   clifford_conjugate(−1)^{g(g+1)/2}   (Ā = reverse∘involution)
    fn by_grade_sign(&self, flip: fn(u32) -> bool) -> Self {
        // per stratum: grade = position in the list, we do not go inside a stratum
        let st = self.0.grade_map(A::NGEN, &mut |g| {
            if flip(g as u32) {
                StratumSign::Neg
            } else {
                StratumSign::Keep
            }
        });
        Mv(st, PhantomData)
    }
    /// Reverse Ã: (−1)^{g(g−1)/2}.
    pub fn reverse(&self) -> Self {
        self.by_grade_sign(|g| (g.wrapping_mul(g.wrapping_sub(1)) / 2) & 1 == 1)
    }
    /// Grade involution Â: (−1)^g (odd grades change sign).
    pub fn grade_involution(&self) -> Self {
        self.by_grade_sign(|g| g & 1 == 1)
    }
    /// Clifford conjugation Ā = reverse∘involution: (−1)^{g(g+1)/2}.
    pub fn clifford_conjugate(&self) -> Self {
        self.by_grade_sign(|g| (g.wrapping_mul(g + 1) / 2) & 1 == 1)
    }
    /// Projection onto grade k: ⟨A⟩ₖ (the other strata zeroed). Per stratum.
    pub fn grade(&self, k: usize) -> Self {
        let st = self.0.grade_map(A::NGEN, &mut |g| {
            if g == k {
                StratumSign::Keep
            } else {
                StratumSign::Drop
            }
        });
        Mv(st, PhantomData)
    }
    /// Even part ⟨A⟩₊ (grades 0,2,4,…) — the home of versors/motors. Per stratum.
    pub fn even(&self) -> Self {
        let st = self.0.grade_map(A::NGEN, &mut |g| {
            if g & 1 == 0 {
                StratumSign::Keep
            } else {
                StratumSign::Drop
            }
        });
        Mv(st, PhantomData)
    }
    /// Odd part ⟨A⟩₋ (grades 1,3,5,…).
    pub fn odd(&self) -> Self {
        self.clone() - self.even()
    }

    // gp with a filter on the grade of the result: wedge takes the highest grade
    // (rank a + rank b), dot the lowest (|rank a − rank b|). The signs are already inside
    // the product — the section only selects the grade.
    fn graded_gp(&self, rhs: &Self, wedge: bool) -> Self {
        let mut out = Self::zero();
        let dim = A::DIM;
        let mut a = 0;
        while a < dim {
            let (ca, ga) = (self.get(a), blade_grade(a));
            let mut b = 0;
            while b < dim {
                let (sign, c) = blade_prod(a, b, A::ZERO_MASK, A::NEG_MASK);
                let gb = blade_grade(b);
                let want = if wedge { ga + gb } else { ga.abs_diff(gb) };
                if sign != 0 && blade_grade(c) == want {
                    let term = ca * rhs.get(b);
                    out.add_at(c, if sign < 0 { -term } else { term });
                }
                b += 1;
            }
            a += 1;
        }
        out
    }
    /// Outer (∧) product = the grade-⟨rank a + rank b⟩ section of gp.
    pub fn wedge(&self, rhs: &Self) -> Self {
        self.graded_gp(rhs, true)
    }
    /// Inner (·) product = the grade-⟨|rank a − rank b|⟩ section of gp.
    pub fn dot(&self, rhs: &Self) -> Self {
        self.graded_gp(rhs, false)
    }

    /// Dual: blade b ↦ complement (top ⊕ b) with the orientation sign.
    /// ★ THE SYSTEM ANCHOR: the sign of `complement_sign` is chosen so that
    ///   b ∧ dual(b) = +I  (see the `//!` at the top of the file and the `orientation_anchor` test).
    /// This is the ONLY free choice; everything else is derived from it.
    /// Grade⟨k⟩ ↦ Grade⟨n−k⟩.
    pub fn dual(&self) -> Self {
        let mut out = Self::zero();
        let top = A::DIM - 1;
        let mut s = 0;
        while s < A::DIM {
            let comp = top ^ s;
            let c = self.get(s);
            out.add_at(comp, if complement_sign(s, comp) < 0 { -c } else { c });
            s += 1;
        }
        out
    }

    /// Multiplication by a scalar — componentwise, by structural recursion.
    pub fn scale(&self, s: S) -> Self {
        Mv(self.0.scale(s), PhantomData)
    }
    /// Left complement — the inverse of dual (the order of the sign arguments is reversed).
    pub fn undual(&self) -> Self {
        let mut out = Self::zero();
        let top = A::DIM - 1;
        let mut s = 0;
        while s < A::DIM {
            let comp = top ^ s;
            let c = self.get(s);
            out.add_at(comp, if complement_sign(comp, s) < 0 { -c } else { c });
            s += 1;
        }
        out
    }
    /// Regressive (meet) product: undual(dual(a) ∧ dual(b)).
    pub fn meet(&self, rhs: &Self) -> Self {
        self.dual().wedge(&rhs.dual()).undual()
    }
    /// Lift the coefficients through the canonical injection S ↪ T (deepen).
    pub fn deepen<T>(&self) -> Mv<A, T>
    where
        T: Ring,
        S: Lift<T>,
    {
        let mut out = Mv::<A, T>::zero();
        let mut b = 0;
        while b < A::DIM {
            out.add_at(b, self.get(b).lift());
            b += 1;
        }
        out
    }
}

// gp and its derivatives are a separate block: structural recursion needs the
// carrier to unfold by the top gen. For concrete algebras the bound
// is resolved by normalization; generic code carries one honest line.
impl<A: Carrier, S: Ring> Mv<A, S>
where
    <A as Carrier>::St<S>: GpRec<A>,
{
    /// Geometric product — the Cayley–Dickson recursion over the storage.
    pub fn gp(&self, rhs: &Self) -> Self {
        Mv(GpRec::<A>::gp_rec(&self.0, &rhs.0), PhantomData)
    }
    /// RAW commutator: ab − ba, WITHOUT ½. The foundation of the Lie bracket — but it does not
    /// carry the ½ itself (½ is localized in `Twist::bracket`/`velocity_at`/`exp`). Antisymmetric
    /// by construction: commutator(a,b) = −commutator(b,a).
    pub fn commutator(&self, rhs: &Self) -> Self {
        self.gp(rhs) - rhs.gp(self)
    }
    /// Sandwich M X M̃ (conjugation by a versor).
    pub fn conjugate(&self, x: &Self) -> Self {
        self.gp(x).gp(&self.reverse())
    }
}

// Inversion of a multivector via the regular representation + Gauss–Jordan
// (Bareiss, division only by the previous pivot). None for non-invertible ones.
/// The matrix of the regular representation: a peano vector of rows, the dimension is
/// the type-level DimP (= 2^NGEN by structural doubling). No Vec —
/// the boundary linear algebra lives on the same carrier as everything else.
type RegMat<A, S> = Vector<<A as Algebra>::DimP, Vector<<A as Algebra>::DimP, S>>;

impl<A: Carrier, S: Ring + Invertible> Mv<A, S>
where
    <A as Carrier>::St<S>: GpRec<A>,
    A::DimP: Storage,
{
    pub fn inverse(&self) -> Option<Self> {
        let dim = A::DIM; // == DimP::LEN — runtime bounds of the Gaussian loops
        let at = |m: &RegMat<A, S>, i: usize, j: usize| -> S {
            *<A::DimP as Storage>::get(<A::DimP as Storage>::get(m, i), j)
        };
        let row =
            |m: &RegMat<A, S>, i: usize| -> Vector<A::DimP, S> { *<A::DimP as Storage>::get(m, i) };
        // a[i][j] = the coefficient (blade i) of gp(self, e_j); b is the identity.
        let cols: RegMat<A, S> = <A::DimP as Storage>::from_fn(|j| {
            let mut ej = Self::zero();
            ej.add_at(j, S::ONE);
            let col = self.gp(&ej);
            <A::DimP as Storage>::from_fn(|i| col.get(i))
        });
        let mut a: RegMat<A, S> =
            <A::DimP as Storage>::from_fn(|i| <A::DimP as Storage>::from_fn(|j| at(&cols, j, i)));
        let mut bmat: RegMat<A, S> = <A::DimP as Storage>::from_fn(|i| {
            <A::DimP as Storage>::from_fn(|j| if i == j { S::ONE } else { S::ZERO })
        });
        let mut prev = S::ONE;
        let mut k = 0;
        while k < dim {
            let mut piv = k;
            while piv < dim && at(&a, piv, k).try_recip().is_none() {
                piv += 1;
            }
            if piv == dim {
                return None; // degenerate → non-invertible
            }
            if piv != k {
                let (ak, ap) = (row(&a, k), row(&a, piv));
                <A::DimP as Storage>::set(&mut a, k, ap);
                <A::DimP as Storage>::set(&mut a, piv, ak);
                let (bk, bp) = (row(&bmat, k), row(&bmat, piv));
                <A::DimP as Storage>::set(&mut bmat, k, bp);
                <A::DimP as Storage>::set(&mut bmat, piv, bk);
            }
            let pivot = at(&a, k, k);
            let mut i = 0;
            while i < dim {
                if i != k {
                    let factor = at(&a, i, k);
                    let pinv = prev.try_recip()?;
                    // Bareiss: the whole row (r_i·pivot − r_k·factor)/prev.
                    let (ai, ak) = (row(&a, i), row(&a, k));
                    let na = <A::DimP as Storage>::from_fn(|jj| {
                        let (x, y) = (
                            *<A::DimP as Storage>::get(&ai, jj),
                            *<A::DimP as Storage>::get(&ak, jj),
                        );
                        (x * pivot - y * factor) * pinv
                    });
                    <A::DimP as Storage>::set(&mut a, i, na);
                    let (bi, bk) = (row(&bmat, i), row(&bmat, k));
                    let nb = <A::DimP as Storage>::from_fn(|jj| {
                        let (x, y) = (
                            *<A::DimP as Storage>::get(&bi, jj),
                            *<A::DimP as Storage>::get(&bk, jj),
                        );
                        (x * pivot - y * factor) * pinv
                    });
                    <A::DimP as Storage>::set(&mut bmat, i, nb);
                }
                i += 1;
            }
            prev = pivot;
            k += 1;
        }
        let det_inv = prev.try_recip()?;
        let mut out = Self::zero();
        let mut i = 0;
        while i < dim {
            out.add_at(i, at(&bmat, i, 0) * det_inv);
            i += 1;
        }
        Some(out)
    }
}

// Versor normalization (drift removal): divide by the Euclidean norm. Degeneracy
// guard (zero check outside the AD path) — as in clifford.
impl<A: Carrier, S: Scalar + StandardPart> Mv<A, S>
where
    <A as Carrier>::St<S>: GpRec<A>,
{
    pub fn normalize(&mut self) {
        let n2 = self.norm_squared();
        if n2.standard_part().is_effective_zero() {
            return;
        }
        let inv = n2.sqrt_explicit().try_recip().unwrap();
        *self = self.scale(inv);
    }
    pub fn normalized(&self) -> Self {
        let mut c = self.clone();
        c.normalize();
        c
    }
}

// CLOSED-FORM exponential of a SIMPLE bivector (B² = scalar), not a series:
//   exp(B) = cos(√u)·1 + sinc(√u)·B,  u = −⟨B²⟩₀   (Euclidean bivector ⇒ u≥0).
// The Study forms sinc_sq/cos_sq are now on Scalar (each carrier has its own
// representation), so a Jet coefficient ⇒ exp differentiates itself.
// The full screw (SE(3)) exp is in the pga module via Motor.
impl<A: Carrier, S: Scalar> Mv<A, S>
where
    <A as Carrier>::St<S>: GpRec<A>,
{
    pub fn exp(&self) -> Self {
        let u = -self.gp(self).get(0); // = |B|² for a Euclidean bivector
        Self::scalar(u.cos_sq()) + self.scale(u.sinc_sq())
    }

    /// ⟨A Ã⟩₀ — the scalar part of gp with the reverse (the squared Euclidean norm).
    pub fn norm_squared(&self) -> S {
        self.gp(&self.reverse()).get(0)
    }
    /// Euclidean norm √⟨A Ã⟩₀.
    pub fn norm(&self) -> S {
        self.norm_squared().sqrt_explicit()
    }
    /// Regularized norm √(‖·‖² + ε²): the derivative is finite through 0.
    pub fn soft_norm(&self, eps: S) -> S {
        (self.norm_squared() + eps * eps).sqrt_explicit()
    }
}

// Abelian group and multiplication-as-gp on Mv itself.
impl<A: Carrier, S: Ring> Add for Mv<A, S> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Mv(self.0 + o.0, PhantomData)
    }
}
impl<A: Carrier, S: Ring> Sub for Mv<A, S> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Mv(self.0 - o.0, PhantomData)
    }
}
impl<A: Carrier, S: Ring> Neg for Mv<A, S> {
    type Output = Self;
    fn neg(self) -> Self {
        Mv(-self.0, PhantomData)
    }
}
impl<A: Carrier, S: Ring> Mul for Mv<A, S>
where
    <A as Carrier>::St<S>: GpRec<A>,
{
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        self.gp(&o)
    }
}

// ── Sections as predicates on blades (without a single blade literal) ─────
// «Take grade-2» / «the force part» is not a list of indices but a predicate
// «does blade b belong to this subspace», derived from the grade and the METRIC
// (A::ZERO_MASK). Predicates compose (And), cardinality falls out
// by counting (like Count for strata).
/// project⟨P⟩: Mv → a narrow Peano vector. The r-th blade passing the predicate → slot r
/// (canonical order = increasing mask). «Blade→axis» is a RANK, not a
/// table of indices. (iso⟨P,Q⟩ with dim⟨P⟩=dim⟨Q⟩ is the identity on the carrier,
/// since both are ranked the same way; hence force→Vec3 is exactly an iso.)
pub fn project<A, P, S, VN>(m: &Mv<A, S>) -> Vector<VN, S>
where
    A: Carrier,
    P: Section<A>,
    S: Ring,
    VN: Storage,
{
    let mut out = VN::from_fn(|_| S::ZERO);
    let mut b = 0;
    let mut r = 0;
    while b < A::DIM {
        if P::contains(b) {
            VN::set(&mut out, r, m.get(b));
            r += 1;
        }
        b += 1;
    }
    out
}

/// inject⟨P⟩: narrow carrier → Mv, components into the same blades of the section (by the same
/// canonical rank), zeros outside P. project⟨P⟩ ∘ inject⟨P⟩ = id.
pub fn inject<A, P, S, VN>(v: &Vector<VN, S>) -> Mv<A, S>
where
    A: Carrier,
    P: Section<A>,
    S: Ring,
    VN: Storage,
{
    let mut out = Mv::zero();
    let mut b = 0;
    let mut r = 0;
    while b < A::DIM {
        if P::contains(b) {
            out.add_at(b, *VN::get(v, r));
            r += 1;
        }
        b += 1;
    }
    out
}

type V3<S> = Vector<Succ<Succ<Succ<Z>>>, S>;
/// A wrench is simply a grade-2 multivector of PGA3 (type alias).
pub type Mpga<S> = Mv<Pga3, S>;

// ── Dual axis markup (⋆₃), not rank-based ────────────────────────
// Axis i ↦ eᵢ (the i-th non-degenerate generator in increasing bit order). Physical quantities
// are marked up through ⋆₃ — the EUCLIDEAN sub-dual (complement in I₃ = the pseudoscalar
// of the Euclidean subspace, WITHOUT e₀), the same one that defines cross. All signs
// come from complement_sign/blade_prod and are COMPUTED. Axes/masks are derived from the metric
// (A::ZERO_MASK), not written out.

// Reading a Vec3 from a (mask, sign) table — shared code for the dual readers.
fn read_dual<S: Ring>(w: &Mpga<S>, blade: fn(usize) -> (usize, i8)) -> V3<S> {
    let mut out = V3::<S>::ZERO;
    let mut i = 0;
    while i < 3 {
        let (mask, sign) = blade(i);
        let c = w.get(mask);
        out[i] = if sign < 0 { -c } else { c };
        i += 1;
    }
    out
}
/// The point at the origin: e₁₂₃ (the Euclidean pseudoscalar, weight 1).
pub fn origin<S: Ring>() -> Mpga<S> {
    let i3 = (Pga3::DIM - 1) & !Pga3::ZERO_MASK;
    let mut p = Mv::zero();
    p.add_at(i3, S::ONE);
    p
}
/// Position of a point: the e₀-trivector part (dually), DIVIDED by the weight
/// (the coefficient of e₁₂₃) — projective normalization rather than a silent «weight 1».
pub fn position<S: Ring + Invertible>(p: &Mpga<S>) -> V3<S> {
    let i3 = (Pga3::DIM - 1) & !Pga3::ZERO_MASK;
    let winv = p.get(i3).try_recip().expect("point weight is zero");
    let mut out = V3::<S>::ZERO;
    let mut i = 0;
    while i < 3 {
        let (mask, sign) = point_blade::<Pga3>(i);
        let c = p.get(mask) * winv;
        out[i] = if sign < 0 { -c } else { c };
        i += 1;
    }
    out
}
/// A point of weight 1 at position v — the inverse of position (position(make_point(v))=v).
pub fn make_point<S: Ring>(v: &V3<S>) -> Mpga<S> {
    let mut p = origin::<S>();
    let mut i = 0;
    while i < 3 {
        let (mask, sign) = point_blade::<Pga3>(i);
        let c = v[i];
        p.add_at(mask, if sign < 0 { -c } else { c });
        i += 1;
    }
    p
}

// Twist/wrench are now built DIRECTLY in the narrow grade-2 carrier (the `screw` module,
// Screw<Contra>/Screw<Co>) via translation_store/make_force_store — the old
// twist_embed/twist_project (which built the full Mv) was removed as dead.
/// Pseudoscalar (the unit highest blade) — the canonical top, from DIM.
pub fn pseudoscalar<S: Ring>() -> Mpga<S> {
    let mut m = Mv::zero();
    m.add_at(Pga3::DIM - 1, S::ONE);
    m
}

// ── Plane (grade-1): normal along the Euclidean generators, offset along e0 ─
// Not a single literal: e0 = ZERO_MASK (the degenerate generator), eᵢ = axis_vec.
pub fn make_plane<S: Ring>(normal: &V3<S>, offset: S) -> Mpga<S> {
    let mut m = Mv::zero();
    m.add_at(Pga3::ZERO_MASK, offset);
    let mut i = 0;
    while i < 3 {
        m.add_at(axis_vec::<Pga3>(i), normal[i]);
        i += 1;
    }
    m
}
pub fn plane_normal<S: Ring>(m: &Mpga<S>) -> V3<S> {
    let mut out = V3::<S>::ZERO;
    let mut i = 0;
    while i < 3 {
        out[i] = m.get(axis_vec::<Pga3>(i));
        i += 1;
    }
    out
}
pub fn plane_offset<S: Ring>(m: &Mpga<S>) -> S {
    m.get(Pga3::ZERO_MASK)
}

// ── Direction (ideal point, weight 0): the e₀-trivector part without e₁₂₃ ──
pub fn make_direction<S: Ring>(v: &V3<S>) -> Mpga<S> {
    let mut m = Mv::zero();
    let mut i = 0;
    while i < 3 {
        let (mask, sign) = point_blade::<Pga3>(i);
        let c = v[i];
        m.add_at(mask, if sign < 0 { -c } else { c });
        i += 1;
    }
    m
}
/// Coordinates of a direction (weight 0 ⇒ no normalization) — the same e₀ markup as a point.
pub fn direction_coords<S: Ring>(m: &Mpga<S>) -> V3<S> {
    read_dual(m, point_blade::<Pga3>)
}
/// Direction/force of a line (Euclidean bivectors) and its moment (radical ones).
pub fn line_direction<S: Ring>(m: &Mpga<S>) -> V3<S> {
    read_dual(m, euclid_blade::<Pga3>)
}
pub fn line_moment<S: Ring>(m: &Mpga<S>) -> V3<S> {
    read_dual(m, radical_blade::<Pga3>)
}

// ── Narrow (store-generic) variants of the dual readers/writers ───────
// They read/write the same blades, but DIRECTLY on the section carrier (GList), without
// extending to the full Mv. For operations closed in the even subalgebra.
fn read_dual_store<S: Ring, St: GList<S>>(st: &St, blade: fn(usize) -> (usize, i8)) -> V3<S> {
    let n = Pga3::NGEN;
    let mut out = V3::<S>::ZERO;
    let mut i = 0;
    while i < 3 {
        let (mask, sign) = blade(i);
        let c = st.get(mask, n, n);
        out[i] = if sign < 0 { -c } else { c };
        i += 1;
    }
    out
}
fn write_dual_store<S: Ring, St: GList<S> + AbelianGroup>(
    v: &V3<S>,
    blade: fn(usize) -> (usize, i8),
) -> St {
    let n = Pga3::NGEN;
    let mut out = St::ZERO;
    let mut i = 0;
    while i < 3 {
        let (mask, sign) = blade(i);
        let c = v[i];
        out.add_at(mask, n, n, if sign < 0 { -c } else { c });
        i += 1;
    }
    out
}
pub fn line_direction_store<S: Ring, St: GList<S>>(st: &St) -> V3<S> {
    read_dual_store(st, euclid_blade::<Pga3>)
}
pub fn line_moment_store<S: Ring, St: GList<S>>(st: &St) -> V3<S> {
    read_dual_store(st, radical_blade::<Pga3>)
}
pub fn make_force_store<S: Ring, St: GList<S> + AbelianGroup>(v: &V3<S>) -> St {
    write_dual_store(v, euclid_blade::<Pga3>)
}
pub fn translation_store<S: Ring, St: GList<S> + AbelianGroup>(v: &V3<S>) -> St {
    write_dual_store(v, radical_blade::<Pga3>)
}
/// The angular half of a screw (ω of a twist / τ of a wrench): the angular markup is the Euclidean
/// blade × the COMPUTED embedding sign σ (see markup::angular_embed_sign).
pub fn angular_store<S: Ring, St: GList<S> + AbelianGroup>(v: &V3<S>) -> St {
    write_dual_store(v, angular_blade::<Pga3>)
}
pub fn angular_from_store<S: Ring, St: GList<S>>(st: &St) -> V3<S> {
    read_dual_store(st, angular_blade::<Pga3>)
}

// ── ⋆₃-cross INSIDE PGA: prove the consistency of moment/⋆₃ with the vector ──
// cross in Pga3 itself (not only in Cl(3,0)). Lift two Vec3s into grade-1
// (Euclidean generators), take the wedge (a Euclidean bivector) and read off its
// components dually (⋆₃) — with the same euclid_blade that marks up force.
fn lift_vec<S: Ring>(v: &V3<S>) -> Mpga<S> {
    let mut m = Mv::zero();
    let mut i = 0;
    while i < 3 {
        m.add_at(axis_vec::<Pga3>(i), v[i]);
        i += 1;
    }
    m
}
/// The cross product via ⋆₃ INSIDE Pga3: ⋆₃(a∧b), componentwise.
pub fn euclid_cross<S: Ring>(a: &V3<S>, b: &V3<S>) -> V3<S> {
    let ab = lift_vec(a).wedge(&lift_vec(b));
    read_dual(&ab, euclid_blade::<Pga3>)
}

// =====================================================================
// PGA3: rigid motions (Motor), se(3) twists, geometric objects.
// Everything on top of ga — in coordinates and through the dual markup, without blade literals.
// Motor::exp is the full screw (SE(3)) exponential via the Study forms (screw =
// rotation+translation), not just a simple rotor.
