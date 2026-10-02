// SPDX-License-Identifier: MIT

//! Strata carriers and operations on them: recursive Void/Leaf/Split (Pascal
//! within a grade), GCons/TNil (the list of grades), a componentwise abelian group,
//! structural operations (recursion over the storage, not over DIM) and the Cayley–Dickson gp.

use super::{Algebra, Gen, Nil, blade_grade, blade_prod};
use bytemuck::{Pod, Zeroable};
use core::fmt::Debug;
use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};
use peano::prelude::*;

// ── Strata carriers ───────────────────────────────────────────────────
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Void; // there is no stratum of this grade
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Leaf<S>(pub S); // one component (the bottom of grade-0)
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Split<Keep, Wear>(pub Keep, pub Wear); // Pascal: keep does not touch the gen, wear does

// ── Leaf count of a form = C(n,k) ──────────────────────────────────────
pub trait Count {
    const N: usize;
}
impl Count for Void {
    const N: usize = 0;
}
impl<S> Count for Leaf<S> {
    const N: usize = 1;
}
impl<K: Count, W: Count> Count for Split<K, W> {
    const N: usize = K::N + W::N;
}

// The grade-2 stratum of Pga3 (the twist/wrench carrier).

// ── Full storage: the list of strata by grade (head = highest grade) ──
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TNil;
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GCons<H, T>(pub H, pub T);

// ── Pod/Zeroable on strata carriers ──────────────────────────────────
// As with peano `Cons`, `Pod` rests on HOMOGENEITY: every real
// `Mv<A, S>` storage is built over a single scalar `S` — all non-ZST leaves
// are `S`, and `Void`/`TNil` are align-1 ZSTs — so the `#[repr(C)]` tree
// is packed with no holes. Heterogeneous alignment (the only source
// of padding) cannot be produced by any `Carrier::St`; `assert_packed` on
// the concrete types catches a regression at compile time.
unsafe impl Zeroable for Void {}
unsafe impl Pod for Void {}
unsafe impl Zeroable for TNil {}
unsafe impl Pod for TNil {}
unsafe impl<S: Zeroable> Zeroable for Leaf<S> {}
unsafe impl<S: Pod> Pod for Leaf<S> {}
unsafe impl<K: Zeroable, W: Zeroable> Zeroable for Split<K, W> {}
unsafe impl<K: Pod, W: Pod> Pod for Split<K, W> {}
unsafe impl<H: Zeroable, T: Zeroable> Zeroable for GCons<H, T> {}
unsafe impl<H: Pod, T: Pod> Pod for GCons<H, T> {}

// Promise bounds on the GAT (GList + group) — the same induction hypothesis,

// ── Abelian group on carriers (componentwise) ──────────────────────
// The carrier's zero is the associated constant AbelianGroup::ZERO; the separate
// Zeroed constructor was a crutch around Leaf, which was not part of the group.

macro_rules! group_pair {
    ($name:ident, $a:ident, $b:ident) => {
        impl<$a: Add<Output = $a> + Copy, $b: Add<Output = $b> + Copy> Add for $name<$a, $b> {
            type Output = Self;
            fn add(self, o: Self) -> Self {
                $name(self.0 + o.0, self.1 + o.1)
            }
        }
        impl<$a: Sub<Output = $a> + Copy, $b: Sub<Output = $b> + Copy> Sub for $name<$a, $b> {
            type Output = Self;
            fn sub(self, o: Self) -> Self {
                $name(self.0 - o.0, self.1 - o.1)
            }
        }
        impl<$a: Add<Output = $a> + Copy, $b: Add<Output = $b> + Copy> AddAssign for $name<$a, $b> {
            fn add_assign(&mut self, o: Self) {
                *self = $name(self.0 + o.0, self.1 + o.1);
            }
        }
        impl<$a: Sub<Output = $a> + Copy, $b: Sub<Output = $b> + Copy> SubAssign for $name<$a, $b> {
            fn sub_assign(&mut self, o: Self) {
                *self = $name(self.0 - o.0, self.1 - o.1);
            }
        }
        impl<$a: Neg<Output = $a> + Copy, $b: Neg<Output = $b> + Copy> Neg for $name<$a, $b> {
            type Output = Self;
            fn neg(self) -> Self {
                $name(-self.0, -self.1)
            }
        }
        impl<$a: AbelianGroup, $b: AbelianGroup> AbelianGroup for $name<$a, $b> {
            const ZERO: Self = $name($a::ZERO, $b::ZERO);
        }
    };
}
group_pair!(Split, K, W);
group_pair!(GCons, H, T);

macro_rules! group_unit {
    ($t:ty, $v:expr) => {
        impl Add for $t {
            type Output = $t;
            fn add(self, _: $t) -> $t {
                $v
            }
        }
        impl Sub for $t {
            type Output = $t;
            fn sub(self, _: $t) -> $t {
                $v
            }
        }
        impl AddAssign for $t {
            fn add_assign(&mut self, _: $t) {
                *self = $v;
            }
        }
        impl SubAssign for $t {
            fn sub_assign(&mut self, _: $t) {
                *self = $v;
            }
        }
        impl Neg for $t {
            type Output = $t;
            fn neg(self) -> $t {
                $v
            }
        }
        impl AbelianGroup for $t {
            const ZERO: Self = $v;
        }
    };
}
group_unit!(Void, Void);
group_unit!(TNil, TNil);

// Leaf is an abelian group too (previously there was only the Zeroed crutch for zero;
// because of that hole the whole strata stack could not be called a group and dragged
// around enumerative bound bundles).
impl<S: Add<Output = S> + Copy> Add for Leaf<S> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Leaf(self.0 + o.0)
    }
}
impl<S: Sub<Output = S> + Copy> Sub for Leaf<S> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Leaf(self.0 - o.0)
    }
}
impl<S: Neg<Output = S> + Copy> Neg for Leaf<S> {
    type Output = Self;
    fn neg(self) -> Self {
        Leaf(-self.0)
    }
}
impl<S: AbelianGroup> AddAssign for Leaf<S> {
    fn add_assign(&mut self, o: Self) {
        *self = Leaf(self.0 + o.0);
    }
}
impl<S: AbelianGroup> SubAssign for Leaf<S> {
    fn sub_assign(&mut self, o: Self) {
        *self = Leaf(self.0 - o.0);
    }
}
impl<S: AbelianGroup> AbelianGroup for Leaf<S> {
    const ZERO: Self = Leaf(S::ZERO);
}

// ── Structural operations: recursion over the storage, not over DIM ───────────

// Scaling by a scalar — componentwise (peano::ScalarMul).
impl<S: Copy> ScalarMul<S> for Void {
    fn scale(self, _: S) -> Void {
        Void
    }
}
impl<S: Copy> ScalarMul<S> for TNil {
    fn scale(self, _: S) -> TNil {
        TNil
    }
}
impl<S: Mul<Output = S> + Copy> ScalarMul<S> for Leaf<S> {
    fn scale(self, s: S) -> Self {
        Leaf(self.0 * s)
    }
}
impl<S: Copy, K: ScalarMul<S> + Copy, W: ScalarMul<S> + Copy> ScalarMul<S> for Split<K, W> {
    fn scale(self, s: S) -> Self {
        Split(self.0.scale(s), self.1.scale(s))
    }
}
impl<S: Copy, H: ScalarMul<S> + Copy, T: ScalarMul<S> + Copy> ScalarMul<S> for GCons<H, T> {
    fn scale(self, s: S) -> Self {
        GCons(self.0.scale(s), self.1.scale(s))
    }
}

/// Per-stratum sign/filter by GRADE: involutions and grade projections decide
/// the fate of a WHOLE stratum (grade = position in the list) and do not go inside a stratum —
/// neither DIM nor blades. Neg and ZERO are structural (strata are abelian groups).
pub enum StratumSign {
    Keep,
    Neg,
    Drop,
}
pub trait GradeMap: Sized {
    fn grade_map(self, head_grade: usize, f: &mut impl FnMut(usize) -> StratumSign) -> Self;
}
impl GradeMap for TNil {
    fn grade_map(self, _g: usize, _f: &mut impl FnMut(usize) -> StratumSign) -> Self {
        TNil
    }
}
impl<H: AbelianGroup, T: GradeMap> GradeMap for GCons<H, T> {
    fn grade_map(self, g: usize, f: &mut impl FnMut(usize) -> StratumSign) -> Self {
        let h = match f(g) {
            StratumSign::Keep => self.0,
            StratumSign::Neg => -self.0,
            StratumSign::Drop => H::ZERO,
        };
        GCons(h, self.1.grade_map(g.wrapping_sub(1), f))
    }
}

// ── The geometric product as Cayley–Dickson recursion ────────────
//
// A = a₀ + a₁·e (e is the top generator, e² = SQ; wear blade = keep·e, with e
// on the RIGHT — the same convention as the increasing order of masks). Moving e
// left through a Rest multivector x yields x̂ (the grade involution: e anticommutes
// with every gen of Rest), whence
//   A·B = (a₀·b₀ + SQ·a₁·b̂₁) + (a₀·b₁ + a₁·b̂₀)·e.
// The signs come FROM THE STRUCTURE (metric SQ + grade parity) — no table,
// no permutation derivation, no DIM. There is still only one freedom (the dual anchor).

/// Fibering of the strata list by the top gen: strata g..1 are Split<keep,wear>,
/// the bottom (grade 0) is entirely keep, it has no wear.
pub trait PeelSplit: Sized {
    type Keeps;
    type Wears;
    fn peel(self) -> (Self::Keeps, Self::Wears);
    fn unpeel(k: Self::Keeps, w: Self::Wears) -> Self;
}
impl<S: Copy> PeelSplit for GCons<Leaf<S>, TNil> {
    type Keeps = GCons<Leaf<S>, TNil>;
    type Wears = TNil;
    fn peel(self) -> (Self::Keeps, Self::Wears) {
        (self, TNil)
    }
    fn unpeel(k: Self::Keeps, _w: TNil) -> Self {
        k
    }
}
impl<K, W, H2, T2> PeelSplit for GCons<Split<K, W>, GCons<H2, T2>>
where
    GCons<H2, T2>: PeelSplit,
{
    type Keeps = GCons<K, <GCons<H2, T2> as PeelSplit>::Keeps>;
    type Wears = GCons<W, <GCons<H2, T2> as PeelSplit>::Wears>;
    fn peel(self) -> (Self::Keeps, Self::Wears) {
        let GCons(Split(k0, w0), tail) = self;
        let (ks, ws) = tail.peel();
        (GCons(k0, ks), GCons(w0, ws))
    }
    fn unpeel(k: Self::Keeps, w: Self::Wears) -> Self {
        GCons(Split(k.0, w.0), <GCons<H2, T2>>::unpeel(k.1, w.1))
    }
}

/// Unfolding of a FULL carrier F(Gen) ≅ (F(Rest), F(Rest)): the keep and wear parts.
/// The keep half of the top stratum is a «ghost» (grade n of an (n−1)-gen algebra,
/// a tree of Void): it is discarded on unfolding and laid down as zero on assembly.
/// The binding `Keeps = GCons<W, Wears>` equates the types of the two halves.
pub trait TopSplit: Sized {
    type Part;
    fn top_split(self) -> (Self::Part, Self::Part);
    fn top_join(k: Self::Part, w: Self::Part) -> Self;
}
impl<G: AbelianGroup, W, H2, T2, WS> TopSplit for GCons<Split<G, W>, GCons<H2, T2>>
where
    GCons<H2, T2>: PeelSplit<Wears = WS, Keeps = GCons<W, WS>>,
{
    type Part = GCons<W, WS>;
    fn top_split(self) -> (Self::Part, Self::Part) {
        let GCons(Split(_ghost, w_top), tail) = self;
        let (ks, ws) = tail.peel();
        (ks, GCons(w_top, ws))
    }
    fn top_join(k: Self::Part, w: Self::Part) -> Self {
        let GCons(w_top, ws) = w;
        GCons(Split(G::ZERO, w_top), <GCons<H2, T2>>::unpeel(k, ws))
    }
}

/// gp by structural recursion. Requires a FULL carrier (all strata Split);
/// narrow sections go through gp_store (the result type of a product of sections
/// is heterogeneous — a topic for the next iteration).
pub trait GpRec<A>: Sized {
    fn gp_rec(a: &Self, b: &Self) -> Self;
}
// Base: a field (0 generators) — a product of scalars.
impl<S: Ring> GpRec<Nil> for GCons<Leaf<S>, TNil> {
    fn gp_rec(a: &Self, b: &Self) -> Self {
        GCons(Leaf(a.0.0 * b.0.0), TNil)
    }
}
// Step: unfold by the top gen, four sub-products over Rest.
impl<const SQ: i8, Rest, G, W, H2, T2> GpRec<Gen<SQ, Rest>> for GCons<Split<G, W>, GCons<H2, T2>>
where
    Rest: Algebra,
    Self: TopSplit + Copy,
    <Self as TopSplit>::Part: GpRec<Rest> + GradeMap + AbelianGroup,
{
    fn gp_rec(a: &Self, b: &Self) -> Self {
        let (a0, a1) = (*a).top_split();
        let (b0, b1) = (*b).top_split();
        // grade involution of the Rest multivector: odd strata with sign −
        let ginv = |x: <Self as TopSplit>::Part| {
            x.grade_map(Rest::NGEN, &mut |g| {
                if g & 1 == 1 {
                    StratumSign::Neg
                } else {
                    StratumSign::Keep
                }
            })
        };
        let c0 = {
            let kk = GpRec::<Rest>::gp_rec(&a0, &b0);
            if SQ == 0 {
                kk // degenerate gen: e² = 0, wear·wear annihilates
            } else {
                let ww = GpRec::<Rest>::gp_rec(&a1, &ginv(b1));
                if SQ == 1 { kk + ww } else { kk - ww }
            }
        };
        let c1 = GpRec::<Rest>::gp_rec(&a0, &b1) + GpRec::<Rest>::gp_rec(&a1, &ginv(b0));
        TopSplit::top_join(c0, c1)
    }
}

// ── Component access by blade ────────────────────────────────────
// Within a stratum: descend through Split; the highest bit of the blade decides keep/wear.
pub trait Stratum<S> {
    fn get(&self, blade: usize, ngen: usize) -> S;
    fn add_at(&mut self, blade: usize, ngen: usize, v: S);
}
impl<S: Ring> Stratum<S> for Void {
    fn get(&self, _b: usize, _n: usize) -> S {
        S::ZERO // unreachable for a valid blade of its own grade
    }
    fn add_at(&mut self, _b: usize, _n: usize, _v: S) {}
}
impl<S: Ring> Stratum<S> for Leaf<S> {
    fn get(&self, _b: usize, _n: usize) -> S {
        self.0
    }
    // NOT `self.0 += v`. `Ring` requires both `Add` and `AddAssign`, and an
    // implementor is free to make them disagree — clifford's own tracing scalar
    // in `base.rs` does: its `Add` folds `x + 0` and `constant + constant`,
    // while its `AddAssign` folds only `x += 0` and emits a runtime op
    // otherwise. Spelling this `+=` raised the traced op count of the
    // `exp`+`norm_squared` path from 128 to 175 and turned
    // `base::tests::rational_compiles` red. `AddAssign for Leaf<S>` just below
    // is written over `S::add` for the same reason.
    #[expect(
        clippy::assign_op_pattern,
        reason = "`S::add` and `S::add_assign` are not interchangeable for a Ring; see above"
    )]
    fn add_at(&mut self, _b: usize, _n: usize, v: S) {
        self.0 = self.0 + v;
    }
}
impl<S: Ring, K: Stratum<S>, W: Stratum<S>> Stratum<S> for Split<K, W> {
    fn get(&self, blade: usize, ngen: usize) -> S {
        let bit = ngen - 1;
        if (blade >> bit) & 1 == 1 {
            self.1.get(blade, bit) // wear: the generator is present
        } else {
            self.0.get(blade, bit) // keep: the generator is absent
        }
    }
    fn add_at(&mut self, blade: usize, ngen: usize, v: S) {
        let bit = ngen - 1;
        if (blade >> bit) & 1 == 1 {
            self.1.add_at(blade, bit, v)
        } else {
            self.0.add_at(blade, bit, v)
        }
    }
}

// Between strata: route by grade (= popcount of the blade). The head of the list is
// the highest grade (= NGEN), decreasing from there.
pub trait GList<S> {
    fn get(&self, blade: usize, head_grade: usize, ngen: usize) -> S;
    fn add_at(&mut self, blade: usize, head_grade: usize, ngen: usize, v: S);
}
impl<S: Ring> GList<S> for TNil {
    fn get(&self, _b: usize, _hg: usize, _n: usize) -> S {
        S::ZERO
    }
    fn add_at(&mut self, _b: usize, _hg: usize, _n: usize, _v: S) {}
}
impl<S: Ring, H: Stratum<S>, T: GList<S>> GList<S> for GCons<H, T> {
    fn get(&self, blade: usize, head_grade: usize, ngen: usize) -> S {
        if blade_grade(blade) == head_grade {
            self.0.get(blade, ngen)
        } else {
            self.1.get(blade, head_grade - 1, ngen)
        }
    }
    fn add_at(&mut self, blade: usize, head_grade: usize, ngen: usize, v: S) {
        if blade_grade(blade) == head_grade {
            self.0.add_at(blade, ngen, v)
        } else {
            self.1.add_at(blade, head_grade - 1, ngen, v)
        }
    }
}

// ── Operations CLOSED on a narrow section carrier ─────────────────────
// They work DIRECTLY on St (GList), without extending to the full Mv. For a section
// closed under gp (the even subalgebra: even·even = even by population
// arithmetic) add_at always lands in an existing stratum — an odd output
// NEVER HAPPENS by construction, rather than being «discarded after the fact».
pub fn gp_store<A: Algebra, S: Ring, St: GList<S> + AbelianGroup>(a: &St, b: &St) -> St {
    let mut out = St::ZERO;
    let (dim, n) = (A::DIM, A::NGEN);
    let mut i = 0;
    while i < dim {
        let ca = a.get(i, n, n);
        let mut j = 0;
        while j < dim {
            let (sign, c) = blade_prod(i, j, A::ZERO_MASK, A::NEG_MASK);
            if sign != 0 {
                let term = ca * b.get(j, n, n);
                out.add_at(c, n, n, if sign < 0 { -term } else { term });
            }
            j += 1;
        }
        i += 1;
    }
    out
}
pub fn scale_store<A: Algebra, S: Ring, St: GList<S> + AbelianGroup>(a: &St, s: S) -> St {
    let mut out = St::ZERO;
    let (dim, n) = (A::DIM, A::NGEN);
    let mut b = 0;
    while b < dim {
        out.add_at(b, n, n, a.get(b, n, n) * s);
        b += 1;
    }
    out
}
pub fn sign_store<A: Algebra, S: Ring, St: GList<S> + AbelianGroup>(
    a: &St,
    flip: fn(u32) -> bool,
) -> St {
    let mut out = St::ZERO;
    let (dim, n) = (A::DIM, A::NGEN);
    let mut b = 0;
    while b < dim {
        let g = blade_grade(b) as u32;
        let c = a.get(b, n, n);
        out.add_at(b, n, n, if flip(g) { -c } else { c });
        b += 1;
    }
    out
}
/// A carrier with a single blade (scalar 1, pseudoscalar, …).
pub fn store_basis<A: Algebra, S: Ring, St: GList<S> + AbelianGroup>(blade: usize, c: S) -> St {
    let mut out = St::ZERO;
    let n = A::NGEN;
    out.add_at(blade, n, n, c);
    out
}
/// Read a blade's component from a carrier.
pub fn store_get<S: Ring, St: GList<S>>(a: &St, blade: usize, ngen: usize) -> S {
    a.get(blade, ngen, ngen)
}
