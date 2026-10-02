// SPDX-License-Identifier: MIT

//! Pascal strata as GAT type functions: the grade-K form of an algebra, the full list
//! of grades, narrow sections (Void outside the section) and the Euclidean blade section.
//! The GAT bounds are the induction hypothesis: the recursive impls prove them
//! generically, so bound bundles are not needed.

use super::section::{AllGrades, EvenGrades, Grade2, ScalarGrade};
use super::store::{GCons, GList, GradeMap, Leaf, Split, Stratum, TNil, Void};
use super::{Algebra, Gen, Nil, Pga3};
use peano::prelude::*;

// ── Grade-K stratum of an algebra: the type function S ↦ Store as a GAT ────────────
// The scalar S is a parameter of the GAT, not of the trait: a single bound `A: GradeForm<K>`
// names the stratum for ANY scalar (Rust has no for<S> quantifier). The bounds on the
// GAT are the induction hypothesis: the recursive impls (Split of two Rest strata)
// prove them generically, and bundles of the form «Store: GList+Zeroed+…» on
// every blanket impl die off.
pub trait GradeForm<K> {
    type Store<S: Ring>: Stratum<S> + AbelianGroup + ScalarMul<S>;
}
// Bottom: a field has only grade-0 (scalar); above that, Void.
impl GradeForm<Z> for Nil {
    type Store<S: Ring> = Leaf<S>;
}
impl<K> GradeForm<Succ<K>> for Nil {
    type Store<S: Ring> = Void;
}
// Gen, grade-0: a new generator adds no scalars to grade-0.
impl<const SQ: i8, Rest: GradeForm<Z>> GradeForm<Z> for Gen<SQ, Rest> {
    type Store<S: Ring> = <Rest as GradeForm<Z>>::Store<S>;
}
// Gen, grade-(K+1): the Pascal split.
impl<const SQ: i8, K, Rest> GradeForm<Succ<K>> for Gen<SQ, Rest>
where
    Rest: GradeForm<Succ<K>> + GradeForm<K>,
{
    type Store<S: Ring> = Split<
        <Rest as GradeForm<Succ<K>>>::Store<S>, // keep: the same grade stratum of Rest
        <Rest as GradeForm<K>>::Store<S>,       // wear: one grade lower, raised by the new gen
    >;
}

pub type Bivec<S> = <Pga3 as GradeForm<Succ<Succ<Z>>>>::Store<S>;

pub trait StrataUpTo<K> {
    type Store<S: Ring>: GList<S> + AbelianGroup + ScalarMul<S> + GradeMap;
}
impl<A> StrataUpTo<Z> for A
where
    A: GradeForm<Z>,
{
    type Store<S: Ring> = GCons<<A as GradeForm<Z>>::Store<S>, TNil>;
}
impl<A, K> StrataUpTo<Succ<K>> for A
where
    A: GradeForm<Succ<K>> + StrataUpTo<K>,
{
    type Store<S: Ring> =
        GCons<<A as GradeForm<Succ<K>>>::Store<S>, <A as StrataUpTo<K>>::Store<S>>;
}

// ── GENERAL grade section: a narrow carrier whose strata outside the section are Void ──
// A section is given by grade membership (Member<K> → Keep/Drop). The carrier stores
// STRUCTURALLY only the grades of the section; the remaining strata are Void (0 bytes). The size
// falls out as the dim of the subspace. The motor is a special case (the even section).
pub struct Keep;
pub struct Drop;
/// Membership of grade K in the section (type state Keep/Drop).
pub trait Member<K> {
    type State;
}
/// The stratum of grade K under a state: Keep → real, Drop → Void.
pub trait Cell<A, K> {
    type Store<S: Ring>: Stratum<S> + AbelianGroup + ScalarMul<S>;
}
impl<A, K> Cell<A, K> for Keep
where
    A: GradeForm<K>,
{
    type Store<S: Ring> = <A as GradeForm<K>>::Store<S>;
}
impl<A, K> Cell<A, K> for Drop {
    type Store<S: Ring> = Void;
}
/// The list of strata of section Spec for grades 0..=K (head = grade K).
pub trait MaskStrataG<Spec, K> {
    type Store<S: Ring>: GList<S> + AbelianGroup + ScalarMul<S> + GradeMap;
}
impl<A, Spec> MaskStrataG<Spec, Z> for A
where
    Spec: Member<Z>,
    <Spec as Member<Z>>::State: Cell<A, Z>,
{
    type Store<S: Ring> = GCons<<<Spec as Member<Z>>::State as Cell<A, Z>>::Store<S>, TNil>;
}
impl<A, Spec, K> MaskStrataG<Spec, Succ<K>> for A
where
    Spec: Member<Succ<K>>,
    <Spec as Member<Succ<K>>>::State: Cell<A, Succ<K>>,
    A: MaskStrataG<Spec, K>,
{
    type Store<S: Ring> = GCons<
        <<Spec as Member<Succ<K>>>::State as Cell<A, Succ<K>>>::Store<S>,
        <A as MaskStrataG<Spec, K>>::Store<S>,
    >;
}
/// Storage of section Spec of algebra A over S.
pub type SectionStore<A, Spec, S> = <A as MaskStrataG<Spec, <A as Algebra>::NgenP>>::Store<S>;

// ── Concrete grade sections ─────────────────────────────────────────
impl<K> Member<K> for AllGrades {
    type State = Keep;
}
impl Member<Z> for EvenGrades {
    type State = Keep;
}
impl Member<Succ<Z>> for EvenGrades {
    type State = Drop;
}
impl<K> Member<Succ<Succ<K>>> for EvenGrades
where
    EvenGrades: Member<K>,
{
    type State = <EvenGrades as Member<K>>::State;
}
impl Member<Z> for ScalarGrade {
    type State = Keep;
}
impl<K> Member<Succ<K>> for ScalarGrade {
    type State = Drop;
}
impl Member<Z> for Grade2 {
    type State = Drop;
}
impl Member<Succ<Z>> for Grade2 {
    type State = Drop;
}
impl Member<Succ<Succ<Z>>> for Grade2 {
    type State = Keep;
}
impl<K> Member<Succ<Succ<Succ<K>>>> for Grade2 {
    type State = Drop;
}
// ⚠ Grade2 is NOT ClosedGp: grade2·grade2 → grades 0,2,4 (leaves the stratum).
// So the bivector is an AbelianGroup carrier (it adds and scales),
// but not a subring: closed_gp::<Grade2> does not typecheck. This is physical —
// «multiplying two twists with the geometric product» is not an operation.

pub type EvenStore<A, S> = SectionStore<A, EvenGrades, S>;

// ── BLADE section: a filter WITHIN a grade, not by whole grades ───────────
// The Euclidean subalgebra = the blades that do not touch the DEGENERATE (highest in PGA)
// generator e₀. In the Split tree of a grade stratum, the Wear branch of the highest generator
// is the blades that use it; we cut it to Void. What remains is a subset of the grade:
// grade-2 stores 3 blades (e12,e13,e23), not 6. Closed (eucl·eucl = eucl,
// the e₀ bit = XOR of two zeros = 0), structurally = dim Cl(3,0) = 8.
pub trait EuclGrade<K> {
    type Store<S: Ring>: Stratum<S> + AbelianGroup + ScalarMul<S>;
}
impl EuclGrade<Z> for Nil {
    type Store<S: Ring> = Leaf<S>;
}
impl<K> EuclGrade<Succ<K>> for Nil {
    type Store<S: Ring> = Void;
}
impl<const SQ: i8, Rest> EuclGrade<Z> for Gen<SQ, Rest>
where
    Rest: GradeForm<Z>,
{
    // grade-0 = scalar, no gens involved
    type Store<S: Ring> = <Rest as GradeForm<Z>>::Store<S>;
}
impl<const SQ: i8, K, Rest> EuclGrade<Succ<K>> for Gen<SQ, Rest>
where
    Rest: GradeForm<Succ<K>>,
{
    // keep = blades WITHOUT the highest gen (the full Rest stratum); wear = those touching
    // e₀ → Void. Rest (Cl30) already has no degenerate ones ⇒ not filtered.
    type Store<S: Ring> = Split<<Rest as GradeForm<Succ<K>>>::Store<S>, Void>;
}
pub trait EuclStrata<K> {
    type Store<S: Ring>: GList<S> + AbelianGroup + ScalarMul<S> + GradeMap;
}
impl<A> EuclStrata<Z> for A
where
    A: EuclGrade<Z>,
{
    type Store<S: Ring> = GCons<<A as EuclGrade<Z>>::Store<S>, TNil>;
}
impl<A, K> EuclStrata<Succ<K>> for A
where
    A: EuclGrade<Succ<K>> + EuclStrata<K>,
{
    type Store<S: Ring> =
        GCons<<A as EuclGrade<Succ<K>>>::Store<S>, <A as EuclStrata<K>>::Store<S>>;
}
/// Storage of the Euclidean subalgebra (blades without the degenerate generator).
pub type EuclideanStore<A, S> = <A as EuclStrata<<A as Algebra>::NgenP>>::Store<S>;
