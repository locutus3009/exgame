// SPDX-License-Identifier: MIT

// ── Peano index: lives only in type signatures, has no values ───────────────
// No value-level sugar (`+`, VALUE, value(), new()) — the index is needed
// exclusively as a type parameter (vector length, element number).
use crate::storage::{Cons, Nil};
use core::fmt::Debug;

#[derive(Clone, Copy)]
pub struct Z;
#[derive(Clone, Copy)]
pub struct Succ<N>(pub(crate) N);

pub type N0 = Z;
pub type N1 = Succ<Z>;
pub type N2 = Succ<Succ<Z>>;
pub type N3 = Succ<Succ<Succ<Z>>>;
pub type N4 = Succ<Succ<Succ<Succ<Z>>>>;
pub type N5 = Succ<Succ<Succ<Succ<Succ<Z>>>>>;
pub type N6 = Succ<Succ<Succ<Succ<Succ<Succ<Z>>>>>>;
pub type N10 = Succ<Succ<Succ<Succ<Succ<N5>>>>>;
pub type N12 = Succ<Succ<N10>>;
pub type N24 = Mul<N12, N2>;

pub trait Nat: Clone + Copy + Send + Sync {
    type Root: Nat;
    /// A carrier of exactly N elements of type T — a cons chain (Z → Nil, Succ → Cons).
    /// The GAT hides the quantifier over T inside the trait: a single bound `N: Nat` names
    /// a container for ANY element, without per-type `Storage<T>` bounds
    /// (Rust has no `for<T>` quantifier).
    ///
    /// Send/Sync must be stated explicitly here: for `Cons`/`Nil` they are derived automatically,
    /// but the compiler does not see auto traits through the projection `N::Repr<T>` — in
    /// generic code the carrier would be non-Send even with a Send element.
    type Repr<T: Copy + Debug + Send + Sync>: Copy + Debug + Send + Sync;
    const LEN: usize;
    fn value() -> usize;
}

impl Nat for Z {
    type Root = Z;
    type Repr<T: Copy + Debug + Send + Sync> = Nil;
    const LEN: usize = 0;
    fn value() -> usize {
        0
    }
}

impl<T: Nat> Nat for Succ<T> {
    type Root = T;
    type Repr<E: Copy + Debug + Send + Sync> = Cons<E, T::Repr<E>>;
    const LEN: usize = T::LEN + 1;
    fn value() -> usize {
        1 + Self::Root::value()
    }
}

pub trait PeanoAdd<Rhs> {
    type Sum;
}
// Z + N = N
impl<N> PeanoAdd<N> for Z {
    type Sum = N;
}
// Succ<M> + N = Succ<M + N>
impl<M: PeanoAdd<N>, N> PeanoAdd<N> for Succ<M> {
    type Sum = Succ<M::Sum>;
}

pub type Add<A, B> = <A as PeanoAdd<B>>::Sum;

pub trait PeanoMul<Rhs> {
    type Prod;
}
// Z * N = Z
impl<N> PeanoMul<N> for Z {
    type Prod = Z;
}
// Succ<M> * N = N + (M * N)
impl<M, N> PeanoMul<N> for Succ<M>
where
    M: PeanoMul<N>,       // M * N
    N: PeanoAdd<M::Prod>, // N + (M*N)
{
    type Prod = <N as PeanoAdd<M::Prod>>::Sum;
}

pub type Mul<A, B> = <A as PeanoMul<B>>::Prod;

#[cfg(test)]
mod tests {
    use super::*;
    trait Same<T> {}
    impl<T> Same<T> for T {}
    fn same<A: Same<B>, B>() {}

    #[test]
    fn arith() {
        same::<Add<N2, N3>, N5>(); // will not compile if ≠
        same::<Mul<N2, N3>, N6>();
        same::<Mul<N4, N3>, N12>();
    }
}
