// SPDX-License-Identifier: MIT

use crate::prelude::*;
use core::ops::{Index, IndexMut, Mul};

/// A vector of exactly `N` elements of type `T` (`N` a Peano index).
pub type Vector<N, T> = <N as Nat>::Repr<T>;
pub type Vector2<T> = Vector<N2, T>;
pub type Vector3<T> = Vector<N3, T>;
pub type Vector6<T> = Vector<N6, T>;

// Layout guard for the widths that reach a bytemuck cast (GPU upload). Each is
// a padding-free block of scalars; if the cons-list ever grew padding or the
// `Pod` chain broke, these `const` evaluations would fail to compile.
const _: () = assert_packed::<Vector3<f32>, f32>(3);
const _: () = assert_packed::<Vector6<f32>, f32>(6);
const _: () = assert_packed::<Vector3<f64>, f64>(3);

// --- Scalar multiplication by a ring element (uses the Ring's `Mul`) ---

pub trait ScalarMul<S> {
    fn scale(self, s: S) -> Self;
}
impl<S: Copy> ScalarMul<S> for Nil {
    fn scale(self, _: S) -> Nil {
        Nil
    }
}
impl<S: Copy, T: Mul<S, Output = T>, R: ScalarMul<S>> ScalarMul<S> for Cons<T, R> {
    fn scale(self, s: S) -> Self {
        Cons(self.0 * s, self.1.scale(s))
    }
}

// Runtime access by index (Storage::get/set) and construction by index
// (Storage::from_fn) live on Nat itself — see peano::storage. What remains here
// are only the operations tied to the SHAPE of the carrier (algebra, typed access).

// --- Element access by Peano index (bounds checked at compile time) ---
//
// The recursion over the cons chain is hidden in a private module: only
// `get`/`get_mut` through the `Access` trait stick out. The `At` resolver itself is not re-exported,
// so `.at()`/`.at_mut()` cannot be called from outside.
mod index_impl {
    use super::*;

    // The index is a Peano type too: `At<Z>` is the head, `At<Succ<I>>` recurses into the
    // tail. `Nil` has no implementation, so going out of bounds is a type error.
    pub trait At<I> {
        type Elem;
        fn at(&self) -> &Self::Elem;
        fn at_mut(&mut self) -> &mut Self::Elem;
    }
    impl<T, R> At<Z> for Cons<T, R> {
        type Elem = T;
        fn at(&self) -> &T {
            &self.0
        }
        fn at_mut(&mut self) -> &mut T {
            &mut self.0
        }
    }
    impl<I, T, R: At<I>> At<Succ<I>> for Cons<T, R> {
        type Elem = R::Elem;
        fn at(&self) -> &R::Elem {
            self.1.at()
        }
        fn at_mut(&mut self) -> &mut R::Elem {
            self.1.at_mut()
        }
    }
}

/// Ergonomic `get::<I>()` / `get_mut::<I>()`. The index is a method parameter,
/// so it is given via turbofish and both yield a reference to the cell directly:
///
/// ```ignore
/// let x = *v.get::<Succ<Z>>();   // &Elem      -> read
/// *v.get_mut::<Z>() = 9.0;       // &mut Elem  -> write/mutate
/// ```
pub trait Access {
    fn get<I>(&self) -> &<Self as index_impl::At<I>>::Elem
    where
        Self: index_impl::At<I>,
    {
        <Self as index_impl::At<I>>::at(self)
    }
    fn get_mut<I>(&mut self) -> &mut <Self as index_impl::At<I>>::Elem
    where
        Self: index_impl::At<I>,
    {
        <Self as index_impl::At<I>>::at_mut(self)
    }
}
impl<T, R> Access for Cons<T, R> {}

// --- Cross product (only for 3-dimensional vectors) ---
//
// Defined exactly on `Vector<Succ<Succ<Succ<Z>>>, T>`; for any other length
// the method simply does not exist — this is also a type error, not a runtime check.
pub trait Cross {
    fn cross(self, rhs: Self) -> Self;
}
impl<T: Ring> Cross for Cons<T, Cons<T, Cons<T, Nil>>> {
    fn cross(self, rhs: Self) -> Self {
        let (a1, a2, a3) = (self.0, self.1.0, self.1.1.0);
        let (b1, b2, b3) = (rhs.0, rhs.1.0, rhs.1.1.0);
        Cons(
            a2 * b3 - a3 * b2,
            Cons(a3 * b1 - a1 * b3, Cons(a1 * b2 - a2 * b1, Nil)),
        )
    }
}

// --- Dot product: pairwise multiplication + sum over all components ---

pub trait Dot<T> {
    type Output;
    fn dot(self, rhs: Self) -> Self::Output;
}

impl<T: AbelianGroup> Dot<T> for Nil {
    type Output = T;
    fn dot(self, _: Nil) -> T {
        T::ZERO
    }
}

impl<T, R> Dot<T> for Cons<T, R>
where
    T: Mul<Output = T> + AbelianGroup,
    R: Dot<T, Output = T>,
{
    type Output = T;
    fn dot(self, rhs: Self) -> T {
        self.0 * rhs.0 + self.1.dot(rhs.1)
    }
}

// --- From/Into for arrays: only the lengths that are USED ---
//
// Conversion to/from [T; N] is defined by a macro; the identifiers serve as a length counter.
// A newly needed length is one `impl_vector_array!` line at the bottom.
macro_rules! cons_ty {
    ($t:ty $(,)?) => { Nil };
    ($t:ty, $head:ident $(, $rest:ident)*) => { Cons<$t, cons_ty!($t $(, $rest)*)> };
}
macro_rules! cons_pat {
    () => { Nil };
    ($head:ident $(, $rest:ident)*) => { Cons($head, cons_pat!($($rest),*)) };
}
macro_rules! impl_vector_array {
    ($n:literal: $($x:ident),+) => {
        impl<T: Copy> From<[T; $n]> for cons_ty!(T, $($x),+) {
            fn from(a: [T; $n]) -> Self {
                let [$($x),+] = a;
                crate::vector![$($x),+]
            }
        }
        impl<T: Copy> From<cons_ty!(T, $($x),+)> for [T; $n] {
            fn from(a:cons_ty!(T, $($x),+)) -> [T; $n] {
                match a {
                    cons_pat!($($x),+) => [$($x),+],
                }
            }
        }
    };
}
impl_vector_array!(1: a0);
impl_vector_array!(2: a0, a1);
impl_vector_array!(3: a0, a1, a2);
impl_vector_array!(6: a0, a1, a2, a3, a4, a5);
impl_vector_array!(12: a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
impl_vector_array!(24: a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
                       a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23);

// --- Access by usize index: recursion over the cons chain, any length ---
//
// The base case is a vector of length 1, the step is a homogeneous Cons<T, Cons<T, _>>. Going out of
// bounds panics (as in Seq); the static check is left to get::<I>().
impl<T> Index<usize> for Cons<T, Nil> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        match i {
            0 => &self.0,
            _ => panic!("Index: index out of bounds"),
        }
    }
}
impl<T, R> Index<usize> for Cons<T, Cons<T, R>>
where
    Cons<T, R>: Index<usize, Output = T>,
{
    type Output = T;
    fn index(&self, i: usize) -> &T {
        if i == 0 { &self.0 } else { &self.1[i - 1] }
    }
}

impl<T> IndexMut<usize> for Cons<T, Nil> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        match i {
            0 => &mut self.0,
            _ => panic!("IndexMut: index out of bounds"),
        }
    }
}
impl<T, R> IndexMut<usize> for Cons<T, Cons<T, R>>
where
    Cons<T, R>: IndexMut<usize, Output = T>,
{
    fn index_mut(&mut self, i: usize) -> &mut T {
        if i == 0 {
            &mut self.0
        } else {
            &mut self.1[i - 1]
        }
    }
}

/// `vector![a, b, c]` builds a `Vector<Succ<Succ<Succ<Z>>>, _>`.
#[macro_export]
macro_rules! vector {
    () => { Nil };
    ($x:expr $(, $rest:expr)* $(,)?) => { Cons($x, vector![$($rest),*]) };
}

pub trait SplitVector<const N: usize, T> {
    fn split(self) -> [T; N];
}

impl<T: Copy> SplitVector<2, T> for Cons<T, Cons<T, Nil>> {
    fn split(self) -> [T; 2] {
        [self[0], self[1]]
    }
}
impl<T: Copy> SplitVector<3, T> for Cons<T, Cons<T, Cons<T, Nil>>> {
    fn split(self) -> [T; 3] {
        [self[0], self[1], self[2]]
    }
}
impl<T: Copy> SplitVector<6, T> for Cons<T, Cons<T, Cons<T, Cons<T, Cons<T, Cons<T, Nil>>>>>> {
    fn split(self) -> [T; 6] {
        [self[0], self[1], self[2], self[3], self[4], self[5]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A generic function requiring an abelian group: proves that
    // Vector<N3, f64> really does climb the tower up to AbelianGroup.
    fn sum_group<G: AbelianGroup>(a: G, b: G) -> G {
        a + b
    }

    #[test]
    fn peano_vector() {
        // The vector type is given by its Peano length; it stores exactly 3 f64s.
        let v: Vector<N3, f64> = vector![1.0, 2.0, 3.0];
        let w: Vector<N3, f64> = vector![10.0, 20.0, 30.0];

        // The length is known at compile time from the shape of the cons chain.
        assert_eq!(<Vector<N3, f64> as StorageLen>::LEN, 3);

        // Componentwise addition — via the AbelianGroup blanket.
        let s = sum_group(v, w);
        assert_eq!(s, vector![11.0, 22.0, 33.0]);

        // Subtraction and negation.
        assert_eq!(w - v, vector![9.0, 18.0, 27.0]);
        assert_eq!(-v, vector![-1.0, -2.0, -3.0]);

        // The group's neutral element comes from the Scalar level (f64: ConstZero).
        let z = <Vector<N3, f64> as AbelianGroup>::ZERO;
        assert_eq!(z, vector![0.0, 0.0, 0.0]);
        assert_eq!(sum_group(v, z), v);

        // Scaling by a scalar from the ring (uses the element's Mul).
        assert_eq!(v.scale(2.0), vector![2.0, 4.0, 6.0]);

        // f64 is a full Scalar: transcendental operations are available.
        assert_eq!(4.0_f64.sqrt_explicit(), 2.0);
        assert_eq!(<f64 as MulMonoid>::ONE, 1.0);
        assert_eq!(2.0_f64.try_recip(), Some(0.5));
        assert_eq!(0.0_f64.try_recip(), None);

        // Different lengths are different types: the next line will not compile,
        //   let _ = v + vector![1.0, 2.0];   // Cons<_,Cons<_,Cons<_,Nil>>> != Cons<_,Cons<_,Nil>>
    }

    #[test]
    fn indexing() {
        let mut v: Vector<N3, f64> = vector![10.0, 20.0, 30.0];

        // Reading by a type index: get yields &Elem.
        assert_eq!(*v.get::<N0>(), 10.0);
        assert_eq!(*v.get::<N1>(), 20.0);
        assert_eq!(*v.get::<N2>(), 30.0);

        // Writing through get_mut — directly a &mut to the cell.
        *v.get_mut::<N1>() = 99.0;
        assert_eq!(v, vector![10.0, 99.0, 30.0]);

        // In-place mutation through the same reference.
        *v.get_mut::<N2>() += 1.0;
        assert_eq!(*v.get::<N2>(), 31.0);

        // Going out of bounds is a compile error (Nil has no At), for example:
        //   v.get::<Succ<Succ<Succ<Z>>>>();   // index 3 in a vector of length 3
    }

    #[test]
    fn cross_product() {
        let x: Vector<N3, f64> = vector![1.0, 0.0, 0.0];
        let y: Vector<N3, f64> = vector![0.0, 1.0, 0.0];
        let z: Vector<N3, f64> = vector![0.0, 0.0, 1.0];

        // Right-handed triple: x × y = z, y × z = x, z × x = y.
        assert_eq!(x.cross(y), z);
        assert_eq!(y.cross(z), x);
        assert_eq!(z.cross(x), y);

        // Antisymmetry and collinearity.
        assert_eq!(y.cross(x), -z);
        assert_eq!(x.cross(x), <Vector<N3, f64> as AbelianGroup>::ZERO);

        let a: Vector<N3, f64> = vector![1.0, 2.0, 3.0];
        let b: Vector<N3, f64> = vector![4.0, 5.0, 6.0];
        assert_eq!(a.cross(b), vector![-3.0, 6.0, -3.0]);

        // cross is defined only for length 3: Vector<N2,_> has no such method.
    }
}
