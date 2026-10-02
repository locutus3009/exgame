// SPDX-License-Identifier: MIT

// =====================
// A vector over Scalar whose length is our Peano index.
//
// The storage is a recursive cons list whose length is governed by the Peano index:
//   Z          -> Nil
//   Succ<N>    -> Cons<T, Repr(N)>
// so `Vector<Succ<Succ<Succ<Z>>>, f64>` stores exactly three f64s, and two
// vectors of different length are simply different types (they cannot be mixed up).
//
// Through the AbelianGroup blanket, Cons/Nil themselves become an abelian group
// (componentwise), so Vector<N, T> is a module over the ring T: it can be added,
// subtracted, negated and scaled by a scalar from T.

use crate::prelude::*;
use bytemuck::{Pod, Zeroable};
use core::{
    fmt::Debug,
    ops::{Add, AddAssign, Neg, Sub, SubAssign},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Nil;
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Cons<T, R>(pub T, pub R);

unsafe impl Zeroable for Nil {}
unsafe impl Pod for Nil {}

unsafe impl<T: Zeroable, R: Zeroable> Zeroable for Cons<T, R> {}
// `Pod` promises "no padding". A cons-list built by `Nat::Repr<T>` is always
// homogeneous — the same `T` repeated, closed by the ZST `Nil` — so every
// element shares one alignment and the block is packed tight (`size_of::<T>()`
// is a multiple of `align_of::<T>()` for any Pod `T`; the trailing `Nil` is
// align-1, size-0). Padding could only appear for a hand-built heterogeneous
// `Cons<A, B>` with mismatched alignment, which no `Repr` can produce.
// `assert_packed` below turns any such layout surprise into a compile error.
unsafe impl<T: Pod, R: Pod> Pod for Cons<T, R> {}

/// Compile-time proof that storage `V` is exactly `n` copies of scalar `S`,
/// packed with no padding — the layout `Pod` silently assumes. Drop a
/// `const _: () = assert_packed::<V, S>(n);` next to every concrete storage
/// type headed for a bytemuck cast (GPU upload, `cast_slice`, …): a padded or
/// ZST-contaminated layout then fails to compile instead of corrupting bytes.
/// The `V: Pod, S: Pod` bounds also force the whole `Pod` chain to resolve, so
/// the assert doubles as the check that "the whole tower lines up".
pub const fn assert_packed<V: Pod, S: Pod>(n: usize) {
    assert!(
        core::mem::size_of::<V>() == n * core::mem::size_of::<S>(),
        "storage is not a packed [S; n]: padding or a stray non-ZST field",
    );
    assert!(
        n == 0 || core::mem::align_of::<V>() == core::mem::align_of::<S>(),
        "storage alignment differs from the scalar alignment",
    );
}

// Runtime operations on the carrier Nat::Repr: construction by index, reading,
// writing. A separate NON-const trait: from_fn takes a closure, and calling a closure
// cannot be expressed in a const context. The methods are generic over T — quantifying over the element type
// is free; the bound only needs `N: Storage`. The implementation is structural recursion
// over Z/Succ (it covers every Nat; generic code has no access to this knowledge, so
// the bound is written explicitly).
pub trait Storage: Nat {
    fn from_fn_at<T: Copy + Debug + Send + Sync>(
        f: &mut impl FnMut(usize) -> T,
        base: usize,
    ) -> Self::Repr<T>;
    fn from_fn<T: Copy + Debug + Send + Sync>(mut f: impl FnMut(usize) -> T) -> Self::Repr<T> {
        Self::from_fn_at(&mut f, 0)
    }
    fn get<T: Copy + Debug + Send + Sync>(v: &Self::Repr<T>, i: usize) -> &T;
    fn set<T: Copy + Debug + Send + Sync>(v: &mut Self::Repr<T>, i: usize, x: T);
}
impl Storage for Z {
    fn from_fn_at<T: Copy + Debug + Send + Sync>(_: &mut impl FnMut(usize) -> T, _: usize) -> Nil {
        Nil
    }
    fn get<T: Copy + Debug + Send + Sync>(_: &Nil, _: usize) -> &T {
        panic!("Storage::get: index out of bounds")
    }
    fn set<T: Copy + Debug + Send + Sync>(_: &mut Nil, _: usize, _: T) {
        panic!("Storage::set: index out of bounds")
    }
}
impl<N: Storage> Storage for Succ<N> {
    fn from_fn_at<T: Copy + Debug + Send + Sync>(
        f: &mut impl FnMut(usize) -> T,
        base: usize,
    ) -> Self::Repr<T> {
        Cons(f(base), N::from_fn_at(f, base + 1))
    }
    fn get<T: Copy + Debug + Send + Sync>(v: &Self::Repr<T>, i: usize) -> &T {
        if i == 0 { &v.0 } else { N::get(&v.1, i - 1) }
    }
    fn set<T: Copy + Debug + Send + Sync>(v: &mut Self::Repr<T>, i: usize, x: T) {
        if i == 0 {
            v.0 = x
        } else {
            N::set(&mut v.1, i - 1, x)
        }
    }
}

// Compile-time length of a storage value.
pub trait StorageLen {
    const LEN: usize;
}
impl StorageLen for Nil {
    const LEN: usize = 0;
}
impl<T, R: StorageLen> StorageLen for Cons<T, R> {
    const LEN: usize = R::LEN + 1;
}

// --- Component-wise abelian-group structure on the storage ---

impl Add for Nil {
    type Output = Nil;
    fn add(self, _: Nil) -> Nil {
        Nil
    }
}
impl<T: Add<Output = T>, R: Add<Output = R>> Add for Cons<T, R> {
    type Output = Cons<T, R>;
    fn add(self, rhs: Self) -> Self {
        Cons(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub for Nil {
    type Output = Nil;
    fn sub(self, _: Nil) -> Nil {
        Nil
    }
}
impl<T: Sub<Output = T>, R: Sub<Output = R>> Sub for Cons<T, R> {
    type Output = Cons<T, R>;
    fn sub(self, rhs: Self) -> Self {
        Cons(self.0 - rhs.0, self.1 - rhs.1)
    }
}

impl Neg for Nil {
    type Output = Nil;
    fn neg(self) -> Nil {
        Nil
    }
}
impl<T: Neg<Output = T>, R: Neg<Output = R>> Neg for Cons<T, R> {
    type Output = Cons<T, R>;
    fn neg(self) -> Self {
        Cons(-self.0, -self.1)
    }
}

impl AddAssign for Nil {
    fn add_assign(&mut self, _: Nil) {}
}
impl<T: AddAssign, R: AddAssign> AddAssign for Cons<T, R> {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
        self.1 += rhs.1;
    }
}

impl SubAssign for Nil {
    fn sub_assign(&mut self, _: Nil) {}
}
impl<T: SubAssign, R: SubAssign> SubAssign for Cons<T, R> {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
        self.1 -= rhs.1;
    }
}

impl AbelianGroup for Nil {
    const ZERO: Nil = Nil;
}
impl<H: AbelianGroup, Tl: AbelianGroup> AbelianGroup for Cons<H, Tl> {
    const ZERO: Self = Cons(H::ZERO, Tl::ZERO);
}
