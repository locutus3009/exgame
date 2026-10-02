// SPDX-License-Identifier: MIT

use peano::prelude::*;

/// Canonical injection R ↪ S = section of StandardPart (value in place, zero
/// fibers), recursive over the tower. One impl per ring; never per-container.
/// NOT a third morphism — the ring-level half of `deepen`.
pub trait Lift<S>: Sized {
    fn lift(self) -> S;
}

// The identity lives at the BOTTOM of the tower (EffectiveZero = residue field): no
// tie to Float — the bottom may be a symbolic scalar (Sym) or a fixed-point one.
impl<T> Lift<T> for T
where
    T: EffectiveZero,
{
    #[inline]
    fn lift(self) -> T {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::markup::axis_vec;
    use crate::pga3::{Motor, Twist};
    use crate::{Mv, Pga3};
    use core::{
        fmt::Debug,
        ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
    };
    use num_rational::Rational32;
    use num_traits::{One, ToPrimitive, Zero};
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{LazyLock, RwLock};

    #[derive(PartialEq, Debug, Copy, Clone)]
    enum Handle {
        Constant(Rational32),
        Runtime(usize),
    }

    #[derive(Debug, Copy, Clone)]
    struct MyScalar(f64, Handle);

    impl FromRational for MyScalar {
        fn from_u32(n: u32) -> Self {
            Self(n as f64, Handle::Constant(Rational32::new_raw(n as i32, 1)))
        }
        fn from_rational(num: u32, den: u32) -> Self {
            Self(
                (num as f64) / (den as f64),
                Handle::Constant(Rational32::new_raw(num as i32, den as i32)),
            )
        }
    }

    static COUNTER: AtomicUsize = AtomicUsize::new(1);
    static CONSTANTS: LazyLock<RwLock<HashMap<Rational32, usize>>> =
        LazyLock::new(|| RwLock::new(HashMap::new()));
    static COMPARISONS: AtomicUsize = AtomicUsize::new(0);

    fn trace_constant(val: &MyScalar) {
        let mut guard = CONSTANTS.write().unwrap();
        if let Handle::Constant(v) = val.1 {
            guard
                .entry(v)
                .and_modify(|counter| *counter += 1)
                .or_insert(1);
        }
    }

    impl MyScalar {
        fn new_runtime(value: f64) -> Self {
            let handle = Handle::Runtime(COUNTER.fetch_add(1, Ordering::SeqCst));
            Self(value, handle)
        }
    }

    impl Add for MyScalar {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            match (self, rhs) {
                // x + 0
                (MyScalar(_, Handle::Constant(c)), other) if c.is_zero() => other,
                (other, MyScalar(_, Handle::Constant(c))) if c.is_zero() => other,

                // constant + constant
                (MyScalar(_, Handle::Constant(left)), MyScalar(_, Handle::Constant(right))) => {
                    let a = left.reduced();
                    let b = right.reduced();
                    let c = a + b;
                    MyScalar(c.to_f64().unwrap(), Handle::Constant(c))
                }

                // x + y
                (a, b) => {
                    trace_constant(&a);
                    trace_constant(&b);

                    Self::new_runtime(a.0 + b.0)
                }
            }
        }
    }

    impl Mul for MyScalar {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            match (self, rhs) {
                // x * 0
                (MyScalar(_, Handle::Constant(c)), _) if c.is_zero() => {
                    MyScalar(0.0, Handle::Constant(Rational32::ZERO))
                }
                (_, MyScalar(_, Handle::Constant(c))) if c.is_zero() => {
                    MyScalar(0.0, Handle::Constant(Rational32::ZERO))
                }

                // x * 1
                (MyScalar(_, Handle::Constant(c)), other) if c.is_one() => other,
                (other, MyScalar(_, Handle::Constant(c))) if c.is_one() => other,

                // constant * constant
                (MyScalar(_, Handle::Constant(left)), MyScalar(_, Handle::Constant(right))) => {
                    let a = left.reduced();
                    let b = right.reduced();
                    let c = a * b;
                    MyScalar(c.to_f64().unwrap(), Handle::Constant(c))
                }

                // x * y
                (a, b) => {
                    trace_constant(&a);
                    trace_constant(&b);

                    Self::new_runtime(a.0 * b.0)
                }
            }
        }
    }

    impl Sub for MyScalar {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            match (self, rhs) {
                // x - 0
                (other, MyScalar(_, Handle::Constant(c))) if c.is_zero() => other,

                // constant - constant
                (MyScalar(_, Handle::Constant(left)), MyScalar(_, Handle::Constant(right))) => {
                    let a = left.reduced();
                    let b = right.reduced();
                    let c = a - b;
                    MyScalar(c.to_f64().unwrap(), Handle::Constant(c))
                }

                // x - y
                (a, b) => {
                    trace_constant(&a);
                    trace_constant(&b);

                    Self::new_runtime(a.0 - b.0)
                }
            }
        }
    }

    impl Div for MyScalar {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            match (self, rhs) {
                (_, MyScalar(_, Handle::Constant(c))) if c.is_zero() => {
                    panic!()
                }

                // 0 / x
                (MyScalar(_, Handle::Constant(c)), _) if c.is_zero() => {
                    MyScalar(0.0, Handle::Constant(c))
                }

                // x / 1
                (other, MyScalar(_, Handle::Constant(c))) if c.is_one() => other,

                // constant / constant
                (MyScalar(_, Handle::Constant(left)), MyScalar(_, Handle::Constant(right))) => {
                    let a = left.reduced();
                    let b = right.reduced();
                    let c = a / b;
                    MyScalar(c.to_f64().unwrap(), Handle::Constant(c))
                }

                // x / y
                (a, b) => {
                    trace_constant(&a);
                    trace_constant(&b);

                    Self::new_runtime(a.0 / b.0)
                }
            }
        }
    }

    impl Neg for MyScalar {
        type Output = Self;
        fn neg(self) -> Self {
            match self {
                MyScalar(_, Handle::Constant(c)) => {
                    let res = -c;
                    MyScalar(res.to_f64().unwrap(), Handle::Constant(res))
                }
                _ => Self::new_runtime(-self.0),
            }
        }
    }

    impl AddAssign for MyScalar {
        fn add_assign(&mut self, rhs: Self) {
            if let Handle::Constant(c) = rhs.1
                && c.is_zero()
            {
                return;
            }

            *self = {
                trace_constant(self);
                trace_constant(&rhs);

                Self::new_runtime(self.0 + rhs.0)
            };
        }
    }

    impl SubAssign for MyScalar {
        fn sub_assign(&mut self, rhs: Self) {
            if let Handle::Constant(c) = rhs.1
                && c.is_zero()
            {
                return;
            }

            trace_constant(self);
            trace_constant(&rhs);

            *self = Self::new_runtime(self.0 - rhs.0);
        }
    }

    impl AbelianGroup for MyScalar {
        const ZERO: Self = Self(0.0, Handle::Constant(Rational32::new_raw(0, 1)));
    }

    impl MulMonoid for MyScalar {
        const ONE: Self = Self(1.0, Handle::Constant(Rational32::new_raw(1, 1)));
    }

    impl EffectiveZero for MyScalar {
        fn is_effective_zero(self) -> bool {
            COMPARISONS.fetch_add(1, Ordering::SeqCst);
            self.0.abs() < 1e-9
        }
    }

    impl Ring for MyScalar {}
    impl Commutative for MyScalar {}
    impl Invertible for MyScalar {
        fn try_recip(self) -> Option<Self> {
            match self {
                MyScalar(_, Handle::Runtime(_)) => {
                    if !self.is_effective_zero() {
                        Some(Self::new_runtime(1.0 / self.0))
                    } else {
                        None
                    }
                }
                MyScalar(_, Handle::Constant(c)) if c.is_zero() => None,
                MyScalar(_, Handle::Constant(c)) => {
                    let res = Rational32::ONE / c;
                    Some(MyScalar(res.to_f64().unwrap(), Handle::Constant(res)))
                }
            }
        }
    }

    impl Scalar for MyScalar {
        #[inline(always)]
        fn powf_explicit(self, n: Self) -> Self {
            trace_constant(&self);
            trace_constant(&n);

            Self::new_runtime(self.0.powf(n.0))
        }

        #[inline(always)]
        fn powi_explicit(self, n: i32) -> Self {
            match self {
                MyScalar(_, Handle::Constant(c)) => {
                    let res = c.pow(n);
                    MyScalar(c.to_f64().unwrap(), Handle::Constant(res))
                }
                _ => {
                    let mut guard = CONSTANTS.write().unwrap();
                    let e = Rational32::new(n, 1);
                    guard
                        .entry(e)
                        .and_modify(|counter| *counter += 1)
                        .or_insert(1);

                    Self::new_runtime(self.0.powi(n))
                }
            }
        }

        #[inline(always)]
        fn sqrt_explicit(self) -> Self {
            trace_constant(&self);

            Self::new_runtime(self.0.sqrt())
        }

        #[inline(always)]
        fn sin_explicit(self) -> Self {
            match self {
                MyScalar(_, Handle::Constant(c)) if c.is_zero() => MyScalar::from_u32(0),
                _ => {
                    trace_constant(&self);

                    Self::new_runtime(self.0.sin())
                }
            }
        }

        #[inline(always)]
        fn cos_explicit(self) -> Self {
            trace_constant(&self);

            Self::new_runtime(self.0.cos())
        }

        #[inline(always)]
        fn exp_explicit(self) -> Self {
            trace_constant(&self);

            Self::new_runtime(self.0.exp())
        }

        #[inline(always)]
        fn ln_explicit(self) -> Self {
            trace_constant(&self);

            Self::new_runtime(self.0.ln())
        }

        #[inline(always)]
        fn atan2_explicit(self, x: Self) -> Self {
            trace_constant(&self);
            trace_constant(&x);

            Self::new_runtime(self.0.atan2(x.0))
        }

        // Study-forms (sinc_sq / cos_sq / dsinc_sq / half_angle_sq) are NOT
        // shader builtins, so they are deliberately left un-delegated: the
        // default trait impls expand them into sin/cos/sqrt/atan2 (which ARE
        // shader builtins and stay recorded as single ops) plus arithmetic and
        // a representation-select comparison. The trace therefore carries the
        // full expansion a shader would emit, not one opaque node.
    }

    impl StandardPart for MyScalar {
        type Real = Self;
        #[inline]
        fn standard_part(self) -> Self {
            self
        }
    }

    #[test]
    fn rational_compiles() {
        let a: MyScalar = MyScalar::from_u32(1);
        assert_eq!(a.0, 1.0);
        assert_eq!(a.1, Handle::Constant(Rational32::ONE));

        let b: MyScalar = MyScalar::from_rational(1, 2);
        assert_eq!(b.0, 0.5);
        assert_eq!(b.1, Handle::Constant(Rational32::new(1, 2)));

        let c = a + b;
        assert_eq!(c.0, 1.5);
        assert_eq!(c.1, Handle::Constant(Rational32::new(3, 2)));
        let counter = COUNTER.load(Ordering::SeqCst);
        assert_eq!(counter, 1);

        {
            let guard = CONSTANTS.read().unwrap();
            assert_eq!(guard.len(), 0, "{guard:?}");
        }

        {
            let g = Twist::new(
                &Vector3::from([
                    MyScalar::new_runtime(0.0),
                    MyScalar::new_runtime(0.0),
                    MyScalar::new_runtime(0.0),
                ]),
                &Vector3::from([
                    MyScalar::new_runtime(0.3),
                    MyScalar::new_runtime(-0.2),
                    MyScalar::new_runtime(0.5),
                ]),
            );
            let m = Motor::exp(&g);
            let mm = m.as_mv().norm_squared();
            assert!((mm.0 - 1.0).abs() < 1e-9, "⟨M M̃⟩₀ = {mm:?}");
            // Total recorded ops of the whole exp+norm trace. 224 → 128 at
            // cutover A7: the strata-based even carrier (8 components, closed-form
            // Study exponential on the narrow section, twist→motor embedding by direct
            // copy of the grade-2 slots) cuts the work almost in half compared with the
            // flat 16-slot Mv.
            assert_eq!(mm.1, Handle::Runtime(128));

            let guard = CONSTANTS.read().unwrap();
            // Constants traced by the un-delegated Study-form expansions:
            //   1/2          — the (p/2) factor in exp's bivector part,
            //   2/1  (TWO)   — the 2·l³ denominator of dsinc_sq's closed form,
            //   1/100000     — DSINC_SCALE in dsinc_sq's branch test.
            assert_eq!(guard.len(), 3, "{guard:?}");
            assert_eq!(*guard.get(&Rational32::new(1, 2)).unwrap(), 1);
            assert_eq!(*guard.get(&Rational32::new(2, 1)).unwrap(), 1);
            assert_eq!(*guard.get(&Rational32::new(1, 100_000)).unwrap(), 1);
            assert_eq!(COMPARISONS.load(Ordering::SeqCst), 3);
        }

        {
            // ωz = π/2 integrated over dt=1 → 90° about +Z (the ½ lives in Twist::exp).
            // The e1 plane maps to the e2 plane.
            let g = Twist::new(
                &Vector3::from([
                    MyScalar::new_runtime(0.0),
                    MyScalar::new_runtime(0.0),
                    MyScalar::new_runtime(0.0),
                ]),
                &Vector3::from([
                    MyScalar::new_runtime(0.0),
                    MyScalar::new_runtime(0.0),
                    MyScalar::new_runtime(core::f64::consts::FRAC_PI_2),
                ]),
            );
            let m = g.exp(MyScalar::new_runtime(1.0));
            let mut e1 = Mv::<Pga3, MyScalar>::zero();
            e1.set(axis_vec::<Pga3>(0), MyScalar::new_runtime(1.0));
            let out = m.conjugate_mv(&e1);
            assert!(
                (out.get(axis_vec::<Pga3>(1)).0.abs() - 1.0).abs() < 1e-9,
                "e1→e2 expected, got e1={:?} e2={:?}",
                out.get(axis_vec::<Pga3>(0)),
                out.get(axis_vec::<Pga3>(1))
            );
            // 491 → 497 in A2 (negations of the derived markup); 497 → 368 at
            // cutover A7: exp on the even carrier + sandwich through the full Mv,
            // but without the dead arithmetic of the wide exp path.
            assert_eq!(out.get(axis_vec::<Pga3>(1)).1, Handle::Runtime(368));

            let guard = CONSTANTS.read().unwrap();
            // Cumulative across both blocks (CONSTANTS/COMPARISONS are never reset).
            assert_eq!(guard.len(), 3, "{guard:?}");
            assert_eq!(*guard.get(&Rational32::new(1, 2)).unwrap(), 3);
            assert_eq!(*guard.get(&Rational32::new(2, 1)).unwrap(), 2);
            assert_eq!(*guard.get(&Rational32::new(1, 100_000)).unwrap(), 2);
            assert_eq!(COMPARISONS.load(Ordering::SeqCst), 6);
        }
    }
}
