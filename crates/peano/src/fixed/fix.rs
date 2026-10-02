// SPDX-License-Identifier: MIT

use crate::prelude::*;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use fixed::traits::{Fixed, FixedSigned};

trait ConstBits: Sized {
    fn const_from_bits(bits: i128) -> Self;
    fn const_to_bits(self) -> i128;
    fn const_from_int(n: i128) -> Self;
}

macro_rules! impl_constbits {
    ($fx:ident, $b:ty, $le:ident) => {
        impl<Frac> ConstBits for fixed::$fx<Frac>
        where
            Frac: fixed::types::extra::$le,
        {
            fn const_from_bits(bits: i128) -> Self {
                Self::from_bits(bits as $b)
            }
            fn const_to_bits(self) -> i128 {
                self.to_bits() as i128
            }
            fn const_from_int(n: i128) -> Self {
                Self::const_from_int(n as $b)
            }
        }
    };
}
impl_constbits!(FixedI128, i128, LeEqU128);
impl_constbits!(FixedI64, i64, LeEqU64);
impl_constbits!(FixedI32, i32, LeEqU32);
impl_constbits!(FixedI16, i16, LeEqU16);
impl_constbits!(FixedI8, i8, LeEqU8);

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Fix<T: Fixed>(T);

macro_rules! impl_fix_conv {
    ($($ty:ty),* $(,)?) => {$(
        impl<T: Fixed> From<$ty> for Fix<T> {
            fn from(val: $ty) -> Self { Fix(T::from_num(val)) }
        }
        impl<T: Fixed> From<Fix<T>> for $ty {
            fn from(val: Fix<T>) -> Self { val.0.to_num() }
        }
    )*};
}

impl_fix_conv!(f32, f64, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

impl<T: Fixed + ConstBits> EffectiveZero for Fix<T> {
    fn is_effective_zero(self) -> bool {
        self.0.const_to_bits() == 0
    }
}
impl<T: Fixed + ConstBits> StandardPart for Fix<T> {
    type Real = Self;
    fn standard_part(self) -> Self {
        self
    }
}
impl<T: Fixed> FromRational for Fix<T>
where
    T: Fixed + ConstBits,
{
    fn from_u32(n: u32) -> Self {
        Fix(T::const_from_int(n as i128))
    }
    fn from_rational(num: u32, den: u32) -> Self {
        let f = T::FRAC_NBITS;
        // with f ≤ 96 and num < 2^32 the shift is guaranteed to fit in i128
        let scaled = (num as i128) << f; // safe if f + 32 ≤ 127
        Fix(T::const_from_bits(scaled / (den as i128)))
    }
}
impl<T: Fixed> Mul for Fix<T> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}
impl<T: Fixed> MulMonoid for Fix<T> {
    const ONE: Self = Self(T::TRY_ONE.unwrap());
}
impl<T: Fixed> Add for Fix<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}
impl<T: Fixed> AddAssign for Fix<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self(self.0 + rhs.0);
    }
}
impl<T: Fixed> Sub for Fix<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}
impl<T: Fixed> Div for Fix<T> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}
impl<T: Fixed> SubAssign for Fix<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self(self.0 - rhs.0);
    }
}
impl<T: Fixed + FixedSigned> Neg for Fix<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}
// The algebraic tower starts with `AbelianGroup`, which requires the carrier to be
// Send + Sync (see peano::abelian). For `Fix<T>` these are derived from `T`, so
// the bound is simply requested of `T` — there is no need to assert it via `unsafe impl`.
// All concrete `fixed::FixedI*` satisfy it, so the restriction is not visible
// at the point of use.
impl<T: Fixed + FixedSigned + Send + Sync> AbelianGroup for Fix<T> {
    const ZERO: Self = Fix(T::ZERO);
}
impl<T: Fixed + FixedSigned + Send + Sync> Ring for Fix<T> {}
impl<T: Fixed + FixedSigned + Send + Sync> Commutative for Fix<T> {}
impl<T: Fixed + FixedSigned + Send + Sync> Invertible for Fix<T> {
    fn try_recip(self) -> Option<Self> {
        Some(Fix(self.0.checked_recip()?))
    }
}
impl<T: Fixed + FixedSigned + ConstBits + Send + Sync> Scalar for Fix<T> {
    fn powf_explicit(self, _n: Self) -> Self {
        panic!("This function must not be used");
    }
    fn powi_explicit(self, n: i32) -> Self {
        let (mut base, mut e) = if n < 0 {
            (
                self.try_recip()
                    .expect("powi_explicit: recip of non-invertible base"),
                n.unsigned_abs(), // correct for i32::MIN, unlike -n
            )
        } else {
            (self, n as u32)
        };
        let mut acc = <Self as MulMonoid>::ONE; // 0^0 = 1 by convention
        while e > 0 {
            if e & 1 == 1 {
                acc = acc * base;
            }
            e >>= 1;
            if e > 0 {
                base = base * base;
            }
        }
        acc
    }
    fn sqrt_explicit(self) -> Self {
        Self(self.0.sqrt())
    }
    fn sin_explicit(self) -> Self {
        panic!("This function must not be used");
    }
    fn cos_explicit(self) -> Self {
        panic!("This function must not be used");
    }
    fn exp_explicit(self) -> Self {
        panic!("This function must not be used");
    }
    fn ln_explicit(self) -> Self {
        panic!("This function must not be used");
    }
    fn atan2_explicit(self, _x: Self) -> Self {
        panic!("This function must not be used");
    }
}
