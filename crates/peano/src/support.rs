// SPDX-License-Identifier: MIT

use num_traits::Float;

/// A narrow constant-source trait — ONLY it is required to be const
pub trait FromRational {
    fn from_u32(n: u32) -> Self;
    fn from_rational(num: u32, den: u32) -> Self;
}

// ── REAL TRACE ──────────────────────────────────────────────────────────────
/// The "effective zero" tolerance at the bottom of the tower. On f64 the threshold is 1e-9, on f32 it is 1e-5
/// (tuned on the dsinc_sq seam in Motor — see the pga3::motor tests), because
/// after-cancellation comparisons with zero are not exact.
///
/// The tolerance has two roles sharing one constant: (1) choosing the representation at the
/// Taylor/closed-form seam (sinc/cos/dsinc/half_angle in Motor) — a numerical
/// accuracy threshold; (2) protection against degeneracy on dimensional quantities (the norm in
/// Mv::normalize, recip of zero, the distance in gravity, the pivot/residual in implicit
/// Newton) — a domain threshold. They coincide because the sim is normalized to O(1); as
/// a consequence, on f64 the threshold sets the absolute floor of Newton convergence (see the
/// free-body test). Category (2) lives only on f64 (the sim); on f32 today
/// only category (1) is instantiated.
pub trait EffectiveZero: Copy {
    fn is_effective_zero(self) -> bool;
}

/// Standard part: the recursive projection of the ring onto the residue field (the bottom of the tower).
/// This is the iterated augmentation π: R ⊕ m → R. Used ONLY for:
///   1. representation-select in the Motor Study helpers (both branches are differentiable),
///   2. the degeneracy guard in normalize (zero-check off the AD path).
///
/// NOT for projecting an axis in ad (that uses base()/component(), one level).
pub trait StandardPart: Copy {
    type Real: EffectiveZero;
    fn standard_part(self) -> Self::Real;
}

impl EffectiveZero for f64 {
    fn is_effective_zero(self) -> bool {
        self.abs() < 1e-9
    }
}

impl EffectiveZero for f32 {
    fn is_effective_zero(self) -> bool {
        self.abs() < 1e-5
    }
}

impl FromRational for f64 {
    fn from_u32(n: u32) -> Self {
        n as Self
    }

    fn from_rational(num: u32, den: u32) -> Self {
        let a = num as Self;
        let b = den as Self;
        a / b
    }
}

impl FromRational for f32 {
    fn from_u32(n: u32) -> Self {
        n as Self
    }

    fn from_rational(num: u32, den: u32) -> Self {
        let a = num as Self;
        let b = den as Self;
        a / b
    }
}

impl<T> StandardPart for T
where
    T: Float + EffectiveZero,
{
    type Real = T;
    #[inline]
    fn standard_part(self) -> T {
        self
    }
}

#[allow(non_camel_case_types)]
pub enum RationalConstant {
    TWO,
    HALF,
    SINC_C1,
    SINC_C2,
    SINC_C3,
    COS_C1,
    COS_C2,
    COS_C3,
    DSINC_SCALE,
    DSINC_C0,
    DSINC_C1,
    DSINC_C2,
    DSINC_C3,
    HALF_C2,
    HALF_C3,
}

// TODO: Replace with constants when stable Rust support will land.
pub trait FromRationalConstant: FromRational + Sized {
    fn get_constant(t: RationalConstant) -> Self {
        match t {
            RationalConstant::TWO => Self::from_u32(2),
            RationalConstant::HALF => Self::from_rational(1, 2),
            RationalConstant::SINC_C1 => Self::from_rational(1, 6),
            RationalConstant::SINC_C2 => Self::from_rational(1, 120),
            RationalConstant::SINC_C3 => Self::from_rational(1, 5_040),
            RationalConstant::COS_C1 => Self::from_rational(1, 2),
            RationalConstant::COS_C2 => Self::from_rational(1, 24),
            RationalConstant::COS_C3 => Self::from_rational(1, 720),
            RationalConstant::DSINC_SCALE => Self::from_rational(1, 100_000),
            RationalConstant::DSINC_C0 => Self::from_rational(1, 6),
            RationalConstant::DSINC_C1 => Self::from_rational(1, 60),
            RationalConstant::DSINC_C2 => Self::from_rational(1, 1_680),
            RationalConstant::DSINC_C3 => Self::from_rational(1, 90_720),
            RationalConstant::HALF_C2 => Self::from_rational(1, 3),
            RationalConstant::HALF_C3 => Self::from_rational(4, 45),
        }
    }
}

impl<T> FromRationalConstant for T where T: FromRational {}
