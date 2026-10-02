// SPDX-License-Identifier: MIT

use crate::prelude::*;
use core::ops::Div;
use num_traits::Float;

// ── SCALAR — transcendentals, needed for the Lie group exp/log ──────────────
pub trait Scalar:
    Commutative + Invertible + Div<Output = Self> + FromRational + StandardPart
{
    fn powf_explicit(self, n: Self) -> Self;
    fn powi_explicit(self, n: i32) -> Self;
    fn sqrt_explicit(self) -> Self;
    fn sin_explicit(self) -> Self;
    fn cos_explicit(self) -> Self;
    fn exp_explicit(self) -> Self;
    fn ln_explicit(self) -> Self;
    fn atan2_explicit(self, x: Self) -> Self;

    /// The larger of the two. Explicit, because the symbolic carrier cannot
    /// answer a comparison during tracing — it lowers it into a single
    /// select instruction rather than into a branch of the decision tree.
    ///
    /// Panics by default: for a carrier with derivatives (the AD tower) max
    /// is piecewise-discontinuous in the derivative, and there is no meaningful answer. Carriers
    /// that need max (machine floats, fixed, symbolic) override it.
    fn max_explicit(self, _other: Self) -> Self {
        panic!("max_explicit: this carrier does not support comparison")
    }

    /*
    // ── Study-form transcendentals (parametrised by u = l²) ───────────────
    // Closed-form exp/log of an SE(3) versor needs sin(√u)/√u, cos(√u) and
    // d(sinc)/du as functions of u = l² (no √ of the angle). These are removable
    // 0/0 singularities at u = 0, so each picks its representation by the seam:
    // a Taylor series near 0 (the only stable arm there), the closed form away —
    // both arms differentiable so an AD tower (Tangent) sees no kink. The series
    // coefficients are baked per monomorphisation as associated consts.
    //
    // Carrier-specific scalars override these to record one op instead of
    // expanding the arithmetic (e.g. a symbolic tracer emitting one SPIR-V node).
    const TWO: Self = Self::from_u32(2);
    /// ½ — the exp/bracket convention coefficient (se(3) ↔ bivector). A constant
    /// rather than a runtime from_rational call: baked in by monomorphization, like the series.
    const HALF: Self = Self::from_rational(1, 2);
    // sinc_sq series (Taylor of sin(√u)/√u in u near 0).
    const SINC_C1: Self = Self::from_rational(1, 6);
    const SINC_C2: Self = Self::from_rational(1, 120);
    const SINC_C3: Self = Self::from_rational(1, 5040);
    // cos_sq series (Taylor of cos(√u) in u near 0).
    const COS_C1: Self = Self::from_rational(1, 2);
    const COS_C2: Self = Self::from_rational(1, 24);
    const COS_C3: Self = Self::from_rational(1, 720);
    // dsinc_sq series (Taylor of d(sin(√u)/√u)/du near 0).
    const DSINC_SCALE: Self = Self::from_rational(1, 100_000);
    const DSINC_C0: Self = Self::from_rational(1, 6);
    const DSINC_C1: Self = Self::from_rational(1, 60);
    const DSINC_C2: Self = Self::from_rational(1, 1680);
    const DSINC_C3: Self = Self::from_rational(1, 90720);
    // half_angle_sq series (Taylor of l² in δ = 1 − cos l).
    const HALF_C2: Self = Self::from_rational(1, 3);
    const HALF_C3: Self = Self::from_rational(4, 45);
    */

    /// sin(√u)/√u as a function of u = l², smooth at 0 (= 1).
    #[inline]
    fn sinc_sq(self) -> Self {
        // representation-select: both arms differentiable and agree at the seam
        if self.standard_part().is_effective_zero() {
            Self::ONE
                - self
                    * (Self::get_constant(RationalConstant::SINC_C1)
                        - self
                            * (Self::get_constant(RationalConstant::SINC_C2)
                                - self * Self::get_constant(RationalConstant::SINC_C3)))
        } else {
            let l = self.sqrt_explicit();
            l.sin_explicit() / l
        }
    }

    /// cos(√u) as a function of u = l².
    #[inline]
    fn cos_sq(self) -> Self {
        // representation-select: both arms differentiable and agree at the seam
        if self.standard_part().is_effective_zero() {
            Self::ONE
                - self
                    * (Self::get_constant(RationalConstant::COS_C1)
                        - self
                            * (Self::get_constant(RationalConstant::COS_C2)
                                - self * Self::get_constant(RationalConstant::COS_C3)))
        } else {
            self.sqrt_explicit().cos_explicit()
        }
    }

    /// d(sinc_sq)/du; series near 0 is mandatory (closed form is 0/0 there).
    #[inline]
    fn dsinc_sq(self) -> Self {
        // representation-select: series is the only stable path near 0
        if (self * Self::get_constant(RationalConstant::DSINC_SCALE))
            .standard_part()
            .is_effective_zero()
        {
            -Self::get_constant(RationalConstant::DSINC_C0)
                + self
                    * (Self::get_constant(RationalConstant::DSINC_C1)
                        - self
                            * (Self::get_constant(RationalConstant::DSINC_C2)
                                - self * Self::get_constant(RationalConstant::DSINC_C3)))
        } else {
            let l = self.sqrt_explicit();
            (l * l.cos_explicit() - l.sin_explicit())
                / (Self::get_constant(RationalConstant::TWO) * l * l * l)
        }
    }

    /// u = l² from `self` = scalar a = cos l and euclidean bivector norm² s2 = sin²l.
    #[inline]
    fn half_angle_sq(self, s2: Self) -> Self {
        let delta = Self::ONE - self;
        // representation-select: both arms differentiable and agree at the seam
        if delta.standard_part().is_effective_zero() {
            delta
                * (Self::get_constant(RationalConstant::TWO)
                    + delta
                        * (Self::get_constant(RationalConstant::HALF_C2)
                            + delta * Self::get_constant(RationalConstant::HALF_C3)))
        } else {
            let l = s2.sqrt_explicit().atan2_explicit(self);
            l * l
        }
    }
}

impl<T> Scalar for T
where
    T: Commutative + Invertible + Div<Output = Self> + Float + FromRational + StandardPart,
{
    #[inline(always)]
    fn powf_explicit(self, n: Self) -> Self {
        self.powf(n)
    }

    #[inline(always)]
    fn powi_explicit(self, n: i32) -> Self {
        self.powi(n)
    }

    #[inline(always)]
    fn sqrt_explicit(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn sin_explicit(self) -> Self {
        self.sin()
    }

    #[inline(always)]
    fn cos_explicit(self) -> Self {
        self.cos()
    }

    #[inline(always)]
    fn exp_explicit(self) -> Self {
        self.exp()
    }

    #[inline(always)]
    fn ln_explicit(self) -> Self {
        self.ln()
    }

    #[inline(always)]
    fn atan2_explicit(self, x: Self) -> Self {
        self.atan2(x)
    }

    #[inline(always)]
    fn max_explicit(self, other: Self) -> Self {
        Float::max(self, other)
    }
}

// The Study-form tests lived in clifford::pga3::motor until the forms moved here
// (A7); tests follow their subject. For the representation-select seams and f32 thresholds see the
// comments at EffectiveZero in support.rs.
#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn sinc_sq_one_at_zero() {
        assert!(close(0.0f64.sinc_sq(), 1.0, 1e-15));
    }

    #[test]
    fn sinc_sq_branches_agree_around_threshold() {
        for &u in &[0.0_f64, 1e-12, 1e-10, 9e-9, 1e-8, 1.1e-8, 1e-7, 1e-6, 1e-5] {
            let l = u.sqrt();
            let reference = if l == 0.0 { 1.0 } else { l.sin() / l };
            assert!(close(u.sinc_sq(), reference, 1e-13), "u={u}");
        }
    }

    #[test]
    fn sinc_sq_decreases_from_one() {
        let (a, b, c) = (0.01f64.sinc_sq(), 0.5f64.sinc_sq(), 2.0f64.sinc_sq());
        assert!(a < 1.0 && a > b && b > c, "{a},{b},{c}");
    }

    #[test]
    fn sinc_sq_zero_at_pi_squared() {
        use core::f64::consts::PI;
        assert!(close((PI * PI).sinc_sq(), 0.0, 1e-14));
    }

    #[test]
    fn cos_sq_branches_agree_around_threshold() {
        for &u in &[0.0_f64, 1e-12, 1e-10, 9e-10, 1e-9, 1.1e-9, 1e-8, 1e-6, 1e-4] {
            let reference = u.sqrt().cos();
            assert!(close(u.cos_sq(), reference, 1e-13), "u={u}");
        }
    }

    #[test]
    fn cos_sq_endpoints() {
        use core::f64::consts::PI;
        assert!(close(0.0f64.cos_sq(), 1.0, 1e-15));
        assert!(close((PI * PI).cos_sq(), -1.0, 1e-14));
    }

    #[test]
    fn dsinc_sq_at_zero() {
        assert!(close(0.0f64.dsinc_sq(), -1.0 / 6.0, 1e-13));
    }

    #[test]
    fn dsinc_sq_is_derivative_of_sinc_sq() {
        let u = 0.5_f64;
        let h = 1e-7;
        let fd = ((u + h).sinc_sq() - (u - h).sinc_sq()) / (2.0 * h);
        assert!(close(u.dsinc_sq(), fd, 1e-7));
    }

    // ── f32 threshold tuning ──────────────────────────────────────────────
    // The f32 EffectiveZero threshold (support.rs) decides where the Study
    // helpers hand off from their Taylor arm to the closed form. These sweep
    // the seam at f32 precision so the constant can be tuned. dsinc_sq is the
    // binding case: its closed form subtracts l·cos l − sin l ≈ −l³/3, which
    // sheds roughly log10(1/u) digits to cancellation, so pushing the series
    // out (a larger threshold) buys f32 more than it ever did f64. A failing
    // assertion prints the worst u and error — that is the tuning signal.

    fn sinc_ref(u: f64) -> f64 {
        let l = u.sqrt();
        if l == 0.0 { 1.0 } else { l.sin() / l }
    }
    fn cos_ref(u: f64) -> f64 {
        u.sqrt().cos()
    }
    fn dsinc_ref(u: f64) -> f64 {
        let l = u.sqrt();
        if l == 0.0 {
            -1.0 / 6.0
        } else {
            (l * l.cos() - l.sin()) / (2.0 * l * l * l)
        }
    }

    // worst absolute error of an f32 helper vs its f64 reference over a
    // geometric sweep of u in [lo, hi]; returns (error, u-at-worst).
    fn sweep_err(
        lo: f64,
        hi: f64,
        helper: impl Fn(f32) -> f32,
        reference: impl Fn(f64) -> f64,
    ) -> (f64, f64) {
        let steps = 400;
        let (mut worst, mut at) = (0.0f64, lo);
        for k in 0..=steps {
            let t = k as f64 / steps as f64;
            let u = (lo.ln() * (1.0 - t) + hi.ln() * t).exp();
            let err = (helper(u as f32) as f64 - reference(u)).abs();
            if err > worst {
                worst = err;
                at = u;
            }
        }
        (worst, at)
    }

    #[test]
    fn sinc_sq_f32_smooth_across_seam() {
        let (err, at) = sweep_err(1e-7, 10.0, |u| u.sinc_sq(), sinc_ref);
        assert!(err < 1e-6, "max |sinc_sq f32 − ref| = {err} at u={at}");
    }

    #[test]
    fn cos_sq_f32_smooth_across_seam() {
        let (err, at) = sweep_err(1e-7, 10.0, |u| u.cos_sq(), cos_ref);
        assert!(err < 1e-6, "max |cos_sq f32 − ref| = {err} at u={at}");
    }

    #[test]
    fn dsinc_sq_f32_smooth_across_seam() {
        // binding case — closed-form cancellation dominates near the seam.
        let (err, at) = sweep_err(1e-4, 10.0, |u| u.dsinc_sq(), dsinc_ref);
        assert!(err < 1e-6, "max |dsinc_sq f32 − ref| = {err} at u={at}");
    }

    #[test]
    fn half_angle_sq_f32_smooth_across_seam() {
        // log()'s atan2 branch: recover u = l² from a = cos l and s2 = sin²l.
        // The seam is at delta = 1 − a (≈ l²/2) effective-zero, i.e. small l.
        // Unlike dsinc, the closed atan2 arm is cancellation-free (it forms l
        // directly), while the SERIES arm leans on delta = 1 − a, which loses
        // precision when a rounds toward 1 — so here a *smaller* threshold is
        // if anything safer, the opposite of dsinc. Error is absolute, because
        // u only ever feeds sinc/dsinc and scales the generator: a 1e-8 miss on
        // a 1e-8 angle is irrelevant downstream even though its relative size
        // is huge. The sweep stops short of π (a → −1, s2 → 0), naturally
        // ill-conditioned for any representation.
        let steps = 400;
        let (mut worst, mut at) = (0.0f64, 0.0f64);
        for k in 0..=steps {
            let t = k as f64 / steps as f64;
            let l = (1e-4_f64.ln() * (1.0 - t) + 3.0_f64.ln() * t).exp();
            let (a, s2) = (l.cos(), l.sin() * l.sin());
            let got = (a as f32).half_angle_sq(s2 as f32) as f64;
            let err = (got - l * l).abs();
            if err > worst {
                worst = err;
                at = l;
            }
        }
        assert!(
            worst < 1e-5,
            "max |half_angle_sq f32 − l²| = {worst} at l={at}"
        );
    }
}
