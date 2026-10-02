// SPDX-License-Identifier: MIT

use crate::Lift;
use bytemuck::{Pod, Zeroable};
use core::fmt::{Debug, Formatter};
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use peano::prelude::*;

// Gradient over GradN axes over R — our Peano vector.
type Grad<GradN, R> = Vector<GradN, R>;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Tangent<GradN: Nat, R: Copy + Debug + Send + Sync> {
    value: R,
    grad: Grad<GradN, R>,
}

// value and grad are both over the same scalar R; `#[repr(C)]` lays them out back to
// back with no holes: [value, grad₀, …, grad_{GradN-1}]. Manual impls (not derive): derive
// would require a spurious `GradN: Pod`. `'static` is a Pod requirement.
unsafe impl<GradN: Nat, R: Copy + Debug + Send + Sync + Zeroable> Zeroable for Tangent<GradN, R> where
    Grad<GradN, R>: Zeroable
{
}
unsafe impl<GradN: Nat + 'static, R: Copy + Debug + Send + Sync + Pod> Pod for Tangent<GradN, R> where
    Grad<GradN, R>: Pod
{
}

// The kernel's Jacobian carrier: value + 24 partials over f32 — 25 packed
// slots (100 bytes). A layout regression will not compile.
const _: () = assert_packed::<Tangent<N24, f32>, f32>(25);

// Copy/Debug by hand: derive would attach a spurious GradN: Copy/Debug bound.
impl<GradN: Nat, R: Copy + Debug + Send + Sync> Debug for Tangent<GradN, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "Tangent({:?}, {:?})", self.value, self.grad)
    }
}
impl<GradN: Nat, R: Copy + Debug + Send + Sync + PartialEq> PartialEq for Tangent<GradN, R>
where
    Grad<GradN, R>: PartialEq,
{
    fn eq(&self, o: &Self) -> bool {
        self.value == o.value && self.grad == o.grad
    }
}

// Gradient carrier: our vector with the operations AD needs (sum/difference/
// negation, scaling by scalar R, recursive zero).
pub trait GradVec<R>: ScalarMul<R> + AbelianGroup {}
impl<R, G> GradVec<R> for G where G: ScalarMul<R> + AbelianGroup {}

impl<GradN: Nat, R: Ring> Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    /// Canonical embedding η: R → R ⊕ Rᴺ (zero gradient).
    pub const fn embed(value: R) -> Self {
        Self {
            value,
            grad: Grad::<GradN, R>::ZERO,
        }
    }
    /// From a value and an explicit gradient vector (seeding a variable for a test).
    pub const fn from_grad(value: R, grad: Grad<GradN, R>) -> Self {
        Self { value, grad }
    }
    /// The base (real) part.
    pub const fn base(&self) -> R {
        self.value
    }
    /// The whole gradient.
    pub const fn grad(&self) -> Grad<GradN, R> {
        self.grad
    }
    /// k-th gradient component (runtime index — for per-axis projections).
    /// Not const: runtime access goes through Storage (closure/recursion over Nat).
    pub fn component(&self, k: usize) -> R
    where
        GradN: Storage,
    {
        *GradN::get(&self.grad, k)
    }
}

// Bounds are minimal: const is needed only for the carrier's from_u32/from_rational; ZERO
// is an associated constant, and the non-const AbelianGroup impl is enough for it.
impl<GradN: Nat, R: FromRational + AbelianGroup> FromRational for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    fn from_u32(n: u32) -> Self {
        Self {
            value: R::from_u32(n),
            grad: Grad::<GradN, R>::ZERO,
        }
    }
    fn from_rational(num: u32, den: u32) -> Self {
        Self {
            value: R::from_rational(num, den),
            grad: Grad::<GradN, R>::ZERO,
        }
    }
}

impl<GradN: Nat, R: Ring> Add for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            value: self.value + rhs.value,
            grad: self.grad + rhs.grad,
        }
    }
}
impl<GradN: Nat, R: Ring> Sub for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            value: self.value - rhs.value,
            grad: self.grad - rhs.grad,
        }
    }
}
impl<GradN: Nat, R: Ring> Neg for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            grad: -self.grad,
        }
    }
}
// Product rule: (uv)' = u'v + uv'.
impl<GradN: Nat, R: Ring> Mul for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            value: self.value * rhs.value,
            grad: rhs.grad.scale(self.value) + self.grad.scale(rhs.value),
        }
    }
}
impl<GradN: Nat, R: Ring + Invertible> Div for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    type Output = Self;
    // Division of a dual number is DEFINED via the inverse: (a + a'ε)/(b + b'ε) =
    // (a + a'ε)·(b + b'ε)⁻¹, and `try_recip` already carries the rule for the gradient.
    // Expanding this into a quotient by hand would duplicate the derivative formula
    // and lose the single point where a non-invertible denominator is caught.
    // The `*` here is not the typo the lint is hunting for.
    #[expect(
        clippy::suspicious_arithmetic_impl,
        reason = "division is defined as multiplication by the inverse; see comment"
    )]
    fn div(self, rhs: Self) -> Self {
        self * rhs.try_recip().expect("division by a non-invertible value")
    }
}
impl<GradN: Nat, R: Ring> AddAssign for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<GradN: Nat, R: Ring> SubAssign for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<GradN: Nat, R: Ring> AbelianGroup for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    const ZERO: Self = Self {
        value: R::ZERO,
        grad: Grad::<GradN, R>::ZERO,
    };
}
impl<GradN: Nat, R: Ring> MulMonoid for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    const ONE: Self = Self {
        value: R::ONE,
        grad: Grad::<GradN, R>::ZERO,
    };
}
impl<GradN: Nat, R: Ring> Ring for Tangent<GradN, R> where Grad<GradN, R>: GradVec<R> {}
impl<GradN: Nat, R: Ring> Commutative for Tangent<GradN, R> where Grad<GradN, R>: GradVec<R> {}

// Neumann inversion: ε is nilpotent ⇒ the series terminates after the 1st term.
// 1/(v₀+n) = (1/v₀)(1 − n/v₀) ⇒ grad = g·(−v²).
impl<GradN: Nat, R: Ring + Invertible> Invertible for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    fn try_recip(self) -> Option<Self> {
        let v = self.value.try_recip()?;
        Some(Self {
            value: v,
            grad: self.grad.scale(-(v * v)),
        })
    }
}

impl<GradN: Nat, R: Ring + StandardPart> StandardPart for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    type Real = R::Real;
    fn standard_part(self) -> Self::Real {
        self.value.standard_part()
    }
}

// Full Scalar: derivatives of transcendentals — chain rule, grad·f' via scale.
impl<GradN: Nat, R: Scalar> Scalar for Tangent<GradN, R>
where
    Grad<GradN, R>: GradVec<R>,
{
    fn exp_explicit(self) -> Self {
        let value = self.value.exp_explicit(); // f = f' for exp
        Self {
            value,
            grad: self.grad.scale(value),
        }
    }
    fn ln_explicit(self) -> Self {
        let d = R::ONE / self.value; // f' = 1/a
        Self {
            value: self.value.ln_explicit(),
            grad: self.grad.scale(d),
        }
    }
    fn sqrt_explicit(self) -> Self {
        let value = self.value.sqrt_explicit();
        let d = R::ONE / (value + value); // f' = 1/(2√a)
        Self {
            value,
            grad: self.grad.scale(d),
        }
    }
    fn sin_explicit(self) -> Self {
        let d = self.value.cos_explicit();
        Self {
            value: self.value.sin_explicit(),
            grad: self.grad.scale(d),
        }
    }
    fn cos_explicit(self) -> Self {
        let d = -self.value.sin_explicit();
        Self {
            value: self.value.cos_explicit(),
            grad: self.grad.scale(d),
        }
    }
    fn powf_explicit(self, n: Self) -> Self {
        (n * self.ln_explicit()).exp_explicit()
    }
    fn atan2_explicit(self, x: Self) -> Self {
        let (y0, x0) = (self.value, x.value);
        let inv = R::ONE / (x0 * x0 + y0 * y0); // ∂ = (x·dy − y·dx)/(x²+y²)
        Self {
            value: y0.atan2_explicit(x0),
            grad: self.grad.scale(x0 * inv) - x.grad.scale(y0 * inv),
        }
    }
    fn powi_explicit(self, n: i32) -> Self {
        if n < 0 {
            return self.try_recip().unwrap().powi_explicit(-n);
        }
        let mut result = Self::ONE;
        let mut base = self;
        let mut e = n as u32;
        while e > 0 {
            if e & 1 == 1 {
                result = result * base;
            }
            base = base * base;
            e >>= 1;
        }
        result
    }
}

// One axis (ordinary derivative) — Peano length 1.
// Recursive Lift branch: R ↪ Tangent<GradN, S> = embed R.lift() with zero
// gradient. Lifts up the AD tower; the identity lives at the bottom (EffectiveZero).
impl<GradN: Nat, R: Lift<S>, S: Ring> Lift<Tangent<GradN, S>> for R
where
    Grad<GradN, S>: GradVec<S>,
{
    fn lift(self) -> Tangent<GradN, S> {
        Tangent::embed(self.lift())
    }
}

pub type Jet1<R> = Tangent<N1, R>;
pub type Jet6<R> = Tangent<N6, R>;
pub type Jet12<R> = Tangent<N12, R>;
pub type Jet24 = Tangent<N24, f64>;
pub type Jet24Tower = Tangent<N24, Tangent<N24, f64>>;

// ── AD ATOM #2: truncated polynomial, ONE variable, order K (carrier: ℕ, truncated)
//    R[ε]/(ε^{K+1}). For higher derivatives along one axis.
// pub struct Trunc<R, const K: usize> { c: [R; /* K+1, concrete size */] }
//    impl the same stages; mul = convolution truncating degree > K; try_recip = Neumann up to order K

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::FRAC_PI_2;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    type J1 = Tangent<N1, f64>;
    type J2 = Tangent<N1, Tangent<N1, f64>>;
    type J3 = Tangent<N1, Tangent<N1, Tangent<N1, f64>>>;

    fn var(a: f64) -> J1 {
        Tangent::from_grad(a, vector![1.0])
    }

    // sqrt/ln/atan2 are singular at the origin: the AD derivative divides by zero.
    // Legacy `Jet` used plain division there (NaN/inf gradient, NO panic), so a
    // downstream `standard_part().is_effective_zero()` guard can discard it. Lock
    // that: these must NOT panic on a zero value (regression — try_recip().unwrap()
    // used to abort the camera example before the guard could fire).
    #[test]
    fn sqrt_at_zero_does_not_panic() {
        let r = var(0.0).sqrt_explicit();
        assert_eq!(r.base(), 0.0); // value is clean; only the gradient is non-finite
    }

    #[test]
    fn ln_at_zero_does_not_panic() {
        let _ = var(0.0).ln_explicit(); // must not panic
    }
    // J2: re.base=a re.grad=f', du.base=1 → second derivative in du.grad
    fn var2(a: f64) -> J2 {
        Tangent::from_grad(
            Tangent::from_grad(a, vector![1.0]),
            vector![Tangent::from_grad(1.0, vector![0.0])],
        )
    }
    fn var3(a: f64) -> J3 {
        let l = |x: f64, d: f64| Tangent::<N1, f64>::from_grad(x, vector![d]);
        Tangent::from_grad(
            Tangent::from_grad(l(a, 1.0), vector![l(1.0, 0.0)]),
            vector![Tangent::from_grad(l(1.0, 0.0), vector![l(0.0, 0.0)])],
        )
    }

    fn foo<T: Scalar>(a: T, b: T) -> T {
        a * b + a / b
    }

    #[test]
    fn simple() {
        let a: J1 = Tangent::from_grad(10.0, vector![2.0]);
        let b: J1 = Tangent::from_grad(2.0, vector![2.0]);
        let c = foo(a, b);
        assert_eq!(c.base(), 25.0);
        assert_eq!(c.component(0), 20.0);
    }

    // Ported from experiments/peano when the ad copy there was removed: the same
    // first order, but the derivative is read by a TYPED index
    // grad().get::<Z>() (Access) rather than the runtime component(0).
    #[test]
    fn autodiff() {
        let var = |a: f64| J1::from_grad(a, vector![1.0]); // dx/dx = 1
        let d = |j: J1| *j.grad().get::<Z>(); // derivative (axis 0)

        // Product rule: d(x²) = 2x.
        let y = var(3.0) * var(3.0);
        assert_eq!(y.base(), 9.0);
        assert_eq!(d(y), 6.0);

        // Inverse: d(1/x) = −1/x².
        let y = J1::ONE / var(2.0);
        assert_eq!(y.base(), 0.5);
        assert_eq!(d(y), -0.25);

        // Transcendentals: √, chain rule, exp∘ln.
        let y = var(4.0).sqrt_explicit();
        assert!((y.base() - 2.0).abs() < 1e-12 && (d(y) - 0.25).abs() < 1e-12);
        let y = (var(1.0) * var(1.0)).sin_explicit(); // d sin(x²) = 2x·cos(x²)
        assert!((d(y) - 2.0 * 1.0_f64.cos()).abs() < 1e-12);
        let y = var(3.0).ln_explicit().exp_explicit(); // identity: f=3, f'=1
        assert!((y.base() - 3.0).abs() < 1e-12 && (d(y) - 1.0).abs() < 1e-12);
    }

    // ── first order ─────────────────────────────────────────────────────────

    #[test]
    fn constant_has_zero_derivative() {
        let c = J1::embed(7.0) * J1::embed(3.0);
        assert_eq!(c.component(0), 0.0);
    }

    #[test]
    fn product_rule_x_squared() {
        let y = var(3.0) * var(3.0); // d(x²)=2x
        assert_eq!(y.base(), 9.0);
        assert_eq!(y.component(0), 6.0);
    }

    #[test]
    fn reciprocal() {
        let y = J1::ONE / var(2.0); // d(1/x) = -1/x²
        assert_eq!(y.base(), 0.5);
        assert_eq!(y.component(0), -0.25);
    }

    #[test]
    fn sqrt_deriv() {
        let y = var(4.0).sqrt_explicit();
        assert!(close(y.base(), 2.0));
        assert!(close(y.component(0), 0.25));
    }

    #[test]
    fn exp_ln_roundtrip_is_identity() {
        let y = var(3.0).ln_explicit().exp_explicit();
        assert!(close(y.base(), 3.0));
        assert!(close(y.component(0), 1.0));
    }

    #[test]
    fn sin_cos_at_zero() {
        let s = var(0.0).sin_explicit();
        let c = var(0.0).cos_explicit();
        assert!(close(s.base(), 0.0) && close(s.component(0), 1.0));
        assert!(close(c.base(), 1.0) && close(c.component(0), 0.0));
    }

    #[test]
    fn chain_rule_sin_of_square() {
        let y = (var(1.0) * var(1.0)).sin_explicit();
        assert!(close(y.base(), 1.0_f64.sin()));
        assert!(close(y.component(0), 2.0 * 1.0_f64.cos()));
    }

    #[test]
    fn powi_positive_negative_zero() {
        let p3 = var(2.0).powi_explicit(3);
        assert_eq!(p3.base(), 8.0);
        assert_eq!(p3.component(0), 12.0);
        let pm2 = var(2.0).powi_explicit(-2);
        assert!(close(pm2.base(), 0.25));
        assert!(close(pm2.component(0), -0.25));
        let p0 = var(5.0).powi_explicit(0);
        assert_eq!(p0.base(), 1.0);
        assert_eq!(p0.component(0), 0.0);
    }

    #[test]
    fn powf_matches_powi() {
        let pf = var(2.0).powf_explicit(J1::embed(3.0));
        assert!(close(pf.base(), 8.0));
        assert!(close(pf.component(0), 12.0));
    }

    #[test]
    fn atan2_roundtrip_and_derivative() {
        let t = var(0.7);
        let y = t.sin_explicit().atan2_explicit(t.cos_explicit());
        assert!(close(y.base(), 0.7));
        assert!(close(y.component(0), 1.0));
    }

    #[test]
    fn atan2_partials() {
        let x0 = 2.0;
        let y0 = 1.0;
        let g = var(y0).atan2_explicit(J1::embed(x0));
        assert!(close(g.base(), y0.atan2(x0)));
        assert!(close(g.component(0), x0 / (x0 * x0 + y0 * y0)));
    }

    // ── second order (towers) ────────────────────────────────────────────────
    // Layout: y.base() = inner Tangent (value, ∂/∂inner); y.component(0) = outer
    // derivative (∂/∂outer, ∂²). So f = base().base(), f' = component(0).base(),
    // f'' = component(0).component(0).

    #[test]
    fn second_deriv_cubic() {
        let y = var2(2.0).powi_explicit(3); // x³: f=8, f'=12, f''=12
        assert!(close(y.base().base(), 8.0));
        assert!(close(y.base().component(0), 12.0));
        assert!(close(y.component(0).base(), 12.0)); // mixed symmetry
        assert!(close(y.component(0).component(0), 12.0));
    }

    #[test]
    fn second_deriv_exp() {
        let y = var2(0.0).exp_explicit();
        assert!(close(y.base().base(), 1.0));
        assert!(close(y.base().component(0), 1.0));
        assert!(close(y.component(0).component(0), 1.0));
    }

    #[test]
    fn second_deriv_sin_at_pi_half() {
        let y = var2(FRAC_PI_2).sin_explicit(); // f=1, f'=0, f''=-1
        assert!(close(y.base().base(), 1.0));
        assert!(close(y.base().component(0), 0.0));
        assert!(close(y.component(0).component(0), -1.0));
    }

    #[test]
    fn second_deriv_reciprocal() {
        let y = J2::ONE / var2(2.0); // 1/x: f=.5, f'=-.25, f''=.25
        assert!(close(y.base().base(), 0.5));
        assert!(close(y.base().component(0), -0.25));
        assert!(close(y.component(0).component(0), 0.25));
    }

    #[test]
    fn mixed_partial_x2y() {
        // f(x,y)=x²y at (3,5): f=45, ∂x=30, ∂y=9, ∂x∂y=6. x on outer ε, y on inner.
        let x: J2 = Tangent::from_grad(
            Tangent::from_grad(3.0, vector![1.0]),
            vector![Tangent::from_grad(0.0, vector![0.0])],
        );
        let y: J2 = Tangent::from_grad(
            Tangent::from_grad(5.0, vector![0.0]),
            vector![Tangent::from_grad(1.0, vector![0.0])],
        );
        let f = x * x * y;
        assert!(close(f.base().base(), 45.0));
        assert!(close(f.base().component(0), 30.0));
        assert!(close(f.component(0).base(), 9.0));
        assert!(close(f.component(0).component(0), 6.0));
    }

    // ── third order ───────────────────────────────────────────────────────────

    #[test]
    fn third_deriv_cubic_is_constant_six() {
        let y = var3(2.0).powi_explicit(3);
        assert!(close(y.base().base().base(), 8.0));
        assert!(close(y.component(0).component(0).component(0), 6.0));
    }

    #[test]
    fn third_deriv_exp_composes() {
        let y = var3(0.0).exp_explicit();
        assert!(close(y.base().base().base(), 1.0));
        assert!(close(y.component(0).component(0).component(0), 1.0));
    }

    // ── flat Jet<N>: gradient in one pass ─────────────────────────────────────

    fn flat_seed<N: Storage>(a: f64, axis: usize) -> Tangent<N, f64>
    where
        Grad<N, f64>: GradVec<f64>,
    {
        Tangent::from_grad(a, N::from_fn(|k| if k == axis { 1.0 } else { 0.0 }))
    }
    fn flat_all_ones<N: Storage>(a: f64) -> Tangent<N, f64>
    where
        Grad<N, f64>: GradVec<f64>,
    {
        Tangent::from_grad(a, N::from_fn(|_| 1.0))
    }
    fn assert_silent_axes<N: Storage>(y: &Tangent<N, f64>, active: usize)
    where
        Grad<N, f64>: GradVec<f64>,
    {
        for k in 0..N::LEN {
            if k != active {
                assert!(close(y.component(k), 0.0), "axis {k} leaked");
            }
        }
    }

    #[test]
    fn flat_value_in_grade0() {
        let x = flat_seed::<N6>(2.0, 2);
        let y = x.exp_explicit();
        assert!(close(y.base(), 2.0_f64.exp()));
    }

    #[test]
    fn flat_single_axis_derivative() {
        let axis = 3;
        let x = flat_seed::<N6>(1.5, axis);
        let y = x.exp_explicit();
        assert!(close(y.component(axis), 1.5_f64.exp()));
        assert_silent_axes(&y, axis);
    }

    #[test]
    fn flat_six_axes_independent() {
        let x = flat_all_ones::<N6>(0.7);
        let y = x.exp_explicit();
        let g = 0.7_f64.exp();
        for k in 0..6 {
            assert!(close(y.component(k), g));
        }
    }

    #[test]
    fn flat_gradient_of_product_is_per_axis() {
        let axis = 1;
        let x = flat_seed::<N3>(3.0, axis);
        let y = x * x;
        assert!(close(y.base(), 9.0));
        assert!(close(y.component(axis), 6.0));
        assert_silent_axes(&y, axis);
    }

    #[test]
    fn flat_sin_per_axis_derivative() {
        let axis = 0;
        let x = flat_seed::<N6>(FRAC_PI_2, axis);
        let y = x.sin_explicit();
        assert!(close(y.base(), 1.0));
        assert!(close(y.component(axis), 0.0));
        let z = x.cos_explicit();
        assert!(close(z.base(), 0.0));
        assert!(close(z.component(axis), -1.0));
    }

    #[test]
    fn flat_chain_rule_holds_per_axis() {
        let axis = 4;
        let x = flat_seed::<N6>(1.0, axis);
        let y = (x * x).sin_explicit();
        assert!(close(y.base(), 1.0_f64.sin()));
        assert!(close(y.component(axis), 2.0 * 1.0_f64.cos()));
        assert_silent_axes(&y, axis);
    }

    #[test]
    fn flat_powi_per_axis() {
        let axis = 2;
        let x = flat_seed::<N6>(2.0, axis);
        let y = x.powi_explicit(3);
        assert!(close(y.base(), 8.0));
        assert!(close(y.component(axis), 12.0));
        assert_silent_axes(&y, axis);
    }

    #[test]
    fn flat_reciprocal_per_axis() {
        let axis = 1;
        let x = flat_seed::<N4>(2.0, axis);
        let y = Tangent::<N4, f64>::ONE / x;
        assert!(close(y.base(), 0.5));
        assert!(close(y.component(axis), -0.25));
        assert_silent_axes(&y, axis);
    }

    #[test]
    fn from_grad_and_powi_neg() {
        let x: Tangent<N1, f64> = Tangent::from_grad(2.0, vector![1.0]); // var at 2
        // 1/x : value 0.5, grad -0.25
        let inv = x.powi_explicit(-1);
        assert!((inv.base() - 0.5).abs() < 1e-12);
        assert!((inv.component(0) - (-0.25)).abs() < 1e-12);
    }

    #[test]
    fn standard_part_descends_to_bottom() {
        // tower value 3.0, all grads 0
        let leaf: Tangent<N2, f64> = Tangent::embed(3.0);
        let tower: Tangent<N2, Tangent<N2, f64>> = Tangent::embed(leaf);
        let r: f64 = tower.standard_part();
        assert_eq!(r, 3.0);
        assert!((0.0f64).is_effective_zero());
        assert!(!(1.0f64).is_effective_zero());
    }
}
