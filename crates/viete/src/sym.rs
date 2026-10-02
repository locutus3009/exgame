// SPDX-License-Identifier: MIT

use num_rational::Rational32;

use crate::engine::{emit_bin, emit_max, emit_select, emit_un, fork, neg_of};
use crate::ir::{BranchTest, Instr};
use num_traits::{One, Signed, Zero};
use peano::prelude::*;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Handle {
    Constant(Rational32),
    Input(u32),
    Param(u32),
    Runtime(u32),
}

unsafe impl bytemuck::Zeroable for Handle {}
unsafe impl bytemuck::Pod for Handle {}

/// Symbolic scalar: a pure handle, no f64 shadow. Operations push graph events.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Sym(pub(crate) Handle);

impl Sym {
    /// A closure input occupying input-slot `slot`.
    pub fn input(slot: u32) -> Self {
        Sym(Handle::Input(slot))
    }
    /// A param occupying param-slot `i`. Pure constructor (no engine
    /// interaction, callable before tracing) — like `from_u32`/`input`. The
    /// engine resolves it to `Input(INPUTS + i)` at trace time.
    pub fn param(i: u32) -> Self {
        Sym(Handle::Param(i))
    }
    /// One-hot selector / Kronecker delta: `1` if `selector == lane`, else `0`.
    /// Straight-line (emits `Instr::Select`, no `is_effective_zero` fork), so it
    /// never branches the decision trie; lowers to a Lua `select(...)` helper /
    /// SPIR-V `OpSelect`. `lane` is a compile-time constant (the active axis
    /// index); the returned `1`/`0` are pooled constant references. Used to build
    /// a one-hot AD seed `e_selector` from a single runtime selector.
    pub fn select(selector: Sym, lane: u32) -> Sym {
        Sym::r(emit_select(selector.0, lane))
    }
    fn r(handle: Handle) -> Self {
        Sym(handle)
    }
}

impl PartialEq for Sym {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl FromRational for Sym {
    fn from_u32(n: u32) -> Self {
        Sym(Handle::Constant(Rational32::new_raw(n as i32, 1)))
    }
    fn from_rational(num: u32, den: u32) -> Self {
        Sym(Handle::Constant(Rational32::new_raw(
            num as i32, den as i32,
        )))
    }
}

impl Add for Sym {
    type Output = Sym;
    fn add(self, rhs: Sym) -> Sym {
        match (self.0, rhs.0) {
            (Handle::Constant(c), _) if c.is_zero() => rhs,
            (_, Handle::Constant(c)) if c.is_zero() => self,
            (Handle::Constant(a), Handle::Constant(b)) => Sym(Handle::Constant((a + b).reduced())),
            (a, b) => Sym::r(emit_bin(a, b, Instr::Add)),
        }
    }
}

impl Sub for Sym {
    type Output = Sym;
    fn sub(self, rhs: Sym) -> Sym {
        match (self.0, rhs.0) {
            (_, Handle::Constant(c)) if c.is_zero() => self,
            (Handle::Constant(a), Handle::Constant(b)) => Sym(Handle::Constant((a - b).reduced())),
            (Handle::Constant(c), _) if c.is_zero() => -rhs,
            (a, b) if a == b => Sym(Handle::Constant(Rational32::zero())),
            (a, b) => Sym::r(emit_bin(a, b, Instr::Sub)),
        }
    }
}

impl Mul for Sym {
    type Output = Sym;
    fn mul(self, rhs: Sym) -> Sym {
        match (self.0, rhs.0) {
            (Handle::Constant(c), _) if c.is_zero() => Sym(Handle::Constant(Rational32::zero())),
            (_, Handle::Constant(c)) if c.is_zero() => Sym(Handle::Constant(Rational32::zero())),
            (Handle::Constant(c), _) if c.is_one() => rhs,
            (_, Handle::Constant(c)) if c.is_one() => self,
            // Multiplication by -1 is exactly a negation (and lets the -1 constant
            // drop out of the pool). Goes through Sym::neg, so Neg(Neg(x)) folds too.
            (Handle::Constant(c), _) if c == -Rational32::one() => -rhs,
            (_, Handle::Constant(c)) if c == -Rational32::one() => -self,
            (Handle::Constant(a), Handle::Constant(b)) => Sym(Handle::Constant((a * b).reduced())),
            (a, b) => Sym::r(emit_bin(a, b, Instr::Mul)),
        }
    }
}

impl Div for Sym {
    type Output = Sym;
    fn div(self, rhs: Sym) -> Sym {
        match (self.0, rhs.0) {
            // statically-zero divisor: explicit, classifiable panic (caught by the driver)
            (_, Handle::Constant(c)) if c.is_zero() => panic!("viete::fatal::div_by_zero"),
            // 0 / x folds to 0 (assumes a nonzero divisor — same Div-by-runtime
            // assertion as the runtime arm; only diverges from f64 if x is itself
            // zero at runtime, which the design treats as a precondition).
            (Handle::Constant(c), _) if c.is_zero() => Sym(Handle::Constant(Rational32::zero())),
            (_, Handle::Constant(c)) if c.is_one() => self,
            (Handle::Constant(a), Handle::Constant(b)) => Sym(Handle::Constant((a / b).reduced())),
            (a, b) if a == b => Sym(Handle::Constant(Rational32::one())),
            (a, b) => {
                // Fork the exact-zero guard (suppressed when b is proven nonzero).
                // The b==0 arm panics -> Fatal{DivByZero} leaf, which the post-build
                // `collapse_fatal_branches` pass turns into a branchless fatal-status
                // slot (same route as NonInvertible). One mechanism for all fatals.
                if fork(b, BranchTest::ExactZero) {
                    panic!("viete::fatal::div_by_zero")
                } else {
                    Sym(emit_bin(a, b, Instr::Div))
                }
            }
        }
    }
}

impl Neg for Sym {
    type Output = Sym;
    fn neg(self) -> Sym {
        match self.0 {
            Handle::Constant(c) => Sym(Handle::Constant(-c)),
            a => {
                if let Some(inner) = neg_of(a) {
                    Sym(inner) // Neg(Neg(x)) -> x
                } else {
                    Sym::r(emit_un(a, Instr::Neg))
                }
            }
        }
    }
}

impl AddAssign for Sym {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Sym {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl AbelianGroup for Sym {
    const ZERO: Self = Sym(Handle::Constant(Rational32::new_raw(0, 1)));
}
impl MulMonoid for Sym {
    const ONE: Self = Sym(Handle::Constant(Rational32::new_raw(1, 1)));
}
impl Ring for Sym {}
impl Commutative for Sym {}

impl StandardPart for Sym {
    type Real = Self;
    fn standard_part(self) -> Self {
        self
    }
}

impl EffectiveZero for Sym {
    fn is_effective_zero(self) -> bool {
        fork(self.0, BranchTest::EffectiveZero)
    }
}

impl Invertible for Sym {
    fn try_recip(self) -> Option<Self> {
        match self.0 {
            Handle::Constant(c) if c.is_zero() => None,
            Handle::Constant(c) => Some(Sym(Handle::Constant((Rational32::one() / c).reduced()))),
            _ => {
                if self.is_effective_zero() {
                    None // zero arm; consumer's .unwrap() will panic -> caught as Fatal
                } else {
                    // recip = Div(ONE, x); emit directly, UNguarded — invertibility
                    // already established by the is_effective_zero check above.
                    Some(Sym(emit_bin(Sym::ONE.0, self.0, Instr::Div)))
                }
            }
        }
    }
}

impl Scalar for Sym {
    fn powf_explicit(self, n: Self) -> Self {
        Sym::r(emit_bin(self.0, n.0, Instr::Powf))
    }
    fn powi_explicit(self, n: i32) -> Self {
        match self.0 {
            Handle::Constant(c) => Sym(Handle::Constant(c.pow(n))),
            a => {
                let exp = Handle::Constant(Rational32::new_raw(n, 1));
                Sym::r(emit_bin(a, exp, Instr::Powi))
            }
        }
    }
    fn sqrt_explicit(self) -> Self {
        if let Handle::Constant(c) = self.0
            && c.is_negative()
        {
            panic!("viete::fatal::neg_sqrt");
        }
        Sym::r(emit_un(self.0, Instr::Sqrt))
    }
    fn sin_explicit(self) -> Self {
        if let Handle::Constant(c) = self.0
            && c.is_zero()
        {
            return Self::ZERO;
        }
        Sym::r(emit_un(self.0, Instr::Sin))
    }
    fn cos_explicit(self) -> Self {
        if let Handle::Constant(c) = self.0
            && c.is_zero()
        {
            return Self::ONE;
        }
        Sym::r(emit_un(self.0, Instr::Cos))
    }
    fn exp_explicit(self) -> Self {
        Sym::r(emit_un(self.0, Instr::Exp))
    }
    fn ln_explicit(self) -> Self {
        if let Handle::Constant(c) = self.0
            && c.is_negative()
        {
            panic!("viete::fatal::neg_ln");
        }
        Sym::r(emit_un(self.0, Instr::Ln))
    }
    fn atan2_explicit(self, x: Self) -> Self {
        Sym::r(emit_bin(self.0, x.0, Instr::Atan2))
    }
    /// One `Select`, not a fork: a comparison of two runtime values has no
    /// answer at trace time, and forking on it would double the trie for
    /// something the GPU decides in one instruction.
    fn max_explicit(self, other: Self) -> Self {
        Sym::r(emit_max(self.0, other.0))
    }
    // Study-form helpers (sinc_sq / cos_sq / dsinc_sq / half_angle_sq) are NOT
    // overridden: the default Scalar impls expand them into the ops above plus an
    // is_effective_zero fork, which is exactly the trace we want.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Event, drain_events_for_test};
    use crate::ir::{Instr, Operand};

    /// `max` is a straight-line select, not a fork: the trie must stay a single
    /// leaf, and the value must be right in both orders. A comparison of two
    /// runtime values has no answer at trace time — forking on it would double
    /// the kernel for something the GPU decides in one instruction.
    #[test]
    fn max_is_straight_line_and_picks_the_larger() {
        let traced = crate::api::Tracer::builder()
            .fn_name("maxprobe")
            .flatten()
            .build()
            .trace::<_>(2, 0, 1, &[], |inp: &Vec<Sym>, _p: &Vec<Sym>| {
                vec![inp[0].max_explicit(inp[1])]
            });
        assert_eq!(traced.leaf_count(), 1, "max must not fork the trie");
        assert_eq!(traced.run_lua(&[3.0, 7.0], &[]).unwrap()[0], 7.0);
        assert_eq!(traced.run_lua(&[7.0, 3.0], &[]).unwrap()[0], 7.0);
    }

    #[test]
    fn runtime_add_emits_instr_and_constant_folds() {
        crate::engine::reset_for_test();
        let a = Sym::input(0);
        let b = Sym::input(1);
        let _c = a + b; // runtime + runtime -> Instr::Add(0, Input0, Input1)
        // constant folding: 2 + 3 collapses, no instr
        let _f = Sym::from_u32(2) + Sym::from_u32(3);
        let events = drain_events_for_test();
        assert_eq!(events.len(), 1, "only the runtime add emits");
        match &events[0] {
            Event::Instr(Instr::Add(0, Operand::Input(0), Operand::Input(1))) => {}
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn add_zero_is_identity() {
        crate::engine::reset_for_test();
        let a = Sym::input(0);
        let r = a + Sym::from_u32(0); // (Sym::ZERO is defined in Task 5)
        assert!(matches!(r.0, Handle::Input(0)));
        assert!(drain_events_for_test().is_empty());
    }

    #[test]
    fn sqrt_emits_and_is_effective_zero_forks() {
        crate::engine::reset_for_test();
        crate::engine::ENGINE.with(|c| c.borrow_mut().start_path(vec![]));

        let x = Sym::input(0);
        let _s = x.sqrt_explicit(); // -> Instr::Sqrt(0, Input0)
        let decided = x.is_effective_zero(); // first frontier fork -> true, records a Branch
        assert!(decided, "frontier fork takes the true arm first");

        let events = crate::engine::drain_events_for_test();
        assert!(matches!(events[0], Event::Instr(Instr::Sqrt(0, _))));
        assert!(matches!(events[1], Event::Branch(_, _, _)));
        // the flip (false) was enqueued for later exploration
        let pending_len = crate::engine::ENGINE.with(|c| c.borrow().pending.len());
        assert_eq!(pending_len, 1);
    }
}
