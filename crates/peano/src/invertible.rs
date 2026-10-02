// SPDX-License-Identifier: MIT

use crate::prelude::{EffectiveZero, Ring};
use core::ops::Div;
use num_traits::Float;

// ── INVERTIBILITY — a separate, stronger rung, NOT a property of the monoid ─────
/// Division. Partial (try_recip → Option), because even in a field 0 is not invertible,
/// and in Z/n the non-coprime elements are not invertible. THIS, not Monoid, bounds inversion.
pub trait Invertible: Ring {
    fn try_recip(self) -> Option<Self>;
}

impl<T> Invertible for T
where
    T: Ring + Div<Output = Self> + EffectiveZero + Float,
{
    fn try_recip(self) -> Option<Self> {
        if !self.is_effective_zero() {
            Some(Self::ONE / self)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn try_recip_f32_threshold() {
        // Category-(2) EffectiveZero: try_recip answers None at the carrier's effective
        // zero (f32 threshold = 1e-5, see support.rs).
        assert!((1e-3_f32).try_recip().is_some()); // above 1e-5
        assert!((1e-6_f32).try_recip().is_none()); // below 1e-5 → not invertible
    }
}
