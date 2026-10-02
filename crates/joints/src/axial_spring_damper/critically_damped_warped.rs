// SPDX-License-Identifier: MIT

use super::{AxialSpringDamper, AxialSpringDamperType, AxialSpringForce};
use crate::{Joint, JointFromParams};
use aristotle::{Epoch, World};
use bytemuck::Pod;
use clifford::Lift;
use peano::prelude::*;
use std::sync::Arc;

#[derive(Debug)]
pub struct CriticallyDampedWarped;

impl<T> JointFromParams<T> for CriticallyDampedWarped
where
    T: Scalar + StandardPart + Pod,
{
    fn shader_name(&self) -> &'static str {
        "CriticallyDampedWarped"
    }
    fn n_params(&self) -> usize {
        10
    }
    fn build_from_params(&self, world: Arc<World>, params: &[T]) -> Joint<T> {
        AxialSpringDamper::<T>::builder(world.clone(), Self)
            .a(Vector3::from([params[0], params[1], params[2]]))
            .b(Vector3::from([params[3], params[4], params[5]]))
            .rest(params[6])
            .stiffness(params[7])
            .damping(params[8])
            .softening(params[9])
            .build()
    }
}

impl CriticallyDampedWarped {
    /// Critically-damped Padé step for the camera spring. Variables are in the
    /// body's coordinate (radial distance / radial velocity).
    ///
    /// The formula here is the static-target form, derived from implicit Euler
    /// on `ẍ + 2ω₀·ẋ + ω₀²(x − r) = 0` with the identity
    /// `(1 + 2ω₀dt + ω₀²dt²) = (1 + ω₀dt)²`.
    ///
    /// MOVING-TARGET HANDLING. The caller (`linear_wrench`) damps the *relative*
    /// closing rate `unit·(va − vb)`, and `CameraFieldArc::accumulate` drives the
    /// intermediate through the predictive point `λ = 2·p_interm − p_real`;
    /// together these cancel the cascade drag-lag without the absolute-damping
    /// `c0_min` floor the previous production path needed. See the spec
    /// `docs/architecture/records/2026-06-01-camera-spring-pga-echo-design.md`.
    ///
    /// SIGN CONVENTION. `AxialSpringForce::force` is TENSION-POSITIVE — the same
    /// convention as the trait default `k·(d−rest) + c·v` — because
    /// `AxialSpringDamper::wrenches` applies `wa = unit·(−f)`. So a stretched
    /// spring (`x > rest`) must return a POSITIVE force. The old camera path used
    /// the opposite (acceleration `ẍ = −(…)`) convention with `unit·f`; porting it
    /// here flips the leading sign, hence the `+(…)` below.
    #[inline]
    fn calculate<T: Scalar>(&self, c0: T, x: T, v: T, rest: T, dt: T) -> T {
        //        let omega_0 = (c0 / self.m).sqrt_explicit();
        //        let omega_02 = omega_0.powi_explicit(2);

        let omega_0 = c0.sqrt_explicit();
        let omega_02 = c0;

        // With Pade approximation
        (omega_02 * (x - rest) + (T::ONE + T::ONE) * omega_0 * v + omega_02 * dt * v)
            / (T::ONE + omega_0 * dt).powi_explicit(2)
    }
}

impl<T: Scalar> AxialSpringForce<T> for CriticallyDampedWarped {
    fn force<S>(
        &self,
        stiffness: S,
        distance: S,
        rest_length: S,
        _damping: S,
        velocity: S,
        epoch: &Epoch<T>,
    ) -> S
    where
        S: Scalar + StandardPart,
        T: Lift<S>,
    {
        // TODO: optimization, don't convert here, store values somewhere
        let dt: S = (*epoch.dt()).lift();
        let warp: S = (*epoch.warp()).lift();
        let c0 = stiffness / warp.powi_explicit(2);
        self.calculate(c0, distance, velocity, rest_length, dt)
    }

    fn into_enum(self) -> AxialSpringDamperType {
        AxialSpringDamperType::CriticallyDampedWarped(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::axial_spring_damper::SimpleSpringDamper;
    use aristotle::Epoch;

    /// CONTRACT: `AxialSpringForce::force` is tension-positive — the same sign
    /// convention `AxialSpringDamper::wrenches` relies on (`wa = unit·(−f)`),
    /// as fixed by the trait default `k·(d−rest) + c·v`. A STRETCHED spring
    /// (d > rest) must report a POSITIVE force, identical in sign to
    /// `SimpleSpringDamper`. The critically-damped variant must not invert it.
    #[test]
    fn force_sign_matches_simple_spring_when_stretched() {
        // dt = 0 ⇒ Padé denominator = 1 and the velocity terms vanish, isolating
        // the restoring term's sign. warp = 1 ⇒ c0 = stiffness.
        let epoch = Epoch::standalone(0.0, 1.0);
        let f_simple = SimpleSpringDamper.force(10.0, 4.0, 2.0, 0.0, 0.0, &epoch);
        let f_crit = CriticallyDampedWarped.force(10.0, 4.0, 2.0, 0.0, 0.0, &epoch);

        assert!(f_simple > 0.0, "sanity: simple spring is tension-positive");
        assert!(
            f_crit > 0.0,
            "critically-damped force must be tension-positive too, got {f_crit}"
        );
    }
}
