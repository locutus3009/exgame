// SPDX-License-Identifier: MIT

//! First-order forward-AD pushforward for constraint functions.
//!
//! Differentiates a user `Dynamics::eval` verbatim by instantiating it on
//! `S = Tangent<K, R>`: perturbs poses by the retraction `exp(δ)∘M` and
//! velocities additively `V + δ`, evaluates once, and projects the typed output
//! `[Wrench<S>; N]` back to `Wrench<R>` along the fiber axes. The tower never
//! escapes. 24-axis layout for two bodies: pose A 0..6, pose B 6..12,
//! vel A 12..18, vel B 18..24.

use super::super::{GradVec, Tangent};
use super::{Motor, Twist, Wrench};
use crate::Lift;
use peano::prelude::*;

/// The island's poses lifted onto the `GRAD`-axis tangent tower and seeded, one
/// per body. `GRAD` is the total axis count, `Mul<BODIES, N12>` at the call site.
type SeededPoses<BODIES, GRAD, R> = Vector<BODIES, Motor<Tangent<GRAD, R>>>;
/// The same for velocities.
type SeededVels<BODIES, GRAD, R> = Vector<BODIES, Twist<Tangent<GRAD, R>>>;
/// One body's row of the Jacobian: `∂(wrench on this body)/∂(axis k)` for every
/// one of the `GRAD` axes.
type WrenchRow<GRAD, R> = Vector<GRAD, Wrench<R>>;
/// [`Differential::jacobian`]'s answer: each body's wrench at the current state,
/// and each body's Jacobian row.
type ValueAndJacobian<BODIES, GRAD, R> = (
    Vector<BODIES, Wrench<R>>,
    Vector<BODIES, WrenchRow<GRAD, R>>,
);

/// Which fiber level to read when projecting a tower wrench down one level.
#[derive(Clone, Copy)]
enum ProjAxis {
    Value,
    Deriv(usize),
}

pub trait Dynamics<N: Nat, R: Scalar> {
    type Context;
    /// Poly-instantiated in the scalar S (= R, Tangent<N1,R>, Tangent<DOFS,R>).
    /// `R: Lift<S>` lets constants in `Context` lift into S as dual-free values.
    fn eval<S: Scalar + StandardPart>(
        &self,
        poses: &Vector<N, Motor<S>>,
        vels: &Vector<N, Twist<S>>,
        context: &Self::Context,
    ) -> Vector<N, Wrench<S>>
    where
        R: Lift<S>;
}

pub struct Differential<'a, BODIES: Nat, R: Scalar> {
    poses: &'a Vector<BODIES, &'a Motor<R>>,
    vels: &'a Vector<BODIES, &'a Twist<R>>,
}

impl<'a, BODIES: Storage + 'a, R: Scalar + StandardPart + FromRational>
    Differential<'a, BODIES, R>
{
    pub fn at(
        poses: &'a Vector<BODIES, &'a Motor<R>>,
        vels: &'a Vector<BODIES, &'a Twist<R>>,
    ) -> Self {
        Self { poses, vels }
    }

    /// Scalar seed: value 0, unit derivative in axis `axis`.
    fn seed<K: Storage>(axis: usize) -> Tangent<K, R>
    where
        Vector<K, R>: GradVec<R>,
    {
        let one_hot = K::from_fn(|i| if i == axis { R::ONE } else { R::ZERO });
        Tangent::from_grad(R::ZERO, one_hot)
    }

    /// Twist seed: unit duals in axes `offset..offset+6`, value 0.
    fn seed_twist<K: Storage>(offset: usize) -> Twist<Tangent<K, R>>
    where
        Vector<K, R>: GradVec<R>,
    {
        Twist::new(
            &vector![
                Self::seed(offset),
                Self::seed(offset + 1),
                Self::seed(offset + 2),
            ],
            &vector![
                Self::seed(offset + 3),
                Self::seed(offset + 4),
                Self::seed(offset + 5),
            ],
        )
    }

    /// Project one tower-wrench down one fiber level (value or a single axis).
    /// One-level `base()` / `component(k)` — NOT `standard_part` (which would
    /// descend to the bottom and drop the outer derivative on Hessian towers).
    fn project_wrench<K: Storage>(w: &Wrench<Tangent<K, R>>, axis: ProjAxis) -> Wrench<R>
    where
        Vector<K, R>: GradVec<R>,
    {
        let f = w.force();
        let t = w.torque();
        let pick = |x: Tangent<K, R>| -> R {
            match axis {
                ProjAxis::Value => x.base(),
                ProjAxis::Deriv(k) => x.component(k),
            }
        };
        Wrench::new(
            &vector![pick(f[0]), pick(f[1]), pick(f[2])],
            &vector![pick(t[0]), pick(t[1]), pick(t[2])],
        )
    }

    /// Value AND full Jacobian in one AD pass. `value[b]` is body b's wrench at
    /// the current state; `jac[b][k]` is ∂(wrench on b)/∂(axis k).
    pub fn jacobian<D>(
        &self,
        d: &D,
        context: &D::Context,
    ) -> ValueAndJacobian<BODIES, Mul<BODIES, N12>, R>
    where
        D: Dynamics<BODIES, R>,
        BODIES: PeanoMul<N12>,
        Mul<BODIES, N12>: Storage,
        Vector<Mul<BODIES, N12>, R>: GradVec<R>,
        R: Lift<Tangent<Mul<BODIES, N12>, R>>,
    {
        let m: SeededPoses<BODIES, Mul<BODIES, N12>, R> = BODIES::from_fn(|i| {
            BODIES::get(self.poses, i)
                .deepen()
                .compose(&Motor::exp(&Self::seed_twist::<Mul<BODIES, N12>>(i * 6)))
        });
        let v: SeededVels<BODIES, Mul<BODIES, N12>, R> = BODIES::from_fn(|i| {
            BODIES::get(self.vels, i).deepen()
                + Self::seed_twist::<Mul<BODIES, N12>>(<Mul<BODIES, N12>>::value() / 2 + i * 6)
        });
        let w = d.eval(&m, &v, context);
        let value: Vector<BODIES, Wrench<R>> =
            BODIES::from_fn(|b| Self::project_wrench(BODIES::get(&w, b), ProjAxis::Value));
        let jac: Vector<BODIES, WrenchRow<Mul<BODIES, N12>, R>> = BODIES::from_fn(|b| {
            <Mul<BODIES, N12>>::from_fn(|k| {
                Self::project_wrench(BODIES::get(&w, b), ProjAxis::Deriv(k))
            })
        });
        (value, jac)
    }

    /// Single-direction derivative: value and the directional derivative along
    /// `(pose_dir, vel_dir)` in one Jet<1> pass.
    pub fn directional_derivative<D>(
        &self,
        pose_dir: &Vector<BODIES, Twist<R>>,
        vel_dir: &Vector<BODIES, Twist<R>>,
        d: &D,
        context: &D::Context,
    ) -> (Vector<BODIES, Wrench<R>>, Vector<BODIES, Wrench<R>>)
    where
        D: Dynamics<BODIES, R>,
        R: Lift<Tangent<N1, R>>,
    {
        let eps = Self::seed::<N1>(0);
        let m: Vector<BODIES, Motor<Tangent<N1, R>>> = BODIES::from_fn(|i| {
            // tangent twist of body i: value 0, ε-part = pose_dir[i]
            let dir = BODIES::get(pose_dir, i).deepen::<Tangent<N1, R>>() * eps;
            BODIES::get(self.poses, i)
                .deepen()
                .compose(&Motor::exp(&dir))
        });
        let v: Vector<BODIES, Twist<Tangent<N1, R>>> = BODIES::from_fn(|i| {
            BODIES::get(self.vels, i).deepen::<Tangent<N1, R>>()
                + BODIES::get(vel_dir, i).deepen::<Tangent<N1, R>>() * eps
        });
        let w = d.eval(&m, &v, context);
        let value: Vector<BODIES, Wrench<R>> =
            BODIES::from_fn(|b| Self::project_wrench(BODIES::get(&w, b), ProjAxis::Value));
        let dir: Vector<BODIES, Wrench<R>> =
            BODIES::from_fn(|b| Self::project_wrench(BODIES::get(&w, b), ProjAxis::Deriv(0)));
        (value, dir)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Dof, Point};
    use super::*;

    /// Two-body coupling (production-shape), generic in S. Wrench-per-body; the
    /// force depends on world anchor positions of BOTH bodies + linear velocities,
    /// torque on angular velocities. Body B uses a − sign so off-diagonal
    /// body→body blocks are nonzero. Intentionally not frame-invariant.
    struct Toy;
    impl Dynamics<N2, f64> for Toy {
        type Context = ();
        fn eval<S: Scalar + StandardPart>(
            &self,
            p: &Vector<N2, Motor<S>>,
            v: &Vector<N2, Twist<S>>,
            _: &(),
        ) -> Vector<N2, Wrench<S>>
        where
            f64: Lift<S>,
        {
            let pa = p[0]
                .conjugate(&Point::new(Vector3::from([S::ONE, S::ZERO, S::ZERO])))
                .coords();
            let pb = p[1]
                .conjugate(&Point::new(Vector3::from([S::ZERO, S::ONE, S::ZERO])))
                .coords();
            let la = v[0].linear();
            let lb = v[1].linear();
            let aa = v[0].angular();
            let ab = v[1].angular();
            let wa = Wrench::new(
                &Vector3::from([
                    pa[0] + pb[0] + la[0] + lb[0],
                    pa[1] + pb[1] + la[1] + lb[1],
                    pa[2] + pb[2] + la[2] + lb[2],
                ]),
                &(aa + ab),
            );
            let wb = Wrench::new(
                &Vector3::from([
                    pa[0] - pb[0] + la[0] - lb[0],
                    pa[1] - pb[1] + la[1] - lb[1],
                    pa[2] - pb[2] + la[2] - lb[2],
                ]),
                &(aa - ab),
            );
            vector![wa, wb]
        }
    }

    fn close_w(a: &Wrench<f64>, b: &Wrench<f64>, eps: f64) -> bool {
        let (af, bf, at, bt) = (a.force(), b.force(), a.torque(), b.torque());
        (0..3).all(|i| (af[i] - bf[i]).abs() < eps && (at[i] - bt[i]).abs() < eps)
    }

    fn fd_w(plus: &Wrench<f64>, minus: &Wrench<f64>, h: f64) -> Wrench<f64> {
        let (pf, mf, pt, mt) = (plus.force(), minus.force(), plus.torque(), minus.torque());
        let d = |a: f64, b: f64| (a - b) / (2.0 * h);
        Wrench::new(
            &Vector3::from([d(pf[0], mf[0]), d(pf[1], mf[1]), d(pf[2], mf[2])]),
            &Vector3::from([d(pt[0], mt[0]), d(pt[1], mt[1]), d(pt[2], mt[2])]),
        )
    }

    fn base() -> ([Motor<f64>; 2], [Twist<f64>; 2]) {
        let ma = Twist::new(
            &Vector3::from([1.0, -2.0, 0.5]),
            &Vector3::from([0.3, 0.1, -0.2]),
        )
        .exp(1.0);
        let mb = Twist::new(
            &Vector3::from([-1.0, 0.5, 2.0]),
            &Vector3::from([-0.1, 0.4, 0.2]),
        )
        .exp(1.0);
        let va = Twist::new(
            &Vector3::from([0.2, -0.3, 0.5]),
            &Vector3::from([0.4, -0.1, 0.2]),
        );
        let vb = Twist::new(
            &Vector3::from([-0.5, 0.1, 0.3]),
            &Vector3::from([0.2, 0.3, -0.4]),
        );
        ([ma, mb], [va, vb])
    }

    #[test]
    fn jacobian_matches_finite_differences() {
        let (m, v) = base();
        let pr = vector![&m[0], &m[1]];
        let vr = vector![&v[0], &v[1]];
        let diff = Differential::<N2, f64>::at(&pr, &vr);
        let (_value, jac) = diff.jacobian(&Toy, &());

        let h = 1e-6;
        for k in 0..24 {
            let (wp, wm) = if k < 12 {
                let (body, dof) = (k / 6, Dof::ALL[k % 6]);
                let b = Twist::basis(dof);
                let mut mp = m;
                let mut mm = m;
                mp[body] = m[body].compose(&Motor::exp(&(b * h))); // world/left exp
                mm[body] = m[body].compose(&Motor::exp(&(b * (-h))));
                (
                    Toy.eval(&vector![mp[0], mp[1]], &vector![v[0], v[1]], &()),
                    Toy.eval(&vector![mm[0], mm[1]], &vector![v[0], v[1]], &()),
                )
            } else {
                let (body, dof) = ((k - 12) / 6, Dof::ALL[(k - 12) % 6]);
                let b = Twist::basis(dof);
                let mut vp = v;
                let mut vm = v;
                vp[body] = v[body] + b * h;
                vm[body] = v[body] + b * (-h);
                (
                    Toy.eval(&vector![m[0], m[1]], &vector![vp[0], vp[1]], &()),
                    Toy.eval(&vector![m[0], m[1]], &vector![vm[0], vm[1]], &()),
                )
            };
            for body in 0..2 {
                let want = fd_w(&wp[body], &wm[body], h);
                assert!(close_w(&jac[body][k], &want, 1e-6), "body {body}, axis {k}");
            }
        }
    }

    /// Trivial dynamics: force = linear velocity of the body.
    struct VelReadout;
    impl Dynamics<N1, f64> for VelReadout {
        type Context = ();
        fn eval<S: Scalar + StandardPart>(
            &self,
            _poses: &Vector<N1, Motor<S>>,
            vels: &Vector<N1, Twist<S>>,
            _: &Self::Context,
        ) -> Vector<N1, Wrench<S>>
        where
            f64: Lift<S>,
        {
            let v = vels[0].linear();
            vector![Wrench::new(&v, &Vector3::ZERO)]
        }
    }

    #[test]
    fn jacobian_velocity_readout_is_identity_block() {
        let pose = Motor::<f64>::identity();
        let vel = Twist::<f64>::zero();
        let poses = vector![&pose];
        let vels = vector![&vel];
        let diff = Differential::<N1, f64>::at(&poses, &vels);
        let (_value, jac) = diff.jacobian(&VelReadout, &());
        // velocity DOFs are axes 6..12; vx is axis 6 → unit force in x.
        assert!(
            (jac[0][6].force()[0] - 1.0).abs() < 1e-9,
            "{:?}",
            jac[0][6].force()
        );
        // a pose axis (0) produces no force from a pure velocity readout.
        assert!(jac[0][0].force()[0].abs() < 1e-9);
    }
}
