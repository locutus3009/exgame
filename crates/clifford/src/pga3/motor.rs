// SPDX-License-Identifier: MIT

//! Motor — an element of the even subalgebra (versor), a structurally narrow carrier
//! (EvenStore: odd strata are Void, 8 components). The Study forms exp/log work
//! directly on the narrow carrier.

use super::{Conjugatable, Twist};
use crate::algebra::mv::Mv;
use crate::algebra::mv::{
    line_direction_store, line_moment_store, make_force_store, translation_store,
};
use crate::algebra::store::{GList, GradeMap};
use crate::algebra::store::{gp_store, scale_store, sign_store, store_basis, store_get};
use crate::algebra::strata::{EvenStore, MaskStrataG};
use crate::algebra::{Algebra, Pga3};
use bytemuck::{Pod, Zeroable};
use peano::prelude::*;

type M<S> = Mv<Pga3, S>;
type V3<S> = Vector<Succ<Succ<Succ<Z>>>, S>;

// ── Motor — AN ELEMENT OF THE EVEN SUBALGEBRA (versor), structurally narrow ──────
// A motor is an even multivector (grades 0,2,4): it has NO odd grades,
// so its storage is EvenStore (odd strata are Void), STRUCTURALLY 8 components,
// not 16. Operations go through the full Mv (motors also multiply odd
// objects), but the motor ITSELF stores only the even part.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Motor<S: Ring>(<Pga3 as EvenCarrier>::St<S>);

// The narrow carrier of the even section (the general MaskStrataG<EvenGrades>): the scalar is a
// parameter of the GAT, the promises ride on the MaskStrataG GAT — the former bundle is not needed.
use crate::algebra::section::EvenGrades;
pub trait EvenCarrier: Algebra {
    type St<S: Ring>: GList<S> + AbelianGroup + ScalarMul<S> + GradeMap;
}
impl<A> EvenCarrier for A
where
    A: Algebra + MaskStrataG<EvenGrades, A::NgenP>,
{
    type St<S: Ring> = EvenStore<A, S>;
}

// EvenStore is Pod ⇒ Motor is Pod: a newtype over a Pod carrier, `#[repr(C)]` = its own
// layout with no holes. `'static` is a Pod requirement.
unsafe impl<S: Ring> Zeroable for Motor<S> where <Pga3 as EvenCarrier>::St<S>: Zeroable {}
unsafe impl<S: Ring + 'static> Pod for Motor<S> where <Pga3 as EvenCarrier>::St<S>: Pod {}

// A motor over f32 — the even section, 8 packed slots (grades 0,2,4).
const _: () = peano::prelude::assert_packed::<Motor<f32>, f32>(8);

/// Embed a grade-2 carrier into the even one: a copy of the six bivector slots.
/// The zero grades (0, 4) are not touched at all — no arithmetic on zeros.
fn even_from_grade2<S: Ring>(g2: &impl GList<S>) -> <Pga3 as EvenCarrier>::St<S> {
    let n = Pga3::NGEN;
    let mut st = <<Pga3 as EvenCarrier>::St<S> as AbelianGroup>::ZERO;
    let mut b = 0;
    while b < Pga3::DIM {
        if b.count_ones() == 2 {
            st.add_at(b, n, n, g2.get(b, n, n));
        }
        b += 1;
    }
    st
}

impl<S: Ring> core::fmt::Debug for Motor<S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Motor({:?})", self.0)
    }
}
impl<S: Ring> PartialEq for Motor<S>
where
    <Pga3 as EvenCarrier>::St<S>: PartialEq,
{
    fn eq(&self, o: &Self) -> bool {
        self.0 == o.0
    }
}

impl<S: Ring> Motor<S> {
    /// Narrow a full Mv down to its even part (odd grades are discarded).
    pub fn from_mv(m: &M<S>) -> Self {
        let mut st = <<Pga3 as EvenCarrier>::St<S> as AbelianGroup>::ZERO;
        let n = Pga3::NGEN;
        let mut b = 0;
        while b < Pga3::DIM {
            if (b.count_ones() as usize) & 1 == 0 {
                st.add_at(b, n, n, m.get(b));
            }
            b += 1;
        }
        Motor(st)
    }
    /// Widen to a full Mv (odd blades are zero).
    pub fn as_mv(&self) -> M<S> {
        let mut m = M::zero();
        let n = Pga3::NGEN;
        let mut b = 0;
        while b < Pga3::DIM {
            if (b.count_ones() as usize) & 1 == 0 {
                m.add_at(b, self.0.get(b, n, n));
            }
            b += 1;
        }
        m
    }
    /// The identity motor = scalar 1 (grade-0, even) — directly on the carrier.
    pub fn identity() -> Self {
        Motor(store_basis::<Pga3, S, _>(0, S::ONE))
    }
    /// Decompose a motor into the 8 raw components of the even carrier — a direct traversal
    /// of the blades with EVEN population (grade 0,2,4), ordered by increasing blade.
    /// «Honest» (de)serialization: an array slot ↔ exactly one storage blade,
    /// without reindexing; when translated into a shader, the blade arithmetic folds
    /// into a bare access by variable index. Paired with `from_components` — the roundtrip
    /// is exact. Needed at the GPU boundary: the host lays out the f32 pose components into a buffer,
    /// and the kernel trace assembles `Motor<Sym>` from the inputs in THE SAME order.
    pub fn to_components(&self) -> [S; 8] {
        let n = Pga3::NGEN;
        let mut out = [S::ZERO; 8];
        let mut i = 0;
        let mut b = 0;
        while b < Pga3::DIM {
            if (b.count_ones() as usize) & 1 == 0 {
                out[i] = self.0.get(b, n, n);
                i += 1;
            }
            b += 1;
        }
        out
    }
    /// Assemble a motor from 8 raw components (the inverse of `to_components`, the same
    /// blade order). The odd strata of the carrier are Void; they are not here.
    pub fn from_components(c: &[S; 8]) -> Self {
        let mut st = <<Pga3 as EvenCarrier>::St<S> as AbelianGroup>::ZERO;
        let n = Pga3::NGEN;
        let mut i = 0;
        let mut b = 0;
        while b < Pga3::DIM {
            if (b.count_ones() as usize) & 1 == 0 {
                st.add_at(b, n, n, c[i]);
                i += 1;
            }
            b += 1;
        }
        Motor(st)
    }
    /// Composition of ACTIONS «first self, then rhs» (pipeline order);
    /// algebraically rhs·self: the sandwich (rhs·self)·X·(rhs·self)̃ applies self
    /// first. Cheat sheet for integrators: the body-frame increment
    /// `pose_new = pose·δ` is written `δ.compose(&pose)`; the world/left
    /// `pose_new = exp·pose` is written `pose.compose(&exp)`. The order is pinned by
    /// newton's world-momentum oracles. CLOSED in the even subalgebra
    /// (even·even = even) — directly on the narrow carrier, without the full Mv.
    pub fn compose(&self, rhs: &Self) -> Self {
        Motor(gp_store::<Pga3, S, _>(&rhs.0, &self.0))
    }
    /// Inverse = reverse: a sign by grade, it does NOT change the population ⇒ even
    /// stays even. Directly on the carrier.
    pub fn inverse(&self) -> Self {
        Motor(sign_store::<Pga3, S, _>(&self.0, |g| {
            (g.wrapping_mul(g.wrapping_sub(1)) / 2) & 1 == 1
        }))
    }
    /// The sandwich M·X·M̃ moves any (including an ODD) object — the result
    /// is odd and does NOT fit into the even carrier, so lifting into the full Mv here
    /// is structurally REQUIRED and honest (case A, the only remaining lift).
    pub fn conjugate_mv(&self, x: &M<S>) -> M<S> {
        self.as_mv().conjugate(x)
    }
    /// The sandwich M·X·M̃ over any Conjugatable object (pure gp — Ring).
    pub fn conjugate<T: Conjugatable<S>>(&self, x: &T) -> T {
        T::from_mv(self.conjugate_mv(&x.to_mv()))
    }
    pub fn deepen<T: Ring>(&self) -> Motor<T>
    where
        S: crate::Lift<T>,
    {
        Motor::from_mv(&self.as_mv().deepen())
    }
}

impl<S: Scalar + StandardPart> Motor<S> {
    /// Closed-form SCREW exponential of a bivector B via Study numbers —
    /// EMERGENTLY EVEN: every factor is even (B is grade-2, I is grade-4,
    /// b² is grade-0⊕4), gp_store is closed, so the whole Study form lives ON
    /// the narrow even carrier. Evenness here is the TYPE of the result, not an after-the-fact
    /// check; not a single as_mv. The input B is a bivector (even ⇒ from_mv is exact).
    /// Integrate a twist generator into a motor (the public form of the facade).
    /// grade-2 ⊂ even: the embedding is a direct copy of the six bivector slots
    /// (even_from_grade2), WITHOUT a detour through the full 16-slot Mv (widen→narrow
    /// would run arithmetic over the zero grades).
    pub fn exp(g: &Twist<S>) -> Self {
        Self::exp_even(even_from_grade2(g.store()))
    }

    /// Boundary form: the bivector is given as a full Mv (tests, log roundtrip).
    pub fn exp_bivector(b: &M<S>) -> Self {
        Self::exp_even(Motor::from_mv(b).0)
    }

    fn exp_even(bs: <Pga3 as EvenCarrier>::St<S>) -> Self {
        let n = Pga3::NGEN;
        let top = Pga3::DIM - 1;
        let half = S::get_constant(RationalConstant::HALF);
        let b2 = gp_store::<Pga3, S, _>(&bs, &bs); // grade-0 ⊕ grade-4
        let u = -store_get(&b2, 0, n);
        let p = store_get(&b2, top, n);
        let (cu, su, du) = (u.cos_sq(), u.sinc_sq(), u.dsinc_sq());
        let i_mv = store_basis::<Pga3, S, _>(top, S::ONE); // pseudoscalar (grade-4)
        let ib = gp_store::<Pga3, S, _>(&i_mv, &bs);
        type St<S> = <Pga3 as EvenCarrier>::St<S>;
        let scalar_part = store_basis::<Pga3, S, St<S>>(0, cu)
            + scale_store::<Pga3, S, St<S>>(&i_mv, p * half * su);
        let bivector_part =
            scale_store::<Pga3, S, St<S>>(&bs, su) - scale_store::<Pga3, S, St<S>>(&ib, p * du);
        Motor(scalar_part + bivector_part)
    }

    /// Versor normalization — ⟨M M̃⟩₀ and division, DIRECTLY on the carrier (closed).
    /// The logarithm as a twist (the inverse of exp; a pure generator, without ½).
    pub fn log(&self) -> Twist<S> {
        Twist::from_mv(&self.log_bivector())
    }

    pub fn normalize(&mut self) {
        *self = self.normalized();
    }

    pub fn normalized(&self) -> Self {
        type St<S> = <Pga3 as EvenCarrier>::St<S>;
        let rev = sign_store::<Pga3, S, St<S>>(&self.0, |g| {
            (g.wrapping_mul(g.wrapping_sub(1)) / 2) & 1 == 1
        });
        let n2 = store_get(&gp_store::<Pga3, S, St<S>>(&self.0, &rev), 0, Pga3::NGEN);
        if n2.standard_part().is_effective_zero() {
            return *self;
        }
        let inv = n2.sqrt_explicit().try_recip().unwrap();
        Motor(scale_store::<Pga3, S, St<S>>(&self.0, inv))
    }
    /// The rotational part: scalar + Euclidean bivector, normalized. On the carrier.
    pub fn rotation_part(&self) -> Self {
        type St<S> = <Pga3 as EvenCarrier>::St<S>;
        let scalar = store_basis::<Pga3, S, St<S>>(0, store_get(&self.0, 0, Pga3::NGEN));
        let euclid: St<S> = make_force_store(&line_direction_store(&self.0));
        Motor(scalar + euclid).normalized()
    }

    /// Logarithm: the bivector B with exp(B)=self. CLOSED (a motor is even, a bivector
    /// grade-2 is even) ⇒ the whole reconstruction is on the narrow carrier, without as_mv.
    /// Returns the bivector as a full Mv at the boundary (for exp/tests).
    pub fn log_bivector(&self) -> M<S> {
        type St<S> = <Pga3 as EvenCarrier>::St<S>;
        let n = Pga3::NGEN;
        let top = Pga3::DIM - 1;
        let a = store_get(&self.0, 0, n); // ⟨M⟩₀ = cos l
        let dir = line_direction_store(&self.0); // Euclidean bivector as a Vec3
        let s2 = dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2];
        let u = a.half_angle_sq(s2);
        let (sn, ds) = (u.sinc_sq(), u.dsinc_sq());
        let inv_sn = S::ONE / sn;
        // I·B_eucl on the carrier (I is grade-4, B_eucl is grade-2 ⇒ even).
        let b_eucl: St<S> = scale_store::<Pga3, S, St<S>>(&make_force_store(&dir), inv_sn);
        let i_mv = store_basis::<Pga3, S, St<S>>(top, S::ONE);
        let ib = gp_store::<Pga3, S, St<S>>(&i_mv, &b_eucl);
        let p = (store_get(&self.0, top, n) + store_get(&self.0, top, n)) * inv_sn;
        let ideal_m = line_moment_store(&self.0);
        let ib_moment = line_moment_store(&ib);
        let mut ideal = V3::<S>::ZERO;
        let mut i = 0;
        while i < 3 {
            let val = (ideal_m[i] + p * ds * ib_moment[i]) * inv_sn;
            ideal[i] = val;
            i += 1;
        }
        let angular = dir.scale(inv_sn);
        // The result is a bivector (grade-2, even): assemble it on the carrier, hand out an Mv.
        let biv: St<S> = translation_store::<S, St<S>>(&ideal) + make_force_store(&angular);
        Motor(biv).as_mv()
    }
}

// The se(3) twist lives in the `screw` module as Screw<Contra> over the shared Grade<2>
// carrier (together with Wrench = Screw<Co>): the difference is in the transformation law
// under a motor, the variance is in the type. It is no longer here — we do not duplicate it.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pga3::Twist;
    use num_traits::Float;

    #[test]
    fn motor_is_the_even_subalgebra_structurally() {
        use crate::algebra::Pga3;
        use crate::algebra::mv::Mv;
        use crate::algebra::section::{Grade, Or, section_dim};
        use crate::pga3::Motor;
        use core::mem::size_of;
        type Two = Succ<Succ<Z>>;
        type Four = Succ<Succ<Succ<Succ<Z>>>>;
        // The even subalgebra = Grade0 ∪ Grade2 ∪ Grade4 (via the Or combinators),
        // the dimension falls out by counting: 1+6+1 = 8, not 16.
        type Even = Or<Grade<Z>, Or<Grade<Two>, Grade<Four>>>;
        assert_eq!(section_dim::<Pga3, Even>(), 8);

        // Motor STRUCTURALLY stores 8 components (odd strata are Void, 0 bytes),
        // while the full multivector stores 16. A motor ≠ a multivector structurally.
        assert_eq!(size_of::<Motor<f64>>(), 8 * size_of::<f64>());
        assert_eq!(size_of::<Mv<Pga3, f64>>(), 16 * size_of::<f64>());

        // And this holds algebraically: the exp of a bivector is even — its odd part is zero.
        let mut biv = Mv::<Pga3, f64>::zero();
        biv.set(6, 0.5); // an arbitrary Euclidean bivector (grade 2)
        biv.set(9, 0.3); // and a radical one (screw)
        let m = Motor::exp_bivector(&biv).as_mv();
        assert_eq!(m.odd(), Mv::<Pga3, f64>::zero()); // there are no odd grades
        assert_eq!(m.even(), m); // a motor = its own even part
    }

    #[test]
    fn components_roundtrip_and_agree_with_mv() {
        // A screw motor: decomposing into 8 components and back is an exact roundtrip, and the same
        // assembly via as_mv (the full Mv) gives the same motor ⇒ the blade order of
        // to/from_components agrees with the even layout of Mv.
        let m = Motor::exp(&Twist::new(
            &vector![0.3f64, -0.2, 0.5],
            &vector![0.1, 0.4, -0.2],
        ));
        let comps = m.to_components();
        let back = Motor::from_components(&comps);
        assert_eq!(back, m, "from∘to != id");

        // Each component is an even blade of the full Mv, by increasing blade.
        let mv = m.as_mv();
        let mut i = 0;
        for b in 0..16 {
            if (b as u32).count_ones() & 1 == 0 {
                assert!((comps[i] - mv.get(b)).abs() < 1e-15, "blade {b} slot {i}");
                i += 1;
            }
        }
        assert_eq!(i, 8, "expected exactly 8 even components");
    }

    #[test]
    fn pga_log_exp_roundtrip() {
        use crate::pga3::Motor;
        use crate::pga3::Twist;
        let zero3: Vector<N3, f64> = vector![0.0, 0.0, 0.0];
        // Screw: rotation + translation. log(exp(B)) = B (for a unit versor).
        let g = Twist::new(&vector![0.3, -0.2, 0.5], &vector![0.1, 0.4, -0.2]);
        let b = g.as_mv();
        let m = Motor::exp_bivector(&b);
        let back = m.log_bivector();
        for s in 0..16 {
            assert!((back.get(s) - b.get(s)).abs() < 1e-9, "slot {s}");
        }
        // Motor·inverse = identity.
        let prod = m.compose(&m.inverse());
        assert!((prod.as_mv().get(0) - 1.0).abs() < 1e-9);
        let _ = zero3;
    }

    #[test]
    fn pga_rigid_motions() {
        use crate::pga3::Twist;
        use crate::pga3::{Motor, Point};
        fn close3(a: &Vector<N3, f64>, b: [f64; 3]) -> bool {
            (*a.get::<Z>() - b[0]).abs() < 1e-9
                && (*a.get::<Succ<Z>>() - b[1]).abs() < 1e-9
                && (*a.get::<Succ<Succ<Z>>>() - b[2]).abs() < 1e-9
        }
        let zero3: Vector<N3, f64> = vector![0.0, 0.0, 0.0];

        // The identity does not move a point.
        let p = Point::new(vector![1.0, 2.0, 3.0]);
        let idp = Point::from_mv(Motor::<f64>::identity().conjugate_mv(&p.as_mv()));
        assert!(close3(&idp.coords(), [1.0, 2.0, 3.0]));

        // Rotation by +90° about z: (1,0,0) → (0,1,0).
        let gz = Twist::new(&zero3, &vector![0.0, 0.0, core::f64::consts::FRAC_PI_2]);
        let mz = gz.exp(1.0);
        let rp = Point::from_mv(mz.conjugate_mv(&Point::new(vector![1.0, 0.0, 0.0]).as_mv()));
        assert!(
            close3(&rp.coords(), [0.0, 1.0, 0.0]),
            "rot: {:?}",
            rp.coords()
        );

        // A pure translation by (2,0,0) moves the origin to (2,0,0).
        let gt = Twist::new(&vector![2.0, 0.0, 0.0], &zero3);
        let mt = gt.exp(1.0);
        let tp = Point::from_mv(mt.conjugate_mv(&Point::new(zero3).as_mv()));
        assert!(
            close3(&tp.coords(), [2.0, 0.0, 0.0]),
            "trans: {:?}",
            tp.coords()
        );
    }

    #[test]
    fn ad_through_full_motor() {
        // AD THROUGH the whole SE(3) motor: rotating the point (1,0,0) about z by angle θ gives
        // (cos θ, sin θ, 0); d/dθ at θ=0 gives (0, 1, 0). The derivative flows through
        // twist_embed → Study-form exp → sandwich → point coordinates, with no special code.
        use crate::Jet1 as Jet;
        use crate::pga3::Point;
        use crate::pga3::Twist;
        type J = Jet<f64>;
        let c0 = J::embed(0.0);
        let th = J::from_grad(0.0, vector![1.0]); // θ = the variable

        let g = Twist::new(&vector![c0, c0, c0], &vector![c0, c0, th]); // ω_z = θ
        let m = g.exp(J::embed(1.0));
        let p = Point::new(vector![J::embed(1.0), c0, c0]);
        let out = Point::from_mv(m.conjugate_mv(&p.as_mv())).coords();

        let x = *out.get::<Z>();
        let y = *out.get::<Succ<Z>>();
        assert!((x.base() - 1.0).abs() < 1e-9); // cos0=1
        assert!(y.base().abs() < 1e-9); // sin0=0
        // ∂x/∂θ = −sin0 = 0, ∂y/∂θ = cos0 = 1.
        assert!(x.grad().get::<Z>().abs() < 1e-9);
        assert!((*y.grad().get::<Z>() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn even_closed_ops_stay_on_narrow_carrier() {
        use crate::algebra::Pga3;
        use crate::algebra::mv::Mv;
        use crate::pga3::Motor;
        use crate::pga3::Twist;
        // Two motors (screws).
        let a = Motor::exp(&Twist::new(
            &vector![0.1, -0.2, 0.3],
            &vector![0.4, 0.1, -0.2],
        ));
        let b = Motor::exp(&Twist::new(
            &vector![0.2, 0.1, -0.1],
            &vector![-0.3, 0.2, 0.1],
        ));

        // compose is CLOSED: the narrow gp_store == the wide gp with narrowing, componentwise.
        // The wide gp is the Cayley–Dickson recursion, the narrow one a blade loop: the order
        // of summation differs, so equality is WITH A TOLERANCE ~ε (as in
        // bracket_pure_grade2), not bitwise. Order: compose(a,b) = b·a
        // («a, then b» — a pipeline of actions).
        let narrow = a.compose(&b).as_mv();
        let wide = b.as_mv().gp(&a.as_mv());
        let mut blade = 0;
        while blade < 16 {
            assert!(
                (narrow.get(blade) - wide.get(blade)).abs() < 1e-12,
                "blade {blade}: {} vs {}",
                narrow.get(blade),
                wide.get(blade)
            );
            blade += 1;
        }

        // inverse on the carrier == reverse with narrowing (the signs are exact — bitwise).
        assert_eq!(a.inverse(), Motor::from_mv(&a.as_mv().reverse()));

        // M·M⁻¹ = identity (closed composition; numerically ⇒ with a tolerance).
        let prod = a.compose(&a.inverse()).as_mv();
        assert!((prod.get(0) - 1.0).abs() < 1e-9);
        for s in 1..16 {
            assert!(prod.get(s).abs() < 1e-9);
        }

        // exp is even BY CONSTRUCTION (not after the fact): its widening has no
        // odd grades — but the code never computed them either; the loop ran over the even carrier.
        assert_eq!(a.as_mv().odd(), Mv::<Pga3, f64>::zero());
    }

    #[test]
    fn point_transport_anchors_translator_sign_and_half() {
        // The sign and the ½ coefficient of the translator are fixed by GEOMETRY (the point moves to +r),
        // not by a desired sign of the moment. (The parallel-axis theorem itself for the wrench is proven on
        // the narrow crate::pga3::Wrench in wrench_parallel_axis_on_narrow_carrier.)
        use crate::algebra::Pga3;
        use crate::algebra::mv::{Mv, euclid_cross, origin, position};
        use crate::pga3::Twist;
        type P = Mv<Pga3, f64>;
        let zero3 = <Vector<N3, f64> as AbelianGroup>::ZERO;

        // Translator «shift by +r»: T = 1 + ½B, where B is the translational bivector
        // (= a twist with linear part r). T̃ = T.reverse().
        let translator = |r: &Vector<N3, f64>, c: f64| -> P {
            let b: P = Twist::new(r, &<Vector<N3, f64> as AbelianGroup>::ZERO).as_mv();
            P::scalar(1.0) + P::scalar(c).gp(&b)
        };
        let transport = |w: &P, r: &Vector<N3, f64>| -> P {
            let t = translator(r, 0.5);
            t.gp(w).gp(&t.reverse())
        };

        let r: Vector<N3, f64> = vector![1.0, 2.0, 3.0];
        let f: Vector<N3, f64> = vector![2.0, -1.0, 3.0];

        // (1) T moves the origin exactly to +r; an asymmetric r catches crossed axes.
        assert_eq!(position(&transport(&origin::<f64>(), &r)), r);
        // (2) position(origin) = 0 (zero e₀ part).
        assert_eq!(position(&origin::<f64>()), zero3);
        // (3) Weight ≠ 1: a doubled point gives THE SAME position — projective normalization.
        let pt2: P = P::scalar(2.0).gp(&transport(&origin::<f64>(), &r));
        assert_eq!(position(&pt2), r);
        // (4) The ½ coefficient is FORCED by the anchor: the sandwich gives a shift of 2c·r; ==r requires
        // c=½. c=⅓ breaks the anchor (the point moves to ⅔r ≠ r).
        let t_bad = translator(&r, 1.0 / 3.0);
        let moved_bad = t_bad.gp(&origin::<f64>()).gp(&t_bad.reverse());
        assert_ne!(position(&moved_bad), r);

        // ⋆₃ ↔ cross agree INSIDE Pga3.
        assert_eq!(euclid_cross(&r, &f), r.cross(f));
    }

    // ── Restored invariants of the legacy facade (A7): the motor group, ─────
    // f32 seams of log∘exp, AD towers through exp. Slots go through the derived markup.
    use crate::algebra::Pga3;
    use crate::algebra::markup::axis_vec;

    fn z_rotation_of_e1(omega_z: f64, dt: f64) -> (f64, f64) {
        // angle = ω_z·dt; returns the coefficients (e1, e2) after the sandwich.
        let zero3 = <Vector<N3, f64> as AbelianGroup>::ZERO;
        let g = Twist::new(&zero3, &vector![0.0, 0.0, omega_z]);
        let m = g.exp(dt);
        let mut e1 = crate::algebra::mv::Mv::<Pga3, f64>::zero();
        e1.set(axis_vec::<Pga3>(0), 1.0);
        let out = m.conjugate_mv(&e1);
        (out.get(axis_vec::<Pga3>(0)), out.get(axis_vec::<Pga3>(1)))
    }

    #[test]
    fn rotation_pi_is_180deg() {
        // ωz=π at dt=1 → 180° → e1 → −e1 (pins ½ away from the point π/2).
        let (e1, e2) = z_rotation_of_e1(core::f64::consts::PI, 1.0);
        assert!(
            (e1 - (-1.0)).abs() < 1e-9 && e2.abs() < 1e-9,
            "e1={e1} e2={e2}"
        );
    }

    #[test]
    fn dt_scales_angle_linearly() {
        // ωz=π/2 over dt=2 = 180° (angle = ω·dt) — dt is linear, ½ is not glued to π/2.
        let (e1, e2) = z_rotation_of_e1(core::f64::consts::FRAC_PI_2, 2.0);
        assert!(
            (e1 - (-1.0)).abs() < 1e-9 && e2.abs() < 1e-9,
            "e1={e1} e2={e2}"
        );
    }

    fn approx_twist(a: &Twist<f64>, b: &Twist<f64>) -> bool {
        let (al, bl) = (a.linear(), b.linear());
        let (aa, ba) = (a.angular(), b.angular());
        (0..3).all(|i| (al[i] - bl[i]).abs() < 1e-9 && (aa[i] - ba[i]).abs() < 1e-9)
    }

    #[test]
    fn roundtrip_large_angle() {
        // near-π rotation + translation: log∘exp = id (the atan2 branch).
        let b = Twist::new(&vector![0.5, -0.3, 0.2], &vector![2.5, 0.4, -0.6]);
        let m = Motor::exp(&b);
        let back = m.log();
        assert!(
            approx_twist(&back, &b),
            "{:?} vs {:?}",
            back.angular(),
            b.angular()
        );
    }

    #[test]
    fn log_exp_roundtrip_screw() {
        // roundtrip AT THE SCREW LEVEL (through the linear/angular readers, not slots).
        let b = Twist::new(&vector![0.3, -0.2, 0.1], &vector![0.4, 0.5, -0.6]);
        let m = Motor::exp(&b);
        let back = m.log();
        assert!(
            approx_twist(&back, &b),
            "{:?} vs {:?}",
            back.linear(),
            b.linear()
        );
    }

    #[test]
    fn rotor_is_unit() {
        let zero3 = <Vector<N3, f64> as AbelianGroup>::ZERO;
        let g = Twist::new(&zero3, &vector![0.3f64, -0.2, 0.5]);
        let m = Motor::exp(&g);
        let mm = m.as_mv().norm_squared();
        assert!((mm - 1.0).abs() < 1e-9, "⟨M M̃⟩₀ = {mm}");
    }

    #[test]
    fn log_of_identity_is_zero() {
        let z = Motor::<f64>::identity().log();
        assert!(approx_twist(&z, &Twist::zero()));
    }

    // ── f32: log∘exp through both arms of the seams ─────────────────────────────────
    fn approx_twist32(a: &Twist<f32>, b: &Twist<f32>, eps: f32) -> bool {
        let (al, bl) = (a.linear(), b.linear());
        let (aa, ba) = (a.angular(), b.angular());
        (0..3).all(|i| (al[i] - bl[i]).abs() < eps && (aa[i] - ba[i]).abs() < eps)
    }

    #[test]
    fn log_exp_roundtrip_f32_small_angle() {
        // a tiny twist — the Taylor arm of each helper; a shifted seam
        // would show up as roundtrip drift.
        let b = Twist::new(&vector![1e-3f32, -5e-4, 2e-4], &vector![3e-3, -1e-3, 2e-3]);
        let m = Motor::exp(&b);
        let back = m.log();
        assert!(
            approx_twist32(&back, &b, 1e-5),
            "{:?} vs {:?}",
            back.angular(),
            b.angular()
        );
    }

    #[test]
    fn log_exp_roundtrip_f32_large_angle() {
        let b = Twist::new(&vector![0.5f32, -0.3, 0.2], &vector![2.5, 0.4, -0.6]);
        let m = Motor::exp(&b);
        let back = m.log();
        assert!(
            approx_twist32(&back, &b, 1e-3),
            "{:?} vs {:?}",
            back.angular(),
            b.angular()
        );
    }

    #[test]
    fn rotor_is_unit_f32() {
        let zero3 = <Vector<N3, f32> as AbelianGroup>::ZERO;
        let g = Twist::new(&zero3, &vector![0.3f32, -0.2, 0.5]);
        let m = Motor::exp(&g);
        let mm = m.as_mv().norm_squared();
        assert!((mm - 1.0).abs() < 1e-6, "⟨M M̃⟩₀ = {mm}");
    }

    // ── AD towers through exp ────────────────────────────────────────────────
    #[test]
    fn exp_second_derivative_tower() {
        // exp on the Hessian tower Tangent<1, Tangent<1, f64>>: the scalar part
        // = cos(ωz/2); its 2nd derivative with respect to ωz is −¼cos(ωz/2). Read
        // via component(0).component(0) — a double single-level peel,
        // which standard_part would collapse. A regression guard for the ad projection.
        use crate::Tangent;
        type J2 = Tangent<N1, Tangent<N1, f64>>;
        let wz0 = core::f64::consts::FRAC_PI_3;
        let inner = Tangent::<N1, f64>::from_grad(wz0, vector![1.0]);
        let theta: J2 = Tangent::from_grad(inner, vector![Tangent::from_grad(1.0, vector![0.0])]);
        let zero3 = <Vector<N3, J2> as AbelianGroup>::ZERO;
        let g = Twist::<J2>::new(&zero3, &vector![J2::ZERO, J2::ZERO, theta]);
        let m = g.exp(J2::ONE);
        let scalar = m.as_mv().get(0); // cos(ωz/2)
        let value = scalar.base().base();
        let f2 = scalar.component(0).component(0);
        assert!((value - (wz0 / 2.0).cos()).abs() < 1e-9, "value {value}");
        assert!((f2 - (-0.25 * (wz0 / 2.0).cos())).abs() < 1e-9, "f''={f2}");
    }

    #[test]
    fn exp_jet_seed_is_one_plus_delta() {
        // Nilpotent seed δ (re=0, du=1) in the bivector e12: exp(δ) = 1 + δ
        // exactly, the derivative ∂exp/∂ε = δ is preserved (neither sin nor sqrt is called
        // at the hard zero of the standard part).
        use crate::Tangent;
        type J = Tangent<N1, f64>;
        let e12 = axis_vec::<Pga3>(0) | axis_vec::<Pga3>(1);
        let mut b = crate::algebra::mv::Mv::<Pga3, J>::zero();
        b.set(e12, J::from_grad(0.0, vector![1.0]));
        let m = Motor::exp(&Twist::from_mv(&b));
        let mv = m.as_mv();
        assert!(
            (mv.get(0).standard_part() - 1.0).abs() < 1e-15,
            "re(scalar)=1"
        );
        assert!(mv.get(e12).standard_part().abs() < 1e-15, "re(e12)=0");
        assert!(
            (mv.get(e12).component(0) - 1.0).abs() < 1e-15,
            "du(e12)=generator"
        );
        assert!(mv.get(0).component(0).abs() < 1e-15, "du(scalar)=0");
    }
}
