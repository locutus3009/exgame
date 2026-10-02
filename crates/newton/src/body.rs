// SPDX-License-Identifier: MIT

use crate::Inertia;
use aristotle::{World, WorldKey};
use bytemuck::Pod;
use clifford::pga3::{Motor, Point, Twist, Wrench};
use peano::prelude::*;
use std::sync::Arc;

// ============================================================================
// RIGID BODY — state + INSTANTANEOUS dynamics (no time step)
// ============================================================================
//
// The body knows its velocity, gyroscopic term and energy at the current instant.
// Advancing in time is NOT its concern: that is done by the Integrator (integrator.rs).
// Hence RigidBody does not require transcendental scalar functions — sin/cos are needed
// only for exp, and exp lives in the integrator. This is exactly the layer-cut boundary.

/// One contribution to a body's gather: the joint value slot + which of the two halves flows into
/// this body (`0` = the wrench on end `a` of the joint, `1` = on end `b`). Everything is a `WorldKey`
/// (Arc handle, `'static`) + an index, so the list is fully owned and does not drag
/// a lifetime along.
#[derive(Debug, Clone)]
pub struct GatherTerm<T: Scalar + Pod> {
    pub key: WorldKey<[Wrench<T>; 2]>,
    pub slot: u8,
}

/// Rigid-body state: pose (motor), body-frame momentum (co-screw),
/// inertial properties.
#[derive(Debug)]
pub struct RigidBody<T: Scalar + Pod> {
    pub world: Arc<World>,
    pub pose: WorldKey<Motor<T>>,
    pub momentum: WorldKey<Wrench<T>>,
    pub inertia: Inertia<T>,
    pub effective_size: WorldKey<T>,
    world_momentum: WorldKey<Wrench<T>>,
    world_velocity: WorldKey<Twist<T>>,
    /// Buffer of EXTERNAL forces (gravity, fields) — phase 4 of the mechanism writes here, this is
    /// the SEED of the gather. Lives in World storage (like pose/momentum), zeroed at
    /// the start of each step. The island holds a CLONE of this key in its step-time map.
    pub accum_wrench: WorldKey<Wrench<T>>,
    /// Per-body OUTPUT of the gather: total world wrench = external + Σ joint forces at
    /// the current iterate. `enqueue` accumulates here; the integrator reads it when assembling
    /// the residual / in an explicit step. A CPU stand-in for the future GPU gather (ACCELERATOR §I.6).
    pub total_wrench: WorldKey<Wrench<T>>,
    /// Long-lived incidence cache: the joint contributions summed into
    /// `total_wrench` (gather phase 3). Rebuilt ONLY on a topology change of the
    /// island (`Island::recompute_incidence`) via `&mut` — not during the step. This is
    /// the structural side of the cut: the step writes data slots (WorldKey) via `&`, while
    /// this cache changes only under the write lock. Cheap to clone (Arc bump).
    pub incident_terms: Arc<[GatherTerm<T>]>,
    /// Baked gather incidence rows — one per round (a long list of
    /// joints is cut by row capacity). Rebuilt together with
    /// `incident_terms`, by the same trigger: a row addresses slots by index, and
    /// it can only stay in sync with them if it is baked alongside.
    pub gather_rows: Arc<[WorldKey<crate::accelerator::row::Row>]>,
    /// Kernel PRE input: midpoint pose `snap ∘ exp(½dt·vmid)` (implicit) / current pose
    /// (explicit). The integrator writes into PRE, `enqueue` reads it as `base`.
    pub midpoint_pose: WorldKey<Motor<T>>,
    /// Kernel PRE input: solver velocity — vmid (implicit) / world velocity (explicit).
    pub solve_vel: WorldKey<Twist<T>>,
}

impl<T> RigidBody<T>
where
    T: Scalar + StandardPart + Pod,
{
    fn get_effective_size(inertia: &Inertia<T>) -> T {
        let three = T::ONE + T::ONE + T::ONE;
        let four = three + T::ONE;
        let mass = inertia.mass();

        // PI is ~3, density is ~1
        (mass / four).powf_explicit(T::ONE / three)
    }

    /// A body at the origin, at rest.
    #[inline]
    pub fn new(world: Arc<World>, inertia: Inertia<T>) -> Self {
        let effective_size = Self::get_effective_size(&inertia);

        let binding = world.clone();
        let mut map_pose = binding.write();
        let mut map_momentum = binding.write();
        let mut map_size = binding.write();
        let mut map_vel = binding.write();
        Self {
            world,
            pose: map_pose.add(Motor::identity()),
            momentum: map_momentum.add(Wrench::zero()),
            inertia,
            effective_size: map_size.add(effective_size),
            world_momentum: map_momentum.add(Wrench::zero()),
            world_velocity: map_vel.add(Twist::new(&Vector3::ZERO, &Vector3::ZERO)),
            accum_wrench: map_momentum.add(Wrench::zero()),
            total_wrench: map_momentum.add(Wrench::zero()),
            // A fresh body is a singleton with no joints: empty incidence cache.
            incident_terms: Arc::from(Vec::<GatherTerm<T>>::new()),
            gather_rows: Arc::from(Vec::new()),
            midpoint_pose: map_pose.add(Motor::identity()),
            solve_vel: map_vel.add(Twist::new(&Vector3::ZERO, &Vector3::ZERO)),
        }
    }

    pub fn with_size(self, effective_size: T) -> Self {
        self.effective_size.write(effective_size);
        self
    }

    pub fn body_at_with_mass(world: Arc<World>, x: &Vector3<T>, mass: T) -> RigidBody<T> {
        Self::body_at_with_speed_and_mass(world, x, &Vector3::ZERO, mass)
    }

    pub fn body_at_with_speed_and_mass(
        world: Arc<World>,
        x: &Vector3<T>,
        v: &Vector3<T>,
        mass: T,
    ) -> RigidBody<T> {
        let b = RigidBody::new(world.clone(), Inertia::isotropic(world, mass, T::ONE));

        {
            b.pose.write(Twist::new(x, &Vector3::ZERO).exp(T::ONE));
            // The pose is a pure translation (identity rotation): the body-frame linear
            // velocity coincides with the world-frame one, and the body-frame momentum = inertia · twist(v, 0).
            let v_twist = Twist::new(v, &Vector3::ZERO);
            b.momentum.write(b.inertia.apply(&v_twist));
        }

        b
    }
}

impl<T> RigidBody<T>
where
    T: Scalar + StandardPart + Pod,
{
    /// Velocity (twist) from momentum: V = I⁻¹·P. BODY frame.
    #[inline]
    pub fn velocity(&self) -> Twist<T> {
        self.inertia.apply_inverse(&self.momentum.read())
    }

    /// World (spatial) twist of the body: `Ad_M · V_body` (= `M·V·M̃`).
    /// Symmetric to `world_momentum`; in this form the joints receive the velocity already
    /// in the WORLD frame, and the Ad conversion body→world lives here, at the boundary.
    #[inline]
    pub fn world_velocity(&self) -> WorldKey<Twist<T>> {
        self.world_velocity
            .write(self.pose.read().conjugate(&self.velocity()));
        self.world_velocity.clone()
    }

    /// The same slot WITHOUT recomputing it — for baking a row that will address
    /// it later. `world_velocity()` is a recompute-and-hand-back; a baker wants
    /// the address, not the value, and taking the value here would write a state
    /// nobody asked for.
    #[inline]
    pub fn world_velocity_slot(&self) -> WorldKey<Twist<T>> {
        self.world_velocity.clone()
    }

    #[inline]
    pub fn position(&self) -> Vector3<T> {
        self.pose
            .read()
            .conjugate(&Point::new(Vector3::ZERO))
            .coords()
    }

    /// Kinetic energy E = ½⟨P, V⟩ (the pairing via the dual — accounts for both the
    /// linear and the angular part).
    #[inline]
    pub fn kinetic_energy(&self) -> T {
        let half = T::ONE / (T::ONE + T::ONE);
        half * self.momentum.read().power(&self.velocity())
    }

    /// Gyroscopic term Ṗ_gyro = −ad*_V P — the COADJOINT bracket (the dual
    /// form, the same as Co::transport). The sign is pinned by the precession oracle
    /// (`free_precession_world_momentum_converges`): Euler L̇ = L×Ω = −ad*_Ω L
    /// under the (−) pairing convention (⟨ad*_t w, s⟩ = −⟨w,`[t,s]`⟩). On a purely angular
    /// momentum it coincides with the old flat ½`[P,V]` (which is why that one survived);
    /// on a MIXED one it diverges, and the duality corrects the old form
    /// (finding A3). It carries no power, it sets the precession.
    #[inline]
    pub fn gyroscopic(&self) -> Wrench<T> {
        -self.momentum.read().coad_bracket(&self.velocity())
    }

    /// Momentum in the world frame (conserved for a free body). A wrench is a
    /// co-screw, so body→world goes via the COADJOINT `Ad*_M = transport(M) under the Co law`, not
    /// via the sandwich `M·P·M̃` (that one is for twists): under rotation with angular momentum they diverge.
    #[inline]
    pub fn world_momentum(&self) -> WorldKey<Wrench<T>> {
        self.world_momentum
            .write(self.momentum.read().transport(&self.pose.read()));
        self.world_momentum.clone()
    }
}

// ============================================================================
// Tests (instantaneous dynamics; integration is tested in integrator.rs)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Gyro of purely linear motion = 0 (the commutator of translational bivectors
    /// is zero: e0i·e0j contains e0² = 0).
    #[test]
    fn gyroscopic_of_pure_linear_is_zero() {
        let world = Arc::new(World::builder().usual::<f32>());
        let inertia = Inertia::diagonal(world.clone(), 2.0, [1.0, 1.0, 1.0]);
        let body = RigidBody::new(world, inertia.clone());
        body.momentum.write(inertia.apply(&Twist::new(
            &Vector3::from([3.0, -1.0, 2.0]),
            &Vector3::from([0.0, 0.0, 0.0]),
        )));

        let g = body.gyroscopic();
        for c in g.force().split().iter().chain(g.torque().split().iter()) {
            assert!(approx(*c, 0.0, 1e-12));
        }
    }

    /// With an identity pose the world momentum coincides with the body-frame one.
    #[test]
    fn world_momentum_at_identity_equals_body() {
        let world = Arc::new(World::builder().usual::<f32>());
        let inertia = Inertia::diagonal(world.clone(), 1.0f32, [2.0, 3.0, 4.0]);
        let body = RigidBody::new(world, inertia);
        body.momentum.write(Wrench::new(
            &Vector3::from([1.0f32, 2.0, 3.0]),
            &Vector3::from([4.0, 5.0, 6.0]),
        ));

        let w = body.world_momentum();
        assert_eq!(w.read().force().split(), [1.0, 2.0, 3.0]);
        assert_eq!(w.read().torque().split(), [4.0, 5.0, 6.0]);
    }

    // ========================================================================
    // body_at_with_speed_and_mass
    // ========================================================================

    use clifford::pga3::Point;

    fn approx_vec(a: Vector3<f32>, b: Vector3<f32>, eps: f32) -> bool {
        a.split()
            .iter()
            .zip(b.split().iter())
            .all(|(x, y)| (x - y).abs() < eps)
    }

    fn position_of(b: &RigidBody<f32>) -> Vector3<f32> {
        b.pose.read().conjugate(&Point::new(Vector3::ZERO)).coords()
    }

    #[test]
    fn body_at_with_speed_and_mass_position_is_x() {
        let b = RigidBody::body_at_with_speed_and_mass(
            Arc::new(World::builder().usual::<f32>()),
            &Vector3::from([3.0, -2.0, 1.5]),
            &Vector3::ZERO,
            4.0,
        );
        let p = position_of(&b);
        assert!(
            approx_vec(p, Vector3::from([3.0, -2.0, 1.5]), 1e-12),
            "p = {p:?}"
        );
    }

    #[test]
    fn body_at_with_speed_and_mass_zero_velocity_has_zero_momentum() {
        let b = RigidBody::body_at_with_speed_and_mass(
            Arc::new(World::builder().usual::<f32>()),
            &Vector3::ZERO,
            &Vector3::ZERO,
            2.0f32,
        );
        assert_eq!(b.momentum.read(), Wrench::zero());
    }

    #[test]
    fn body_at_with_speed_and_mass_recovers_velocity() {
        let v = Vector3::from([0.5, -1.0, 2.5]);
        let b = RigidBody::body_at_with_speed_and_mass(
            Arc::new(World::builder().usual::<f32>()),
            &Vector3::ZERO,
            &v,
            3.0,
        );
        let v_recovered = b.velocity().linear();
        assert!(approx_vec(v_recovered, v, 1e-12), "got {v_recovered:?}");
    }

    #[test]
    fn body_at_with_speed_and_mass_world_momentum_is_mass_times_velocity() {
        let v = Vector3::from([1.0, 2.0, -3.0]);
        let m = 5.0;
        let b = RigidBody::body_at_with_speed_and_mass(
            Arc::new(World::builder().usual::<f32>()),
            &Vector3::from([7.0, 0.0, 0.0]),
            &v,
            m,
        );
        let f = b.world_momentum().read().force();
        let expected = v.scale(m);
        assert!(
            approx_vec(f, expected, 1e-12),
            "got {f:?}, expected {expected:?}"
        );
    }

    #[test]
    fn body_at_with_speed_and_mass_mass_is_set() {
        let b = RigidBody::body_at_with_speed_and_mass(
            Arc::new(World::builder().usual::<f32>()),
            &Vector3::ZERO,
            &Vector3::ZERO,
            7.5,
        );
        assert!(approx(b.inertia.mass(), 7.5, 1e-12));
    }
}

#[cfg(test)]
mod tests_with_joints {
    use super::*;
    use crate::RigidBody;
    use crate::{
        Inertia, Integrator,
        integrator::{ExplicitEuler, SymplecticEuler},
    };
    use aristotle::{Epoch, World};
    use clifford::pga3::Twist;
    use joints::{
        AxialSpringDamper, CriticallyDampedWarped, JointImpl, PerpendicularDamperWarped,
        SimpleSpringDamper, TorsionalDamperWarped, anchor_point,
    };
    use std::f32::consts::FRAC_PI_2;
    use std::sync::Arc;

    fn placed_at(world: Arc<World>, m: Inertia<f32>, x: f32, y: f32, z: f32) -> RigidBody<f32> {
        let body = RigidBody::new(world, m);
        body.pose
            .write(Twist::new(&Vector3::from([x, y, z]), &Vector3::ZERO).exp(1.0));
        body
    }

    fn separation(a: &RigidBody<f32>, b: &RigidBody<f32>) -> f32 {
        // anchors at the body centres for these tests.
        let pa = anchor_point(&a.pose.read(), &Vector3::ZERO);
        let pb = anchor_point(&b.pose.read(), &Vector3::ZERO);
        pa.join(&pb).weight_norm()
    }

    fn crit(world: Arc<World>, rest: f32, k: f32) -> AxialSpringDamper<f32> {
        AxialSpringDamper::builder(world, CriticallyDampedWarped)
            .rest(rest)
            .stiffness(k)
            .build_raw()
    }

    /// A stretched critically-damped spring must PULL the bodies together
    /// (separation decreases), not push them apart. Inverted sign ⇒ "flies away".
    #[test]
    fn stretched_critically_damped_spring_pulls_inward() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        let b = placed_at(world.clone(), m, 4.0, 0.0, 0.0); // rest = 2 ⇒ stretched
        let joint = crit(world, 2.0, 10.0);

        let d0 = separation(&a, &b);
        let integ = SymplecticEuler;
        let dt = 0.01;
        for _ in 0..50 {
            let epoch = Epoch::standalone(dt, 1.0);
            let [wa, wb] = joint
                .wrench(
                    &vector![a.pose.read(), b.pose.read()],
                    &vector![a.world_velocity().read(), b.world_velocity().read()],
                    &epoch,
                )
                .split();
            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);
        }

        assert!(
            separation(&a, &b) < d0,
            "stretched spring must pull inward: {d0} → {}",
            separation(&a, &b)
        );
    }

    /// Critically damped ⇒ settles to rest length without flying away.
    #[test]
    fn critically_damped_spring_settles_to_rest_length() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        let b = placed_at(world.clone(), m, 4.0, 0.0, 0.0);
        let joint = crit(world, 2.0, 10.0);

        let integ = SymplecticEuler;
        let dt = 0.01;
        for _ in 0..4000 {
            let epoch = Epoch::standalone(dt, 1.0);
            let [wa, wb] = joint
                .wrench(
                    &vector![a.pose.read(), b.pose.read()],
                    &vector![a.world_velocity().read(), b.world_velocity().read()],
                    &epoch,
                )
                .split();
            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);
        }

        let d = separation(&a, &b);
        assert!(
            (d - 2.0).abs() < 0.05,
            "should settle near rest length 2, got {d}"
        );
    }

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Spatial (world) twist of a body: Ad_pose(body_twist). Used to evaluate the
    /// physical power a world-frame wrench delivers to the body.
    fn spatial_twist(b: &RigidBody<f32>) -> Twist<f32> {
        b.pose.read().conjugate(&b.velocity())
    }

    /// The damping must use the TRUE closing rate of the anchor points. At rest
    /// length the conservative term vanishes, so the instantaneous power the
    /// joint delivers to the pair is exactly `−c·ḋ²`, where `ḋ` is the closing
    /// rate computed from the WORLD (spatial) twists. A body that is both offset
    /// from the origin and rotating has a body-frame twist that differs from its
    /// spatial twist; feeding the former to `unit.power` gets `ḋ` wrong (wrong
    /// magnitude, sign-flippable), so the damper pumps. This pins the frame down.
    #[test]
    fn damped_offset_spring_dissipates_true_closing_rate() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);

        // B at the origin, at rest.
        let b = RigidBody::new(world.clone(), m.clone());

        // A translated to [0, 2, 0] and SPINNING about +Z. (Even a pure-
        // translation offset makes the spatial twist differ from the body twist.)
        let a = RigidBody::new(world.clone(), m.clone());
        a.pose
            .write(Twist::new(&Vector3::from([0.0, 2.0, 0.0]), &Vector3::ZERO).exp(1.0));
        a.momentum
            .write(m.apply(&Twist::new(&Vector3::ZERO, &Vector3::from([0.0, 0.0, 1.0]))));

        // anchor_a local [1,0,0] → world [1,2,0]; anchor_b at origin ⇒ d = √5.
        // rest = √5 ⇒ conservative term is zero, leaving pure damping power.
        let damping = 3.0;
        let rest = 5.0_f32.sqrt();
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::from([1.0, 0.0, 0.0]))
            .b(Vector3::ZERO)
            .rest(rest)
            .stiffness(10.0)
            .damping(damping)
            .build_raw();

        // True closing rate from the spatial twists, and the dissipation it implies.
        let pa = anchor_point(&a.pose.read(), &joint.a());
        let pb = anchor_point(&b.pose.read(), &joint.b());
        let unit = Wrench::from_line(&pa.join(&pb)) * (1.0 / pa.join(&pb).weight_norm());
        let d_dot_true = unit.power(&(spatial_twist(&a) - spatial_twist(&b)));
        assert!(
            d_dot_true.abs() > 1e-6,
            "test must exercise a nonzero closing rate"
        );
        let expected_power = -damping * d_dot_true * d_dot_true;

        let [wa, wb] = joint
            .wrench(
                &vector![a.pose.read(), b.pose.read()],
                &vector![a.world_velocity().read(), b.world_velocity().read()],
                &Epoch::standalone(0.01, 1.0),
            )
            .split();
        let power = wa.power(&spatial_twist(&a)) + wb.power(&spatial_twist(&b));
        assert!(
            (power - expected_power).abs() < 1e-5,
            "damping power {power} ≠ −c·ḋ² = {expected_power} (closing rate in wrong frame)"
        );
    }

    #[test]
    fn damped_offset_spring_dissipates_true_closing_rate_for_jacobian() {
        use clifford::Jet12;

        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);

        // B at the origin, at rest.
        let b = RigidBody::new(world.clone(), m.clone());

        // A translated to [0, 2, 0] and SPINNING about +Z. (Even a pure-
        // translation offset makes the spatial twist differ from the body twist.)
        let a = RigidBody::new(world.clone(), m.clone());
        a.pose
            .write(Twist::new(&Vector3::from([0.0, 2.0, 0.0]), &Vector3::ZERO).exp(1.0));
        a.momentum
            .write(m.apply(&Twist::new(&Vector3::ZERO, &Vector3::from([0.0, 0.0, 1.0]))));

        // anchor_a local [1,0,0] → world [1,2,0]; anchor_b at origin ⇒ d = √5.
        // rest = √5 ⇒ conservative term is zero, leaving pure damping power.
        let damping = 3.0;
        let rest = 5.0_f32.sqrt();
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::from([1.0, 0.0, 0.0]))
            .b(Vector3::ZERO)
            .rest(rest)
            .stiffness(10.0)
            .damping(damping)
            .build_raw();

        // True closing rate from the spatial twists, and the dissipation it implies.
        let pa = anchor_point(&a.pose.read(), &joint.a());
        let pb = anchor_point(&b.pose.read(), &joint.b());
        let unit = Wrench::from_line(&pa.join(&pb)) * (1.0 / pa.join(&pb).weight_norm());
        let d_dot_true = unit.power(&(spatial_twist(&a) - spatial_twist(&b)));
        assert!(
            d_dot_true.abs() > 1e-6,
            "test must exercise a nonzero closing rate"
        );
        let expected_power = -damping * d_dot_true * d_dot_true;

        // deepen() is useful for PGA3 constants such as positions, lines, planes.
        // for true differentiation we need to provide correct DOF projections
        // for both motor and twist spaces
        let a_pose: Motor<Jet12<f32>> = a.pose.read().deepen();
        let b_pose = b.pose.read().deepen();
        let a_velocity = a.world_velocity().read().deepen();
        let b_velocity = b.world_velocity().read().deepen();

        // Calculated on lifted scalar T->Jet<12, T>!
        let [wa, wb] = joint
            .wrench(
                &vector![a_pose, b_pose],
                &vector![a_velocity, b_velocity],
                &Epoch::standalone(0.01, 1.0),
            )
            .split();
        // w[0] will be w(x) itself
        // w[i] where i>0  will be derivatives of w(x) in respect to all degrees of freedom

        // Take the value (standard part) of the Jet<12, T> result; the gradient
        // components carry the derivatives, which this test does not assert on.
        let power =
            (wa.power(&spatial_twist(&a).deepen()) + wb.power(&spatial_twist(&b).deepen())).base();
        assert!(
            (power - expected_power).abs() < 1e-5,
            "damping power {power:?} ≠ −c·ḋ² = {expected_power:?} (closing rate in wrong frame)"
        );
    }

    // Distance between anchors — dogfoods the new geometric API.
    fn separation2(a: &RigidBody<f32>, j: &AxialSpringDamper<f32>, b: &RigidBody<f32>) -> f32 {
        let pa = anchor_point(&a.pose.read(), &j.a());
        let pb = anchor_point(&b.pose.read(), &j.b());
        pa.join(&pb).weight_norm()
    }

    /// Anchors at distance L₀ at rest → zero force.
    #[test]
    fn at_rest_length_no_force() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone()); // at the origin
        let b = placed_at(world.clone(), m, 2.0, 0.0, 0.0);
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::ZERO)
            .b(Vector3::ZERO)
            .rest(2.0)
            .stiffness(10.0)
            .damping(1.0f32)
            .build_raw();

        let [wa, wb] = joint
            .wrench(
                &vector![a.pose.read(), b.pose.read()],
                &vector![a.world_velocity().read(), b.world_velocity().read()],
                &Epoch::standalone(0.1, 0.1),
            )
            .split();
        // Soft-norm gives d_soft = √(L₀²+ε²) ≈ L₀ + ε²/2L₀, so at rest
        // a regularization force residual of order k·ε²/2L₀ remains (~1e-12 at
        // ε=1e-6). This is a deliberate softening, not a bug — the tolerance is relaxed to 1e-5.
        for c in wa.force().split().iter().chain(wb.force().split().iter()) {
            assert!(
                approx(*c, 0.0, 1e-5),
                "force at rest ~zero (up to the ε² residual)"
            );
        }
    }

    /// Newton's third law: with anchors at the CoM the motion is purely translational,
    /// the total world linear momentum is conserved (we start from rest → 0).
    #[test]
    fn conserves_total_linear_momentum() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        let b = placed_at(world.clone(), m, 4.0, 0.0, 0.0); // stretched (rest=2)
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::ZERO)
            .b(Vector3::ZERO)
            .rest(2.0)
            .stiffness(8.0)
            .damping(0.5f32)
            .build_raw();

        let integ = SymplecticEuler;
        let dt = 0.005;
        for _ in 0..2000 {
            let epoch = Epoch::standalone(dt, 1.0);
            let [wa, wb] = joint
                .wrench(
                    &vector![a.pose.read(), b.pose.read()],
                    &vector![a.world_velocity().read(), b.world_velocity().read()],
                    &epoch,
                )
                .split();
            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);

            let total = a.world_momentum().read().force() + b.world_momentum().read().force();
            assert!(total.dot(total) < 1e-12, "Σ linear momentum ≠ 0: {total:?}");
        }
    }

    /// A damped spring settles to its rest length.
    #[test]
    fn damped_spring_settles_to_rest_length() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        let b = placed_at(world.clone(), m, 4.0, 0.0, 0.0); // stretched by 2
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::ZERO)
            .b(Vector3::ZERO)
            .rest(2.0)
            .stiffness(10.0)
            .damping(4.0f32)
            .build_raw();

        let integ = SymplecticEuler;
        let dt = 0.005;
        for _ in 0..4000 {
            let epoch = Epoch::standalone(dt, 1.0);
            // t = 20
            let [wa, wb] = joint
                .wrench(
                    &vector![a.pose.read(), b.pose.read()],
                    &vector![a.world_velocity().read(), b.world_velocity().read()],
                    &epoch,
                )
                .split();
            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);
        }

        assert!(
            approx(separation2(&a, &joint, &b), 2.0, 0.05),
            "settled at {}, expected ≈2",
            separation2(&a, &joint, &b)
        );
    }

    /// A stretched spring (without a damper) first pulls the bodies together — the distance
    /// decreases. This also confirms the direction of the force (attraction).
    #[test]
    fn stretched_spring_pulls_inward() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        let b = placed_at(world.clone(), m, 4.0, 0.0, 0.0);
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::ZERO)
            .b(Vector3::ZERO)
            .rest(2.0)
            .stiffness(10.0)
            .damping(0.0f32)
            .build_raw();

        let d0 = separation2(&a, &joint, &b);
        let integ = SymplecticEuler;
        let dt = 0.005;
        for _ in 0..100 {
            let epoch = Epoch::standalone(dt, 1.0);

            // a short interval before a possible overshoot
            let [wa, wb] = joint
                .wrench(
                    &vector![a.pose.read(), b.pose.read()],
                    &vector![a.world_velocity().read(), b.world_velocity().read()],
                    &epoch,
                )
                .split();
            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);
        }
        assert!(
            separation2(&a, &joint, &b) < d0,
            "a stretched spring must contract"
        );
    }

    /// A force on an anchor offset from the CoM creates a torque → the body starts to rotate.
    /// The torque now comes from the Plücker coordinates of the line of action (the dual),
    /// not from a manual p×F; having passed the pullback into body.velocity(), it gives a non-zero ω.
    #[test]
    fn offset_anchor_induces_rotation() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        let b = placed_at(world.clone(), m, 4.0, 0.0, 0.0);
        // anchor A is offset along Y → stretching along X gives a torque about Z
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::from([0.0, 1.0, 0.0]))
            .b(Vector3::ZERO)
            .rest(1.0f32)
            .stiffness(10.0)
            .damping(0.0)
            .build_raw();

        let integ = SymplecticEuler;
        let dt = 0.005;
        for _ in 0..50 {
            let epoch = Epoch::standalone(dt, 1.0);
            let [wa, wb] = joint
                .wrench(
                    &vector![a.pose.read(), b.pose.read()],
                    &vector![a.world_velocity().read(), b.world_velocity().read()],
                    &epoch,
                )
                .split();
            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);
        }

        let omega = a.velocity().angular();
        assert!(
            omega.dot(omega) > 1e-6,
            "an offset anchor must spin the body"
        );
    }

    // Total energy of a pair of bodies with a joint: kinetic energy of both + spring potential.
    fn total_energy(a: &RigidBody<f32>, j: &AxialSpringDamper<f32>, b: &RigidBody<f32>) -> f32 {
        a.kinetic_energy() + b.kinetic_energy() + j.potential_energy(&a.pose.read(), &b.pose.read())
    }

    /// CONSERVATIVE INVARIANT: a spring without a damper (c=0). The total energy
    /// (kinetic + potential) does NOT drift over a long coarse run under
    /// SymplecticEuler — it oscillates within a bounded band around E₀.
    #[test]
    fn undamped_spring_conserves_total_energy() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        let b = placed_at(world.clone(), m, 3.5, 0.0, 0.0); // stretched (rest=2)
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::ZERO)
            .b(Vector3::ZERO)
            .rest(2.0)
            .stiffness(6.0)
            .damping(0.0f32)
            .build_raw();

        let e0 = total_energy(&a, &joint, &b);
        let integ = SymplecticEuler;
        let dt = 0.005;
        let mut e_min = e0;
        let mut e_max = e0;
        for _ in 0..8000 {
            let epoch = Epoch::standalone(dt, 1.0);

            // t = 40 — a long run
            let [wa, wb] = joint
                .wrench(
                    &vector![a.pose.read(), b.pose.read()],
                    &vector![a.world_velocity().read(), b.world_velocity().read()],
                    &epoch,
                )
                .split();
            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);
            let e = total_energy(&a, &joint, &b);
            e_min = e_min.min(e);
            e_max = e_max.max(e);
        }

        let band = (e_max - e_min) / e0;
        assert!(
            band < 0.05,
            "energy must stay within a narrow band: band = {band}"
        );
        let e_end = total_energy(&a, &joint, &b);
        assert!(
            (e_end - e0).abs() / e0 < 0.05,
            "E: {e0} → {e_end} (no drift)"
        );
    }

    /// DISSIPATIVE MONOTONIC: with a damper (c>0) the total energy decreases
    /// MONOTONICALLY and tends to the rest energy.
    #[test]
    fn damped_spring_dissipates_monotonically() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        let b = placed_at(world.clone(), m, 3.5, 0.0, 0.0);
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::ZERO)
            .b(Vector3::ZERO)
            .rest(2.0f32)
            .stiffness(6.0)
            .damping(2.0)
            .build_raw();

        let integ = SymplecticEuler;
        let dt = 0.002; // finer: monotonicity is sensitive to the step
        let mut e_prev = total_energy(&a, &joint, &b);
        let e0 = e_prev;

        for _ in 0..5000 {
            let epoch = Epoch::standalone(dt, 1.0);

            // t = 10
            let [wa, wb] = joint
                .wrench(
                    &vector![a.pose.read(), b.pose.read()],
                    &vector![a.world_velocity().read(), b.world_velocity().read()],
                    &epoch,
                )
                .split();
            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);
            let e = total_energy(&a, &joint, &b);
            assert!(e <= e_prev + 1e-6, "energy jumped up: {e_prev} → {e}");
            e_prev = e;
        }

        let e_end = total_energy(&a, &joint, &b);
        assert!(
            e_end < e0 * 0.5,
            "the damper must dissipate noticeably: {e0} → {e_end}"
        );
        assert!(e_end >= 0.0, "energy is non-negative: {e_end}");
    }

    /// ExplicitEuler on the same conservative spring drifts NOTICEABLY in
    /// energy — the remaining difference between the schemes after the flip (both hold momentum to machine precision,
    /// but the explicit one pumps energy in under force). The reference contrast.
    #[test]
    fn explicit_euler_drifts_energy_on_spring() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        let b = placed_at(world.clone(), m, 3.5, 0.0, 0.0);
        let joint = AxialSpringDamper::builder(world, SimpleSpringDamper)
            .a(Vector3::ZERO)
            .b(Vector3::ZERO)
            .rest(2.0f32)
            .stiffness(6.0)
            .damping(0.0)
            .build_raw();

        let e0 = total_energy(&a, &joint, &b);
        let integ = ExplicitEuler;
        let dt = 0.005;
        let mut e_max: f32 = e0;
        for _ in 0..8000 {
            let epoch = Epoch::standalone(dt, 1.0);

            let [wa, wb] = joint
                .wrench(
                    &vector![a.pose.read(), b.pose.read()],
                    &vector![a.world_velocity().read(), b.world_velocity().read()],
                    &epoch,
                )
                .split();
            integ.step(&a, wa, &epoch);
            integ.step(&b, wb, &epoch);
            e_max = e_max.max(total_energy(&a, &joint, &b));
        }
        assert!(
            (e_max - e0) / e0 > 0.05,
            "explicit Euler must pump energy noticeably more than the symplectic one"
        );
    }

    fn approx3(a: Vector3<f32>, b: Vector3<f32>, eps: f32) -> bool {
        a.split()
            .iter()
            .zip(b.split().iter())
            .all(|(x, y)| (x - y).abs() < eps)
    }

    /// Brakes the TRANSVERSE relative velocity, computed in the WORLD frame.
    /// A sits at the origin but is ROTATED 90° about +Z with a body-frame linear
    /// velocity +X̂ ⇒ its WORLD velocity is +Ŷ. The line of sight is along X
    /// (B at [10,0,0]), so +Ŷ is purely transverse ⇒ force = −b·[0,1,0] =
    /// [0,−3,0]. A body-frame implementation would see +X̂ (radial) and brake
    /// nothing — this pins the world-frame `twist_under` path.
    #[test]
    fn brakes_transverse_velocity_in_world_frame() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());

        a.pose
            .write(Twist::new(&Vector3::ZERO, &Vector3::from([0.0, 0.0, FRAC_PI_2])).exp(1.0));
        a.momentum
            .write(m.apply(&Twist::new(&Vector3::from([1.0, 0.0, 0.0]), &Vector3::ZERO)));
        let b = RigidBody::new(world.clone(), m);
        b.pose
            .write(Twist::new(&Vector3::from([10.0, 0.0, 0.0]), &Vector3::ZERO).exp(1.0));

        let damping = 3.0;
        let joint = PerpendicularDamperWarped::new_raw(world, damping, 1e-6);
        let [wa, wb] = joint
            .wrench(
                &vector![a.pose.read(), b.pose.read()],
                &vector![a.world_velocity().read(), b.world_velocity().read()],
                &Epoch::standalone(0.01, 1.0),
            )
            .split();

        assert!(
            approx3(wa.force(), Vector3::from([0.0, -3.0, 0.0]), 1e-5),
            "transverse brake force wrong: {:?}",
            wa.force()
        );
        assert!(
            approx3(wb.force(), Vector3::from([0.0, 3.0, 0.0]), 1e-5),
            "third law: {:?}",
            wb.force()
        );
    }

    /// Leaves RADIAL relative velocity untouched (that is the axial spring's job).
    /// Line along X, A moving +X̂ (world) ⇒ zero transverse force.
    #[test]
    fn ignores_radial_velocity() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());

        a.momentum
            .write(m.apply(&Twist::new(&Vector3::from([1.0, 0.0, 0.0]), &Vector3::ZERO)));
        let b = RigidBody::new(world.clone(), m);
        b.pose
            .write(Twist::new(&Vector3::from([10.0, 0.0, 0.0]), &Vector3::ZERO).exp(1.0));

        let joint = PerpendicularDamperWarped::new_raw(world, 3.0, 1e-6);
        let [wa, _wb] = joint
            .wrench(
                &vector![a.pose.read(), b.pose.read()],
                &vector![a.world_velocity().read(), b.world_velocity().read()],
                &Epoch::standalone(0.01, 1.0),
            )
            .split();
        assert!(
            approx3(wa.force(), Vector3::ZERO, 1e-5),
            "radial motion must not be braked: {:?}",
            wa.force()
        );
    }

    /// Dissipative as a pair even when bodies are offset from the origin AND
    /// rotating: power = −b·|v_perp|² ≤ 0.
    #[test]
    fn dissipative_under_rotation_offset_from_origin() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);
        let a = RigidBody::new(world.clone(), m.clone());
        a.pose.write(
            Twist::new(
                &Vector3::from([0.0, 3.0, 0.0]),
                &Vector3::from([0.0, 0.0, FRAC_PI_2]),
            )
            .exp(1.0),
        );
        a.momentum.write(m.apply(&Twist::new(
            &Vector3::from([0.4, 0.1, 0.0]),
            &Vector3::from([0.0, 0.0, 1.0]),
        )));
        let b = RigidBody::new(world.clone(), m.clone());
        b.pose
            .write(Twist::new(&Vector3::from([5.0, 0.0, 0.0]), &Vector3::ZERO).exp(1.0));
        b.momentum.write(m.apply(&Twist::new(
            &Vector3::from([0.0, -0.2, 0.0]),
            &Vector3::from([0.0, 1.0, 0.0]),
        )));

        let joint = PerpendicularDamperWarped::new_raw(world, 4.0, 1e-6);
        let [wa, wb] = joint
            .wrench(
                &vector![a.pose.read(), b.pose.read()],
                &vector![a.world_velocity().read(), b.world_velocity().read()],
                &Epoch::standalone(0.01, 1.0),
            )
            .split();
        let power = wa.power(&spatial_twist(&a)) + wb.power(&spatial_twist(&b));
        assert!(
            power <= 1e-5,
            "perpendicular damper pumped: power = {power}"
        );
    }

    /// A pure couple opposing the RELATIVE angular velocity (world frame): zero
    /// force, third-law paired, and the power it delivers is exactly −b·|ω_rel|².
    /// Body A is ROTATED (so body-frame ≠ spatial-frame), pinning the world-frame
    /// requirement: a body-frame `ω` would mis-orient the couple and break the
    /// power identity.
    #[test]
    fn pure_couple_dissipates_relative_angular_velocity() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m = Inertia::isotropic(world.clone(), 1.0, 1.0);

        // A: rotated 90° about +Z, spinning about BODY +X ⇒ world ω = +Y.
        let a = RigidBody::new(world.clone(), m.clone());
        a.pose
            .write(Twist::new(&Vector3::ZERO, &Vector3::from([0.0, 0.0, FRAC_PI_2])).exp(1.0));
        a.momentum
            .write(m.apply(&Twist::new(&Vector3::ZERO, &Vector3::from([1.0, 0.0, 0.0]))));
        // B: identity, at rest.
        let b = RigidBody::new(world.clone(), m);

        let damping = 3.0;
        let joint = TorsionalDamperWarped::new(world, damping);
        let [wa, wb] = joint
            .wrench(
                &vector![a.pose.read(), b.pose.read()],
                &vector![a.world_velocity().read(), b.world_velocity().read()],
                &Epoch::standalone(0.01, 1.0),
            )
            .split();

        // Pure couple ⇒ no force; third law ⇒ wb = −wa.
        assert!(
            approx3(wa.force(), Vector3::ZERO, 1e-12),
            "couple must have zero force, got {:?}",
            wa.force()
        );
        assert!(
            approx3(wb.torque(), -wa.torque(), 1e-12),
            "third law broken: {:?} vs −{:?}",
            wb.torque(),
            wa.torque()
        );

        // Dissipation must equal −b·|ω_rel|², with ω_rel from WORLD twists.
        let w_rel = (spatial_twist(&a) - spatial_twist(&b)).angular();
        let expected = -damping * (w_rel[0] * w_rel[0] + w_rel[1] * w_rel[1] + w_rel[2] * w_rel[2]);
        assert!(expected.abs() > 1e-6, "test must exercise a nonzero ω_rel");
        let power = wa.power(&spatial_twist(&a)) + wb.power(&spatial_twist(&b));
        assert!(
            (power - expected).abs() < 1e-5,
            "power {power} ≠ −b·|ω_rel|² = {expected}"
        );
    }
}
