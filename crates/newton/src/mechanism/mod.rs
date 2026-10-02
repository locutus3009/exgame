// SPDX-License-Identifier: MIT

//! Mechanism: a multigraph of bodies orchestrating the step (pre-step → forces → integration),
//! with storage by connectivity islands. The types are split across files; here there is only
//! the module assembly, the re-export of the public API and integration tests of the whole
//! mechanism.

mod component;
mod force_field;
mod island;
mod islands;
// The file is named after its structure (Mechanism), like the other submodules; the type
// is re-exported, so the module sharing its parent's name is intentional here.
#[allow(clippy::module_inception)]
mod mechanism;

pub use component::{Component, Entity, Inert};
pub use force_field::ForceField;
pub use mechanism::{Mechanism, StructureError};

// ============================================================================
// Integration tests of the whole mechanism
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Accelerator, Inertia, RigidBody, integrator::ImplicitIntegrator};
    use async_trait::async_trait;

    // Each test builds its own accelerator (the single-GPU sharing is a
    // production concern; isolated tests just need a working one).
    fn accel(world: Arc<World>) -> Arc<Accelerator<f32>> {
        Arc::new(Accelerator::builder(world).build())
    }
    use aristotle::{Epoch, World, WorldId, WorldKey};
    use clifford::pga3::{Point, Twist, Wrench};
    use indexmap::IndexMap;
    use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
    use peano::prelude::*;
    use std::collections::HashSet;
    use std::sync::{Arc, RwLock};

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // X coordinate of the body centre (apply the pose to the origin).
    async fn pos(b: &RigidBody<f32>) -> f32 {
        b.pose.read().conjugate(&Point::new(Vector3::ZERO)).coords()[0]
    }

    // A spacecraft body placed at (x,0,0).
    fn body_at(world: Arc<World>, x: f32) -> RigidBody<f32> {
        body_at_with_mass(world, x, 1.0)
    }

    // A spacecraft body with a given mass at (x,0,0). For centroid tests, where mass
    // is the main observed parameter.
    fn body_at_with_mass(world: Arc<World>, x: f32, mass: f32) -> RigidBody<f32> {
        let b = RigidBody::new(world.clone(), Inertia::isotropic(world, mass, 1.0));
        b.pose
            .write(Twist::new(&Vector3::from([x, 0.0, 0.0]), &Vector3::ZERO).exp(1.0));
        b
    }

    fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
        AxialSpringDamper::builder(world, SimpleSpringDamper)
            .rest(rest)
            .stiffness(k)
            .damping(c)
            .build()
    }

    #[tokio::test]
    async fn add_connect_query() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));

        let id1 = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let id2 = m.add_body(Inert::new(body_at(world.clone(), 2.0))).await;
        m.connect(id1, vec![(spring(world, 2.0, 5.0, 0.0), id2)])
            .await;

        assert!(m.inspect_body(id1, async |_| {}).await.is_some());
        assert!(m.inspect_body(WorldId::get(), async |_| {}).await.is_none());
    }

    #[tokio::test]
    #[should_panic(expected = "duplicate world id")]
    async fn duplicate_world_id_panics() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let en1 = Inert::new(body_at(world.clone(), 0.0));
        let mut en2 = Inert::new(body_at(world.clone(), 1.0));
        en2.id = en1.id;

        m.add_body(en1).await;
        m.add_body(en2).await;
    }

    #[tokio::test]
    #[should_panic(expected = "self-loop forbidden")]
    async fn self_loop_panics() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));

        let id1 = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let world = Arc::new(World::builder().usual::<f32>());
        m.connect(id1, vec![(spring(world, 1.0, 1.0, 0.0), id1)])
            .await;
    }

    /// A damped spring inside a mechanism settles to its rest length — the same
    /// physics as in the joint.rs tests, but run through the step orchestration.
    #[tokio::test]
    async fn damped_pair_settles_via_mechanism() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));

        let id1 = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let id2 = m.add_body(Inert::new(body_at(world.clone(), 4.0))).await; // stretched (rest=2)
        m.connect(id1, vec![(spring(world, 2.0, 10.0, 4.0), id2)])
            .await;

        for _ in 0..4000 {
            m.step(&Epoch::standalone(0.005, 1.0)).await.unwrap(); // t = 20
        }

        // distance between the bodies ≈ rest_length
        let c1 = m
            .inspect_body(id1, async |x1| {
                // centre positions: apply the pose to the origin
                x1.pose
                    .read()
                    .conjugate(&Point::new(Vector3::ZERO))
                    .coords()
            })
            .await
            .unwrap();
        let c2 = m
            .inspect_body(id2, async |x2| {
                x2.pose
                    .read()
                    .conjugate(&Point::new(Vector3::ZERO))
                    .coords()
            })
            .await
            .unwrap();
        let dist =
            ((c2[0] - c1[0]).powi(2) + (c2[1] - c1[1]).powi(2) + (c2[2] - c1[2]).powi(2)).sqrt();
        assert!(approx(dist, 2.0, 0.1), "settled at {dist}, expected ≈2");
    }

    /// A constant force field (thrust along +Y on all bodies) accelerates a body —
    /// checks the ForceField path through the orchestration.
    #[tokio::test]
    async fn force_field_accelerates() {
        #[derive(Debug)]
        struct UniformPush;

        #[async_trait]
        impl ForceField<f32, f32> for UniformPush {
            async fn accumulate(
                &self,
                bodies: &IndexMap<WorldId, Box<dyn Component<f32>>>,
                out: &IndexMap<WorldId, WorldKey<Wrench<f32>>>,
                _epoch: &Epoch<f32>,
                _origin: &Vector3<f32>,
            ) {
                for i in 0..bodies.len() {
                    let tmp = &out[i];
                    tmp.write(
                        tmp.read() + Wrench::new(&Vector3::from([0.0, 5.0, 0.0]), &Vector3::ZERO),
                    );
                }
            }
        }

        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let id1 = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        m.add_field(Box::new(UniformPush)).await;

        for _ in 0..50 {
            m.step(&Epoch::standalone(0.01, 1.0)).await.unwrap(); // t = 0.5
        }

        let v = m
            .inspect_body(id1, async |b| b.velocity().linear())
            .await
            .unwrap();
        assert!(v[1] > 0.0, "the field must accelerate along +Y: {v:?}");
        // p_y = F·t = 5·0.5 = 2.5, m=1 → v_y ≈ 2.5
        assert!(approx(v[1], 2.5, 0.05), "v_y ≈ 2.5, got {}", v[1]);
    }

    /// PULLBACK world→body in `Mechanism::step` (phase 5).
    ///
    /// `ForceField` returns a WORLD wrench with the Plücker `τ_o = r×F` — exactly the
    /// form that comes from `GravityPropagator`/`Joint::wrenches` via
    /// `Wrench::from_line(dual)`. For a central force applied at the CoM
    /// of an off-axis body, the body-frame `τ_COM ≡ 0`, so after ONE step
    /// the body-frame angular momentum MUST stay zero — this is exactly the
    /// pullback invariant. Without it `τ_o` leaks into spin about the CoM.
    ///
    /// The invariant is ARCHITECTURAL (it lives in Mechanism, not in the integrator): it must
    /// hold identically for ALL three integrators.
    async fn assert_mechanism_pulls_world_wrench_back_to_body(
        world: Arc<World>,
        integ: ImplicitIntegrator<f32>,
    ) {
        // A force field imitating a central force at the CoM of an off-axis body:
        // F in world axes + τ_o = r×F (as with Wrench::from_line(dual)).
        #[derive(Debug)]
        struct CentralWorldPull {
            r: Vector3<f32>,
            f: Vector3<f32>,
        }

        #[async_trait]
        impl ForceField<f32, f32> for CentralWorldPull {
            async fn accumulate(
                &self,
                bodies: &IndexMap<WorldId, Box<dyn Component<f32>>>,
                out: &IndexMap<WorldId, WorldKey<Wrench<f32>>>,
                _epoch: &Epoch<f32>,
                origin: &Vector3<f32>,
            ) {
                // The field sets the force at the ABSOLUTE point `r`; build the torque about the island
                // origin (poses are now local): r_local = r − origin.
                let rl = self.r - *origin;
                let tau = Vector3::from([
                    rl[1] * self.f[2] - rl[2] * self.f[1],
                    rl[2] * self.f[0] - rl[0] * self.f[2],
                    rl[0] * self.f[1] - rl[1] * self.f[0],
                ]);
                for i in 0..bodies.len() {
                    let tmp = &out[i];
                    tmp.write(tmp.read() + Wrench::new(&self.f, &tau));
                }
            }
        }

        let r = Vector3::from([10.0, 5.0, 0.0]);
        let f = Vector3::from([-1.0, 0.0, 0.0]);

        let m = Mechanism::<f32, f32>::new(integ);
        let body = RigidBody::new(world.clone(), Inertia::isotropic(world.clone(), 1.0, 0.4));
        body.pose.write(Twist::new(&r, &Vector3::ZERO).exp(1.0));
        let id1 = m.add_body(Inert::new(body)).await;
        m.add_field(Box::new(CentralWorldPull { r, f })).await;

        m.step(&Epoch::standalone(0.5, 1.0)).await.unwrap();

        let l = m
            .inspect_body(id1, async |b| b.momentum.read().torque())
            .await
            .unwrap();
        for k in 0..3 {
            assert!(
                approx(l[k], 0.0, 1e-6),
                "body angular momentum component {k} ≠ 0: {l:?} \
                 (Mechanism did not fold τ_o = r×F into the body frame)"
            );
        }
    }

    #[tokio::test]
    async fn mechanism_pulls_world_wrench_back_to_body_lie() {
        let world = Arc::new(World::builder().usual::<f32>());
        assert_mechanism_pulls_world_wrench_back_to_body(
            world.clone(),
            ImplicitIntegrator::LieEuler(accel(world.clone())),
        )
        .await;
    }

    #[tokio::test]
    async fn mechanism_pulls_world_wrench_back_to_body_symplectic() {
        let world = Arc::new(World::builder().usual::<f32>());
        assert_mechanism_pulls_world_wrench_back_to_body(
            world.clone(),
            ImplicitIntegrator::SymplecticEuler(accel(world.clone())),
        )
        .await;
    }

    #[tokio::test]
    async fn mechanism_pulls_world_wrench_back_to_body_explicit() {
        let world = Arc::new(World::builder().usual::<f32>());
        assert_mechanism_pulls_world_wrench_back_to_body(
            world.clone(),
            ImplicitIntegrator::ExplicitEuler(accel(world.clone())),
        )
        .await;
    }

    /// Migration: detach carries the body away WITH ITS BEHAVIOR and cuts joints; attach to another
    /// mechanism. The behavior (pre_step counter) moves together with the body.
    #[tokio::test]
    async fn detach_carries_behavior_and_breaks_joints() {
        // A behavior with a pre_step call counter — observable portable state.
        #[derive(Debug)]
        struct Counter {
            ticks: RwLock<u32>,
        }
        impl Counter {
            fn step(&self) {
                let mut guard = self.ticks.write().unwrap();
                *guard += 1;
            }
        }

        let world = Arc::new(World::builder().usual::<f32>());

        let ticks = Arc::new(Counter {
            ticks: RwLock::new(0),
        });

        let m1 =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));

        let ticks_cloned = ticks.clone();
        let id1 = m1.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let id2 = m1
            .add_body(Entity::new(body_at(world.clone(), 2.0), move |_, _| {
                ticks_cloned.step()
            }))
            .await;
        m1.connect(id1, vec![(spring(world.clone(), 2.0, 5.0, 0.0), id2)])
            .await;

        m1.step(&Epoch::standalone(0.01, 1.0)).await.unwrap(); // tick = 1
        assert_eq!(*ticks.ticks.read().unwrap(), 1);

        // detach body 2: carries the Counter away, cuts joint 1↔2.
        let migrated = m1.detach(id2).await.expect("body 2 exists");
        assert!(m1.inspect_body(id2, async |_| {}).await.is_none());

        // m1 has no joints after detach — step does not panic (no dangling keys).
        m1.step(&Epoch::standalone(0.01, 1.0)).await.unwrap(); // Counter left → ticks does not grow from m1

        let m2 =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m2.attach(migrated).await;
        m2.step(&Epoch::standalone(0.01, 1.0)).await.unwrap(); // tick = 2: the behavior moved and works
        assert_eq!(
            *ticks.ticks.read().unwrap(),
            2,
            "Counter must move together with the body"
        );
    }

    /// split: bodies 2,3 are split off into a new mechanism with a DIFFERENT integrator type.
    /// The internal edge 2↔3 travels with them; the boundary 1↔2 is cut; the external one stays.
    #[tokio::test]
    async fn split_classifies_edges_and_changes_integrator() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));

        let id1 = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let id2 = m.add_body(Inert::new(body_at(world.clone(), 2.0))).await;
        let id3 = m.add_body(Inert::new(body_at(world.clone(), 4.0))).await;
        m.connect(id1, vec![(spring(world.clone(), 2.0, 5.0, 0.0), id2)])
            .await; // boundary (1 stays, 2 leaves)
        m.connect(id2, vec![(spring(world.clone(), 1.0, 5.0, 0.0), id3)])
            .await; // internal, STRETCHED (dist=2, rest=1)

        // Split off 2 and 3 under SymplecticEuler (a different type than m's).
        let detached: Mechanism<f32, f32> = m
            .split(
                &[id2, id3],
                ImplicitIntegrator::SymplecticEuler(accel(world.clone())),
            )
            .await;

        // Parent: only body 1 remains, with no joints (the boundary one is cut).
        assert!(m.inspect_body(id1, async |_| {}).await.is_some());
        assert!(m.inspect_body(id2, async |_| {}).await.is_none());
        assert!(m.inspect_body(id3, async |_| {}).await.is_none());

        // New mechanism: bodies 2,3 are present.
        assert!(detached.inspect_body(id2, async |_| {}).await.is_some());
        assert!(detached.inspect_body(id3, async |_| {}).await.is_some());

        // Both mechanisms are stepped independently by their integrators without panicking.
        m.step(&Epoch::standalone(0.01, 1.0)).await.unwrap();
        detached.step(&Epoch::standalone(0.01, 1.0)).await.unwrap();

        // The internal joint 2↔3 moved and works: the stretched spring contracts.
        let d_before = {
            let c2 = detached.inspect_body(id2, pos).await.unwrap();
            let c3 = detached.inspect_body(id3, pos).await.unwrap();
            (c3 - c2).abs()
        };
        for _ in 0..200 {
            detached.step(&Epoch::standalone(0.005, 1.0)).await.unwrap();
        }
        let d_after = {
            let c2 = detached.inspect_body(id2, pos).await.unwrap();
            let c3 = detached.inspect_body(id3, pos).await.unwrap();
            (c3 - c2).abs()
        };
        assert!(d_after < d_before, "the internal spring 2↔3 must contract");
    }

    /// merge: two mechanisms with DIFFERENT integrator types merge; the primary's
    /// integrator survives. The linking edge crosses the boundary.
    #[tokio::test]
    async fn merge_different_integrators() {
        let world = Arc::new(World::builder().usual::<f32>());
        let ship =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let id1 = ship.add_body(Inert::new(body_at(world.clone(), 0.0))).await;

        let planet_sys =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let id2 = planet_sys
            .add_body(Inert::new(body_at(world.clone(), 5.0)))
            .await;

        // Landing: merge planet_sys into ship, joint 10↔20.
        ship.merge(planet_sys, (spring(world, 5.0, 1.0, 0.5), id1, id2))
            .await;

        // Both bodies are now in ship; the step runs with ship's integrator (SymplecticEuler) without panicking.
        assert!(ship.inspect_body(id1, async |_| {}).await.is_some());
        assert!(ship.inspect_body(id2, async |_| {}).await.is_some());
        ship.step(&Epoch::standalone(0.01, 1.0)).await.unwrap();
    }

    #[tokio::test]
    #[should_panic(expected = "must cross")]
    async fn merge_link_must_cross_boundary() {
        let world = Arc::new(World::builder().usual::<f32>());

        let a =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));

        let id1 = a.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let id2 = a.add_body(Inert::new(body_at(world.clone(), 1.0))).await;

        let b =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));

        let _body = b.add_body(Inert::new(body_at(world.clone(), 2.0))).await;

        // Joint 1↔2 — both ends in `a`, does not cross the boundary.
        a.merge(b, (spring(world, 1.0, 1.0, 0.0), id1, id2)).await;
    }

    #[tokio::test]
    async fn connectivity_diagnostics() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        assert!(
            m.is_connected().await,
            "an empty mechanism is degenerately connected"
        );
        assert_eq!(m.components().await.len(), 0);

        let id1 = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        assert!(m.is_connected().await, "a single body is connected");
        assert_eq!(m.components().await.len(), 1);

        // A second unconnected body → two components, disconnected (but this is LEGAL).
        let id2 = m.add_body(Inert::new(body_at(world.clone(), 5.0))).await;
        assert!(!m.is_connected().await);
        assert_eq!(m.components().await.len(), 2);

        // Connected them → one component again.
        m.connect(id1, vec![(spring(world, 5.0, 1.0, 0.0), id2)])
            .await;
        assert!(m.is_connected().await);
        assert_eq!(m.components().await.len(), 1);
    }

    /// Motivating scenario: detect disconnected subgraphs and split one of
    /// them off into a separate mechanism via split.
    #[tokio::test]
    async fn detect_components_then_split() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        // Island A: bodies 1—2 are connected.
        let id1 = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let id2 = m.add_body(Inert::new(body_at(world.clone(), 2.0))).await;
        m.connect(id1, vec![(spring(world.clone(), 2.0, 1.0, 0.0), id2)])
            .await;
        // Island B: bodies 3—4 are connected, but not connected to A.
        let id3 = m.add_body(Inert::new(body_at(world.clone(), 10.0))).await;
        let id4 = m.add_body(Inert::new(body_at(world.clone(), 12.0))).await;
        m.connect(id3, vec![(spring(world.clone(), 2.0, 1.0, 0.0), id4)])
            .await;

        let comps = m.components().await;
        assert_eq!(comps.len(), 2, "two islands");

        // Split one of the components (the one containing body 3) into its own mechanism.
        let island = comps.iter().find(|c| c.contains(&id3)).unwrap().clone();
        let detached = m
            .split(
                &island,
                ImplicitIntegrator::ExplicitEuler(accel(world.clone())),
            )
            .await;

        // Each mechanism is now connected.
        assert!(
            m.is_connected().await,
            "the remainder (island A) is connected"
        );
        assert!(
            detached.is_connected().await,
            "the split-off one (island B) is connected"
        );
        assert_eq!(m.components().await.len(), 1);
        assert_eq!(detached.components().await.len(), 1);
    }

    // ========================================================================
    // Connectivity islands: the partition is the storage, the step runs over islands.
    // ========================================================================

    /// Regression for the original bug (mechanism.rs: one step_all for the whole disconnected
    /// mechanism). Two spring pairs NOT connected to each other in one
    /// mechanism each settle to their own rest length — so the integrator received
    /// one connected island at a time, not a disconnected "mush" of four bodies.
    #[tokio::test]
    async fn disconnected_islands_step_independently() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        // Island A: a spring pair, stretched (rest=2, dist=4).
        let a1 = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let a2 = m.add_body(Inert::new(body_at(world.clone(), 4.0))).await;
        m.connect(a1, vec![(spring(world.clone(), 2.0, 10.0, 4.0), a2)])
            .await;
        // Island B: another pair, far away, NOT connected to A.
        let b1 = m.add_body(Inert::new(body_at(world.clone(), 100.0))).await;
        let b2 = m.add_body(Inert::new(body_at(world.clone(), 104.0))).await;
        m.connect(b1, vec![(spring(world.clone(), 2.0, 10.0, 4.0), b2)])
            .await;

        assert_eq!(m.components().await.len(), 2, "two islands");
        assert!(!m.is_connected().await);

        for _ in 0..4000 {
            m.step(&Epoch::standalone(0.005, 1.0)).await.unwrap(); // t = 20
        }

        let da = {
            let c1 = m.inspect_body(a1, pos).await.unwrap();
            let c2 = m.inspect_body(a2, pos).await.unwrap();
            (c2 - c1).abs()
        };
        let db = {
            let c1 = m.inspect_body(b1, pos).await.unwrap();
            let c2 = m.inspect_body(b2, pos).await.unwrap();
            (c2 - c1).abs()
        };
        assert!(
            approx(da, 2.0, 0.1),
            "island A settled at {da}, expected ≈2"
        );
        assert!(
            approx(db, 2.0, 0.1),
            "island B settled at {db}, expected ≈2"
        );
    }

    /// `connect` across a boundary merges islands; `detach` of a bridge body cuts them.
    /// There is no manual connectivity recomputation in the test — the operations themselves
    /// uphold the invariant.
    #[tokio::test]
    async fn connect_merges_detach_splits() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let a = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let b = m.add_body(Inert::new(body_at(world.clone(), 2.0))).await;
        let c = m.add_body(Inert::new(body_at(world.clone(), 4.0))).await;
        assert_eq!(m.components().await.len(), 3, "three singletons");

        // Path a—b—c → one island.
        m.connect(a, vec![(spring(world.clone(), 2.0, 1.0, 0.0), b)])
            .await;
        m.connect(b, vec![(spring(world.clone(), 2.0, 1.0, 0.0), c)])
            .await;
        assert!(m.is_connected().await);
        assert_eq!(m.components().await.len(), 1);

        // detach of the bridge b → a and c fall apart into two islands.
        let _detached = m.detach(b).await;
        assert_eq!(m.components().await.len(), 2);
        assert!(!m.is_connected().await);
        let comps = m.components().await;
        assert!(comps.iter().any(|cc| cc.contains(&a) && !cc.contains(&c)));
        assert!(comps.iter().any(|cc| cc.contains(&c) && !cc.contains(&a)));
    }

    /// The partition invariant holds after EVERY operation: no id lies in
    /// two islands, and the islands cover exactly the live bodies.
    #[tokio::test]
    async fn partition_stays_valid_through_operations() {
        async fn assert_partition(m: &Mechanism<f32, f32>, live: &[WorldId]) {
            let comps = m.components().await;
            let mut seen = HashSet::new();
            for cc in &comps {
                for id in cc {
                    assert!(seen.insert(*id), "id {id:?} in two islands");
                }
            }
            let want: HashSet<WorldId> = live.iter().copied().collect();
            assert_eq!(seen, want, "the islands must cover exactly the live bodies");
        }

        let world = Arc::new(World::builder().usual::<f32>());
        let m = Mechanism::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let mut live: Vec<WorldId> = Vec::new();

        let a = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        live.push(a);
        assert_partition(&m, &live).await;
        let b = m.add_body(Inert::new(body_at(world.clone(), 2.0))).await;
        live.push(b);
        let c = m.add_body(Inert::new(body_at(world.clone(), 4.0))).await;
        live.push(c);
        let d = m.add_body(Inert::new(body_at(world.clone(), 6.0))).await;
        live.push(d);
        assert_partition(&m, &live).await;

        m.connect(a, vec![(spring(world.clone(), 2.0, 1.0, 0.0), b)])
            .await;
        m.connect(c, vec![(spring(world.clone(), 2.0, 1.0, 0.0), d)])
            .await;
        assert_partition(&m, &live).await;
        assert_eq!(m.components().await.len(), 2);

        // split island {c,d} into a separate mechanism.
        let det = m
            .split(
                &[c, d],
                ImplicitIntegrator::SymplecticEuler(accel(world.clone())),
            )
            .await;
        live.retain(|x| *x != c && *x != d);
        assert_partition(&m, &live).await;
        assert_partition(&det, &[c, d]).await;

        // detach b → {a,b} falls apart, b leaves.
        let _detached = m.detach(b).await;
        live.retain(|x| *x != b);
        assert_partition(&m, &live).await;

        // merge back across the boundary a—c.
        m.merge(det, (spring(world.clone(), 6.0, 1.0, 0.0), a, c))
            .await;
        live.push(c);
        live.push(d);
        assert_partition(&m, &live).await;
        assert!(m.is_connected().await);
    }

    // ========================================================================
    // Centroid (mass-weighted) — the basis for a KSP-style camera tracking the CoM.
    // ========================================================================

    /// An empty mechanism has no CoM — None, not NaN/a panic.
    #[tokio::test]
    async fn centroid_of_empty_is_none() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m: Mechanism<f32, f32> =
            Mechanism::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        assert!(m.centroid().await.is_none());
    }

    /// A single body — the CoM coincides with its position regardless of mass.
    #[tokio::test]
    async fn centroid_of_single_body_is_its_position() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m.add_body(Inert::new(body_at_with_mass(world.clone(), 3.0, 7.0)))
            .await;
        let c = m.centroid().await.unwrap().coords();
        assert!(approx(c[0], 3.0, 1e-6), "x={}", c[0]);
        assert!(approx(c[1], 0.0, 1e-6));
        assert!(approx(c[2], 0.0, 1e-6));
    }

    /// Two bodies of equal mass — the CoM is exactly in the middle (the geometric centroid as
    /// a special case of the weighted one with equal weights).
    #[tokio::test]
    async fn centroid_of_equal_masses_is_geometric_midpoint() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        m.add_body(Inert::new(body_at(world.clone(), 4.0))).await;
        let c = m.centroid().await.unwrap().coords();
        assert!(approx(c[0], 2.0, 1e-6), "x={}", c[0]);
    }

    /// Mass weighting: the heavy body pulls the CoM toward itself.
    /// m=10 @ x=0, m=1 @ x=11 → CoM = (10·0 + 1·11) / 11 = 1.0
    #[tokio::test]
    async fn centroid_is_pulled_toward_heavier_body() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m.add_body(Inert::new(body_at_with_mass(world.clone(), 0.0, 10.0)))
            .await;
        m.add_body(Inert::new(body_at_with_mass(world.clone(), 11.0, 1.0)))
            .await;
        let c = m.centroid().await.unwrap().coords();
        assert!(approx(c[0], 1.0, 1e-6), "x={} (expected 1.0)", c[0]);
    }

    /// Three bodies in 3D with different masses — the exact barycentre formula.
    #[tokio::test]
    async fn centroid_three_body_3d_weighted() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));

        // m=2 @ (1,0,0), m=1 @ (0,3,0), m=1 @ (0,0,4)
        let b1 = RigidBody::new(world.clone(), Inertia::isotropic(world.clone(), 2.0, 1.0));
        b1.pose
            .write(Twist::new(&Vector3::from([1.0, 0.0, 0.0]), &Vector3::ZERO).exp(1.0));
        let b2 = RigidBody::new(world.clone(), Inertia::isotropic(world.clone(), 1.0, 1.0));
        b2.pose
            .write(Twist::new(&Vector3::from([0.0, 3.0, 0.0]), &Vector3::ZERO).exp(1.0));
        let b3 = RigidBody::new(world.clone(), Inertia::isotropic(world.clone(), 1.0, 1.0));
        b3.pose
            .write(Twist::new(&Vector3::from([0.0, 0.0, 4.0]), &Vector3::ZERO).exp(1.0));

        m.add_body(Inert::new(b1)).await;
        m.add_body(Inert::new(b2)).await;
        m.add_body(Inert::new(b3)).await;

        // Σm = 4. CoM = (2·1+0+0, 0+1·3+0, 0+0+1·4) / 4 = (0.5, 0.75, 1.0).
        let c = m.centroid().await.unwrap().coords();
        assert!(approx(c[0], 0.5, 1e-6), "x={}", c[0]);
        assert!(approx(c[1], 0.75, 1e-6), "y={}", c[1]);
        assert!(approx(c[2], 1.0, 1e-6), "z={}", c[2]);
    }

    /// KSP scenario: fuel in the tank burns off → the tank mass drops → the CoM
    /// drifts toward the dry end of the spacecraft. Direct motivation for a camera
    /// bound to the centroid: it automatically recomposes the frame as
    /// fuel is consumed, without explicit recomputation outside Mechanism.
    #[tokio::test]
    async fn centroid_shifts_toward_dry_end_as_fuel_drains() {
        // A tank with the behavior "drained dm of mass per step" — the simplest representation
        // of consumable fuel.
        #[derive(Debug)]
        struct Tank {
            drain_per_step: f32,
        }
        impl Tank {
            fn pre_step(&mut self, body: &mut RigidBody<f32>) {
                // Do not go below zero — otherwise the very point of mass application is lost.
                let new_mass = (body.inertia.mass() - self.drain_per_step).max(0.1);
                body.inertia = Inertia::isotropic(body.world.clone(), new_mass, 1.0);
            }
        }

        let world = Arc::new(World::builder().usual::<f32>());
        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));

        let tank = Arc::new(RwLock::new(Tank {
            drain_per_step: 0.1,
        }));
        let tank_cloned = tank.clone();

        // A dry nose (mass does not change) and a heavy tail tank with fuel.
        m.add_body(Inert::new(body_at_with_mass(world.clone(), 0.0, 1.0)))
            .await;
        m.add_body(Entity::new(
            body_at_with_mass(world.clone(), 10.0, 10.0),
            move |b, _| {
                let mut guard = tank_cloned.write().unwrap();
                guard.pre_step(b);
            },
        ))
        .await;

        // Before burn-off: CoM = (1·0 + 10·10) / 11 ≈ 9.09 — far in the tail.
        let x_before = m.centroid().await.unwrap().coords()[0];
        assert!(
            approx(x_before, 100.0 / 11.0, 1e-6),
            "initial CoM = {x_before}"
        );

        // 50 steps × 0.1 = 5 units of mass drained from the tank → the tank now has 5.
        for _ in 0..50 {
            m.step(&Epoch::standalone(0.01, 1.0)).await.unwrap();
        }

        // After burn-off: CoM = (1·0 + 5·10) / 6 ≈ 8.33 — shifted toward the nose.
        let x_after = m.centroid().await.unwrap().coords()[0];
        assert!(approx(x_after, 50.0 / 6.0, 1e-6), "CoM after = {x_after}");
        assert!(
            x_after < x_before,
            "the CoM must shift toward the dry end: {x_before} → {x_after}"
        );
    }

    // ========================================================================
    // Centroid velocity — world-frame linear, angular forced to zero.
    // ========================================================================

    #[tokio::test]
    async fn centroid_velocity_of_empty_is_none() {
        let world = Arc::new(World::builder().usual::<f32>());
        let m: Mechanism<f32, f32> =
            Mechanism::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        assert!(m.centroid_velocity().await.is_none());
    }

    #[tokio::test]
    async fn centroid_velocity_of_single_body_at_rest_is_zero() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;
        let v = m.centroid_velocity().await.unwrap().linear();
        assert!(approx(v[0], 0.0, 1e-6));
        assert!(approx(v[1], 0.0, 1e-6));
        assert!(approx(v[2], 0.0, 1e-6));
    }

    #[tokio::test]
    async fn centroid_velocity_of_single_body_equals_its_velocity() {
        let world = Arc::new(World::builder().usual::<f32>());

        let mass = 1.0;
        let v_set = [3.0, -2.0, 1.0];
        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
            world.clone(),
            &Vector3::ZERO,
            &Vector3::from(v_set),
            mass,
        )))
        .await;
        let v = m.centroid_velocity().await.unwrap().linear();
        assert!(approx(v[0], v_set[0], 1e-6), "x: {}", v[0]);
        assert!(approx(v[1], v_set[1], 1e-6), "y: {}", v[1]);
        assert!(approx(v[2], v_set[2], 1e-6), "z: {}", v[2]);
    }

    #[tokio::test]
    async fn centroid_velocity_zero_for_equal_opposite_momenta() {
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
            world.clone(),
            &Vector3::ZERO,
            &Vector3::from([2.0, 0.0, 0.0]),
            1.0,
        )))
        .await;
        m.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
            world.clone(),
            &Vector3::ZERO,
            &Vector3::from([-2.0, 0.0, 0.0]),
            1.0,
        )))
        .await;
        let v = m.centroid_velocity().await.unwrap().linear();
        assert!(approx(v[0], 0.0, 1e-6), "x: {}", v[0]);
        assert!(approx(v[1], 0.0, 1e-6));
        assert!(approx(v[2], 0.0, 1e-6));
    }

    #[tokio::test]
    async fn centroid_velocity_is_mass_weighted() {
        let world = Arc::new(World::builder().usual::<f32>());

        // m=10 at rest + m=1 moving at v=[11,0,0]
        // → v_com.x = (10·0 + 1·11) / 11 = 1.0
        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
            world.clone(),
            &Vector3::ZERO,
            &Vector3::ZERO,
            10.0,
        )))
        .await;
        m.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
            world.clone(),
            &Vector3::ZERO,
            &Vector3::from([11.0, 0.0, 0.0]),
            1.0,
        )))
        .await;
        let v = m.centroid_velocity().await.unwrap().linear();
        assert!(approx(v[0], 1.0, 1e-6), "got {}", v[0]);
        assert!(approx(v[1], 0.0, 1e-6));
        assert!(approx(v[2], 0.0, 1e-6));
    }

    /// Anti-regression. Body with non-identity rotation must report world-frame
    /// velocity, not body-frame. The pre-fix formula returns body-frame and
    /// fails this test.
    #[tokio::test]
    async fn centroid_velocity_uses_world_frame_for_rotated_body() {
        use clifford::pga3::Wrench;
        use std::f32::consts::FRAC_PI_2;

        let world = Arc::new(World::builder().usual::<f32>());

        // 90° rotation around +Z. Per clifford::pga3::screw::tests::
        // right_hand_rotation_about_z, angular = [0, 0, π/2] under .exp(1.0)
        // produces a motor that maps world +X → +Y.
        let body = RigidBody::new(world.clone(), Inertia::isotropic(world.clone(), 1.0, 1.0));
        body.pose.write(
            Twist::new(
                &Vector3::from([0.0, 0.0, 0.0]),
                &Vector3::from([0.0, 0.0, FRAC_PI_2]),
            )
            .exp(1.0),
        );
        // body-frame momentum: linear = [1, 0, 0]. With mass = 1, body-frame
        // velocity is also [1, 0, 0]. World-frame: rotation 90° around +Z
        // maps +X → +Y, so world velocity is [0, 1, 0].
        body.momentum.write(Wrench::new(
            &Vector3::from([1.0, 0.0, 0.0]),
            &Vector3::from([0.0, 0.0, 0.0]),
        ));

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m.add_body(Inert::new(body)).await;
        let v = m.centroid_velocity().await.unwrap().linear();

        assert!(
            approx(v[0], 0.0, 1e-6),
            "x should be ~0 (world-frame), got {}",
            v[0]
        );
        assert!(
            approx(v[1], 1.0, 1e-6),
            "y should be ~1 (world-frame), got {}",
            v[1]
        );
        assert!(approx(v[2], 0.0, 1e-6), "z: {}", v[2]);
    }

    #[tokio::test]
    async fn centroid_velocity_angular_is_always_zero() {
        use clifford::pga3::Wrench;
        let world = Arc::new(World::builder().usual::<f32>());

        let body = RigidBody::new(world.clone(), Inertia::isotropic(world.clone(), 1.0, 0.5));
        body.momentum.write(Wrench::new(
            &Vector3::from([0.0, 0.0, 0.0]),
            &Vector3::from([1.0, 2.0, 3.0]),
        ));

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        m.add_body(Inert::new(body)).await;
        let cv = m.centroid_velocity().await.unwrap();
        let a = cv.angular();
        assert!(approx(a[0], 0.0, 1e-6), "angular[0] = {}", a[0]);
        assert!(approx(a[1], 0.0, 1e-6), "angular[1] = {}", a[1]);
        assert!(approx(a[2], 0.0, 1e-6), "angular[2] = {}", a[2]);
    }

    // ========================================================================
    // update_body — mutating analog of inspect_body.
    // ========================================================================

    #[tokio::test]
    async fn update_body_mutates_and_returns_value() {
        use clifford::pga3::Wrench;
        let world = Arc::new(World::builder().usual::<f32>());

        let m =
            Mechanism::<f32, f32>::new(ImplicitIntegrator::SymplecticEuler(accel(world.clone())));
        let id = m.add_body(Inert::new(body_at(world.clone(), 0.0))).await;

        // Set body-frame momentum via closure; closure also returns a sentinel.
        let returned = m
            .update_body(id, async |b| {
                b.momentum.write(Wrench::new(
                    &Vector3::from([5.0, 0.0, 0.0]),
                    &Vector3::from([0.0, 0.0, 0.0]),
                ));
                42u32
            })
            .await;
        assert_eq!(returned, Some(42));

        // inspect_body sees the mutation.
        let f = m
            .inspect_body(id, async |b| b.momentum.read().force())
            .await
            .unwrap();
        assert!(approx(f[0], 5.0, 1e-6), "got {f:?}");

        // Unknown id returns None and does not panic.
        let none = m.update_body(WorldId::get(), async |_| 0u32).await;
        assert!(none.is_none());
    }
}
