// SPDX-License-Identifier: MIT

//! AR-0103: the cross-mechanism epoch driver.
//!
//! - Mechanisms stepped together by `Driver` follow, step by step and bit for
//!   bit, the trajectories they follow stepped alone.
//! - A structural change to a mechanism inside an epoch is refused with
//!   `StructureError::EpochInProgress` — not queued behind the step, not
//!   deadlocked — and succeeds once the epoch is over.
//! - One body world id cannot be owned by two mechanisms.
//! - The quiescence flush does not cost more submissions than the idle path.

use aristotle::{Epoch, World, WorldId};
use async_trait::async_trait;
use clifford::pga3::Twist;
use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
use newton::field::UniformField;
use newton::integrator::{ImplicitIntegrator, Newton};
use newton::{
    Accelerator, Component, Driver, Inert, Inertia, Mechanism, RigidBody, StructureError,
};
use peano::prelude::*;
use std::sync::Arc;

const K: f32 = 100.0;
const G: f32 = 9.81;
const DT: f32 = 1.0 / 60.0;

fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
    AxialSpringDamper::builder(world, SimpleSpringDamper)
        .rest(rest)
        .stiffness(k)
        .damping(c)
        .build()
}

fn accelerator(world: &Arc<World>) -> Arc<Accelerator<f32>> {
    Arc::new(Accelerator::<f32>::builder(world.clone()).build())
}

/// A COLS×ROWS curtain hanging from its kinematic top row, a few nodes kicked
/// sideways (`seed` scales the kick, so curtains differ), under gravity.
async fn curtain(
    world: Arc<World>,
    accel: &Arc<Accelerator<f32>>,
    cols: usize,
    rows: usize,
    seed: f32,
) -> (Mechanism<f32, f32>, Vec<WorldId>) {
    let h = 0.5 / (cols as f32 - 1.0).max(1.0);
    let m = 1.0 / (cols * rows) as f32;
    let c = 0.7 * 2.0 * (K * (m * 0.5)).sqrt();
    let mech: Mechanism<f32, f32> =
        Mechanism::new(ImplicitIntegrator::Newton(Newton::new(accel.clone())));

    let mut ids = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let pos = Vector3::from([col as f32 * h, 0.0, row as f32 * h]);
            let id = if row == rows - 1 {
                let b = RigidBody::new(world.clone(), Inertia::Kinematic);
                b.pose.write(Twist::new(&pos, &Vector3::ZERO).exp(1.0));
                mech.add_body(Inert::new(b)).await
            } else {
                let vel = if (row * cols + col) % 4 == 1 {
                    Vector3::from([0.0, 0.5 * seed, 0.0])
                } else {
                    Vector3::ZERO
                };
                mech.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                    world.clone(),
                    &pos,
                    &vel,
                    m,
                )))
                .await
            };
            ids.push(id);
        }
    }
    for row in 0..rows {
        for col in 0..cols {
            let a = ids[row * cols + col];
            if col + 1 < cols {
                let b = ids[row * cols + col + 1];
                mech.connect(a, vec![(spring(world.clone(), h, K, c), b)])
                    .await;
            }
            if row + 1 < rows {
                let b = ids[(row + 1) * cols + col];
                mech.connect(a, vec![(spring(world.clone(), h, K, c), b)])
                    .await;
            }
        }
    }
    mech.add_field(Box::new(UniformField::new(Vector3::from([0.0, 0.0, -G]))))
        .await;
    (mech, ids)
}

async fn positions(mech: &Mechanism<f32, f32>, ids: &[WorldId]) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let p = mech.body_absolute_position(*id).await.unwrap();
        out.push([p[0], p[1], p[2]]);
    }
    out
}

/// Different sizes, so the mechanisms' rounds differ in length.
const SHAPES: [(usize, usize, f32); 3] = [(2, 3, 1.0), (3, 3, -0.7), (2, 5, 0.4)];

/// Every step of every mechanism through the driver equals the same step taken
/// alone, on its own world and accelerator — bit for bit.
#[tokio::test]
async fn driver_matches_solo_trajectories_bit_exactly() {
    const STEPS: usize = 25;

    let mut solo = Vec::new();
    for (cols, rows, seed) in SHAPES {
        let world = Arc::new(World::builder().usual::<f32>());
        let accel = accelerator(&world);
        let (mech, ids) = curtain(world, &accel, cols, rows, seed).await;
        let mut trajectory = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            mech.step(&Epoch::standalone(DT, 1.0)).await.unwrap();
            trajectory.push(positions(&mech, &ids).await);
        }
        solo.push(trajectory);
    }

    let world = Arc::new(World::builder().usual::<f32>());
    let accel = accelerator(&world);
    let mut driver = Driver::new(accel.clone());
    let mut together = Vec::new();
    for (cols, rows, seed) in SHAPES {
        let (mech, ids) = curtain(world.clone(), &accel, cols, rows, seed).await;
        let mech = Arc::new(mech);
        driver.add(mech.clone()).await.unwrap();
        together.push((mech, ids));
    }
    let epoch = Epoch::standalone(DT, 1.0);
    for step in 0..STEPS {
        driver.step(&epoch).await.unwrap();
        for (i, (mech, ids)) in together.iter().enumerate() {
            let got = positions(mech, ids).await;
            assert_eq!(
                got, solo[i][step],
                "mechanism {i} diverged from its solo run at step {step}"
            );
        }
    }
}

/// Every structural change, attempted while the driver's epoch is in
/// progress, is refused at once with `EpochInProgress` — and the epoch itself
/// completes. After the epoch the same change goes through.
#[tokio::test]
async fn structural_change_mid_epoch_is_refused() {
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = accelerator(&world);
    let (a, a_ids) = curtain(world.clone(), &accel, 2, 3, 1.0).await;
    let (b, _) = curtain(world.clone(), &accel, 2, 2, -1.0).await;
    let (a, b) = (Arc::new(a), Arc::new(b));
    let mut driver = Driver::new(accel.clone());
    driver.add(a.clone()).await.unwrap();
    driver.add(b.clone()).await.unwrap();
    let epoch = Epoch::standalone(DT, 1.0);
    let frozen = Err(StructureError::EpochInProgress(a.id()));

    // `join!` polls the epoch first, so every mechanism is frozen before the
    // mutation is first polled. Each mutation must resolve WITHOUT waiting for
    // the epoch: it is refused, not queued behind the step's lock.
    let body = || {
        Inert::new(RigidBody::body_at_with_mass(
            world.clone(),
            &Vector3::ZERO,
            1.0,
        ))
    };
    let (stepped, added, connected, removed, split, merged) = tokio::join!(
        driver.step(&epoch),
        a.try_add_body(body()),
        a.try_connect(
            a_ids[0],
            vec![(spring(world.clone(), 0.1, K, 1.0), a_ids[3])]
        ),
        a.try_remove_body(a_ids[0]),
        a.try_split(
            &a_ids[..2],
            ImplicitIntegrator::Newton(Newton::new(accel.clone()))
        ),
        a.try_merge(
            Mechanism::new(ImplicitIntegrator::Newton(Newton::new(accel.clone()))),
            (spring(world.clone(), 0.1, K, 1.0), a_ids[0], WorldId::get()),
        ),
    );
    stepped.unwrap();
    assert_eq!(added.map(|_| ()), frozen);
    assert_eq!(connected, frozen);
    assert_eq!(removed, frozen);
    assert_eq!(split.map(|_| ()), frozen);
    assert_eq!(merged.map_err(|(e, _)| e), frozen);
    assert_eq!(
        a.components().await.concat().len(),
        6,
        "a refused change landed"
    );

    // A bare `Mechanism::step` freezes its own mechanism the same way.
    let (stepped, added) = tokio::join!(a.step(&epoch), a.try_add_body(body()));
    stepped.unwrap();
    assert_eq!(added.map(|_| ()), frozen);

    // Between epochs the structure is open again. (The added body is removed
    // again before stepping: removing a curtain node would change the size of
    // an island whose Newton cache is already built, and that rebuild
    // self-deadlocks in `NewtonCache::sync` — a defect outside this test's
    // subject, recorded in the AR-0103 evidence log.)
    let extra = a.try_add_body(body()).await.unwrap();
    assert_eq!(a.components().await.concat().len(), 7);
    a.try_remove_body(extra).await.unwrap();
    assert_eq!(a.components().await.concat().len(), 6);
    driver.step(&epoch).await.unwrap();
}

/// A component that reports a chosen world id — the only way to present one
/// body id to two mechanisms, which is exactly what must be refused.
struct Aliased {
    id: WorldId,
    body: RigidBody<f32>,
}

#[async_trait]
impl Component<f32, f32> for Aliased {
    fn body(&self) -> &RigidBody<f32> {
        &self.body
    }
    fn body_mut(&mut self) -> &mut RigidBody<f32> {
        &mut self.body
    }
    fn id(&self) -> WorldId {
        self.id
    }
}

#[tokio::test]
async fn one_body_cannot_join_two_mechanisms() {
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = accelerator(&world);
    let newton = || ImplicitIntegrator::Newton(Newton::new(accel.clone()));
    let a: Mechanism<f32, f32> = Mechanism::new(newton());
    let b: Mechanism<f32, f32> = Mechanism::new(newton());
    let id = WorldId::get();
    let aliased = || {
        Box::new(Aliased {
            id,
            body: RigidBody::body_at_with_mass(world.clone(), &Vector3::ZERO, 1.0),
        })
    };

    assert_eq!(a.try_add_body(aliased()).await, Ok(id));
    assert_eq!(
        b.try_add_body(aliased()).await,
        Err(StructureError::AlreadyOwned(id))
    );
    // Within one mechanism too.
    assert_eq!(
        a.try_add_body(aliased()).await,
        Err(StructureError::AlreadyOwned(id))
    );
    assert!(b.components().await.is_empty(), "the refused body landed");

    // Released by removal: now the other mechanism may take it.
    a.try_remove_body(id).await.unwrap();
    assert_eq!(b.try_add_body(aliased()).await, Ok(id));

    // Moved by split, not released: still owned, now by the split-off part.
    let c = b.split(&[id], newton()).await;
    assert_eq!(
        a.try_add_body(aliased()).await,
        Err(StructureError::AlreadyOwned(id))
    );
    // Released when its owner is dropped.
    drop(c);
    assert_eq!(a.try_add_body(aliased()).await, Ok(id));
}

#[tokio::test]
async fn driver_refuses_foreign_and_duplicate_mechanisms() {
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = accelerator(&world);
    let other = accelerator(&world);
    let mut driver: Driver<f32, f32> = Driver::new(accel.clone());

    let ours = Arc::new(Mechanism::new(ImplicitIntegrator::Newton(Newton::new(
        accel.clone(),
    ))));
    let foreign = Arc::new(Mechanism::new(ImplicitIntegrator::Newton(Newton::new(
        other.clone(),
    ))));
    driver.add(ours.clone()).await.unwrap();
    assert_eq!(
        driver.add(ours.clone()).await,
        Err(StructureError::DuplicateMechanism(ours.id()))
    );
    assert_eq!(
        driver.add(foreign.clone()).await,
        Err(StructureError::ForeignAccelerator(foreign.id()))
    );
    assert_eq!(driver.mechanisms().len(), 1);
    assert!(driver.remove(ours.id()).is_some());
    assert!(driver.mechanisms().is_empty());
}

/// The `cloth_grid_stability` frame-spike scenario (2×4 curtain, 100 frames,
/// every 17th at 0.1 s), and three curtains at once: submissions per step
/// through the driver (quiescence flush) against the idle path. The driver
/// must not cost more. Printed for the evidence log.
#[tokio::test]
async fn quiescence_does_not_add_flushes() {
    const FRAMES: usize = 100;
    let dt = |frame: usize| if frame % 17 == 16 { 0.1 } else { DT };

    // One curtain: bare `step` (idle path) versus the driver.
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = accelerator(&world);
    let (mech, _) = curtain(world.clone(), &accel, 2, 4, 1.0).await;
    let before = accel.flushes();
    for frame in 0..FRAMES {
        mech.step(&Epoch::standalone(dt(frame), 1.0)).await.unwrap();
    }
    let idle_one = accel.flushes() - before;

    let world = Arc::new(World::builder().usual::<f32>());
    let accel = accelerator(&world);
    let (mech, _) = curtain(world.clone(), &accel, 2, 4, 1.0).await;
    let mut driver = Driver::new(accel.clone());
    driver.add(Arc::new(mech)).await.unwrap();
    let before = accel.flushes();
    for frame in 0..FRAMES {
        driver
            .step(&Epoch::standalone(dt(frame), 1.0))
            .await
            .unwrap();
    }
    let driven_one = accel.flushes() - before;

    // Three curtains: hand-rolled `join_all` (idle path) versus the driver.
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = accelerator(&world);
    let mut mechs = Vec::new();
    for (cols, rows, seed) in SHAPES {
        mechs.push(curtain(world.clone(), &accel, cols, rows, seed).await.0);
    }
    let before = accel.flushes();
    for frame in 0..FRAMES {
        let epoch = Epoch::standalone(dt(frame), 1.0);
        for r in futures::future::join_all(mechs.iter().map(|m| m.step(&epoch))).await {
            r.unwrap();
        }
    }
    let idle_three = accel.flushes() - before;

    let world = Arc::new(World::builder().usual::<f32>());
    let accel = accelerator(&world);
    let mut driver = Driver::new(accel.clone());
    for (cols, rows, seed) in SHAPES {
        let mech = curtain(world.clone(), &accel, cols, rows, seed).await.0;
        driver.add(Arc::new(mech)).await.unwrap();
    }
    let before = accel.flushes();
    for frame in 0..FRAMES {
        driver
            .step(&Epoch::standalone(dt(frame), 1.0))
            .await
            .unwrap();
    }
    let driven_three = accel.flushes() - before;

    let per_step = |n: usize| n as f64 / FRAMES as f64;
    eprintln!(
        "flushes per step: one curtain idle {:.2} driver {:.2}; three curtains join_all {:.2} driver {:.2}",
        per_step(idle_one),
        per_step(driven_one),
        per_step(idle_three),
        per_step(driven_three),
    );
    assert!(
        driven_one <= idle_one,
        "driver cost more flushes on one curtain: {driven_one} > {idle_one}"
    );
    assert!(
        driven_three <= idle_three,
        "driver cost more flushes on three curtains: {driven_three} > {idle_three}"
    );
}
