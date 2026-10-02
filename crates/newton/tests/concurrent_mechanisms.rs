// SPDX-License-Identifier: MIT

//! Several mechanisms on ONE accelerator, stepping simultaneously, must
//! get exactly the same result as when run individually.
//!
//! This is a requirement on the transport, not on the physics. A stage round is one message
//! occupying a contiguous region of the batch table; rounds of different mechanisms
//! go into the same table and ship in one dispatch. So they can be mixed up
//! with each other in exactly two ways: by shifting the region (then a mechanism
//! computes from someone else's rows) or by distributing the result/fatals back incorrectly
//! (then one mechanism's error poisons another). The test catches both: the result must be
//! BITWISE the same as in a solo run.

use aristotle::{Epoch, World, WorldId};
use clifford::pga3::Twist;
use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
use newton::field::UniformField;
use newton::integrator::{ImplicitIntegrator, Newton};
use newton::{Accelerator, Inert, Inertia, Mechanism, RigidBody};
use peano::prelude::*;
use std::sync::Arc;

const K: f32 = 100.0;
const G: f32 = 9.81;

fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
    AxialSpringDamper::builder(world, SimpleSpringDamper)
        .rest(rest)
        .stiffness(k)
        .damping(c)
        .build()
}

/// A chain of `n` nodes suspended by the top one: different `seed`s give different
/// initial velocities so that mechanisms do not coincide by accident.
async fn chain(
    world: Arc<World>,
    accel: &Arc<Accelerator<f32>>,
    n: usize,
    seed: f32,
) -> (Mechanism<f32, f32>, Vec<WorldId>) {
    let h = 0.25;
    let mass = 1.0 / n as f32;
    let c = 0.7 * 2.0 * (K * (mass * 0.5)).sqrt();
    let mech: Mechanism<f32, f32> =
        Mechanism::new(ImplicitIntegrator::Newton(Newton::new(accel.clone())));

    let mut ids = Vec::with_capacity(n);
    for i in 0..n {
        let pos = Vector3::from([0.0, 0.0, i as f32 * h]);
        let id = if i == n - 1 {
            let b = RigidBody::new(world.clone(), Inertia::Kinematic);
            b.pose.write(Twist::new(&pos, &Vector3::ZERO).exp(1.0));
            mech.add_body(Inert::new(b)).await
        } else {
            mech.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &pos,
                &Vector3::from([seed * (i as f32 + 1.0), 0.0, 0.0]),
                mass,
            )))
            .await
        };
        ids.push(id);
    }
    for i in 0..n - 1 {
        mech.connect(ids[i], vec![(spring(world.clone(), h, K, c), ids[i + 1])])
            .await;
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

/// Three mechanisms of different sizes on a shared accelerator, stepping interleaved,
/// versus the same three stepping one at a time, each on its own accelerator.
#[tokio::test]
async fn mechanisms_sharing_one_accelerator_match_solo_runs() {
    use futures::future::join_all;
    const DT: f32 = 1.0 / 60.0;
    const STEPS: usize = 40;
    // Different sizes — so that the mechanisms' rounds have DIFFERENT lengths: identical ones
    // would pass even with the region shifted by a whole number of rounds.
    const SHAPES: [(usize, f32); 3] = [(4, 0.3), (7, -0.5), (5, 0.8)];

    // Individually: each has its own world and its own accelerator, no foreign messages.
    let mut solo = Vec::new();
    for (n, seed) in SHAPES {
        let world = Arc::new(World::builder().usual::<f32>());
        let accel = Arc::new(Accelerator::<f32>::builder(world.clone()).build());
        let (mech, ids) = chain(world, &accel, n, seed).await;
        for _ in 0..STEPS {
            mech.step(&Epoch::standalone(DT, 1.0)).await.unwrap();
        }
        solo.push(positions(&mech, &ids).await);
    }

    // Together: one world, ONE accelerator, steps via join_all — i.e. their
    // rounds land in one and the same batch table.
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = Arc::new(Accelerator::<f32>::builder(world.clone()).build());
    let mut together = Vec::new();
    for (n, seed) in SHAPES {
        together.push(chain(world.clone(), &accel, n, seed).await);
    }
    let epoch = Epoch::standalone(DT, 1.0);
    for _ in 0..STEPS {
        let futs = together.iter().map(|(m, _)| m.step(&epoch));
        for r in join_all(futs).await {
            r.unwrap();
        }
    }

    for (i, ((mech, ids), want)) in together.iter().zip(&solo).enumerate() {
        let got = positions(mech, ids).await;
        assert_eq!(got.len(), want.len(), "mechanism {i}: different body count");
        for (b, (g, w)) in got.iter().zip(want).enumerate() {
            for axis in 0..3 {
                assert_eq!(
                    g[axis], w[axis],
                    "mechanism {i}, body {b}, axis {axis}: together {} versus individually {} \
                     — the mechanisms' rounds got mixed up",
                    g[axis], w[axis]
                );
            }
        }
    }
}
