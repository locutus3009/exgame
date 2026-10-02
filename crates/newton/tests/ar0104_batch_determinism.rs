// SPDX-License-Identifier: MIT

//! Batch composition must change only timing, never physics
//! (ACCELERATOR.md Part V, the determinism invariant).
//!
//! The same chain is stepped on accelerators that pack its work differently: the
//! default flush policy, a batch-size hint of ONE row (so every message goes out
//! in a flush of its own), and a submission bound of ONE message (so no two
//! messages are ever in flight together). The final positions must be BITWISE
//! identical. A difference would mean a kernel's output depends on where in the
//! batch table its rows landed, or on what else shared the dispatch.

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
const N: usize = 6;
const DT: f32 = 1.0 / 60.0;
const STEPS: usize = 30;

#[derive(Clone, Copy, Debug)]
enum Scheme {
    Newton,
    SymplecticEuler,
    ExplicitEuler,
}

#[derive(Clone, Copy, Debug)]
enum Policy {
    Default,
    OneRowBatches,
    OneMessageInFlight,
}

fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
    AxialSpringDamper::builder(world, SimpleSpringDamper)
        .rest(rest)
        .stiffness(k)
        .damping(c)
        .build()
}

fn accelerator(world: Arc<World>, policy: Policy) -> Arc<Accelerator<f32>> {
    let b = Accelerator::<f32>::builder(world);
    Arc::new(
        match policy {
            Policy::Default => b,
            Policy::OneRowBatches => b.batch_size(1),
            Policy::OneMessageInFlight => b.max_pending(1),
        }
        .build(),
    )
}

/// A hanging chain of `N` nodes, the top one kinematic, under gravity, with a
/// sideways kick so every joint is loaded off-axis. Returns the final
/// positions after `STEPS` steps.
async fn run(scheme: Scheme, policy: Policy) -> Vec<[f32; 3]> {
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = accelerator(world.clone(), policy);
    let integrator = match scheme {
        Scheme::Newton => ImplicitIntegrator::Newton(Newton::new(accel.clone())),
        Scheme::SymplecticEuler => ImplicitIntegrator::SymplecticEuler(accel.clone()),
        Scheme::ExplicitEuler => ImplicitIntegrator::ExplicitEuler(accel.clone()),
    };
    let mech: Mechanism<f32, f32> = Mechanism::new(integrator);

    let h = 0.25;
    let mass = 1.0 / N as f32;
    let c = 0.7 * 2.0 * (K * (mass * 0.5)).sqrt();
    let mut ids: Vec<WorldId> = Vec::with_capacity(N);
    for i in 0..N {
        let pos = Vector3::from([0.0, 0.0, i as f32 * h]);
        let id = if i == N - 1 {
            let b = RigidBody::new(world.clone(), Inertia::Kinematic);
            b.pose.write(Twist::new(&pos, &Vector3::ZERO).exp(1.0));
            mech.add_body(Inert::new(b)).await
        } else {
            mech.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &pos,
                &Vector3::from([0.4 * (i as f32 + 1.0), -0.2, 0.0]),
                mass,
            )))
            .await
        };
        ids.push(id);
    }
    for i in 0..N - 1 {
        mech.connect(ids[i], vec![(spring(world.clone(), h, K, c), ids[i + 1])])
            .await;
    }
    mech.add_field(Box::new(UniformField::new(Vector3::from([0.0, 0.0, -G]))))
        .await;

    let epoch = Epoch::standalone(DT, 1.0);
    for _ in 0..STEPS {
        mech.step(&epoch).await.unwrap();
    }

    let mut out = Vec::with_capacity(N);
    for id in &ids {
        let p = mech.body_absolute_position(*id).await.unwrap();
        out.push([p[0], p[1], p[2]]);
    }
    out
}

async fn assert_policy_independent(scheme: Scheme) {
    let want = run(scheme, Policy::Default).await;
    // The chain must actually have moved, or equality proves nothing.
    assert!(
        want.iter().any(|p| p[0].abs() > 1e-3),
        "{scheme:?}: the chain did not move: {want:?}"
    );
    for policy in [Policy::OneRowBatches, Policy::OneMessageInFlight] {
        let got = run(scheme, policy).await;
        for (b, (g, w)) in got.iter().zip(&want).enumerate() {
            for axis in 0..3 {
                assert_eq!(
                    g[axis].to_bits(),
                    w[axis].to_bits(),
                    "{scheme:?}, {policy:?}, body {b}, axis {axis}: {} versus default {} \
                     — batch composition changed the physics",
                    g[axis],
                    w[axis]
                );
            }
        }
    }
}

#[tokio::test]
async fn newton_chain_is_bit_identical_across_batch_policies() {
    assert_policy_independent(Scheme::Newton).await;
}

#[tokio::test]
async fn symplectic_euler_chain_is_bit_identical_across_batch_policies() {
    assert_policy_independent(Scheme::SymplecticEuler).await;
}

#[tokio::test]
async fn explicit_euler_chain_is_bit_identical_across_batch_policies() {
    assert_policy_independent(Scheme::ExplicitEuler).await;
}
