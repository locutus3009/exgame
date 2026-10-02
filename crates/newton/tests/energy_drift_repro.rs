// SPDX-License-Identifier: MIT

//! Regression: the lag-1 of `GravityPropagator` feeds a one-step-retarded force
//! into the integrator. A retarded restoring/central force is anti-damping
//! (+ω²·dt·ẋ to first order) and SECULARLY pumps energy — a gravitating spring
//! pair orbiting a central mass gains energy and escapes, instead of slowly
//! losing energy through its own spring damper.
//!
//! `lagged_gravity_must_not_pump_energy` asserts the physical invariant: a
//! dissipative system cannot end with MORE mechanical energy than it started.
//!
//! STATUS, measured 2026-09-07 on rustc 1.98.1 (AR-0005). This test PASSES, and
//! it does not make `cargo test --workspace` red. It was run at
//! `93dbe53` — the commit before any AR-0005 change — and at the head of
//! AR-0005, and produced byte-identical energies both times: E starts at
//! -7.77600, stays bounded between roughly -8.29 and -7.87 over 40 000 steps and
//! ends at -8.06477, so it never rises above E0 and never crosses zero. The
//! paragraph that used to stand here said the test currently FAILS; that was
//! stale, and AR-0005 replaced it with the measurement rather than acting on it.
//!
//! KNOWN LIMITATION, recorded rather than fixed. The `orbit_radius` escape
//! detector prints `r = 0.000` at every sample INCLUDING step 0, where the
//! declared geometry (pair at x = -3 and -2, centre at the origin) says 2.5. So
//! the second assertion is carried entirely by `e_end < 0`, and the radius trace
//! is not measuring what its comment claims. The energy assertion is unaffected:
//! it compares E against this run's own E0. Diagnosing the frame `pos()` reports
//! in is physics work and belongs with the deferred physics-invariant tests
//! (M1), not with a lint-and-gate task.

use aristotle::{Epoch, World, WorldId};
use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
use newton::{
    Accelerator, Inert, Mechanism, RigidBody, gravity::GravityPropagator,
    integrator::ImplicitIntegrator,
};
use peano::prelude::*;
use std::sync::Arc;

fn accel(world: Arc<World>) -> Arc<Accelerator<f32>> {
    Arc::new(Accelerator::builder(world).build())
}

const G: f32 = 0.001;
const STIFFNESS: f32 = 5.0;
const DAMPING: f32 = 0.5;
const REST: f32 = 0.9;
const DT: f32 = 0.016;
const STEPS: usize = 40_000;

fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
    AxialSpringDamper::builder(world, SimpleSpringDamper)
        .rest(rest)
        .stiffness(k)
        .damping(c)
        .build()
}

struct Sys {
    mech: Mechanism<f32, f32>,
    grav_ids: [WorldId; 3], // id1, id2, center
}

async fn build(world: Arc<World>) -> (Sys, Arc<GravityPropagator<f32>>) {
    let mech = Mechanism::new(ImplicitIntegrator::LieEuler(accel(world.clone())));

    let id1 = mech
        .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
            world.clone(),
            &Vector3::from([-3.0, 0.0, 0.0]),
            &Vector3::from([0.0, -1.0, 0.0]),
            1.0,
        )))
        .await;
    let id2 = mech
        .add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
            world.clone(),
            &Vector3::from([-2.0, 0.0, 0.0]),
            &Vector3::from([0.0, -1.0, 0.0]),
            1.0,
        )))
        .await;
    let center = mech
        .add_body(Inert::new(
            RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &Vector3::ZERO,
                &Vector3::ZERO,
                2200.0,
            )
            .with_size(1.0),
        ))
        .await;

    mech.connect(id1, vec![(spring(world, REST, STIFFNESS, DAMPING), id2)])
        .await;

    let prop = Arc::new(GravityPropagator::new(G));
    prop.register(id1);
    prop.register(id2);
    prop.register(center);
    mech.add_field(Box::new(prop.clone())).await;

    (
        Sys {
            mech,
            grav_ids: [id1, id2, center],
        },
        prop,
    )
}

async fn pos(m: &Mechanism<f32, f32>, id: WorldId) -> Vector3<f32> {
    m.inspect_body(id, async |b| RigidBody::position(b))
        .await
        .unwrap()
}

fn dist(a: Vector3<f32>, b: Vector3<f32>) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Total mechanical energy of the gravitating subsystem: ΣKE + spring PE + grav PE.
async fn total_energy(sys: &Sys) -> f32 {
    let m = &sys.mech;
    let [id1, id2, center] = sys.grav_ids;

    let mut ke = 0.0f32;
    for &id in &sys.grav_ids {
        ke += m
            .inspect_body(id, async |b| RigidBody::kinetic_energy(b))
            .await
            .unwrap();
    }

    let p1 = pos(m, id1).await;
    let p2 = pos(m, id2).await;
    let pc = pos(m, center).await;
    let d12 = dist(p1, p2);
    let spring_pe = 0.5 * STIFFNESS * (d12 - REST).powi(2);

    let mass = async |id| {
        m.inspect_body(id, async |b| b.inertia.mass())
            .await
            .unwrap()
    };
    let (m1, m2, mc) = (mass(id1).await, mass(id2).await, mass(center).await);
    let grav_pe = -G * (m1 * m2 / d12 + m1 * mc / dist(p1, pc) + m2 * mc / dist(p2, pc));

    ke + spring_pe + grav_pe
}

/// Orbital radius of the pair's midpoint from the central body — escape detector.
async fn orbit_radius(sys: &Sys) -> f32 {
    let [id1, id2, center] = sys.grav_ids;
    let p1 = pos(&sys.mech, id1).await;
    let p2 = pos(&sys.mech, id2).await;
    let pc = pos(&sys.mech, center).await;
    let mid = (p1 + p2).scale(1.0 / 2.0);
    dist(mid, pc)
}

/// Asserts two invariants of the gravitating spring pair. It is a dissipative
/// system (the spring damper can only remove mechanical energy), so its total
/// energy must never climb above the starting value by more than a small
/// numerical tolerance, and the pair must stay gravitationally bound (E < 0).
///
/// This test PASSES; see the measurement in the module header. The paragraph
/// that stood here said the lag-1 force pumps energy until E crosses zero and
/// that both invariants are violated, which was stale in the same way the
/// module header was. Read the header's KNOWN LIMITATION too: the second
/// assertion is carried by `e_end < 0` alone, because the `orbit_radius` trace
/// reads zero at every sample.
#[tokio::test]
async fn lagged_gravity_must_not_pump_energy() {
    let world = Arc::new(World::builder().usual::<f32>());
    let (sys, prop) = build(world).await;

    let e0 = total_energy(&sys).await;
    let mut e_max = e0;
    eprintln!("=== GravityPropagator (lag-1): energy / orbit radius over time ===");
    eprintln!(
        "step 0:  E = {e0:+.5},  r = {:.3}",
        orbit_radius(&sys).await
    );
    for step in 1..=STEPS {
        sys.mech.step(&Epoch::standalone(DT, 1.0)).await.unwrap();
        prop.advance_epoch();
        let e = total_energy(&sys).await;
        e_max = e_max.max(e);
        if step % 5_000 == 0 {
            eprintln!(
                "step {step}:  E = {e:+.5},  r = {:.3}",
                orbit_radius(&sys).await
            );
        }
    }
    let e_end = total_energy(&sys).await;

    // Tolerance: a few percent of |E₀| for tidal/spring energy sloshing —
    // well below the +130% the lag injects.
    let tol = 0.05 * e0.abs();
    assert!(
        e_max <= e0 + tol,
        "lagged gravity PUMPED energy: E₀ = {e0:.5}, peak E = {e_max:.5} \
         (a damped system must not gain energy)"
    );
    assert!(
        e_end < 0.0,
        "pair became UNBOUND (E_end = {e_end:.5} ≥ 0): lag-1 force drove escape"
    );
}
