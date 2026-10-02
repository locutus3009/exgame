// SPDX-License-Identifier: MIT

//! Regression test for the stability of the cloth from the `cloth_grid` demo (2026-07-21).
//!
//! The demo blew apart, and this did not reproduce on a pair of bodies in f64 — it needs exactly its
//! parameters: carrier type `f32`, a grid with a pinned top row, gravity
//! with wind, and a VARIABLE step taken from real frame time.
//!
//! Why the step is decisive here. A cloth node weighs `MASS_TOTAL/(COLS·ROWS) = 1/18`
//! kg with `K = 100 N/m`, i.e. `ω = sqrt(2K/m) ≈ 60 rad/s`, and the natural
//! period is about 0.1 s. Any step of comparable scale passes through whole
//! periods at once, and what is required of the scheme is not accuracy (there can be none), but that
//! it does not fall apart and remains usable for the following frames.

use aristotle::{Epoch, World, WorldId};
use clifford::pga3::Twist;
use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
use newton::field::UniformField;
use newton::integrator::{ImplicitIntegrator, Newton};
use newton::{Accelerator, Inert, Inertia, Mechanism, RigidBody};
use peano::prelude::*;
use std::cmp::Ordering;
use std::sync::Arc;

// The grid is deliberately small: both regressions caught here — in step control and in line
// search — do not depend on size, while the GEMM cost grows as m².
const COLS: usize = 2;
const ROWS: usize = 4;
const K: f32 = 100.0;
const ZETA: f32 = 0.7;
const MASS_TOTAL: f32 = 1.0;
const KICK_SPEED: f32 = 0.5;
const WIDTH: f32 = 0.5;
const G: f32 = 9.81;
const WIND: f32 = 1.0;
const MAX_DT: f32 = 0.1;

fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
    AxialSpringDamper::builder(world, SimpleSpringDamper)
        .rest(rest)
        .stiffness(k)
        .damping(c)
        .build()
}

/// One curtain — as in the demo: a uniform COLS×ROWS grid in the XZ plane,
/// the top row kinematic, a few nodes receive a gust of wind.
async fn build_curtain(
    world: Arc<World>,
    accel: &Arc<Accelerator<f32>>,
) -> (Mechanism<f32, f32>, Vec<WorldId>) {
    let h = WIDTH / (COLS as f32 - 1.0);
    let m = MASS_TOTAL / (COLS as f32 * ROWS as f32);
    let c = ZETA * 2.0 * (K * (m * 0.5)).sqrt();

    let mech: Mechanism<f32, f32> =
        Mechanism::new(ImplicitIntegrator::Newton(Newton::new(accel.clone())));

    let mut ids: Vec<WorldId> = Vec::with_capacity(COLS * ROWS);
    for row in 0..ROWS {
        for col in 0..COLS {
            let pos = Vector3::from([col as f32 * h, 0.0, row as f32 * h]);
            let id = if row == ROWS - 1 {
                let b = RigidBody::new(world.clone(), Inertia::Kinematic);
                b.pose.write(Twist::new(&pos, &Vector3::ZERO).exp(1.0));
                mech.add_body(Inert::new(b)).await
            } else {
                // Gust: a few nodes blow in +Y, as in the demo.
                let vel = if (row * COLS + col) % 4 == 1 {
                    Vector3::from([0.0, KICK_SPEED, 0.0])
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

    for row in 0..ROWS {
        for col in 0..COLS {
            let a = ids[row * COLS + col];
            if col + 1 < COLS {
                let b = ids[row * COLS + col + 1];
                mech.connect(a, vec![(spring(world.clone(), h, K, c), b)])
                    .await;
            }
            if row + 1 < ROWS {
                let b = ids[(row + 1) * COLS + col];
                mech.connect(a, vec![(spring(world.clone(), h, K, c), b)])
                    .await;
            }
        }
    }

    mech.add_field(Box::new(UniformField::new(Vector3::from([0.0, 0.0, -G]))))
        .await;
    mech.add_field(Box::new(UniformField::new(Vector3::from([0.0, WIND, 0.0]))))
        .await;
    (mech, ids)
}

/// The largest coordinate over all nodes — a blow-up detector not tied to
/// a specific trajectory.
async fn worst_coordinate(mech: &Mechanism<f32, f32>, ids: &[WorldId]) -> f32 {
    let mut worst: f32 = 0.0;
    for id in ids {
        if let Some(p) = mech.body_absolute_position(*id).await {
            for axis in 0..3 {
                let v = p[axis].abs();
                // NaN is unordered with respect to everything, so it too must end up
                // here: the blow-up detector must notice a non-finite coordinate.
                if !matches!(
                    v.partial_cmp(&worst),
                    Some(Ordering::Less | Ordering::Equal)
                ) {
                    worst = v;
                }
            }
        }
    }
    worst
}

/// A curtain under a variable step, with frame drops up to `MAX_DT`, must
/// remain a connected rag rather than blow apart.
///
/// The cloth hangs from the top row and cannot go further than its length plus sag:
/// five spans of `h = 0.25` is 1.25 m, so the 5 m threshold catches precisely
/// a blow-up, not oscillations.
#[tokio::test]
async fn cloth_survives_frame_time_spikes() {
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = Arc::new(Accelerator::<f32>::builder(world.clone()).build());
    let (mech, ids) = build_curtain(world, &accel).await;

    // Frame profile: mostly 60 Hz, but every 17th frame drops to
    // MAX_DT, as in the demo on a window switch or garbage collection.
    for frame in 0..100 {
        let dt = if frame % 17 == 16 { MAX_DT } else { 1.0 / 60.0 };
        mech.step(&Epoch::standalone(dt, 1.0)).await.unwrap();

        let worst = worst_coordinate(&mech, &ids).await;
        assert!(
            worst < 5.0,
            "curtain blew apart at frame {frame} (dt={dt}): max coordinate {worst}"
        );
    }
}

/// A large step. Not about accuracy — about the scheme not falling apart.
///
/// At `dt = 10 s` a cloth node (`ω ≈ 60 rad/s`) passes through about a hundred natural
/// periods per step, so there can be no meaningful trajectory here in principle;
/// there is exactly one requirement — remain a finite connected rag and not fly away.
///
/// The subdivision limit is also visible here: it halves the step at most `MAX_DEPTH = 8`
/// times, i.e. the minimum span is `dt/256`. At `dt = 10 s` that is 39 ms, and
/// that is enough for the initial guess; but the margin is finite, and the test pins down where we stand.
#[tokio::test]
async fn cloth_survives_absurdly_large_steps() {
    for &dt in &[1.0f32, 10.0] {
        let world = Arc::new(World::builder().usual::<f32>());
        let accel = Arc::new(Accelerator::<f32>::builder(world.clone()).build());
        let (mech, ids) = build_curtain(world, &accel).await;

        // One huge step, then back to a normal frame: the island must
        // not only survive it but also remain usable afterwards.
        mech.step(&Epoch::standalone(dt, 1.0)).await.unwrap();
        let after_spike = worst_coordinate(&mech, &ids).await;
        for _ in 0..60 {
            mech.step(&Epoch::standalone(1.0 / 60.0, 1.0))
                .await
                .unwrap();
        }
        let settled = worst_coordinate(&mech, &ids).await;

        println!(
            "dt={dt:>5.1}  after spike {after_spike:>12.4e}  one second later {settled:>12.4e}"
        );
        assert!(
            after_spike < 100.0,
            "step dt={dt} broke the scheme: max coordinate {after_spike}"
        );
        assert!(
            settled < 100.0,
            "after step dt={dt} the island did not recover: {settled}"
        );
    }
}

/// INVARIANT: lengthening the step reduces accuracy, but NOT stability.
///
/// The implicit midpoint is A-stable, and the subdivision inside the integrator must
/// make up for whatever is missing. So for any `dt` the suspended cloth must
/// remain a hanging rag: accuracy on a large step may be anything,
/// but energy must not be pumped in from step to step.
///
/// The metric is the extent of the grid. For cloth suspended by its top row it is bounded by its
/// length (5 spans of 0.25 m) plus sag, so growth of the extent from step to
/// step is precisely pumping.
///
/// Catches a specific regression: subdivision used to continue only while it
/// improved `best`, which makes sense only with an EXACT linear-system solver. With
/// an approximate one `best` hits its floor, subdivision gave up at depth 1-2,
/// and a knowingly unconverged step was committed — and that does not conserve energy, it pumps
/// it. Observed as `1.25 → 2.66 → 14.5 → 250 → 692 → 19106 → NaN`.
#[tokio::test]
async fn longer_steps_lose_accuracy_but_not_stability() {
    for &dt in &[1.0f32 / 60.0, 0.5, 1.0] {
        let world = Arc::new(World::builder().usual::<f32>());
        let accel = Arc::new(Accelerator::<f32>::builder(world.clone()).build());
        let (mech, ids) = build_curtain(world, &accel).await;

        let mut trail = Vec::new();
        for _ in 0..20 {
            mech.step(&Epoch::standalone(dt, 1.0)).await.unwrap();
            trail.push(worst_coordinate(&mech, &ids).await);
        }
        let last = *trail.last().unwrap();
        println!("dt={dt:>6.3}  final extent {last:>10.3e}");
        assert!(
            last < 10.0,
            "dt={dt}: cloth pumped up to an extent of {last} (trajectory {trail:?})"
        );
    }
}

/// A 30-second step is ~300 natural periods of the cloth at once. The requirement
/// is unchanged: not accuracy, but that the scheme stays finite.
///
/// Regression test for a destructive line search: while the trial step wrote into the same slots
/// as the iterate, a rejected trial was not rolled back, eight halvings accumulated
/// `dv·(1 + ½ + ¼ + …)`, Newton never converged, and at the bottom of the recursion
/// an unconverged step was committed. Here this produced NaN.
#[tokio::test]
async fn cloth_survives_a_thirty_second_step() {
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = Arc::new(Accelerator::<f32>::builder(world.clone()).build());
    let (mech, ids) = build_curtain(world, &accel).await;
    mech.step(&Epoch::standalone(30.0, 1.0)).await.unwrap();
    let worst = worst_coordinate(&mech, &ids).await;
    assert!(worst < 100.0, "30 s step broke the scheme: {worst}");
}
