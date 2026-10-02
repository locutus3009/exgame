// SPDX-License-Identifier: MIT

//! How the step grows with the number of cloth nodes.
//!
//! This is THE main characteristic of the machinery: on a small grid the step is bound by
//! submit latency and barely depends on what is being computed, while on a large one it is bound
//! by whatever grows as `m²` (the Newton–Schulz block products, matrix
//! assembly and what the solver reads from the slots on the CPU). The curve shows where
//! one gives way to the other.
//!
//! `#[ignore]` — this is a measurement, not a regression: run
//! `cargo test --release -p newton --test cloth_scaling -- --ignored --nocapture`.

use aristotle::{Epoch, World, WorldId};
use clifford::pga3::Twist;
use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
use newton::field::UniformField;
use newton::integrator::{ImplicitIntegrator, Newton};
use newton::{Accelerator, Inert, Inertia, Mechanism, RigidBody};
use peano::prelude::*;
use std::sync::Arc;
use std::time::Instant;

const K: f32 = 100.0;
const ZETA: f32 = 0.7;
const MASS_TOTAL: f32 = 1.0;
const KICK_SPEED: f32 = 0.5;
const WIDTH: f32 = 0.5;
const G: f32 = 9.81;
const WIND: f32 = 1.0;

fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
    AxialSpringDamper::builder(world, SimpleSpringDamper)
        .rest(rest)
        .stiffness(k)
        .damping(c)
        .build()
}

/// A COLS×ROWS curtain: the top row is kinematic, the rest are unknowns.
/// Springs along both grid axes, as in `cloth_grid_stability`.
async fn curtain(
    world: Arc<World>,
    accel: &Arc<Accelerator<f32>>,
    cols: usize,
    rows: usize,
) -> Mechanism<f32, f32> {
    let h = WIDTH / (cols as f32 - 1.0);
    let m = MASS_TOTAL / (cols as f32 * rows as f32);
    let c = ZETA * 2.0 * (K * (m * 0.5)).sqrt();

    let mech: Mechanism<f32, f32> =
        Mechanism::new(ImplicitIntegrator::Newton(Newton::new(accel.clone())));

    let mut ids: Vec<WorldId> = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let pos = Vector3::from([col as f32 * h, 0.0, row as f32 * h]);
            let id = if row == rows - 1 {
                let b = RigidBody::new(world.clone(), Inertia::Kinematic);
                b.pose.write(Twist::new(&pos, &Vector3::ZERO).exp(1.0));
                mech.add_body(Inert::new(b)).await
            } else {
                let vel = if (row * cols + col) % 4 == 1 {
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
    mech.add_field(Box::new(UniformField::new(Vector3::from([0.0, WIND, 0.0]))))
        .await;
    mech
}

#[tokio::test]
#[ignore]
async fn cloth_step_cost_by_body_count() {
    const DT: f32 = 1.0 / 60.0;
    const WARMUP: usize = 20;
    let steps: usize = std::env::var("CLOTH_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);

    println!(
        "{:>5} {:>5} {:>5} {:>4} {:>9} {:>9} {:>9} {:>9} {:>11}",
        "cols", "rows", "bodies", "m", "median", "mean", "p95", "max", "median/m"
    );
    // The ceiling today is 5 columns; beyond that the grid no longer computes.
    let sizes: Vec<(usize, usize)> = match std::env::var("CLOTH_N") {
        Ok(v) => vec![(v.parse().unwrap(), 2 * v.parse::<usize>().unwrap())],
        Err(_) => vec![(2, 4), (3, 6), (4, 8), (5, 10)],
    };
    for &(cols, rows) in &sizes {
        let world = Arc::new(World::builder().usual::<f32>());
        let accel = Arc::new(Accelerator::<f32>::builder(world.clone()).build());
        let mech = curtain(world, &accel, cols, rows).await;

        for _ in 0..WARMUP {
            mech.step(&Epoch::standalone(DT, 1.0)).await.unwrap();
        }
        // Each step separately: subdivision is a rare and heavy event, so
        // a mean over a handful of steps says more about whether it landed in the sample
        // than about the cost of a step. The median is the typical frame, the tail is the cost of subdivision.
        let mut ms: Vec<f64> = Vec::with_capacity(steps);
        for _ in 0..steps {
            let t0 = Instant::now();
            mech.step(&Epoch::standalone(DT, 1.0)).await.unwrap();
            ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let mean: f64 = ms.iter().sum::<f64>() / ms.len() as f64;
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ms[ms.len() / 2];
        let p95 = ms[ms.len() * 95 / 100];
        let max = *ms.last().unwrap();
        let m = cols * (rows - 1);
        println!(
            "{cols:>5} {rows:>5} {:>5} {m:>4} {med:>9.2} {mean:>9.2} {p95:>9.2} {max:>9.2} {:>11.3}",
            cols * rows,
            med / m as f64
        );
    }
}
