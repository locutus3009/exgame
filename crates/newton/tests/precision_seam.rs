// SPDX-License-Identifier: MIT

//! The precision seam between the world position and the f32 kernels (ACCELERATOR.md Part V).
//!
//! The kernels read base-pose Motors as `f32`. The open question was whether the translation
//! part of a base pose survives narrowing to an `f32` shader input when the mechanism sits far
//! from the world origin. The island design answers it in principle: each island keeps an `S`
//! anchor re-bound to its centre of mass after every step, and the body poses the kernels read
//! are LOCAL to that anchor. This file measures it.
//!
//! The same jointed chain is stepped on the GPU path at a reference position and translated by
//! a large offset (1e3, 1e6, 1e9 world units on every axis), and body positions relative to the
//! first body are compared after `STEPS` steps. Two geometries keep two seams apart:
//!
//! - **Kernel seam** (`GRID`): every construction coordinate is a multiple of 64, which `f32`
//!   represents exactly up to 2^30 > 1e9. Construction is then lossless at every offset, so any
//!   divergence can only come from stepping — from the kernels or from the anchor bookkeeping.
//! - **Ingestion seam** (`UNIT`): a unit-scale chain whose spacing is not dyadic. Bodies can only
//!   be placed by writing an ABSOLUTE `f32` pose before `add_body`, so the construction itself
//!   rounds every coordinate to the `f32` grid at the offset, before any anchor exists. The
//!   divergence is measured right after construction, before the first step.
//!
//! A third measurement contrasts the anchor carrier: the same `GRID` chain with `S = f32` in
//! place of the fixed-point `S`. Up to 1e6 only the absolute track of the centre of mass
//! degrades; at 1e9 the rounded anchor leaves the centre of mass, the local poses change and the
//! shape diverges too.

use aristotle::{Epoch, World, WorldId};
use joints::{AxialSpringDamper, Joint, SimpleSpringDamper};
use newton::field::UniformField;
use newton::integrator::{ImplicitIntegrator, Newton};
use newton::{Accelerator, Inert, Mechanism, RigidBody};
use peano::fixed::{Fix, types::I96F32};
use peano::prelude::*;
use std::sync::Arc;

/// The fixed-point anchor carrier the examples use for far-field scenes.
type Fx = Fix<I96F32>;

const N: usize = 5;
/// Kept short: at the 64-unit `GRID` scale the `f32` Newton step hits its relative tolerance
/// only after subdividing `dt` into tens to hundreds of spans (at every offset, the reference
/// included), so one run of 20 steps already takes several seconds.
const STEPS: usize = 20;
const DT: f32 = 1.0 / 60.0;
const OFFSETS: [f32; 3] = [1.0e3, 1.0e6, 1.0e9];

/// One chain shape. Every quantity scales with `h`, so both geometries swing alike.
#[derive(Clone, Copy)]
struct Geometry {
    name: &'static str,
    /// Body spacing along X; also the spring rest length.
    h: f32,
    /// Reference translation of the whole chain (applied on every axis).
    reference: f32,
}

/// Construction coordinates are multiples of 64: exact in `f32` at every offset up to 2^30.
const GRID: Geometry = Geometry {
    name: "grid-64",
    h: 64.0,
    reference: 0.0,
};

/// Unit-scale chain with a non-dyadic spacing: construction rounds at large offsets.
const UNIT: Geometry = Geometry {
    name: "unit-0.3",
    h: 0.3,
    reference: 0.0,
};

fn spring(world: Arc<World>, rest: f32, k: f32, c: f32) -> Joint<f32> {
    AxialSpringDamper::builder(world, SimpleSpringDamper)
        .rest(rest)
        .stiffness(k)
        .damping(c)
        .build()
}

/// A free chain of `N` unit masses along X at `offset` (on every axis), with non-uniform
/// transverse velocities so the springs stretch and swing, under uniform gravity.
async fn chain<S>(geo: Geometry, offset: f32) -> (Mechanism<f32, S>, Vec<WorldId>)
where
    S: Scalar + From<f32> + Into<f32>,
{
    let world = Arc::new(World::builder().usual::<f32>());
    let accel = Arc::new(Accelerator::<f32>::builder(world.clone()).build());
    let mech: Mechanism<f32, S> = Mechanism::new(ImplicitIntegrator::Newton(Newton::new(accel)));

    let h = geo.h;
    let mass = 1.0;
    let k = 50.0;
    let c = 0.1 * 2.0 * (k * mass * 0.5f32).sqrt();
    let mut ids = Vec::with_capacity(N);
    for i in 0..N {
        let pos = Vector3::from([offset + i as f32 * h, offset, offset]);
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        let vel = Vector3::from([
            0.02 * h * i as f32,
            0.15 * h * sign * (i as f32 + 1.0),
            0.05 * h * i as f32,
        ]);
        ids.push(
            mech.add_body(Inert::new(RigidBody::body_at_with_speed_and_mass(
                world.clone(),
                &pos,
                &vel,
                mass,
            )))
            .await,
        );
    }
    for i in 0..N - 1 {
        mech.connect(ids[i], vec![(spring(world.clone(), h, k, c), ids[i + 1])])
            .await;
    }
    mech.add_field(Box::new(UniformField::new(Vector3::from([
        0.0, 0.0, -9.81,
    ]))))
    .await;
    (mech, ids)
}

/// What one run leaves behind.
struct Run {
    /// Position of each body relative to body 0, read from the LOCAL poses (all bodies share
    /// one island and therefore one anchor). Never goes through an absolute `f32` coordinate.
    relative: Vec<[f32; 3]>,
    /// Largest |coordinate| of any local pose, i.e. the size of what the kernels are fed.
    max_local: f32,
    /// Centre of mass in the anchor carrier, lowered to f64 AFTER subtracting the offset.
    centroid_minus_offset: [f64; 3],
    /// First step error, if any.
    error: Option<String>,
}

async fn observe<S>(mech: &Mechanism<f32, S>, ids: &[WorldId], offset: f32) -> Run
where
    S: Scalar + From<f32> + Into<f32> + Into<f64>,
{
    let mut local = Vec::with_capacity(ids.len());
    for id in ids {
        let p = mech
            .inspect_body(*id, async |b: &RigidBody<f32>| b.position())
            .await
            .expect("body present");
        local.push([p[0], p[1], p[2]]);
    }
    let relative = local
        .iter()
        .map(|p| [p[0] - local[0][0], p[1] - local[0][1], p[2] - local[0][2]])
        .collect();
    let max_local = local
        .iter()
        .flat_map(|p| p.iter())
        .fold(0.0f32, |m, v| m.max(v.abs()));
    let com = mech.centroid().await.expect("non-empty mechanism").coords();
    // The subtraction happens in S, so for the fixed-point carrier it is exact.
    let off = S::from(offset);
    let centroid_minus_offset = [0, 1, 2].map(|k| {
        let d: f64 = (com[k] - off).into();
        d
    });
    Run {
        relative,
        max_local,
        centroid_minus_offset,
        error: None,
    }
}

async fn run<S>(geo: Geometry, offset: f32, steps: usize) -> Run
where
    S: Scalar + From<f32> + Into<f32> + Into<f64>,
{
    let (mech, ids) = chain::<S>(geo, offset).await;
    let epoch = Epoch::standalone(DT, 1.0);
    let mut error = None;
    for s in 0..steps {
        if let Err(e) = mech.step(&epoch).await {
            error = Some(format!("step {s}: {e:?}"));
            break;
        }
    }
    let mut r = observe(&mech, &ids, offset).await;
    r.error = error;
    r
}

/// Largest per-coordinate difference of the relative positions. NaN propagates as +inf.
fn divergence(a: &Run, b: &Run) -> f32 {
    let mut worst = 0.0f32;
    for (p, q) in a.relative.iter().zip(&b.relative) {
        for k in 0..3 {
            let d = (p[k] - q[k]).abs();
            worst = if d.is_nan() {
                f32::INFINITY
            } else {
                worst.max(d)
            };
        }
    }
    worst
}

/// Largest per-axis difference of the centre-of-mass displacement from its offset.
fn centroid_divergence(a: &Run, b: &Run) -> f64 {
    (0..3)
        .map(|k| (a.centroid_minus_offset[k] - b.centroid_minus_offset[k]).abs())
        .fold(0.0, f64::max)
}

/// Distance between adjacent `f32` values at `x`.
fn ulp(x: f32) -> f32 {
    let x = x.abs();
    f32::from_bits(x.to_bits() + 1) - x
}

/// One row of a kernel-seam table: the far run at `offset` against the reference run.
struct Row {
    offset: f32,
    relative: f32,
    centroid: f64,
    max_local: f32,
}

/// Steps the `GRID` chain at the reference and at every offset with anchor carrier `S`, and
/// prints the whole table BEFORE any assertion so a failure still reports every measurement.
async fn kernel_table<S>(label: &str) -> (Run, Vec<Row>)
where
    S: Scalar + From<f32> + Into<f32> + Into<f64>,
{
    let reference = run::<S>(GRID, GRID.reference, STEPS).await;
    assert!(
        reference.error.is_none(),
        "reference: {:?}",
        reference.error
    );
    println!(
        "{label} [{}], {STEPS} steps: reference max |local| = {}",
        GRID.name, reference.max_local
    );
    let mut rows = Vec::with_capacity(OFFSETS.len());
    for offset in OFFSETS {
        let far = run::<S>(GRID, offset, STEPS).await;
        assert!(far.error.is_none(), "offset {offset:e}: {:?}", far.error);
        let row = Row {
            offset,
            relative: divergence(&far, &reference),
            centroid: centroid_divergence(&far, &reference),
            max_local: far.max_local,
        };
        println!(
            "  offset {offset:>8.0e}: relative divergence = {:e}, centroid divergence = {:e}, \
             max |local| = {} (f32 ulp at offset = {:e})",
            row.relative,
            row.centroid,
            row.max_local,
            ulp(offset)
        );
        rows.push(row);
    }
    (reference, rows)
}

/// Kernel seam. Construction is exact at every offset, so this isolates stepping. With the
/// fixed-point anchor the local poses the kernels read do not grow with the offset, and the
/// relative positions after `STEPS` steps are BITWISE those of the reference run: translating
/// the anchor does not change a kernel input.
///
/// The centre of mass differs from the reference by `offset * 2^-32` (measured: 2.3e-7, 2.3e-4,
/// 2.3e-1). That is not anchor drift: `Mechanism::centroid` divides the mass-weighted anchor sum
/// by multiplying with `S::ONE / M`, and `1/5` rounds to the `I96F32` grid (2^-32), an error the
/// sum `M * offset` then scales. The bound asserted is that readout error, `M * offset * 2^-33`
/// per axis, plus 2^-20 for the `f32` COM increments the anchor absorbs.
#[tokio::test]
async fn far_offset_does_not_reach_the_kernels() {
    let (reference, rows) = kernel_table::<Fx>("kernel seam, S = I96F32").await;
    for r in &rows {
        assert_eq!(
            r.max_local, reference.max_local,
            "offset {:e}: the kernel inputs depend on the world position",
            r.offset
        );
        assert_eq!(
            r.relative, 0.0,
            "offset {:e}: relative positions diverge from the reference",
            r.offset
        );
        let readout = N as f64 * f64::from(r.offset) * 2f64.powi(-33) + 2f64.powi(-20);
        assert!(
            r.centroid <= readout,
            "offset {:e}: the fixed-point centre of mass drifts by {} > {readout}",
            r.offset,
            r.centroid
        );
    }
}

/// The anchor carrier, contrasted. With `S = f32` the anchor is rounded to the `f32` grid at the
/// offset every time it advances. Measured over `STEPS` steps:
///
/// - 1e3 and 1e6: the shape is still bitwise that of the reference; only the absolute COM track
///   degrades (2.3e-4 and 0.54 world units).
/// - 1e9 (`f32` ulp 64): the rounded anchor no longer sits on the COM, the local poses the
///   kernels read change (max |local| 130.1 against 128.2), and the shape itself diverges by
///   about 13 world units, a fifth of the 64-unit spacing.
///
/// So an `f32` anchor leaks the world position into the kernels once the anchor's ulp is
/// comparable to the motion per step; the fixed-point anchor does not. The assertions pin the
/// exact results at 1e3 and 1e6 and the observed bound at 1e9.
#[tokio::test]
async fn f32_anchor_contrast() {
    let (reference, rows) = kernel_table::<f32>("anchor carrier contrast, S = f32").await;
    for r in &rows[..2] {
        assert_eq!(r.relative, 0.0, "offset {:e}: the shape diverges", r.offset);
        assert_eq!(r.max_local, reference.max_local, "offset {:e}", r.offset);
    }
    let far = &rows[2];
    assert!(
        far.relative <= 16.0,
        "offset 1e9: shape divergence {} exceeds the observed bound",
        far.relative
    );
}

/// Ingestion seam. Bodies enter a mechanism through an absolute `f32` pose, so the
/// construction rounds each coordinate to the `f32` grid at the offset BEFORE the island anchor
/// takes over. Measured right after construction: the relative-position error is bounded by
/// one `f32` ulp at the offset (two half-ulp roundings), and nothing smaller is reachable
/// through the public construction API. At 1e9 the ulp (64) exceeds the whole chain, which
/// collapses onto a single point.
#[tokio::test]
async fn construction_rounds_to_the_f32_grid_at_the_offset() {
    let reference = run::<Fx>(UNIT, UNIT.reference, 0).await;
    println!(
        "ingestion seam [{}], S = I96F32, after construction (0 steps):",
        UNIT.name
    );
    for offset in OFFSETS {
        let far = run::<Fx>(UNIT, offset, 0).await;
        let rel = divergence(&far, &reference);
        let bound = ulp(offset);
        println!(
            "  offset {offset:>8.0e}: relative divergence = {rel:e} (ulp at offset = {bound:e})"
        );
        assert!(
            rel <= bound,
            "offset {offset:e}: construction error {rel} exceeds one ulp {bound}"
        );
    }
}
