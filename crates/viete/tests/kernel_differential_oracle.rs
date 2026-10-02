// SPDX-License-Identifier: MIT

#![cfg(feature = "lua")]

//! Cross-oracle (the kernel-vs-`Differential` gate): the build.rs joint kernel construction
//! — `pose = base ∘ exp(δ)`, δ value 0, UNIT retraction — traced through viete and
//! run, must agree with a direct CPU `Differential::jacobian` at the same
//! (base pose, velocity) point.
//!
//! Why this test exists. viete's own checks (`jacobian_smoke`) only compare the
//! emitted Lua against the trace's OWN f64 carrier — they cannot catch a kernel
//! whose retraction CONVENTION diverges from `Differential` (e.g. baking a `½dt`
//! into the pose columns instead of unit retraction). This is the only check that
//! pins the traced kernel to `Differential`, so the CPU stand-in and the compiled
//! kernel are provably interchangeable fillers of the per-connection Jacobian slot.
//!
//! `base_diff_kernel` below MIRRORS `crates/newton/build.rs::jac_kernel`; keep them
//! in step (input layout NIN=30, unit retraction, same component/axis order).

use aristotle::Epoch;
use aristotle::{World, WorldId};
use bytemuck::Pod;
use clifford::pga3::{Differential, Dynamics, Motor, Twist, Wrench};
use clifford::{Lift, Tangent};
use joints::{AxialSpringDamper, CriticallyDampedWarped, Joint, JointEdge};
use peano::prelude::*;
use std::sync::Arc;
use viete::{InputFact, InputRef, Sym, Tracer};

const NIN: usize = 30; // 12 velocity values + 8+8 base-pose components + dt + warp
const NPARAM: usize = 10; // AxialSpringDamper constants (a3, b3, rest, k, c, soft)
const NOUT_FULL: usize = 300; // 12 components × (value + 24 partials)

fn world() -> Arc<World> {
    Arc::new(
        World::builder()
            .with_storage::<Sym>()
            .with_storage::<Vector3<Sym>>()
            .with_storage::<f64>()
            .with_storage::<Vector3<f64>>()
            .with_storage::<[Wrench<Sym>; 2]>()
            .with_storage::<[Wrench<f64>; 2]>()
            .with_storage::<[[Wrench<Sym>; 24]; 2]>()
            .with_storage::<[[Wrench<f64>; 24]; 2]>()
            .build(),
    )
}

/// Sym joint: constants are PARAM references (order mirrors `params()`). Anchors
/// offset from each COM so the wrench carries torque too (densely-coupled block).
fn build_joint_sym(world: Arc<World>) -> JointEdge<Sym> {
    let j = AxialSpringDamper::<Sym>::builder(world.clone(), CriticallyDampedWarped)
        .a(Vector3::from([Sym::param(0), Sym::param(1), Sym::param(2)]))
        .b(Vector3::from([Sym::param(3), Sym::param(4), Sym::param(5)]))
        .rest(Sym::param(6))
        .stiffness(Sym::param(7))
        .damping(Sym::param(8))
        .softening(Sym::param(9))
        .build_raw();
    JointEdge::new(WorldId::get(), WorldId::get(), Joint::AxialSpringDamper(j))
}

/// f64 joint: the REAL constants (same values the trace lifts into params).
fn build_joint_f64(world: Arc<World>) -> JointEdge<f64> {
    let j = AxialSpringDamper::<f64>::builder(world.clone(), CriticallyDampedWarped)
        .a(Vector3::from([3.0 / 10.0, -1.0 / 5.0, 3.0 / 20.0]))
        .b(Vector3::from([-1.0 / 10.0, 1.0 / 4.0, -1.0 / 20.0]))
        .rest(13.0 / 10.0)
        .stiffness(7.0)
        .damping(2.0)
        .softening(1.0 / 1_000_000.0)
        .build_raw();
    JointEdge::new(WorldId::get(), WorldId::get(), Joint::AxialSpringDamper(j))
}

/// Mirror of `newton/build.rs::jac_kernel`: base pose (8 even-grade components as
/// runtime inputs) ∘ `Motor::exp(δ)` with δ a pure differential (value 0), velocity
/// as 6 twist coords with runtime values. Output: 12 components × [value, ∂/∂x0..23].
fn base_diff_kernel<S>(edge: &JointEdge<S>, inp: &[S]) -> Vec<S>
where
    S: Scalar + StandardPart + FromRational + Copy + Lift<Tangent<N24, S>> + Pod,
{
    // Pose-twist DOF: pure differential, value 0, unit partial in gradient axis k.
    let dvar = |k: usize| -> Tangent<N24, S> {
        let mut g = <Vector<N24, S> as AbelianGroup>::ZERO;
        g[k] = S::ONE;
        Tangent::from_grad(S::ZERO, g)
    };
    // Velocity DOF: runtime value inp[ax-12], unit partial in gradient axis ax.
    let vvar = |ax: usize| -> Tangent<N24, S> {
        let mut g = <Vector<N24, S> as AbelianGroup>::ZERO;
        g[ax] = S::ONE;
        Tangent::from_grad(inp[ax - 12], g)
    };
    // Base pose per body: 8 even-grade Motor components as runtime inputs (value only).
    let base = |off: usize| -> Motor<Tangent<N24, S>> {
        Motor::from_components(&core::array::from_fn(|i| {
            Tangent::<N24, S>::embed(inp[off + i])
        }))
    };

    // Motor::exp — the SAME retraction Differential uses (NOT Twist::exp, which
    // would bake the half-angle ½ and halve every pose column).
    let pose_a = base(12).compose(&Motor::exp(&Twist::new(
        &Vector3::from([dvar(0), dvar(1), dvar(2)]),
        &Vector3::from([dvar(3), dvar(4), dvar(5)]),
    )));
    let pose_b = base(20).compose(&Motor::exp(&Twist::new(
        &Vector3::from([dvar(6), dvar(7), dvar(8)]),
        &Vector3::from([dvar(9), dvar(10), dvar(11)]),
    )));
    let vel_a = Twist::new(
        &Vector3::from([vvar(12), vvar(13), vvar(14)]),
        &Vector3::from([vvar(15), vvar(16), vvar(17)]),
    );
    let vel_b = Twist::new(
        &Vector3::from([vvar(18), vvar(19), vvar(20)]),
        &Vector3::from([vvar(21), vvar(22), vvar(23)]),
    );
    let epoch = Epoch::standalone(inp[28], inp[29]);

    let [wa, wb]: [Wrench<Tangent<N24, S>>; 2] = edge
        .eval::<Tangent<N24, S>>(&vector![pose_a, pose_b], &vector![vel_a, vel_b], &epoch)
        .split();

    let comps: [Tangent<N24, S>; 12] = [
        wa.force()[0],
        wa.force()[1],
        wa.force()[2],
        wa.torque()[0],
        wa.torque()[1],
        wa.torque()[2],
        wb.force()[0],
        wb.force()[1],
        wb.force()[2],
        wb.torque()[0],
        wb.torque()[1],
        wb.torque()[2],
    ];
    let mut out = [S::ZERO; NOUT_FULL];
    let mut idx = 0;
    for c in comps {
        out[idx] = c.base();
        idx += 1;
        for k in 0..24 {
            out[idx] = c.component(k);
            idx += 1;
        }
    }
    out.to_vec()
}

/// The traced base+differential kernel (unit retraction) equals CPU
/// `Differential::jacobian` — value and all 24 columns, both bodies.
#[test]
fn base_differential_kernel_matches_cpu_differential() {
    // ---- fixed evaluation point ----
    // Base poses are REAL Motors (exp of a twist); their 8 components feed the
    // kernel's `base(off)`, and the Motors themselves feed `Differential::at`.
    let base_a = Twist::new(
        &Vector3::from([0.5, -0.3, 0.2]),
        &Vector3::from([0.1, 0.4, -0.2]),
    )
    .exp(1.0);
    let base_b = Twist::new(
        &Vector3::from([-0.4, 0.25, 0.15]),
        &Vector3::from([0.2, -0.1, 0.3]),
    )
    .exp(1.0);
    let vla = [0.40, -0.20, 0.35];
    let vaa = [0.12, -0.28, 0.18];
    let vlb = [-0.30, 0.45, -0.15];
    let vab = [0.22, 0.14, -0.26];
    let vel_a = Twist::new(&Vector3::from(vla), &Vector3::from(vaa));
    let vel_b = Twist::new(&Vector3::from(vlb), &Vector3::from(vab));
    let (dt, warp) = (0.016, 1.30);

    // Input vector (NIN=30): 12 velocity values, 8+8 base-pose components, dt, warp.
    let ca = base_a.to_components();
    let cb = base_b.to_components();
    let mut inputs: Vec<f64> = vec![
        vla[0], vla[1], vla[2], vaa[0], vaa[1], vaa[2], // vel A (lin, ang) -> inp 0..6
        vlb[0], vlb[1], vlb[2], vab[0], vab[1], vab[2], // vel B (lin, ang) -> inp 6..12
    ];
    inputs.extend_from_slice(&ca); // base A -> inp 12..20
    inputs.extend_from_slice(&cb); // base B -> inp 20..28
    inputs.push(dt); // inp 28
    inputs.push(warp); // inp 29
    assert_eq!(inputs.len(), NIN);

    // ---- traced kernel (Lua) ----
    let joint_sym = build_joint_sym(world());
    let traced = Tracer::builder()
        .fn_name("base_diff_oracle")
        .flatten()
        .build()
        .trace::<_>(
            NIN,
            NPARAM,
            NOUT_FULL,
            &[
                (InputRef::Input(28), InputFact::AbsGtEps), // dt
                (InputRef::Input(29), InputFact::AbsGeOne), // warp
                (InputRef::Input(28), InputFact::Positive),
                (InputRef::Input(29), InputFact::Positive),
            ],
            |inp, _param| base_diff_kernel::<Sym>(&joint_sym, inp),
        );
    let edge_f64 = build_joint_f64(world());
    let params: Vec<f64> = edge_f64.params();
    let got = traced.run_lua(&inputs, &params).unwrap();

    // ---- CPU Differential oracle at the SAME point ----
    let epoch = Epoch::standalone(dt, warp);
    let poses = vector![&base_a, &base_b];
    let vels = vector![&vel_a, &vel_b];
    let (value, block) = Differential::<N2, _>::at(&poses, &vels).jacobian(&edge_f64, &epoch);

    // Wrench component j in [f0,f1,f2,t0,t1,t2].
    let wc = |w: Wrench<f64>, j: usize| -> f64 {
        let f = w.force();
        let t = w.torque();
        [f[0], f[1], f[2], t[0], t[1], t[2]][j]
    };
    // Reshape Differential output to the kernel's [12 × 25] layout: component c is
    // (body c/6, wrench-part c%6); axis order matches (pose A/B, vel A/B).
    let mut want = [0.0f64; NOUT_FULL];
    for c in 0..12usize {
        let (body, part) = (c / 6, c % 6);
        want[c * 25] = wc(value[body], part);
        for k in 0..24usize {
            want[c * 25 + 1 + k] = wc(block[body][k], part);
        }
    }

    // ---- compare ----
    let mut nonzero = 0usize;
    let mut worst_rel = 0.0f64;
    let mut worst_i = 0usize;
    for i in 0..NOUT_FULL {
        let rel = (got[i] - want[i]).abs() / want[i].abs().max(1.0);
        if rel > worst_rel {
            worst_rel = rel;
            worst_i = i;
        }
        if want[i].abs() > 1e-9 {
            nonzero += 1;
        }
    }
    assert!(
        worst_rel < 1e-7,
        "kernel vs Differential mismatch at {worst_i}: got={} want={} (rel {worst_rel:e})",
        got[worst_i],
        want[worst_i]
    );
    assert!(
        nonzero >= 250,
        "expected a densely-coupled block, only {nonzero}/{NOUT_FULL} nonzero"
    );
    eprintln!("kernel vs Differential: {nonzero}/{NOUT_FULL} nonzero, worst rel {worst_rel:e}");
}
