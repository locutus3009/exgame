// SPDX-License-Identifier: MIT

#![cfg(feature = "lua")]

//! 24-DOF Jacobian smoke test for viete at shader scale.
//!
//! Traces a real `newton` joint law — `AxialSpringDamper` with the nonlinear
//! `CriticallyDampedWarped` force (ω₀ = √(k/warp²), Padé `(1 + ω₀·dt)²`) — over a
//! 24-variable forward-AD jet (`Tangent<24, Sym>`), emits Lua for the WHOLE
//! Jacobian block, and checks Lua == f64 to floating tolerance.
//!
//! This emulates a shader assembling one full connection's contribution to the
//! global Jacobian: two bodies × (6 pose DOF via the exp-chart + 6 velocity DOF)
//! = 24 independent variables, both wrenches returned by the law, every
//! component, every partial. `dt` and `warp` are runtime inputs (not baked
//! constants), so they flow through the law symbolically with zero gradient.

use aristotle::Epoch;
use aristotle::{World, WorldId};
use bytemuck::Pod;
use clifford::pga3::{Dynamics, Twist, Wrench};
use clifford::{Lift, Tangent};
use joints::{AxialSpringDamper, CriticallyDampedWarped, Joint, JointEdge};
use peano::prelude::*;
use std::sync::Arc;
use viete::{InputRef, Sym, Tracer};

const NIN: usize = 26; // 24 state DOF + dt + warp  (the fixed evaluation point)
const NPARAM: usize = 11; // 10 joint constants + selector (param[10])
const NOUT_FULL: usize = 300; // 12 components × (value + 24 partials) — oracle layout
const NOUT_DIR: usize = 24; // 12 components × (value, directional derivative)

/// Sym joint: the constants are PARAM references, not baked rationals. Built ONCE
/// (pure handles, no instructions) and captured by the trace closure. The
/// `Sym::param(i)` order MUST mirror `AxialSpringDamper::params()`. Anchors offset
/// from each COM so the wrench carries torque too → a densely-coupled block.
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

/// f64 joint: the REAL constants (same values the trace lifts into params). Its
/// `params()` supplies the run_lua param slice; `softening` is set explicitly to
/// the builder default so the slice has exactly NPARAM entries.
fn build_joint_f64(world: Arc<World>) -> JointEdge<f64> {
    let j = AxialSpringDamper::<f64>::builder(world.clone(), CriticallyDampedWarped)
        .a(Vector3::from([3.0 / 10.0, -1.0 / 5.0, 3.0 / 20.0]))
        .b(Vector3::from([-1.0 / 10.0, 1.0 / 4.0, -1.0 / 20.0]))
        .rest(13.0 / 10.0)
        .stiffness(7.0)
        .damping(2.0)
        .softening(1.0 / 1_000_000.0) // == newton::joint::default_softening
        .build_raw();
    JointEdge::new(WorldId::get(), WorldId::get(), Joint::AxialSpringDamper(j))
}

/// FULL-JACOBIAN ORACLE — runs only at f64. Builds per-body state, runs the
/// joint's wrench over the 24-DOF jet, extracts value + full Jacobian. The working
/// scalar is `Tangent<N24, S>` — a forward-AD jet over all 24 state DOF.
fn jac_kernel<S>(edge: &JointEdge<S>, inp: &[S]) -> Vec<S>
where
    S: Scalar + StandardPart + FromRational + Copy + Lift<Tangent<N24, S>> + Pod,
{
    // Seed state DOF k as an AD variable: value inp[k], unit partial in slot k.
    let var = |k: usize| -> Tangent<N24, S> {
        let mut g = <Vector<N24, S> as AbelianGroup>::ZERO;
        g[k] = S::ONE;
        Tangent::from_grad(inp[k], g)
    };

    // Poses via the exp-chart: 6 twist coords per body → a Motor. exp "time" = 1.
    let unit_dt = Tangent::<N24, S>::embed(S::ONE);
    let pose_a = Twist::new(
        &Vector3::from([var(0), var(1), var(2)]),
        &Vector3::from([var(3), var(4), var(5)]),
    )
    .exp(unit_dt);
    let pose_b = Twist::new(
        &Vector3::from([var(6), var(7), var(8)]),
        &Vector3::from([var(9), var(10), var(11)]),
    )
    .exp(unit_dt);

    // World-frame spatial velocities: 6 twist coords per body, used directly.
    let vel_a = Twist::new(
        &Vector3::from([var(12), var(13), var(14)]),
        &Vector3::from([var(15), var(16), var(17)]),
    );
    let vel_b = Twist::new(
        &Vector3::from([var(18), var(19), var(20)]),
        &Vector3::from([var(21), var(22), var(23)]),
    );

    // dt and warp are RUNTIME inputs (not constants): symbolic base-scalar values.
    let epoch = Epoch::standalone(inp[24], inp[25]);

    let [wa, wb]: [Wrench<Tangent<N24, S>>; 2] = edge
        .eval::<Tangent<N24, S>>(&vector![pose_a, pose_b], &vector![vel_a, vel_b], &epoch)
        .split();

    // Both wrenches, all six components each (force + torque). No shortcut: the
    // law returns two wrenches and the assembler emits both.
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

    // Flatten to [value, ∂/∂x0 … ∂/∂x23] per component.
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

/// DIRECTIONAL KERNEL — the only thing traced. One-wide forward-AD jet
/// (`Tangent<N1, Sym>`): the seed is the one-hot `e_selector`, built in-trace from
/// 24 independent `Sym::select(selector, j)` calls (exactly one returns 1). A
/// single directional-derivative accumulator then propagates through exp / motor /
/// wrench — that is the whole compactness win over the 24-wide full kernel.
fn dir_kernel(edge: &JointEdge<Sym>, inp: &[Sym], selector: Sym) -> Vec<Sym> {
    // Seeds are built INLINE (after the exp forks), on purpose: the engine's
    // hoist_invariant pass lifts these branch-invariant selects back to the root
    // prefix, so no manual seeds-first reorder is needed. This test is the
    // integration check that hoisting collapses the post-fork replication.
    let var = |j: usize| -> Tangent<N1, Sym> {
        Tangent::<N1, Sym>::from_grad(inp[j], vector![Sym::select(selector, j as u32)])
    };

    // common part (identical structure to the full kernel, but 1-wide).
    let unit_dt = Tangent::<N1, Sym>::embed(Sym::ONE);
    let pose_a = Twist::new(
        &Vector3::from([var(0), var(1), var(2)]),
        &Vector3::from([var(3), var(4), var(5)]),
    )
    .exp(unit_dt);
    let pose_b = Twist::new(
        &Vector3::from([var(6), var(7), var(8)]),
        &Vector3::from([var(9), var(10), var(11)]),
    )
    .exp(unit_dt);
    let vel_a = Twist::new(
        &Vector3::from([var(12), var(13), var(14)]),
        &Vector3::from([var(15), var(16), var(17)]),
    );
    let vel_b = Twist::new(
        &Vector3::from([var(18), var(19), var(20)]),
        &Vector3::from([var(21), var(22), var(23)]),
    );
    let epoch = Epoch::standalone(inp[24], inp[25]);

    let [wa, wb]: [Wrench<Tangent<N1, Sym>>; 2] = edge
        .eval::<Tangent<N1, Sym>>(&vector![pose_a, pose_b], &vector![vel_a, vel_b], &epoch)
        .split();

    let comps: [Tangent<N1, Sym>; 12] = [
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

    // 12 × (value, directional derivative). value is selector-independent.
    let mut out = [Sym::ZERO; NOUT_DIR];
    let mut idx = 0;
    for c in comps {
        out[idx] = c.base();
        out[idx + 1] = c.component(0);
        idx += 2;
    }
    out.to_vec()
}

#[test]
fn jacobian_24dof_directional_matches_full_f64() {
    let world = Arc::new(
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
    );
    let joint = build_joint_sym(world); // Sym::param(0..10) for the 10 joint constants
    let tracer = Tracer::builder()
        .fn_name("joint_jacobian_dir")
        .flatten()
        .build();
    // dt (inp 24) >> eps; warp (inp 25) >= 1. selector (param[10]) is unconstrained.
    let traced = tracer.trace::<_>(
        NIN,
        NPARAM,
        NOUT_DIR,
        &[
            (InputRef::Input(24), viete::InputFact::AbsGtEps),
            (InputRef::Input(25), viete::InputFact::AbsGeOne),
            (InputRef::Input(24), viete::InputFact::Positive),
            (InputRef::Input(25), viete::InputFact::Positive),
        ],
        |inp, param| dir_kernel(&joint, inp, param[10]), // param[10] = selector
    );

    // Metrics for the DIRECTIONAL trace (compare against the full-Jacobian
    // baseline: ~25,328 instrs / 44 leaves / 14 consts). Refresh out.txt under
    // VIETE_DEMO: `VIETE_DEMO=1 cargo test -p viete --release --test
    // jacobian_smoke -- --nocapture`.
    {
        use viete::Term;
        let tree = traced.tree();
        let leaves = traced.leaf_count();
        let branches = tree
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Term::Branch { .. }))
            .count();
        let instrs: usize = tree.blocks.iter().map(|b| b.instrs.len()).sum();
        let consts = tree.consts.len();
        eprintln!(
            "METRICS(directional) leaves={leaves} branches={branches} instrs={instrs} consts={consts}"
        );
        assert_eq!(branches, 0, "flatten: fully if-converted, no branches");
        assert_eq!(leaves, 1, "flatten: one straight-line block");
        // 4893 → 4741 at the A7 cutover (see the full-trace pins).
        assert_eq!(
            instrs, 4741,
            "directional flat trace instr count regressed (hoist/flatten changed)"
        );
        if std::env::var_os("VIETE_DEMO").is_some() {
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../out.txt");
            std::fs::write(path, traced.emit_lua()).expect("write out.txt");
            eprintln!("DEMO wrote {path}");
        }
    }

    // The 24 selects are built INLINE (post-fork) in dir_kernel; hoist_invariant
    // lifts them out of the replicated branches back to a single copy each.
    // flatten then if-converts every branch, adding the output-merge selects
    // (one Lt-select per live output per branch) plus the dead-arm fatal masks.
    {
        use viete::Instr;
        let selects: usize = traced
            .tree()
            .blocks
            .iter()
            .map(|b| {
                b.instrs
                    .iter()
                    .filter(|i| matches!(i, Instr::Select(..)))
                    .count()
            })
            .sum();
        // 55 pre-flatten selects (24 seed + 11 div-fatal + 20 noninvertible-fatal)
        // + flatten's per-branch output-merge and fatal-mask Lt-selects. Pins the
        // hoist, both fatal sources, and the if-conversion together.
        assert_eq!(
            selects, 357,
            "55 pre-flatten + flatten output-merge/fatal masks"
        );
    }

    // Global tracer-bug invariant: no Unexpected fatal leaves.
    assert_eq!(
        traced.unexpected_count(),
        0,
        "Unexpected fatal leaf = tracer bug"
    );
    eprintln!("Fatal arms: {}", traced.fatal_leaves().len());

    // Fixed evaluation point (same as the full-Jacobian test).
    let inputs: Vec<f64> = vec![
        0.20, -0.30, 0.15, 0.10, 0.25, -0.20, // body A pose twist (lin, ang)
        -0.25, 0.18, -0.12, -0.15, 0.22, 0.30, // body B pose twist
        0.40, -0.20, 0.35, 0.12, -0.28, 0.18, // body A velocity (lin, ang)
        -0.30, 0.45, -0.15, 0.22, 0.14, -0.26, // body B velocity
        0.016, 1.30, // dt, warp (runtime, not constants)
    ];

    let world = Arc::new(
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
    );
    // f64 oracle: the full 24-wide Jacobian, [value, ∂/∂x0..∂/∂x23] per component.
    let edge_f64 = build_joint_f64(world);
    let want: Vec<f64> = jac_kernel::<f64>(&edge_f64, &inputs);
    let consts: Vec<f64> = edge_f64.params(); // 10 joint constants

    // Emulated dispatch: one run_lua per column k, selector = k. Assemble the full
    // Jacobian in the oracle layout: component c at [c*25 .. c*25+25] = value then
    // 24 partials.
    let mut got = [0.0f64; NOUT_FULL];
    for k in 0..24usize {
        let mut params = consts.clone();
        params.push(k as f64); // params[10] = selector
        let col = traced.run_lua(&inputs, &params).unwrap(); // [12 × (value, D_{e_k})]
        for c in 0..12usize {
            if k == 0 {
                got[c * 25] = col[c * 2]; // value (selector-independent; take once)
            }
            got[c * 25 + 1 + k] = col[c * 2 + 1]; // ∂f_c/∂x_k = column k
        }
    }

    // Same floating-noise tolerance band as the full-Jacobian test.
    let mut nonzero = 0usize;
    let mut worst_rel = 0.0f64;
    let mut worst_i = 0usize;
    for i in 0..NOUT_FULL {
        let d = (got[i] - want[i]).abs();
        let rel = d / want[i].abs().max(1.0);
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
        "worst directional-vs-full mismatch at {worst_i}: got={} want={} (rel {worst_rel:e})",
        got[worst_i],
        want[worst_i]
    );
    assert!(
        nonzero >= 250,
        "expected a densely-coupled block, only {nonzero}/{NOUT_FULL} entries nonzero"
    );
    eprintln!("Directional Jacobian: {nonzero}/{NOUT_FULL} nonzero, worst rel error {worst_rel:e}");
}

/// `fast_math` opt-in: reciprocal-CSE (and algebraic cancellations) on the flat
/// block before vectorize. NOT bit-exact — it factors a shared `1/b` out of the
/// many forward-AD quotient-rule divisions, which exposes common structure the
/// divisions hid from CSE. Here it cuts the full Jacobian 13500 -> 10634 (-21.2%)
/// while staying within ~3 ulp of the f64 oracle. The bit-exact path
/// (`jacobian_24dof_full_trace_stats`) is unaffected.
#[test]
fn jacobian_24dof_fast_math_reduces_instrs() {
    let world = Arc::new(
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
    );
    let joint = build_joint_sym(world.clone());
    let inputs: Vec<f64> = vec![
        0.20, -0.30, 0.15, 0.10, 0.25, -0.20, -0.25, 0.18, -0.12, -0.15, 0.22, 0.30, 0.40, -0.20,
        0.35, 0.12, -0.28, 0.18, -0.30, 0.45, -0.15, 0.22, 0.14, -0.26, 0.016, 1.30,
    ];
    let edge_f64 = build_joint_f64(world.clone());
    let want: Vec<f64> = jac_kernel::<f64>(&edge_f64, &inputs);
    let params: Vec<f64> = edge_f64.params();
    let lanes: Vec<Vec<usize>> = (0..12)
        .map(|c| (1..25).map(|k| c * 25 + k).collect())
        .collect();
    let traced = Tracer::builder()
        .fn_name("joint_jacobian_fast")
        .flatten()
        .fast_math()
        .vectorize_lanes(lanes)
        .build()
        .trace::<_>(
            NIN,
            10,
            NOUT_FULL,
            &[
                (InputRef::Input(24), viete::InputFact::AbsGtEps),
                (InputRef::Input(25), viete::InputFact::AbsGeOne),
                (InputRef::Input(24), viete::InputFact::Positive),
                (InputRef::Input(25), viete::InputFact::Positive),
            ],
            |inp, _p| jac_kernel::<Sym>(&joint, inp),
        );
    // algebraically correct: within a few ulp of the f64 oracle (NOT bit-exact).
    let got = traced.run_lua(&inputs, &params).unwrap();
    let mut worst = 0.0f64;
    for i in 0..NOUT_FULL {
        worst = worst.max((got[i] - want[i]).abs() / want[i].abs().max(1.0));
    }
    eprintln!("FAST_MATH worst rel error {worst:e}");
    assert!(
        worst < 1e-7,
        "fast_math must stay algebraically correct, got {worst:e}"
    );
    // instruction-count win vs the bit-exact baseline (13346 after A7).
    let instrs: usize = traced.tree().blocks.iter().map(|b| b.instrs.len()).sum();
    eprintln!("FAST_MATH full-trace instrs={instrs}");
    // 10731 → 10488 at the A7 cutover: the narrow exp/embed paths of the stratified carrier
    // do not emit the dead arithmetic of a wide Mv.
    assert_eq!(instrs, 10488, "fast_math instr count regressed");
    assert!(
        instrs < 13346,
        "fast_math must reduce vs bit-exact baseline"
    );
}

/// The "old" FULL-Jacobian path, now also TRACED (not just the f64 oracle):
/// `jac_kernel` over `Tangent<24, Sym>` — the 24 partials propagate as a single
/// 24-wide forward-AD jet, so the gradient is the natural lane-parallel axis
/// (unlike the directional kernel, which collapsed it to width 1). No selector,
/// no one-hot seed: params are just the 10 joint constants. Emits one flat block
/// computing value + all 24 partials per component in a single run_lua.
#[test]
fn vectorize_lanes_is_only_a_hint() {
    // Deliberately WRONG groupings: (a) cross-component nonsense groups,
    // (b) groups mixing value+partials of different components. Must stay
    // bit-for-bit anyway (the hint only steers grouping; build_vector
    // verifies op-compat per level and Extract reads back per lane).
    let world = Arc::new(
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
    );
    let joint = build_joint_sym(world.clone());
    let inputs: Vec<f64> = vec![
        0.20, -0.30, 0.15, 0.10, 0.25, -0.20, -0.25, 0.18, -0.12, -0.15, 0.22, 0.30, 0.40, -0.20,
        0.35, 0.12, -0.28, 0.18, -0.30, 0.45, -0.15, 0.22, 0.14, -0.26, 0.016, 1.30,
    ];
    let edge_f64 = build_joint_f64(world.clone());
    let want: Vec<f64> = jac_kernel::<f64>(&edge_f64, &inputs);
    let params: Vec<f64> = edge_f64.params();
    // garbage hint: shuffle all 300 indices into arbitrary groups of 4
    let mut idx: Vec<usize> = (0..NOUT_FULL).collect();
    idx.rotate_left(7); // arbitrary permutation seed (deterministic)
    let bad: Vec<Vec<usize>> = idx.chunks(4).map(|c| c.to_vec()).collect();
    let traced = Tracer::builder()
        .fn_name("bad")
        .flatten()
        .vectorize_lanes(bad)
        .build()
        .trace::<_>(
            NIN,
            10,
            NOUT_FULL,
            &[
                (InputRef::Input(24), viete::InputFact::AbsGtEps),
                (InputRef::Input(25), viete::InputFact::AbsGeOne),
                (InputRef::Input(24), viete::InputFact::Positive),
                (InputRef::Input(25), viete::InputFact::Positive),
            ],
            |inp, _p| jac_kernel::<Sym>(&joint, inp),
        );
    let got = traced.run_lua(&inputs, &params).unwrap();
    let mut worst = 0.0f64;
    for i in 0..NOUT_FULL {
        worst = worst.max((got[i] - want[i]).abs() / want[i].abs().max(1.0));
    }
    eprintln!("WRONG-HINT worst rel error {worst:e}");
    assert!(
        worst < 1e-7,
        "wrong hint must still be bit-for-bit, got {worst:e}"
    );
}

#[test]
fn jacobian_24dof_full_trace_stats() {
    let world = Arc::new(
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
    );
    let joint = build_joint_sym(world.clone()); // Sym::param(0..9) for the 10 joint constants
    // Output layout: component c at [c*25 .. c*25+25] = value then 24 partials.
    // The 24 partials of each component are the lane-parallel AD-gradient axis;
    // declare them so the vectorizer groups them (zero-folding hides this from
    // shape inference). 12 groups of 24 → tiled to vec4 inside the pass.
    let lanes: Vec<Vec<usize>> = (0..12)
        .map(|c| (1..25).map(|k| c * 25 + k).collect())
        .collect();
    let tracer = Tracer::builder()
        .fn_name("joint_jacobian_full")
        .flatten()
        .vectorize_lanes(lanes)
        .build();
    let traced = tracer.trace::<_>(
        NIN,
        10,
        NOUT_FULL,
        &[
            (InputRef::Input(24), viete::InputFact::AbsGtEps),
            (InputRef::Input(25), viete::InputFact::AbsGeOne),
            (InputRef::Input(24), viete::InputFact::Positive),
            (InputRef::Input(25), viete::InputFact::Positive),
        ],
        |inp, _param| jac_kernel::<Sym>(&joint, inp),
    );

    // ---- trace + vectorization statistics (branches==0; realized vec4 win) ----
    {
        use viete::{Instr, Term};
        let tree = traced.tree();
        let leaves = traced.leaf_count();
        let branches = tree
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Term::Branch { .. }))
            .count();
        let all: Vec<&Instr> = tree.blocks.iter().flat_map(|b| &b.instrs).collect();
        let instrs = all.len();
        let count = |p: &dyn Fn(&Instr) -> bool| all.iter().filter(|i| p(i)).count();
        // scalar arithmetic still present after vectorization
        let scalar_arith = count(&|i| {
            matches!(
                i,
                Instr::Add(..)
                    | Instr::Sub(..)
                    | Instr::Mul(..)
                    | Instr::Neg(..)
                    | Instr::Fma { .. }
            )
        });
        let scalar_sel = count(&|i| matches!(i, Instr::Select(..)));
        let vec_ops = count(&|i| {
            matches!(
                i,
                Instr::VAdd(..)
                    | Instr::VSub(..)
                    | Instr::VMul(..)
                    | Instr::VNeg(..)
                    | Instr::VFma(..)
                    | Instr::VSelect(..)
            )
        });
        let splats = count(&|i| matches!(i, Instr::Splat(..)));
        let packs = count(&|i| matches!(i, Instr::Pack(..)));
        let extracts = count(&|i| matches!(i, Instr::Extract(..)));
        eprintln!(
            "METRICS(full,vec) leaves={leaves} branches={branches} instrs={instrs}\n  \
             scalar_arith={scalar_arith} scalar_select={scalar_sel} \
             vec_ops={vec_ops} splat={splats} pack={packs} extract={extracts}"
        );
        assert_eq!(branches, 0, "flatten: full Jacobian is fully if-converted");
        assert_eq!(leaves, 1, "flatten: one straight-line block");
        // Realized vec4 vectorization of the AD-gradient axis (bit-for-bit),
        // including VDiv (gradient/divisor lanes; cascades into the dividend
        // chains). After the A7 cutover (stratified carrier, narrow exp paths):
        // scalar remainder 8949 → 8184, vec ops 3338 → 3689 (the vectorizable
        // share grew — the dead scalar work is gone), total 13599 → 13346.
        // Pins the SLP win; refresh if the pass changes.
        assert_eq!(instrs, 13346, "vectorized instr count regressed");
        assert_eq!(scalar_arith, 8184, "scalar arith remainder regressed");
        assert_eq!(vec_ops, 3689, "vector-op count regressed");
        assert_eq!(
            (splats, packs, extracts),
            (269, 610, 290),
            "pack/splat/extract budget"
        );
        // sign-aware lowering elides abs on nonneg/nonpos conds. Every surviving
        // EffectiveZero cond in this kernel proves nonneg/nonpos, so all 29 of
        // the prior math.abs calls are gone.
        let abs_count = traced.emit_lua().matches("math.abs").count();
        assert_eq!(
            abs_count, 0,
            "sign-aware lowering eliminated every abs (was 29), got {abs_count}"
        );
        if std::env::var_os("VIETE_DEMO").is_some() {
            let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../out_full.txt");
            std::fs::write(path, traced.emit_lua()).expect("write out_full.txt");
            eprintln!("DEMO wrote {path}");
        }
    }
    assert_eq!(
        traced.unexpected_count(),
        0,
        "Unexpected fatal leaf = tracer bug"
    );

    // ---- bit-for-bit vs the f64 oracle (same evaluation point) ----
    let inputs: Vec<f64> = vec![
        0.20, -0.30, 0.15, 0.10, 0.25, -0.20, -0.25, 0.18, -0.12, -0.15, 0.22, 0.30, 0.40, -0.20,
        0.35, 0.12, -0.28, 0.18, -0.30, 0.45, -0.15, 0.22, 0.14, -0.26, 0.016, 1.30,
    ];
    let edge_f64 = build_joint_f64(world);
    let want: Vec<f64> = jac_kernel::<f64>(&edge_f64, &inputs);
    let params: Vec<f64> = edge_f64.params(); // 10 joint constants, no selector
    let got = traced.run_lua(&inputs, &params).unwrap(); // value + 24 partials, all 12 comps

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
        "full-trace vs f64 mismatch at {worst_i}: got={} want={} (rel {worst_rel:e})",
        got[worst_i],
        want[worst_i]
    );
    eprintln!(
        "Full Jacobian (traced): {nonzero}/{NOUT_FULL} nonzero, worst rel error {worst_rel:e}"
    );
}
