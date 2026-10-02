// SPDX-License-Identifier: MIT

//! Build-time codegen for the per-joint force/Jacobian kernels: traces each
//! joint type's constraint law once (over `viete::Sym`) into a branch-enumerated
//! IR and emits one GLSL kernel per type (compiled to SPIR-V by shaderc).
//!
//! The kernel I/O layout below is authoritative: the constants `NIN`,
//! `NOUT_FULL` and `NOUT_PLAIN`, the input map and the `*_output_map`
//! functions in this file implement it, and the host-side readers in
//! `src/accelerator/` consume it. (It moved here from the retired
//! `ACCELERATOR` subsystem document in M1.)
//!
//! # Kernel I/O layout (authoritative)
//!
//! **Input** (`NIN = 30`). The pose perturbation is a pure differential (value 0 —
//! the finite pose lives in `base`), so pose-twist DOF carry NO input values; only
//! their 12 AD gradient axes exist. The AD gradient-axis layout is the FIXED newton
//! 24-order (pose A 0..6, pose B 6..12, vel A 12..18, vel B 18..24) and is
//! independent of these value-input indices; velocity value j feeds gradient axis
//! 12+j.
//!
//! ```text
//! 0..12   velocity values (vel A 0..6, vel B 6..12)  → gradient axes 12..24
//! 12..20  base pose of body A: 8 even-grade Motor components
//! 20..28  base pose of body B: 8 even-grade Motor components
//! 28      dt
//! 29      warp
//! ```
//!
//! The pose retraction is `base ∘ Motor::exp(δ)` — the **same** map
//! `Differential::jacobian` uses, so there is NO exp-"time"/`half` input (a `Twist::exp`
//! would bake the geometric half-angle ½ and halve every pose column — the
//! kernel-vs-`Differential` oracle pins this). The `−½dt²/−½dt` column scaling lives
//! in newton's matrix assembler, not the kernel.
//!
//! **Output** (`NOUT_FULL = 300`) — 12 wrench components × 25 slots. Components in
//! order (body A's wrench, then body B's — force x,y,z then torque x,y,z each):
//!
//! ```text
//! [0]=A.fx [1]=A.fy [2]=A.fz [3]=A.tx [4]=A.ty [5]=A.tz
//! [6]=B.fx [7]=B.fy [8]=B.fz [9]=B.tx [10]=B.ty [11]=B.tz
//! ```
//!
//! Component c occupies `out[c*25 .. c*25+25] = [value, ∂/∂axis0 … ∂/∂axis23]`, the
//! 24 partials along the fixed gradient-axis order above. Explicit reads only the
//! value slot of each component (`out[c*25]`); implicit reads the whole 25-wide
//! group.
//!
//! # Structure of this file
//!
//! Two kernels are generated per joint type, plus the per-body `Pre` stage, and
//! they share almost all of their GPU-boundary plumbing. The common pieces live
//! in one place each:
//!   - `build_world` / `build_joint_sym` — the traced `JointEdge`.
//!   - `InSpec` — the input side of the boundary: how many twist/motor rows the
//!     index table carries, which immediate scalars ride in it, how many params
//!     come through the flat buffer. Joints take two bodies + the epoch pair;
//!     `Pre` takes one body + the retraction.
//!   - `preamble` / `body_pre` — the shared GLSL structs, bindings and prologue,
//!     parameterised by that `InSpec` and by the kernel's outputs (`OutBuf`).
//!   - `input_map` / `field` / `wrench_slots` / `motor_slots` — the input-side
//!     Pod recombination, derived from the same `InSpec`.
//!   - `fatal_slots` / `fatal_map` / `write_kernel` / `write_fatal_counts` —
//!     fatal-slot sizing and wiring, file output, and the per-kernel fatal-count
//!     table the host scans by.
//!
//! Only the traced kernel, the output layout (`*_output_map`) and the `InSpec` +
//! `OutBuf` lists differ. New emitters for other functions reuse the same
//! helpers and supply their own kernel + input spec + output map + buffers.

use aristotle::{Epoch, World, WorldId};
use bytemuck::Pod;
use clifford::{
    Lift, Tangent,
    pga3::{Dynamics, Motor, Twist, Wrench},
};
use joints::{
    CriticallyDampedWarped, JointEdge, JointFromParams, PerpendicularDamperWarpedBuilder,
    SimpleSpringDamper, TorsionalDamperWarpedBuilder,
};
use peano::prelude::*;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use viete::{InputFact, InputRef, Sym, Tracer};

// Input layout, NIN = 30 (velocity values, two base poses, dt/warp). The pose
// perturbation uses UNIT retraction (exp-time = 1, matching `Differential`), so
// there is no `half`/`factor` input — the −½dt²/−½dt column scaling lives in the
// matrix assembler. Authoritative table: the module doc above.
const NIN: usize = 30;
// Output layout, NOUT = 300: 12 wrench components × 25 (value + 24 partials).
// Authoritative table: the module doc above.
const NOUT_FULL: usize = 300;
// Plain (value-only) output: just the 12 wrench components, no partials. Used by
// the no-Jacobian connection kernel (explicit families / the residual value pass).
const NOUT_PLAIN: usize = 12;
// `Pre` is per BODY, not per connection, so it has its own compact layout: one
// twist (6) + one pose motor (8) + the retraction = 15 in, and the midpoint pose
// (8 motor components) + the solve velocity (6 twist coords) = 14 out.
const NIN_PRE: usize = 15;
const NOUT_PRE: usize = 14;

/// Reverse-engineer where each constructor coordinate lands in a type's RAW Pod
/// layout — the layout the GPU shader's `cN` fields overlay. `raw` is the Pod cast
/// of a value built with coordinate `k` seeded to the distinct magic number `k+1`
/// (1.0, 2.0, …). Distinct magnitudes make the mapping unambiguous: the slot whose
/// `|value| == k+1` is where coordinate `k` went, and its sign is the store's
/// blade/markup sign. Returns `[(slot, sign); N]` indexed by coordinate. The type
/// itself (its `pos()` + markup) is the source of truth — no hand table to drift.
fn probe_layout<const N: usize>(raw: [f32; N]) -> [(usize, i8); N] {
    core::array::from_fn(|k| {
        let want = (k + 1) as f32;
        for (slot, &v) in raw.iter().enumerate() {
            // Seeds are exact small integers touched only by ±1 markup signs, so
            // the raw slot holds exactly ±(k+1) — exact equality is safe here.
            if v.abs() == want {
                return (slot, if v < 0.0 { -1 } else { 1 });
            }
        }
        panic!("probe seed {want} vanished from the Pod layout (shape changed?)");
    })
}

// ── Pod recombination at the GPU boundary ────────────────────────────────────
// The shader field `cN` overlays the RAW Pod layout of the host type, whose slot
// order (grade-2/even blade packing) and signs (markup::radical/angular_blade) do
// NOT match the argument order of `Twist::new`/`Motor::from_components` that the
// trace numbers inputs by. We do not hand-write the permutation: SEED distinct
// magic numbers into the constructor and read back "which Pod slot, with which
// sign" — the type itself (pos()+markup) stays the single source of truth and the
// table is derived for the current layout on the fly (see `probe_layout`).

/// Wrench/Twist Pod slots: both ends of a screw share the store, so one Wrench
/// probe serves the input twists AND the output wrenches. Constructor coord order
/// is `[a0,a1,a2,b0,b1,b2]` (linear|angular / force|torque).
fn wrench_slots() -> [(usize, i8); 6] {
    probe_layout::<6>(bytemuck::cast(Wrench::<f32>::new(
        &Vector3::from([1.0, 2.0, 3.0]),
        &Vector3::from([4.0, 5.0, 6.0]),
    )))
}

/// Motor (even store) Pod slots, probed via `from_components`; markup carries no
/// signs here.
fn motor_slots() -> [(usize, i8); 8] {
    probe_layout::<8>(bytemuck::cast(Motor::<f32>::from_components(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ])))
}

/// One coordinate → the shader field `<name>.cN`, its sign baked into the rvalue
/// string (a GLSL lvalue can't negate, so the sign always rides the rhs).
fn field(name: &str, (slot, sign): (usize, i8)) -> String {
    if sign < 0 {
        format!("-{name}.c[{slot}]")
    } else {
        format!("{name}.c[{slot}]")
    }
}

/// Index-table row suffixes: `twist_a`/`twist_b`, `motor_a`/`motor_b`. One per
/// body a kernel reads. A connection kernel uses both; a per-body kernel (`Pre`)
/// uses only the first, and keeps the `_a` spelling so the plumbing below stays
/// one code path.
const ROWS: [&str; 2] = ["a", "b"];

/// The input side of a kernel's GPU boundary. The trace numbers its inputs in
/// exactly this order — twist coordinates (6 per row), motor components (8 per
/// row), immediate scalars, then params — and `input_map` / `preamble` /
/// `body_pre` all read the layout off this one description, so a kernel over a
/// different set of inputs only writes a different `InSpec`.
struct InSpec {
    /// Index-table twist rows, fetched from the `twists` buffer (binding 3).
    twists: usize,
    /// Index-table motor rows, fetched from the `motors` buffer (binding 4).
    motors: usize,
    /// Scalars carried IMMEDIATELY in the index row (no slot indirection), in
    /// input order — the epoch pair for joints, the retraction for `Pre`.
    scalars: &'static [&'static str],
    /// Slot-backed scalar params, read through the flat `plain` buffer.
    params: usize,
}

impl InSpec {
    /// The standard connection-kernel layout: both bodies' twist and pose, the
    /// epoch pair, and the joint type's params. `len()` is NIN + n_params.
    fn joint(n_params: usize) -> Self {
        let spec = InSpec {
            twists: 2,
            motors: 2,
            scalars: &["epoch_dt", "epoch_warp"],
            params: n_params,
        };
        // The traced kernels index `inp` by the constants; if the layout and the
        // constant ever disagree the shader would read the wrong slot silently.
        assert_eq!(spec.len(), NIN + n_params, "joint input layout drifted");
        spec
    }

    /// The per-body PRE layout: one body's velocity and pose, plus the
    /// retraction as an immediate. `len()` is NIN_PRE.
    fn pre() -> Self {
        let spec = InSpec {
            twists: 1,
            motors: 1,
            scalars: &["retraction"],
            params: 0,
        };
        assert_eq!(spec.len(), NIN_PRE, "pre input layout drifted");
        spec
    }

    fn twist_name(row: usize) -> String {
        format!("twist_{}", ROWS[row])
    }

    fn motor_name(row: usize) -> String {
        format!("motor_{}", ROWS[row])
    }

    /// How many trace inputs this layout addresses.
    fn len(&self) -> usize {
        self.twists * 6 + self.motors * 8 + self.scalars.len() + self.params
    }
}

/// Input recombination: input `i` → its GLSL rvalue, in the order `InSpec`
/// declares. Twists/motors go through the Pod probes above (slot + sign folded
/// into the rvalue); immediate scalars and params are plain named locals that
/// `body_pre` has already fetched.
fn input_map(spec: &InSpec) -> HashMap<u32, String> {
    let screw = wrench_slots();
    let motor = motor_slots();
    let mut rhs = Vec::with_capacity(spec.len());
    for row in 0..spec.twists {
        let name = InSpec::twist_name(row);
        rhs.extend(screw.iter().map(|&c| field(&name, c)));
    }
    for row in 0..spec.motors {
        let name = InSpec::motor_name(row);
        rhs.extend(motor.iter().map(|&c| field(&name, c)));
    }
    rhs.extend(spec.scalars.iter().map(|s| s.to_string()));
    rhs.extend((0..spec.params).map(|p| format!("param_{p}")));
    rhs.into_iter()
        .enumerate()
        .map(|(i, r)| (i as u32, r))
        .collect()
}

// ── Fatal slots ──────────────────────────────────────────────────────────────
// A guarded division the tracer cannot prove safe leaves a fatal-STATUS operand
// (1.0 when the guard trips, else 0.0) in the flattened kernel's `Return`. Each
// operand gets one slot of the invocation's `Fatals.fatal[]`, the host scans
// the slots after the dispatch, and a non-zero slot fails the message that owns
// the invocation (`accelerator/shaders.rs`, `check`). Nothing here is
// hand-counted: the slot count is read off the trace, the array is sized from
// it, and the same count is published to the host through `fatal_counts.rs`.

/// Fatal slots the traced kernel writes per invocation.
fn fatal_slots(traced: &viete::Trace) -> usize {
    viete::glsl::fatal_slots(traced.tree())
}

/// Declared length of `Fatals.fatal[]` for a kernel writing `fatals` slots. At
/// least 1: GLSL has no zero-length arrays, and the row-driven kernels clear
/// `fatal[0]` unconditionally (`row_head`). A slot past the count is never
/// written by the trace and never scanned by the host.
fn fatal_width(fatals: usize) -> usize {
    fatals.max(1)
}

/// Fatal slot `i` → its GLSL lvalue, for exactly the `fatals` slots the trace
/// writes. `emit_glsl` refuses to emit a fatal operand this map does not cover,
/// so a kernel with more fatal operands than slots fails the build.
fn fatal_map(fatals: usize) -> HashMap<u32, String> {
    (0..fatals as u32)
        .map(|i| (i, format!("fatals.data[idx].fatal[{i}]")))
        .collect()
}

/// The four input facts every joint kernel traces under: dt is `AbsGtEps` and
/// positive, warp is `AbsGeOne` and positive (the engine's input invariants:
/// it never dispatches a non-positive dt or a warp below one).
fn input_facts() -> [(InputRef, InputFact); 4] {
    [
        (InputRef::Input(28), InputFact::AbsGtEps), // dt
        (InputRef::Input(29), InputFact::AbsGeOne), // warp
        (InputRef::Input(28), InputFact::Positive), // dt
        (InputRef::Input(29), InputFact::Positive), // warp
    ]
}

/// One destination the kernel scatters into. Bindings 0..5 are fixed (fatals,
/// index, params, twists, motors); `Own` destinations take bindings from 5
/// upward, in list order. Drives the preamble and the body prologue; the
/// caller's `output_map` must address `{var}.data[idx_{index_field}]` to match.
enum OutBuf {
    /// A buffer of its own: GLSL `struct` + SSBO binding + an `Incidence` slot
    /// index. The connection wrench and Jacobian blocks are these.
    Own {
        /// GLSL comment emitted above the struct (include trailing newlines, or "").
        comment: &'static str,
        /// GLSL struct name, e.g. `WrenchJac`.
        struct_name: &'static str,
        /// Float count of the struct's `c[..]` array (the flat Pod float span).
        floats: usize,
        /// SSBO block type name, e.g. `WrenchData`.
        buffer: &'static str,
        /// Shader variable the block binds to, e.g. `wrenches`.
        var: &'static str,
        /// `Incidence` field holding this connection's per-buffer slot index,
        /// e.g. `conn`; the body fetches it as `idx_<index_field>`.
        index_field: &'static str,
    },
    /// An output that lands back in one of the INPUT buffers the preamble
    /// already binds (`twists` / `motors`): `Pre`'s midpoint pose and solve
    /// velocity are ordinary `Motor` / `Twist` world slots, living in the same
    /// storage as the pose and velocity they are formed from. Binding that
    /// storage a second time would alias it, so the kernel writes through the
    /// binding it holds and this contributes only the `Incidence` slot index.
    Shared { index_field: &'static str },
}

impl OutBuf {
    /// `Incidence` field holding this output's slot index, for both variants.
    fn index_field(&self) -> &'static str {
        match self {
            Self::Own { index_field, .. } | Self::Shared { index_field } => index_field,
        }
    }
}

/// Assemble the shared GLSL preamble: fixed input structs/bindings (twist, motor,
/// params, fatals, incidence) laid out per `spec`, plus the caller's own output
/// buffers. Every generated kernel shares this; only `spec`, `outs` and the
/// traced body differ. `fatals` is the trace's fatal-slot count (`fatal_slots`).
fn preamble(_spec: &InSpec, fatals: usize, outs: &[OutBuf]) -> String {
    let mut s = String::new();
    s.push_str(
        r"struct Twist {
    float c[6];
};

struct Motor {
    float c[8];
};

",
    );
    for o in outs {
        if let OutBuf::Own {
            comment,
            struct_name,
            floats,
            ..
        } = o
        {
            s.push_str(comment);
            s.push_str(&format!(
                "struct {struct_name} {{\n    float c[{floats}];\n}};\n\n"
            ));
        }
    }
    // Every kernel is row-driven: the batch table carries a row index and nothing
    // else, and the row — baked on topology change — holds every slot the kernel
    // touches. Its word order is the `InSpec` order, which `body_pre` mirrors.
    s.push_str(&format!(
        "struct Row {{\n    uint c[{ROW}];\n}};\n\nstruct Incidence {{\n    uint row;\n}};\n\nstruct Fatals {{\n"
    ));
    s.push_str(&format!("    float fatal[{}];\n", fatal_width(fatals)));
    s.push_str(
        r"};

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform Params { uint count; };

layout(set = 0, binding = 0) buffer FatalsData {
    Fatals data[];
} fatals;

layout(set = 0, binding = 1) buffer Index {
    Incidence data[];
} index;

layout(set = 0, binding = 2) buffer RowData {
    Row data[];
} rows;

layout(set = 0, binding = 3) buffer PlainData {
    float data[];
} plain;

layout(set = 0, binding = 4) buffer TwistData {
    Twist data[];
} twists;

layout(set = 0, binding = 5) buffer MotorData {
    Motor data[];
} motors;

",
    );
    let mut binding = 6;
    for o in outs {
        if let OutBuf::Own {
            struct_name,
            buffer,
            var,
            ..
        } = o
        {
            s.push_str(&format!(
                "layout(set = 0, binding = {binding}) buffer {buffer} {{\n    {struct_name} data[];\n}} {var};\n\n"
            ));
            binding += 1;
        }
    }
    s
}

/// The shared body prologue: fetch the invocation's index-table row (twists,
/// motors, each output's slot index, the immediate scalars and the params) into
/// locals the traced code and `input_map`/`output_map` reference by name.
fn body_pre(spec: &InSpec, outs: &[OutBuf]) -> String {
    let mut s = String::new();
    s.push_str(
        r"    uint idx = gl_GlobalInvocationID.x;

    if (idx >= count) return;

    uint r = index.data[idx].row;

",
    );
    // Word cursor into the row. The order here IS the layout, and the host bakers
    // (`rows::pre`, `rows::joint`) mirror it word for word — twists, motors,
    // outputs, scalars, params.
    let mut w = 0usize;
    for (count, ty, buf, name) in [
        (
            spec.twists,
            "Twist",
            "twists",
            InSpec::twist_name as fn(usize) -> String,
        ),
        (
            spec.motors,
            "Motor",
            "motors",
            InSpec::motor_name as fn(usize) -> String,
        ),
    ] {
        if count == 0 {
            continue;
        }
        for row in 0..count {
            s.push_str(&format!(
                "    uint idx_{f} = rows.data[r].c[{w}];\n",
                f = name(row)
            ));
            w += 1;
        }
        for row in 0..count {
            s.push_str(&format!(
                "    {ty} {f} = {buf}.data[idx_{f}];\n",
                f = name(row)
            ));
        }
        s.push('\n');
    }
    for o in outs {
        s.push_str(&format!(
            "    uint idx_{f} = rows.data[r].c[{w}];\n",
            f = o.index_field()
        ));
        w += 1;
    }
    s.push('\n');
    // Scalars are slot-backed like params. They used to ride the batch table as
    // immediates, which is exactly what stopped these rows from being bakeable:
    // `dt`, `warp` and the retraction change every step while the topology does
    // not. As indices into `plain` the row stays static and the step rewrites
    // three floats.
    for name in spec.scalars {
        s.push_str(&format!(
            "    float {name} = plain.data[rows.data[r].c[{w}]];\n"
        ));
        w += 1;
    }
    for k in 0..spec.params {
        s.push_str(&format!(
            "    float param_{k} = plain.data[rows.data[r].c[{w}]];\n"
        ));
        w += 1;
    }
    assert!(w <= ROW, "fixed-arity row layout does not fit ROW = {ROW}");
    s.push('\n');
    s
}

// ── ROW-driven kernels ───────────────────────────────────────────────────────
// A reduction stage cannot use the fixed-arity index row above: its term list is
// island-sized. Instead it reads a ROW — a baked `[u32; ROW]` in World storage
// holding every index it needs — and the batch table carries only which row.
// The kernel is then three pieces of text: `row_head` + the stage's prologue
// (which opens the term loop), the traced body of ONE iteration, and a footer
// closing the loop and storing the accumulator.

/// Words per incidence row. MUST equal `newton::accelerator::row::ROW` and the
/// literal in `aristotle::usual` (both assert it).
const ROW: usize = 128;

/// The preamble every ROW-driven kernel shares. Binding 1 is the batch table,
/// now just a row index; binding 2 is the row storage; `bufs` are the stage's
/// own storages from binding 3 upward, as `(glsl_struct, floats, block, var)`.
/// `fatals` is the trace's fatal-slot count (`fatal_slots`).
fn row_preamble(fatals: usize, bufs: &[(&str, usize, &str, &str)]) -> String {
    let mut s = String::new();
    let mut seen: Vec<&str> = Vec::new();
    for (struct_name, floats, _, _) in bufs {
        // Two buffers may share a struct (e.g. two `Wrench1` storages); declare
        // each shape once.
        if seen.contains(struct_name) {
            continue;
        }
        seen.push(struct_name);
        s.push_str(&format!(
            "struct {struct_name} {{\n    float c[{floats}];\n}};\n\n"
        ));
    }
    s.push_str(&format!(
        "struct Row {{\n    uint c[{ROW}];\n}};\n\nstruct Incidence {{\n    uint row;\n}};\n\nstruct Fatals {{\n    float fatal[{width}];\n}};\n\n",
        width = fatal_width(fatals)
    ));
    s.push_str(
        r"layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform Params { uint count; };

layout(set = 0, binding = 0) buffer FatalsData {
    Fatals data[];
} fatals;

layout(set = 0, binding = 1) buffer Index {
    Incidence data[];
} index;

layout(set = 0, binding = 2) buffer RowData {
    Row data[];
} rows;

",
    );
    for (binding, (struct_name, _, block, var)) in (3..).zip(bufs) {
        s.push_str(&format!(
            "layout(set = 0, binding = {binding}) buffer {block} {{\n    {struct_name} data[];\n}} {var};\n\n"
        ));
    }
    s
}

/// The prologue every ROW-driven kernel shares: bounds guard, fetch the row,
/// clear the fatal slot (which also keeps `Fatals` in the reflected layout —
/// shaderc strips a buffer nothing touches).
fn row_head() -> String {
    r"    uint idx = gl_GlobalInvocationID.x;
    if (idx >= count) return;
    uint r = index.data[idx].row;
    fatals.data[idx].fatal[0] = 0.0;
"
    .to_string()
}

/// The accumulator is a LOCAL of the same GLSL struct type as the storage it is
/// written to (`Wrench1 acc;`, `Block acc;`), so its components are `acc.c[slot]`
/// — exactly the spelling `field()` produces. That is the whole trick: the
/// accumulator is laid out in Pod slot order, the probe's slot+sign mapping
/// applies to it verbatim, and no separate permutation exists anywhere.
///
/// Two spellings, as everywhere else in this file: an INPUT rvalue folds the
/// sign in, an OUTPUT lvalue must not — a GLSL lvalue cannot negate, so the sign
/// travels beside it in the `output_map` tuple.
fn acc_in(probe: (usize, i8)) -> String {
    field("acc", probe)
}

fn acc_out(slot: usize) -> String {
    format!("acc.c[{slot}]")
}

/// One generated kernel: its file stem in `OUT_DIR` and the fatal slots it
/// writes. `main` collects these into the host's fatal-count table.
struct Kernel {
    stem: String,
    fatals: usize,
}

/// Write a generated shader into `OUT_DIR` as `{stem}.glsl`.
fn write_kernel(stem: String, glsl: String, fatals: usize) -> Kernel {
    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = PathBuf::from(out_dir).join(format!("{stem}.glsl"));
    fs::write(&dest_path, glsl).unwrap();
    Kernel { stem, fatals }
}

/// `SimpleSpringDamper_plain` → `SIMPLE_SPRING_DAMPER_PLAIN`.
fn const_name(stem: &str) -> String {
    let mut out = String::new();
    let mut prev_lower = false;
    for ch in stem.chars() {
        if ch.is_ascii_uppercase() && prev_lower {
            out.push('_');
        }
        prev_lower = ch.is_ascii_lowercase() || ch.is_ascii_digit();
        out.push(ch.to_ascii_uppercase());
    }
    out
}

/// Write `fatal_counts.rs` into `OUT_DIR`: one `usize` per generated kernel,
/// the number of fatal slots its host-side check scans. `accelerator/shaders.rs`
/// includes it, so the count the host reads is the count the trace produced.
fn write_fatal_counts(kernels: &[Kernel]) {
    let mut s = String::from(
        "// This file has been automatically generated by newton/build.rs. Do not edit!\n\n",
    );
    for k in kernels {
        s.push_str(&format!(
            "/// Fatal slots `{stem}.glsl` writes per invocation.\npub(super) const {name}: usize = {n};\n",
            stem = k.stem,
            name = const_name(&k.stem),
            n = k.fatals,
        ));
    }
    let out_dir = env::var("OUT_DIR").unwrap();
    fs::write(PathBuf::from(out_dir).join("fatal_counts.rs"), s).unwrap();
}

/// The world the traced joint lives in: `JointEdge::new` allocates its
/// per-connection value + Jacobian slots against these storages.
fn build_world() -> Arc<World> {
    Arc::new(
        World::builder()
            .with_storage::<Sym>()
            .with_storage::<Vector3<Sym>>()
            .with_storage::<[Wrench<Sym>; 2]>()
            .with_storage::<[[Wrench<Sym>; 24]; 2]>()
            .build(),
    )
}

/// Sym joint: the constants are PARAM references, not baked rationals. Built ONCE
/// (pure handles, no instructions) and captured by the trace closure. The
/// `Sym::param(i)` order MUST mirror `AxialSpringDamper::params()`. Anchors offset
/// from each COM so the wrench carries torque too → a densely-coupled block.
fn build_joint_sym(world: Arc<World>, v: &dyn JointFromParams<Sym>) -> JointEdge<Sym> {
    let mut params = Vec::new();
    for i in 0..v.n_params() {
        params.push(Sym::param(i as u32));
    }
    JointEdge::new(
        WorldId::get(),
        WorldId::get(),
        v.build_from_params(world, &params[..]),
    )
}

fn jac_kernel<S>(edge: &JointEdge<S>, inp: &[S]) -> Vec<S>
where
    S: Pod + Scalar + StandardPart + FromRational + Copy + Lift<Tangent<N24, S>>,
{
    // Velocity DOF: runtime value at input `ax - 12`, unit partial in gradient
    // axis `ax` (velocity axes are 12..24; their values are packed at inputs
    // 0..12). Value and gradient-axis indices are decoupled.
    let vvar = |ax: usize| -> Tangent<N24, S> {
        let mut g = <Vector<N24, S> as AbelianGroup>::ZERO;
        g[ax] = S::ONE;
        Tangent::from_grad(inp[ax - 12], g)
    };
    // Pose-twist DOF: PURE differential (value 0), unit partial in axis k. The
    // finite pose is carried by `base`, so the perturbation vanishes at the
    // linearization point — exactly Differential::seed. Standard part of the exp
    // bivector is then a hard zero ⇒ exp's small-angle branch resolves to its
    // Taylor limit at trace time and the trig branch is pruned. No loss of
    // generality: this IS what newton.rs computes (base arbitrary, δ-value = 0).
    let dvar = |k: usize| -> Tangent<N24, S> {
        let mut g = <Vector<N24, S> as AbelianGroup>::ZERO;
        g[k] = S::ONE;
        Tangent::from_grad(S::ZERO, g)
    };

    // Base pose per body: 8 even-grade Motor components as RUNTIME inputs, frozen
    // (value only, zero gradient — snap_pose is not differentiated). Reassembled
    // in the same component order the host serialises them.
    let base = |off: usize| -> Motor<Tangent<N24, S>> {
        Motor::from_components(&core::array::from_fn(|i| {
            Tangent::<N24, S>::embed(inp[off + i])
        }))
    };

    // Pose = base ∘ Motor::exp(δ), EXACTLY as `Differential::jacobian` retracts
    // (NOT `Twist::exp`, which bakes the geometric half-angle ½ and would halve
    // every pose column — the kernel-vs-Differential oracle pins this). δ is the
    // 6-DOF pose perturbation (value 0, gradient axes 0..12); base carries the
    // finite pose. There is no exp-"time": Motor::exp is the unit retraction, and
    // the −½dt²/−½dt column scaling is applied by newton.rs's matrix assembler,
    // NOT baked here. Explicit drives base = actual pose.
    let pose_a = base(12).compose(&Motor::exp(&Twist::new(
        &Vector3::from([dvar(0), dvar(1), dvar(2)]),
        &Vector3::from([dvar(3), dvar(4), dvar(5)]),
    )));
    let pose_b = base(20).compose(&Motor::exp(&Twist::new(
        &Vector3::from([dvar(6), dvar(7), dvar(8)]),
        &Vector3::from([dvar(9), dvar(10), dvar(11)]),
    )));

    // World-frame spatial velocities: 6 twist coords per body, used directly.
    let vel_a = Twist::new(
        &Vector3::from([vvar(12), vvar(13), vvar(14)]),
        &Vector3::from([vvar(15), vvar(16), vvar(17)]),
    );
    let vel_b = Twist::new(
        &Vector3::from([vvar(18), vvar(19), vvar(20)]),
        &Vector3::from([vvar(21), vvar(22), vvar(23)]),
    );

    // dt and warp are RUNTIME inputs (not constants): symbolic base-scalar values.
    let epoch = Epoch::standalone(inp[28], inp[29]);

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

/// Plain per-connection VALUE kernel — the two connection wrenches ONLY, with NO
/// Jacobian and NO AD: the force law is evaluated directly over the scalar `S`
/// (not `Tangent<N24, S>`). Same NIN=30 input layout as `jac_kernel`, but the pose
/// perturbation δ never appears (`pose = base`), so the output is just
/// NOUT_PLAIN=12: the 12 wrench components (force+torque of both bodies). This is
/// the value-only connection kernel that fills the gather's per-connection slot —
/// the explicit families and the Newton residual value pass need exactly this.
fn plain_kernel<S>(edge: &JointEdge<S>, inp: &[S]) -> Vec<S>
where
    S: Pod + Scalar + StandardPart + FromRational + Copy + Lift<S>,
{
    // Base pose per body: 8 even-grade Motor components as RUNTIME inputs. No
    // perturbation (δ = 0 ⇒ pose = base), so no AD gradient — plain evaluation.
    let base = |off: usize| -> Motor<S> {
        Motor::from_components(&core::array::from_fn(|i| inp[off + i]))
    };
    let pose_a = base(12);
    let pose_b = base(20);

    // World-frame spatial velocities: 6 twist coords per body (values at inp 0..12).
    let vel_a = Twist::new(
        &Vector3::from([inp[0], inp[1], inp[2]]),
        &Vector3::from([inp[3], inp[4], inp[5]]),
    );
    let vel_b = Twist::new(
        &Vector3::from([inp[6], inp[7], inp[8]]),
        &Vector3::from([inp[9], inp[10], inp[11]]),
    );

    // dt and warp are RUNTIME inputs (not constants).
    let epoch = Epoch::standalone(inp[28], inp[29]);

    let [wa, wb]: [Wrench<S>; 2] = edge
        .eval::<S>(&vector![pose_a, pose_b], &vector![vel_a, vel_b], &epoch)
        .split();

    // 12 wrench components (force + torque of both bodies), values only.
    vec![
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
    ]
}

/// Output scatter for the full Jacobian kernel — 300 = 12 components × 25 slots
/// `[value, ∂/∂axis0 … ∂/∂axis23]`. Component `c = body*6 + p` (p — wrench coord
/// in constructor order `[f0,f1,f2,t0,t1,t2]`); `out[c*25]` is the value,
/// `out[c*25+1+axis]` the partial. Values land in the value buffer `[Wrench;2]`
/// (as the plain kernel writes), partials in the block buffer `[[Wrench;24];2]`
/// (body outer, 24 axes, 6 slots):
///   value   → values.data[idx_conn].c[body*6 + slot]
///   partial → wrenches.data[idx_wrenches].c[body*144 + axis*6 + slot]
/// Slot+sign come from the same `screw` probe for coord p; the sign of a wrench
/// component is shared by its value and all its partials (all built by
/// `Wrench::new` with one markup). All 300 address DISTINCT slots. The `values`/
/// `wrenches` names and `idx_conn`/`idx_wrenches` fields must match the `OutBuf`s.
fn jacobian_output_map() -> HashMap<u32, (String, i8)> {
    let screw = wrench_slots();
    (0..NOUT_FULL as u32)
        .map(|i| {
            let comp = i as usize / 25; // 0..12: body A wrench 0..6, body B 6..12
            let rem = i as usize % 25; // 0 — value, 1..25 — partial along axis rem-1
            let body = comp / 6; // 0 or 1
            let (slot, sign) = screw[comp % 6];
            let lhs = if rem == 0 {
                format!("values.data[idx_conn].c[{}]", body * 6 + slot)
            } else {
                let axis = rem - 1;
                format!(
                    "wrenches.data[idx_wrenches].c[{}]",
                    body * 144 + axis * 6 + slot
                )
            };
            (i, (lhs, sign))
        })
        .collect()
}

/// Output scatter for the plain value kernel: comps `[fA0..tA2, fB0..tB2]` into
/// the value buffer `[Wrench;2]` (body A in c[0..6], body B in c[6..12]). Coord
/// `i` → the same `screw` slot+sign, shifted by 6 for body B.
fn plain_output_map() -> HashMap<u32, (String, i8)> {
    let screw = wrench_slots();
    (0..NOUT_PLAIN as u32)
        .map(|i| {
            let (slot, sign) = screw[i as usize % 6];
            let global = slot + if i < 6 { 0 } else { 6 };
            (
                i,
                (format!("wrenches.data[idx_wrenches].c[{global}]"), sign),
            )
        })
        .collect()
}

fn generate_jacobian(v: &dyn JointFromParams<Sym>) -> Kernel {
    let joint = build_joint_sym(build_world(), v);
    // The 24 partials of each component are the lane-parallel AD-gradient axis;
    // declare them so the vectorizer groups them (zero-folding hides this from
    // shape inference). 12 groups of 24 → tiled to vec4 inside the pass.
    let lanes: Vec<Vec<usize>> = (0..12)
        .map(|c| (1..25).map(|k| c * 25 + k).collect())
        .collect();
    let traced = Tracer::builder()
        .fn_name(v.shader_name())
        .flatten()
        .vectorize_lanes(lanes)
        .build()
        .trace::<_>(
            NIN,
            v.n_params(),
            NOUT_FULL,
            &input_facts(),
            |inp, _param| jac_kernel::<Sym>(&joint, inp),
        );

    // Two output buffers: the Jacobian block and, alongside it, the wrench value
    // (both written from the one kernel output, as newton's dispatch consumes).
    let outs = [
        OutBuf::Own {
            comment: "// Per-connection Jacobian block: overlays the host Pod `[[Wrench;24];2]`\n\
                      // (body outermost, then the 24 gradient axes, then the 6 Pod wrench slots),\n\
                      // i.e. contiguous f32 at offset `body*144 + axis*6 + slot`. Flat, so the\n\
                      // std430 stride is exactly 4 and the cast is byte-faithful.\n",
            struct_name: "WrenchJac",
            floats: 288,
            buffer: "WrenchData",
            var: "wrenches",
            index_field: "wrenches",
        },
        OutBuf::Own {
            comment: "// Per-connection wrench VALUE: overlays the host Pod `[Wrench;2]` (body A in\n\
                      // c[0..6], body B in c[6..12]), same slot as the plain kernel writes.\n",
            struct_name: "Wrench2",
            floats: 12,
            buffer: "WrenchValData",
            var: "values",
            index_field: "conn",
        },
    ];

    let spec = InSpec::joint(v.n_params());
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        preamble(&spec, fatals, &outs),
        body_pre(&spec, &outs),
        String::new(),
        &input_map(&spec),
        &jacobian_output_map(),
        &fatal_map(fatals),
    );
    write_kernel(format!("{}_jacobian", v.shader_name()), glsl, fatals)
}

/// Value-only sibling of `generate_jacobian`: traces `plain_kernel` (the two
/// connection wrenches, no Jacobian, no AD) into a much smaller kernel and emits
/// `{name}_plain.glsl`. This is the GPU kernel for connection dispatches that ask
/// for the value only (explicit families / the Newton residual value pass) — no
/// 24-wide AD gradient, so no `vectorize_lanes` (the output is 12 flat scalars).
fn generate_plain(v: &dyn JointFromParams<Sym>) -> Kernel {
    let joint = build_joint_sym(build_world(), v);
    let traced = Tracer::builder()
        .fn_name(v.shader_name())
        .flatten()
        .build()
        .trace::<_>(
            NIN,
            v.n_params(),
            NOUT_PLAIN,
            &input_facts(),
            |inp, _param| plain_kernel::<Sym>(&joint, inp),
        );

    // One output buffer: the wrench value `[Wrench;2]`.
    let outs = [OutBuf::Own {
        comment: "",
        struct_name: "Wrench2",
        floats: 12,
        buffer: "WrenchData",
        var: "wrenches",
        index_field: "wrenches",
    }];

    let spec = InSpec::joint(v.n_params());
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        preamble(&spec, fatals, &outs),
        body_pre(&spec, &outs),
        String::new(),
        &input_map(&spec),
        &plain_output_map(),
        &fatal_map(fatals),
    );
    write_kernel(format!("{}_plain", v.shader_name()), glsl, fatals)
}

/// PRE kernel — per BODY, not per connection, and with no AD: form the midpoint
/// pose `pose ∘ exp(retraction·V)` and carry the solve velocity through. The body
/// is `functions::pre` itself, the same code the CPU stand-in runs, so the two
/// cannot drift apart.
///
/// Compact one-body input layout (`InSpec::pre`): 0..6 the world twist in
/// `Twist::new` order, 6..14 the pose motor's even-store components, 14 the
/// retraction — the exp-time folded into the midpoint (½dt for the implicit
/// family, 0 for the explicit ones, where `midpoint = pose` falls out of the
/// same expression). Output: the 8 midpoint motor components, then the solve
/// velocity's 6 twist coordinates.
fn pre_kernel<S>(inp: &[S]) -> Vec<S>
where
    S: Pod + Scalar + StandardPart + FromRational + Copy + Lift<S>,
{
    let vel = Twist::new(
        &Vector3::from([inp[0], inp[1], inp[2]]),
        &Vector3::from([inp[3], inp[4], inp[5]]),
    );
    let pose = Motor::from_components(&core::array::from_fn(|i| inp[6 + i]));

    let (midpoint, solve_vel) = functions::pre(pose, vel, inp[14]);

    // `to_components` is the exact inverse of the `from_components` the pose was
    // assembled with, so both ends speak the same blade order.
    let mut out = midpoint.to_components().to_vec();
    let (l, a) = (solve_vel.linear(), solve_vel.angular());
    out.extend([l[0], l[1], l[2], a[0], a[1], a[2]]);
    out
}

/// Output scatter for the PRE kernel. Both destinations are ordinary world slots
/// of the SAME storages the kernel reads its inputs from, so they are written
/// through bindings 3/4 (see `OutBuf::Shared`):
///   midpoint pose  → motors.data[idx_midpoint].c[motor slot of component k]
///   solve velocity → twists.data[idx_solve_vel].c[screw slot of coord k−8]
/// Slot+sign come from the same probes the input side uses.
fn pre_output_map() -> HashMap<u32, (String, i8)> {
    let motor = motor_slots();
    let screw = wrench_slots();
    (0..NOUT_PRE as u32)
        .map(|i| {
            let k = i as usize;
            let (lhs, sign) = if k < 8 {
                let (slot, sign) = motor[k];
                (format!("motors.data[idx_midpoint].c[{slot}]"), sign)
            } else {
                let (slot, sign) = screw[k - 8];
                (format!("twists.data[idx_solve_vel].c[{slot}]"), sign)
            };
            (i, (lhs, sign))
        })
        .collect()
}

/// Emit `Pre.glsl` — the per-body PRE stage of every dispatch (phase 1 of
/// the implicit step), value-only like `generate_plain` but over one body instead of a
/// connection.
///
/// Traced under NO input facts: unlike the connection kernels this one never
/// sees dt or warp, and the retraction it does see is legitimately zero for the
/// explicit families, so nothing can be promised about it. Both arms of `exp`'s
/// small-angle branch therefore stay live — as they must, since a body at rest
/// has a vanishing twist too.
fn generate_pre() -> Kernel {
    let traced = Tracer::builder()
        .fn_name("Pre")
        .flatten()
        .build()
        .trace::<_>(NIN_PRE, 0, NOUT_PRE, &[], |inp, _param| {
            pre_kernel::<Sym>(inp)
        });

    let spec = InSpec::pre();
    // No buffers of its own: the midpoint pose and the solve velocity go back
    // into the `Motor` / `Twist` storages already bound for the input pose and
    // velocity, at their own slots.
    let outs = [
        OutBuf::Shared {
            index_field: "midpoint",
        },
        OutBuf::Shared {
            index_field: "solve_vel",
        },
    ];

    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        preamble(&spec, fatals, &outs),
        body_pre(&spec, &outs),
        String::new(),
        &input_map(&spec),
        &pre_output_map(),
        &fatal_map(fatals),
    );
    write_kernel("Pre".to_string(), glsl, fatals)
}

/// GATHER — per body: `total = external + Σ` incident connection wrenches.
/// One row per body per round; each term names a `[Wrench; 2]` connection slot
/// and which end of it feeds this body.
///
/// Row layout (authoritative copy in `accelerator::rows::gather`):
///   0 out (Wrench slot) | 1 external (Wrench slot) | 2 accumulate | 3 n_terms
///   4.. term = pair_slot * 2 + end
///
/// `external` and `out` are the SAME storage, so it is bound once and both are
/// read through it — binding it twice would alias it.
fn generate_gather() -> Kernel {
    let traced = Tracer::builder()
        .fn_name("Gather")
        .flatten()
        .build()
        .trace::<_>(12, 0, 6, &[], |inp: &Vec<Sym>, _param: &Vec<Sym>| {
            let acc = Wrench::new(
                &Vector3::from([inp[0], inp[1], inp[2]]),
                &Vector3::from([inp[3], inp[4], inp[5]]),
            );
            let w = Wrench::new(
                &Vector3::from([inp[6], inp[7], inp[8]]),
                &Vector3::from([inp[9], inp[10], inp[11]]),
            );
            let out = functions::gather_term::<Sym>(acc, w);
            let (f, t) = (out.force(), out.torque());
            vec![f[0], f[1], f[2], t[0], t[1], t[2]]
        });

    let screw = wrench_slots();
    // Inputs 0..6 — the accumulator; 6..12 — the term's wrench, read out of the
    // connection pair at the end this body sits on. Both go through the same
    // slot+sign probe; the pair rvalue is spelled out because its slot is offset
    // by the loop-local `e`, which `field()` cannot express.
    let mut inputs: HashMap<u32, String> = HashMap::new();
    for (k, &probe) in screw.iter().enumerate() {
        inputs.insert(k as u32, acc_in(probe));
        let (slot, sign) = probe;
        let rv = format!("pairs.data[p].c[e + {slot}]");
        inputs.insert(6 + k as u32, if sign < 0 { format!("-{rv}") } else { rv });
    }
    let outputs: HashMap<u32, (String, i8)> = (0..6u32)
        .map(|k| {
            let (slot, sign) = screw[k as usize];
            (k, (acc_out(slot), sign))
        })
        .collect();

    // Round 0 seeds from `external`, later rounds from the output they extend —
    // blended by `a1` rather than branched on.
    let pre = row_head()
        + r"    uint idx_out = rows.data[r].c[0];
    uint idx_ext = rows.data[r].c[1];
    float a1 = float(rows.data[r].c[2]);
    uint n = rows.data[r].c[3];
    Wrench1 acc;
    for (uint i = 0u; i < 6u; ++i) acc.c[i] = a1 * total.data[idx_out].c[i] + (1.0 - a1) * total.data[idx_ext].c[i];
    for (uint k = 0u; k < n; ++k) {
        uint t = rows.data[r].c[4 + k];
        uint p = t >> 1;
        uint e = (t & 1u) * 6u;
";
    let footer = r"    }
    for (uint i = 0u; i < 6u; ++i) total.data[idx_out].c[i] = acc.c[i];
"
    .to_string();

    let bufs = [
        ("Wrench1", 6usize, "WrenchData", "total"),
        ("Wrench2", 12usize, "PairData", "pairs"),
    ];
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        row_preamble(fatals, &bufs),
        pre,
        footer,
        &inputs,
        &outputs,
        &fatal_map(fatals),
    );
    write_kernel("Gather".to_string(), glsl, fatals)
}

/// BLOCK MATVEC — per body row: `dv_i = Σ_j x[i*m + j] · rhs[j]`.
///
/// Row layout (authoritative copy in `accelerator::rows::matvec`):
///   0 out (Twist slot) | 1 accumulate | 2 n_terms
///   3.. pairs (x_block, rhs_wrench)
fn generate_block_matvec() -> Kernel {
    let traced = Tracer::builder()
        .fn_name("BlockMatVec")
        .flatten()
        .build()
        .trace::<_>(48, 0, 6, &[], |inp: &Vec<Sym>, _param: &Vec<Sym>| {
            let acc = Twist::new(
                &Vector3::from([inp[0], inp[1], inp[2]]),
                &Vector3::from([inp[3], inp[4], inp[5]]),
            );
            let x: [[Sym; 6]; 6] =
                core::array::from_fn(|r| core::array::from_fn(|c| inp[6 + r * 6 + c]));
            let w = Wrench::new(
                &Vector3::from([inp[42], inp[43], inp[44]]),
                &Vector3::from([inp[45], inp[46], inp[47]]),
            );
            let out = functions::matvec_term::<Sym>(acc, x, w);
            let (l, a) = (out.linear(), out.angular());
            vec![l[0], l[1], l[2], a[0], a[1], a[2]]
        });

    // Twist and Wrench share the screw store, so one probe serves both ends.
    let screw = wrench_slots();
    let mut inputs: HashMap<u32, String> = HashMap::new();
    for (k, &probe) in screw.iter().enumerate() {
        inputs.insert(k as u32, acc_in(probe));
        let (slot, sign) = probe;
        let rv = format!("rhs.data[wk].c[{slot}]");
        inputs.insert(42 + k as u32, if sign < 0 { format!("-{rv}") } else { rv });
    }
    // The block is a plain array — no probe mapping, index `r * 6 + c`.
    for i in 0..36usize {
        inputs.insert(6 + i as u32, format!("blocks.data[xk].c[{i}]"));
    }
    let outputs: HashMap<u32, (String, i8)> = (0..6u32)
        .map(|k| {
            let (slot, sign) = screw[k as usize];
            (k, (acc_out(slot), sign))
        })
        .collect();

    let pre = row_head()
        + r"    uint idx_out = rows.data[r].c[0];
    float a1 = float(rows.data[r].c[1]);
    uint n = rows.data[r].c[2];
    Twist1 acc;
    for (uint i = 0u; i < 6u; ++i) acc.c[i] = a1 * dv.data[idx_out].c[i];
    for (uint k = 0u; k < n; ++k) {
        uint xk = rows.data[r].c[3 + 2u * k];
        uint wk = rows.data[r].c[4 + 2u * k];
";
    let footer = r"    }
    for (uint i = 0u; i < 6u; ++i) dv.data[idx_out].c[i] = acc.c[i];
"
    .to_string();

    let bufs = [
        ("Block", 36usize, "BlockData", "blocks"),
        ("Wrench1", 6usize, "RhsData", "rhs"),
        ("Twist1", 6usize, "DvData", "dv"),
    ];
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        row_preamble(fatals, &bufs),
        pre,
        footer,
        &inputs,
        &outputs,
        &fatal_map(fatals),
    );
    write_kernel("BlockMatVec".to_string(), glsl, fatals)
}

/// GEMM — per output block: `out = Σ_k a[i*m + k] · (sign · b[k*m + j] + diag·2I)`.
///
/// Row layout (authoritative copy in `accelerator::rows::gemm`):
///   0 out_block | 1 accumulate | 2 n_terms | 3 sign (f32 bits)
///   4.. pairs (a_block, b_block * 2 + diag)
fn generate_gemm() -> Kernel {
    let traced = Tracer::builder()
        .fn_name("Gemm")
        .flatten()
        .build()
        .trace::<_>(110, 0, 36, &[], |inp: &Vec<Sym>, _param: &Vec<Sym>| {
            let grab = |off: usize| -> [[Sym; 6]; 6] {
                core::array::from_fn(|r| core::array::from_fn(|c| inp[off + r * 6 + c]))
            };
            let out = functions::gemm_term::<Sym>(grab(0), grab(36), grab(72), inp[108], inp[109]);
            (0..36).map(|i| out[i / 6][i % 6]).collect()
        });

    // Blocks carry NO probe mapping — a plain array, index `r * 6 + c`, so the
    // accumulator, both factors and the output agree by construction.
    let mut inputs: HashMap<u32, String> = HashMap::new();
    for i in 0..36usize {
        inputs.insert(i as u32, format!("acc.c[{i}]"));
        inputs.insert(36 + i as u32, format!("blocks.data[ak].c[{i}]"));
        inputs.insert(72 + i as u32, format!("blocks.data[bk].c[{i}]"));
    }
    inputs.insert(108, "sign".to_string());
    inputs.insert(109, "diag".to_string());
    let outputs: HashMap<u32, (String, i8)> = (0..36u32)
        .map(|i| (i, (format!("acc.c[{i}]"), 1)))
        .collect();

    let pre = row_head()
        + r"    uint idx_out = rows.data[r].c[0];
    float a1 = float(rows.data[r].c[1]);
    uint n = rows.data[r].c[2];
    float sign = uintBitsToFloat(rows.data[r].c[3]);
    Block acc;
    for (uint i = 0u; i < 36u; ++i) acc.c[i] = a1 * blocks.data[idx_out].c[i];
    for (uint k = 0u; k < n; ++k) {
        uint ak = rows.data[r].c[4 + 2u * k];
        uint bt = rows.data[r].c[5 + 2u * k];
        uint bk = bt >> 1;
        float diag = float(bt & 1u);
";
    let footer = r"    }
    for (uint i = 0u; i < 36u; ++i) blocks.data[idx_out].c[i] = acc.c[i];
"
    .to_string();

    let bufs = [("Block", 36usize, "BlockData", "blocks")];
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        row_preamble(fatals, &bufs),
        pre,
        footer,
        &inputs,
        &outputs,
        &fatal_map(fatals),
    );
    write_kernel("Gemm".to_string(), glsl, fatals)
}

/// ASSEMBLE — per output block: sum the connection Jacobians landing in `(i, j)`
/// with the midpoint factors applied as the columns are read, plus the body's
/// mass block on the diagonal.
///
/// Row layout (authoritative copy in `accelerator::rows::assemble`):
///   0 out_block | 1 mass_block | 2 mass_scale | 3 pose_factor | 4 vel_factor
///   5 accumulate | 6 n_terms | 7.. term = jac_slot*4 + row_end*2 + col_end
///
/// The Jacobian storage is `[[Wrench; 24]; 2]`, flat float offset
/// `body*144 + axis*6 + slot`; with `row_end = bi` and `col_end = ce` the pose
/// column `col` is axis `ce*6 + col` and the velocity column is axis
/// `12 + ce*6 + col`, so the host packs `base = bi*144 + ce*36` and the kernel
/// reads `base + col*6 + slot` / `base + 72 + col*6 + slot`.
fn generate_assemble() -> Kernel {
    let traced = Tracer::builder()
        .fn_name("AssembleBlock")
        .flatten()
        .build()
        .trace::<_>(110, 0, 36, &[], |inp: &Vec<Sym>, _param: &Vec<Sym>| {
            let acc: [[Sym; 6]; 6] =
                core::array::from_fn(|r| core::array::from_fn(|c| inp[r * 6 + c]));
            let wrench_at = |off: usize| -> [Wrench<Sym>; 6] {
                core::array::from_fn(|col| {
                    let b = off + col * 6;
                    Wrench::new(
                        &Vector3::from([inp[b], inp[b + 1], inp[b + 2]]),
                        &Vector3::from([inp[b + 3], inp[b + 4], inp[b + 5]]),
                    )
                })
            };
            let out = functions::assemble_term::<Sym>(
                acc,
                wrench_at(36),
                wrench_at(72),
                inp[108],
                inp[109],
            );
            (0..36).map(|i| out[i / 6][i % 6]).collect()
        });

    let screw = wrench_slots();
    let mut inputs: HashMap<u32, String> = HashMap::new();
    for i in 0..36usize {
        inputs.insert(i as u32, format!("acc.c[{i}]"));
    }
    for col in 0..6usize {
        for (k, &(slot, sign)) in screw.iter().enumerate() {
            let wrap = |s: String| if sign < 0 { format!("-{s}") } else { s };
            inputs.insert(
                (36 + col * 6 + k) as u32,
                wrap(format!("jac.data[jk].c[base + {} + {slot}]", col * 6)),
            );
            inputs.insert(
                (72 + col * 6 + k) as u32,
                wrap(format!("jac.data[jk].c[base + 72 + {} + {slot}]", col * 6)),
            );
        }
    }
    inputs.insert(108, "pf".to_string());
    inputs.insert(109, "vf".to_string());
    let outputs: HashMap<u32, (String, i8)> = (0..36u32)
        .map(|i| (i, (format!("acc.c[{i}]"), 1)))
        .collect();

    // `mass_scale` is 1.0 on the diagonal (first round only) and 0.0 elsewhere,
    // so the mass block is added exactly once with no branch.
    let pre = row_head()
        + r"    uint idx_out = rows.data[r].c[0];
    uint idx_mass = rows.data[r].c[1];
    float mass_scale = uintBitsToFloat(rows.data[r].c[2]);
    float pf = plain.data[rows.data[r].c[3]].c[0];
    float vf = plain.data[rows.data[r].c[4]].c[0];
    float a1 = float(rows.data[r].c[5]);
    uint n = rows.data[r].c[6];
    Block acc;
    for (uint i = 0u; i < 36u; ++i) acc.c[i] = a1 * blocks.data[idx_out].c[i] + mass_scale * blocks.data[idx_mass].c[i];
    for (uint k = 0u; k < n; ++k) {
        uint t = rows.data[r].c[7 + k];
        uint jk = t >> 2;
        uint base = ((t >> 1) & 1u) * 144u + (t & 1u) * 36u;
";
    let footer = r"    }
    for (uint i = 0u; i < 36u; ++i) blocks.data[idx_out].c[i] = acc.c[i];
"
    .to_string();

    // `plain` carries the two midpoint factors. They are the ONLY per-sub-step
    // values the assembly needs, and holding them as bits inside the row made the
    // row depend on the sub-step: every change of `half` reallocated all `m²` rows
    // and dropped the old ones, which at 66 unknowns was a third of the step's CPU
    // time. As slot indices the row depends on the topology alone.
    let bufs = [
        ("Scalar", 1usize, "ScalarData", "plain"),
        ("Block", 36usize, "BlockData", "blocks"),
        ("WrenchJac", 288usize, "JacData", "jac"),
    ];
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        row_preamble(fatals, &bufs),
        pre,
        footer,
        &inputs,
        &outputs,
        &fatal_map(fatals),
    );
    write_kernel("AssembleBlock".to_string(), glsl, fatals)
}

/// PER-BODY POST. `diagonal` picks which angular-inertia storage is bound and how
/// the tensor's nine entries are fetched; everything else is identical, so the two
/// kernels come out of one emitter.
///
/// Row layout (authoritative copy in `accelerator::rows::body_post`):
///   0 n_bodies | 1 half slot | 2 floor2 slot
///   3 + 9b + 0..9 — midpoint_pose, solve_vel, mass, angular, snap_mom,
///                   total_wrench, mass_out, rhs_out, scale_out
/// BODY POST with the wrench GATHER folded in.
///
/// `Gather` sums a body's incident connection wrenches into `total_wrench`, and
/// `BodyPost` reads exactly that for exactly that body. Same invocation, same
/// body, nothing crossing between them — which is what makes these two, alone
/// among the stages, fusible into one kernel rather than merely one submission.
/// The sum stays a local; `total_wrench` is not written at all on this path.
///
/// The accumulate loop is hand-written GLSL because `emit_glsl` places ONE traced
/// body, and this kernel needs a loop and then a tail. The loop touches no Pod
/// slot mapping: a wrench sum is component-wise in slot order, so `acc.c[i] +=
/// pair.c[e + i]` is the whole of it, and only the tail — which the tracer emits —
/// needs `wrench_slots`.
///
/// ONE body per row, unlike the unfused `BodyPost` which packs thirteen: the term
/// list is per body. That leaves `ROW - 12 = 116` incident connections, and a body
/// with more than that keeps the two-stage path (see `rows::body_post`).
///
/// Row layout (authoritative copy in `accelerator::rows::body_post`):
///   0 hstep | 1 floor2 | 2 midpoint | 3 solve_vel | 4 mass | 5 angular
///   6 snap_mom | 7 external | 8 mass_out | 9 rhs_out | 10 scale_out | 11 n_terms
///   12.. `pair_slot * 2 + end`
fn generate_body_post_gathered(diagonal: bool) -> Kernel {
    let name = if diagonal {
        "BodyPostGatheredDiagonal"
    } else {
        "BodyPostGatheredFull"
    };
    let traced = Tracer::builder()
        .fn_name(name)
        .flatten()
        .build()
        .trace::<_>(38, 0, 48, &[], |inp: &Vec<Sym>, _param: &Vec<Sym>| {
            let pose = Motor::from_components(&core::array::from_fn(|i| inp[i]));
            let vel = Twist::new(
                &Vector3::from([inp[8], inp[9], inp[10]]),
                &Vector3::from([inp[11], inp[12], inp[13]]),
            );
            let angular: [[Sym; 3]; 3] =
                core::array::from_fn(|i| core::array::from_fn(|j| inp[15 + i * 3 + j]));
            let snap = Wrench::new(
                &Vector3::from([inp[24], inp[25], inp[26]]),
                &Vector3::from([inp[27], inp[28], inp[29]]),
            );
            let total = Wrench::new(
                &Vector3::from([inp[30], inp[31], inp[32]]),
                &Vector3::from([inp[33], inp[34], inp[35]]),
            );
            let (block, rhs, scale) = functions::body_post::<Sym>(
                pose, vel, inp[14], angular, snap, total, inp[36], inp[37],
            );
            let mut out: Vec<Sym> = (0..36).map(|i| block[i / 6][i % 6]).collect();
            let (rf, rt) = (rhs.force(), rhs.torque());
            let (sf, st) = (scale.force(), scale.torque());
            out.extend([rf[0], rf[1], rf[2], rt[0], rt[1], rt[2]]);
            out.extend([sf[0], sf[1], sf[2], st[0], st[1], st[2]]);
            out
        });

    let screw = wrench_slots();
    let motor = motor_slots();
    let mut inputs: HashMap<u32, String> = HashMap::new();
    for (k, &probe) in motor.iter().enumerate() {
        inputs.insert(k as u32, field("motors.data[mk]", probe));
    }
    for (k, &probe) in screw.iter().enumerate() {
        inputs.insert(8 + k as u32, field("twists.data[vk]", probe));
        inputs.insert(24 + k as u32, field("wrenches.data[snap_k]", probe));
        // The gather's result, as a local rather than a slot.
        inputs.insert(30 + k as u32, field("acc", probe));
    }
    inputs.insert(14, "plain.data[mass_k].c[0]".to_string());
    for i in 0..3usize {
        for j in 0..3usize {
            let rv = if diagonal {
                if i == j {
                    format!("ang.data[ak].c[{i}]")
                } else {
                    "0.0".to_string()
                }
            } else {
                format!("ang.data[ak].c[{}]", i * 3 + j)
            };
            inputs.insert((15 + i * 3 + j) as u32, rv);
        }
    }
    inputs.insert(36, "hstep".to_string());
    inputs.insert(37, "floor2".to_string());

    let mut outputs: HashMap<u32, (String, i8)> = (0..36u32)
        .map(|i| (i, (format!("blocks.data[mass_out].c[{i}]"), 1)))
        .collect();
    for k in 0..6u32 {
        let (slot, sign) = screw[k as usize];
        outputs.insert(36 + k, (format!("wrenches.data[rhs_out].c[{slot}]"), sign));
        outputs.insert(
            42 + k,
            (format!("wrenches.data[scale_out].c[{slot}]"), sign),
        );
    }

    let pre = row_head()
        + r"    float hstep = plain.data[rows.data[r].c[0]].c[0];
    float floor2 = plain.data[rows.data[r].c[1]].c[0];
    uint mk = rows.data[r].c[2];
    uint vk = rows.data[r].c[3];
    uint mass_k = rows.data[r].c[4];
    uint ak = rows.data[r].c[5];
    uint snap_k = rows.data[r].c[6];
    uint ext_k = rows.data[r].c[7];
    uint mass_out = rows.data[r].c[8];
    uint rhs_out = rows.data[r].c[9];
    uint scale_out = rows.data[r].c[10];
    uint n = rows.data[r].c[11];
    Wrench1 acc;
    for (uint i = 0u; i < 6u; ++i) acc.c[i] = wrenches.data[ext_k].c[i];
    for (uint k = 0u; k < n; ++k) {
        uint t = rows.data[r].c[12u + k];
        uint p = t >> 1;
        uint e = (t & 1u) * 6u;
        for (uint i = 0u; i < 6u; ++i) acc.c[i] += pairs.data[p].c[e + i];
    }
";
    let ang_floats = if diagonal { 3usize } else { 9usize };
    let bufs = [
        ("Motor", 8usize, "MotorData", "motors"),
        ("Twist1", 6usize, "TwistData", "twists"),
        ("Wrench1", 6usize, "WrenchData", "wrenches"),
        ("Block", 36usize, "BlockData", "blocks"),
        ("Ang", ang_floats, "AngData", "ang"),
        ("Scalar1", 1usize, "PlainData", "plain"),
        ("Wrench2", 12usize, "PairData", "pairs"),
    ];
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        row_preamble(fatals, &bufs),
        pre,
        String::new(),
        &inputs,
        &outputs,
        &fatal_map(fatals),
    );
    write_kernel(name.to_string(), glsl, fatals)
}

fn generate_body_post(diagonal: bool) -> Kernel {
    let name = if diagonal {
        "BodyPostDiagonal"
    } else {
        "BodyPostFull"
    };
    let traced = Tracer::builder()
        .fn_name(name)
        .flatten()
        .build()
        .trace::<_>(38, 0, 48, &[], |inp: &Vec<Sym>, _param: &Vec<Sym>| {
            let pose = Motor::from_components(&core::array::from_fn(|i| inp[i]));
            let vel = Twist::new(
                &Vector3::from([inp[8], inp[9], inp[10]]),
                &Vector3::from([inp[11], inp[12], inp[13]]),
            );
            let angular: [[Sym; 3]; 3] =
                core::array::from_fn(|i| core::array::from_fn(|j| inp[15 + i * 3 + j]));
            let snap = Wrench::new(
                &Vector3::from([inp[24], inp[25], inp[26]]),
                &Vector3::from([inp[27], inp[28], inp[29]]),
            );
            let total = Wrench::new(
                &Vector3::from([inp[30], inp[31], inp[32]]),
                &Vector3::from([inp[33], inp[34], inp[35]]),
            );
            let (block, rhs, scale) = functions::body_post::<Sym>(
                pose, vel, inp[14], angular, snap, total, inp[36], inp[37],
            );
            let mut out: Vec<Sym> = (0..36).map(|i| block[i / 6][i % 6]).collect();
            let (rf, rt) = (rhs.force(), rhs.torque());
            let (sf, st) = (scale.force(), scale.torque());
            out.extend([rf[0], rf[1], rf[2], rt[0], rt[1], rt[2]]);
            out.extend([sf[0], sf[1], sf[2], st[0], st[1], st[2]]);
            out
        });

    let screw = wrench_slots();
    let motor = motor_slots();
    let mut inputs: HashMap<u32, String> = HashMap::new();
    for (k, &probe) in motor.iter().enumerate() {
        inputs.insert(k as u32, field("motors.data[mk]", probe));
    }
    for (k, &probe) in screw.iter().enumerate() {
        inputs.insert(8 + k as u32, field("twists.data[vk]", probe));
        inputs.insert(24 + k as u32, field("wrenches.data[snap_k]", probe));
        inputs.insert(30 + k as u32, field("wrenches.data[total_k]", probe));
    }
    inputs.insert(14, "plain.data[mass_k].c[0]".to_string());
    for i in 0..3usize {
        for j in 0..3usize {
            // A diagonal inertia is fetched as a diagonal matrix: the off-diagonal
            // entries are literal zeros, so the trace folds them away entirely.
            let rv = if diagonal {
                if i == j {
                    format!("ang.data[ak].c[{i}]")
                } else {
                    "0.0".to_string()
                }
            } else {
                format!("ang.data[ak].c[{}]", i * 3 + j)
            };
            inputs.insert((15 + i * 3 + j) as u32, rv);
        }
    }
    inputs.insert(36, "hstep".to_string());
    inputs.insert(37, "floor2".to_string());

    let mut outputs: HashMap<u32, (String, i8)> = (0..36u32)
        .map(|i| (i, (format!("blocks.data[mass_out].c[{i}]"), 1)))
        .collect();
    for k in 0..6u32 {
        let (slot, sign) = screw[k as usize];
        outputs.insert(36 + k, (format!("wrenches.data[rhs_out].c[{slot}]"), sign));
        outputs.insert(
            42 + k,
            (format!("wrenches.data[scale_out].c[{slot}]"), sign),
        );
    }

    let pre = row_head()
        + r"    uint nb = rows.data[r].c[0];
    float hstep = plain.data[rows.data[r].c[1]].c[0];
    float floor2 = plain.data[rows.data[r].c[2]].c[0];
    for (uint b = 0u; b < nb; ++b) {
        uint o = 3u + 9u * b;
        uint mk = rows.data[r].c[o + 0u];
        uint vk = rows.data[r].c[o + 1u];
        uint mass_k = rows.data[r].c[o + 2u];
        uint ak = rows.data[r].c[o + 3u];
        uint snap_k = rows.data[r].c[o + 4u];
        uint total_k = rows.data[r].c[o + 5u];
        uint mass_out = rows.data[r].c[o + 6u];
        uint rhs_out = rows.data[r].c[o + 7u];
        uint scale_out = rows.data[r].c[o + 8u];
";
    let footer = "    }\n".to_string();

    let ang_floats = if diagonal { 3usize } else { 9usize };
    let bufs = [
        ("Motor", 8usize, "MotorData", "motors"),
        ("Twist1", 6usize, "TwistData", "twists"),
        ("Wrench1", 6usize, "WrenchData", "wrenches"),
        ("Block", 36usize, "BlockData", "blocks"),
        ("Ang", ang_floats, "AngData", "ang"),
        ("Scalar1", 1usize, "PlainData", "plain"),
    ];
    // `Motor::inverse` divides, yet this trace currently leaves NO fatal operand
    // (its count in `fatal_counts.rs` is 0). The count is read off the trace, so
    // neither case needs special handling here.
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        row_preamble(fatals, &bufs),
        pre,
        footer,
        &inputs,
        &outputs,
        &fatal_map(fatals),
    );
    write_kernel(name.to_string(), glsl, fatals)
}

/// Take a list of joint types (unit structs) and expand each into the
/// `(value, type_name)` tuple `main` iterates over. A `path` fragment works in
/// both positions: as the unit-struct value and as the turbofish type argument.
macro_rules! joints {
    ($($t:path),* $(,)?) => {
        vec![$(Box::new($t)),*]
    };
}

/// BLOCK REDUCE — `Σ ‖diag·I − b‖²_F` over a list of blocks, one partial per row.
///
/// The solver needs two scalars off the block arrays: how far `A·X` is from the
/// identity, and whether `X` is still finite. Both were `m²` `WorldKey::read`s on
/// the CPU — each one a lock on the whole block storage plus a host-access guard —
/// evaluated inside the Newton–Schulz loop, i.e. the term that grew fastest with
/// body count after the message count was fixed.
///
/// ONE round, not a tree: a row folds up to `MAX_TERMS` blocks, so the host is
/// left with `⌈m²/126⌉` scalars to add up — 17 at 45 unknowns against 2025 block
/// reads. A second GPU round to fold those would cost a dispatch to save a
/// handful of adds.
///
/// Row layout (authoritative copy in `accelerator::rows::reduce`):
///   0 out (T slot) | 1 n_terms | 2.. `block * 2 + diag`
fn generate_block_reduce() -> Kernel {
    let traced = Tracer::builder()
        .fn_name("BlockReduce")
        .flatten()
        .build()
        .trace::<_>(38, 0, 1, &[], |inp: &Vec<Sym>, _param: &Vec<Sym>| {
            let b: [[Sym; 6]; 6] =
                core::array::from_fn(|r| core::array::from_fn(|c| inp[1 + r * 6 + c]));
            vec![functions::reduce_sq_term::<Sym>(inp[0], b, inp[37])]
        });

    // Blocks carry no probe mapping — a plain array indexed `r * 6 + c`.
    let mut inputs: HashMap<u32, String> = HashMap::new();
    inputs.insert(0, "acc".to_string());
    for i in 0..36usize {
        inputs.insert(1 + i as u32, format!("blocks.data[bk].c[{i}]"));
    }
    inputs.insert(37, "diag".to_string());
    let outputs: HashMap<u32, (String, i8)> =
        [(0u32, ("acc".to_string(), 1))].into_iter().collect();

    let pre = row_head()
        + r"    uint idx_out = rows.data[r].c[0];
    uint n = rows.data[r].c[1];
    float acc = 0.0;
    for (uint k = 0u; k < n; ++k) {
        uint t = rows.data[r].c[2 + k];
        uint bk = t >> 1;
        float diag = float(t & 1u);
";
    let footer = r"    }
    partials.data[idx_out].c[0] = acc;
"
    .to_string();

    // `Scalar` is the flat `T` storage seen as a one-float struct: same layout,
    // and it lets the shared row preamble declare it like any other buffer.
    let bufs = [
        ("Block", 36usize, "BlockData", "blocks"),
        ("Scalar", 1usize, "ScalarData", "partials"),
    ];
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        row_preamble(fatals, &bufs),
        pre,
        footer,
        &inputs,
        &outputs,
        &fatal_map(fatals),
    );
    write_kernel("BlockReduce".to_string(), glsl, fatals)
}

/// BLOCK COPY — `dst = src`, one row per block.
///
/// The solver publishes its accepted approximate inverse on a committed span and
/// restores it on a rejected one, `m²` blocks each way. As host loops those were
/// `m²` slot reads plus `m²` writes per span; here they are one dispatch.
///
/// One row per block rather than a term list: there is no reduction, so nothing
/// is carried between blocks, and a row per block keeps one invocation per copy.
///
/// Row layout (authoritative copy in `accelerator::rows::copy`):
///   0 dst (Block slot) | 1 src (Block slot)
fn generate_block_copy() -> Kernel {
    let traced = Tracer::builder()
        .fn_name("BlockCopy")
        .flatten()
        .build()
        .trace::<_>(36, 0, 36, &[], |inp: &Vec<Sym>, _param: &Vec<Sym>| {
            let b: [[Sym; 6]; 6] =
                core::array::from_fn(|r| core::array::from_fn(|c| inp[r * 6 + c]));
            let out = functions::copy_block::<Sym>(b);
            (0..36).map(|i| out[i / 6][i % 6]).collect()
        });

    // Blocks carry no probe mapping — a plain array, index `r * 6 + c`.
    let inputs: HashMap<u32, String> = (0..36u32)
        .map(|i| (i, format!("blocks.data[idx_src].c[{i}]")))
        .collect();
    let outputs: HashMap<u32, (String, i8)> = (0..36u32)
        .map(|i| (i, (format!("blocks.data[idx_dst].c[{i}]"), 1)))
        .collect();

    let pre = row_head()
        + r"    uint idx_dst = rows.data[r].c[0];
    uint idx_src = rows.data[r].c[1];
";
    let bufs = [("Block", 36usize, "BlockData", "blocks")];
    let fatals = fatal_slots(&traced);
    let glsl = traced.emit_glsl(
        row_preamble(fatals, &bufs),
        pre,
        String::new(),
        &inputs,
        &outputs,
        &fatal_map(fatals),
    );
    write_kernel("BlockCopy".to_string(), glsl, fatals)
}

fn main() {
    let joints: Vec<Box<dyn JointFromParams<Sym>>> = joints![
        CriticallyDampedWarped,
        SimpleSpringDamper,
        PerpendicularDamperWarpedBuilder,
        TorsionalDamperWarpedBuilder,
    ];
    // Two kernels per joint type: the full value+Jacobian block and the value-only
    // plain kernel (connection dispatch without a Jacobian).
    let mut kernels: Vec<Kernel> = Vec::new();
    kernels.extend(joints.iter().map(|c| generate_jacobian(c.as_ref())));
    kernels.extend(joints.iter().map(|c| generate_plain(c.as_ref())));

    kernels.push(generate_pre());
    kernels.push(generate_gather());
    kernels.push(generate_block_matvec());
    kernels.push(generate_gemm());
    kernels.push(generate_assemble());
    kernels.push(generate_block_reduce());
    kernels.push(generate_block_copy());
    kernels.push(generate_body_post(true));
    kernels.push(generate_body_post(false));
    kernels.push(generate_body_post_gathered(true));
    kernels.push(generate_body_post_gathered(false));

    write_fatal_counts(&kernels);

    println!("cargo:rerun-if-changed=build.rs");
}
