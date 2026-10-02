# Accelerator storage contract & kernel-runner `enqueue`

> **Migrated design record.** This document was written under the previous
> process and lived in `docs/superpowers/specs/`, which was deleted. It was kept
> because `ACCELERATOR` (the accelerator design document, retired in M1) cites it for the storage contract's
> rationale — the two aliasing bugs that produced it — which that document does not
> itself carry. It is history, not
> obligation: its `Status:` line records what was true when it was written, and
> where it disagrees with the code or with `ACCELERATOR` (the accelerator design document, retired in M1),
> they win.

Date: 2026-07-19
Status: design approved, pending implementation
Related: `/ACCELERATOR` (retired in M1) (Parts I–V)

## Problem

The in-progress migration moves per-body simulation state into World's dense
per-type storage behind `WorldKey<T>` handles. Two bugs surfaced, both rooted in
the same unfinished idea — the accelerator's intermediate result buffers have no
persistent home:

1. `Island::wrenches` was retyped to `IndexMap<WorldId, WorldKey<Wrench<T>>>` but
   nothing ever populated it (fixed separately: body-owned `accum_wrench`,
   cloned into the island map at `insert_body`, zeroed each step — never
   `clear()`ed).
2. `Newton::assemble_matrix` did `let total = external.clone()` then wrote back
   through the clone. With `WorldKey` values that clone is a **shallow handle
   copy** — `total` aliased the `accum_wrench` slots and accumulated joint forces
   into them across Newton iterates → runaway momentum.

Both say the same thing: the per-body accumulated wrench (and, ahead of it, the
per-connection Jacobian) is **accelerator output** that will arrive from the GPU
into World storage. It must live in persistent, World-backed, domain-owned slots
— not be conjured as a value scratch or an aliasing clone.

## Goal

Fix the storage contract for the accelerator's output streams so that:

- the integrator side becomes **100% accelerator-shaped now** — every
  `Dynamics::eval()` / `Differential::jacobian` call (all four integrators) moves
  *inside* `Accelerator::enqueue`, the CPU stand-in for the shader;
- swapping that CPU stand-in for a batched GPU dispatch later never changes an
  integrator — same `enqueue` signature, same World-backed output slots;
- the immediate aliasing/runaway bug is gone and the suite is green.

### Non-goals

- The GPU dispatch, batching, quiescence flush, `GpuVec`, coherency barriers
  (ACCELERATOR Parts II–IV) — unchanged, still deferred.
- The **fatal** stream — it is a crash indicator handled *inside* the
  accelerator; it does not enter the domain contract.

## Decisions

### D1 — Ownership: domain-owned slots

Slot data always lives in World's dense per-type storage; the `WorldKey` handle
is held by the domain object whose cardinality matches the stream, and frees on
drop. This matches how `pose` / `momentum` / `accum_wrench` already work; no
separate accelerator registry to keep aligned with topology.

- **Per-body** streams → `RigidBody<T>` fields.
- **Per-connection** streams → `JointEdge<T>` fields.

### D2 — The slots

`RigidBody<T>` gains (all `WorldKey<…>`):

| slot | type | role | writer | reader |
|---|---|---|---|---|
| `accum_wrench` *(landed)* | `Wrench<T>` | external-field forces (mechanism phase 4); the gather **seed** | `ForceField::accumulate` | integrator seed step |
| `total_wrench` *(new)* | `Wrench<T>` | per-body **gather output** = external + Σ joint wrench | `enqueue` (accumulate) | integrator (residual / explicit step) |
| `midpoint_pose` *(new)* | `Motor<T>` | PRE: `snap_pose ∘ exp(½dt·vmid)` (implicit) / current pose (explicit) — kernel `base` input | integrator PRE | `enqueue` |
| `solve_vel` *(new)* | `Twist<T>` | vmid (implicit) / world velocity (explicit) — kernel `vel` input | integrator PRE | `enqueue` |

`JointEdge<T>` gains:

| slot | type | role | writer | reader |
|---|---|---|---|---|
| `jacobian` *(new)* | `[[Wrench<T>; 24]; 2]` | per-connection 24-column block (implicit only); plain scatter, no gather | `enqueue` | `assemble_matrix` |

The Jacobian block is `[[Wrench<T>; 24]; 2]` = 288 scalars — exactly the 288
partials of `NOUT_FULL = 300` (the remaining 12 are the per-body value, which
goes to the gather, not this block). World storage registers this type via
`World::builder().usual::<T>()`.

**Not in the domain contract:** the fatal flag (accelerator-internal) and the
per-connection *value* buffer + true segmented gather (accelerator-internal;
`enqueue` accumulates straight into `total_wrench` for now and gains the buffer +
segmented reduction later, invisibly to integrators).

### D3 — `enqueue` is the kernel-runner

All joint force-law evaluation moves inside `enqueue`. Signature is **final now**
(so the CPU→GPU swap touches no integrator):

```rust
pub async fn enqueue<T: Scalar + Pod>(
    &self,
    edge:   &JointEdge<T>,                            // which kernel + params_keys()
    poses:  [WorldKey<Motor<T>>; 2],                  // midpoint_pose / pose
    vels:   [WorldKey<Twist<T>>; 2],                  // solve_vel / world velocity
    totals: [WorldKey<Wrench<T>>; 2],                 // per-body value output — accumulate into
    jacobian: Option<WorldKey<[[Wrench<T>; 24]; 2]>>, // per-connection output — None = skip
    epoch:  &Epoch<T>,                                // dt + warp
)
```

- `jacobian: Option<WorldKey<block>>` is both the switch and the destination:
  `Some(edge.jacobian_key())` = compute + write there; `None` = skip. There is no
  ambiguous "where does the block go" and `enqueue` never reaches into
  `JointEdge` internals — every kernel input and output is an explicit slot.
- **No `factor` parameter** — see D4.

CPU stand-in body (later replaced by a batched dispatch writing the *same* slots):

```
let [wa, wb] = edge.eval(poses, vels, epoch).split();
totals[0] += wa;  totals[1] += wb;                    // collapsed gather
if let Some(jac) = jacobian {
    let block = Differential::at(poses, vels).jacobian(edge, epoch).1;
    jac.write(block);
}
```

### D4 — Retraction: unit exp-time, drop `factor`

Adopt ACCELERATOR Part V's stated target. The kernel's pose perturbation goes to
**unit retraction** (`exp(1·δ)`), matching `Differential::jacobian`. The `−½dt²`
(pose) / `−½dt` (vel) column scaling stays where it already is — in
`assemble_matrix`. Consequences:

- `enqueue` has **no `factor` parameter**; `build.rs` kernel input 30 (`half`) is
  removed → `NIN` drops 31 → 30.
- **The CPU path is already this convention** (`Differential` at exp-time 1 +
  `assemble` scaling), so no solver numerics change on the CPU side. The only
  edit is `build.rs`: the kernel stops baking `½dt`, aligning it with the
  already-correct CPU.
- Gate: the **CPU-vs-kernel oracle test** (Part V) — pick a joint instance and a
  random (base pose, velocity), compute value + all 24 Jacobian columns two ways
  (CPU `Differential::jacobian` vs the compiled Lua kernel through viete's
  executor) and assert agreement. This lands with the `build.rs` change; it is
  the thing that certifies the two fillers of the `jacobian` slot are
  interchangeable.

## Data flow

Mechanism `step` phase 4 (unchanged) accumulates external field forces into each
body's `accum_wrench`.

### Implicit (`Newton`) — per span, per Newton iterate at current `vmid`

1. **PRE** (per body): `midpoint_pose ← snap_pose ∘ exp(½dt·vmid)`,
   `solve_vel ← vmid`.
2. **Seed gather** (per body): `total_wrench ← accum_wrench`.
   *(Replaces the aliasing `total = external.clone()`.)*
3. **`enqueue(edge, …, Some(edge.jacobian_key())).await`** per joint:
   `total_wrench[a] += wa`, `[b] += wb`; writes `edge.jacobian`.
4. After all edges: residual reads per-body `total_wrench` + inertia; matrix
   reads each `edge.jacobian` + inertia diagonal; solve; update `vmid`.
5. Loop 3–4 to convergence, then `commit_span`.

The two edge loops in `Newton` today — the residual assembly (`edge.eval`) and
`assemble_matrix` (`Differential::jacobian`) — collapse into **one** `enqueue`
pass filling the slots; residual and matrix then read the slots.

### Explicit (`bridge`, Euler / Symplectic / Lie) — per step

1. **PRE** (per body): `midpoint_pose ← current pose`, `solve_vel ← world_velocity`.
2. **Seed gather**: `total_wrench ← accum_wrench`.
3. **`enqueue(edge, …, None).await`** per joint: value only; accumulate into
   `total_wrench`.
4. Per body: read `total_wrench`, pull back to body frame, `Integrator::step`.

## Lifecycle & topology

- New `RigidBody` slots allocated in `RigidBody::new` alongside
  `pose`/`momentum`/`accum_wrench`; freed on drop.
- New `JointEdge.jacobian` slot allocated when the edge is constructed (it has
  access to the bodies' `World`); freed on drop.
- The island's step-time `wrenches` map holds **clones** of the body-owned
  `accum_wrench` (the external-forces buffer that `ForceField::accumulate`
  writes and the integrator seeds from): populated at `insert_body`, carried at
  `merge_from`, dropped at `remove_body`, persistent while island topology is
  stable (never `clear()`ed), values zeroed each step. This is the already-landed
  fix for bug 1.
- The other new per-body slots (`total_wrench`, `midpoint_pose`, `solve_vel`)
  need no map — the integrator reaches them per body via `bodies[id].body()`
  during PRE / seed / read, keyed by the same `order`/`bodies` iteration it
  already uses.

## What is implemented now vs deferred

**Now (this spec):**
- All `RigidBody` + `JointEdge` slots (D2).
- `enqueue` as the CPU kernel-runner (D3): value gather for all four integrators,
  Jacobian block for implicit.
- Integrator refactor: PRE + seed + `enqueue` loop + read-slots in `Newton` and
  `bridge`; remove inline `eval`/`Differential`.
- `build.rs`: unit retraction, drop input 30 (D4) + the CPU-vs-kernel oracle test.
- Bug-1 and bug-2 fixes fall out of the above; suite green.

**Deferred (unchanged from ACCELERATOR Parts II–IV):** batched GPU dispatch,
per-connection value buffer + segmented gather, PRE/POST as separate GPU stages,
fatal stream, `GpuVec`, coherency, quiescence flush.

## ACCELERATOR updates (document retired in M1)

- Appendix: `NIN 31 → 30`; remove input 30 (`half`/`factor`); note pose
  perturbation is unit retraction.
- Part V: mark the retraction-time mismatch **resolved** (unit retraction);
  fold the oracle test from "to build" into this spec's deliverables.
- Implementation status: `enqueue` moves from "no-op placeholder" to "CPU
  kernel-runner; integrator side accelerator-complete".

## Test plan

- Existing suite stays green (the migration regressions — camera far-target
  blow-up, `newton_free_body_conserves_*`, `kinematic_body_does_not_stall_*` —
  are all downstream of bug 2 and must pass).
- New: CPU-`Differential`-vs-compiled-kernel oracle test (value + 24 columns) in
  newton's runtime suite, driving the emitted Lua through viete's executor.
- New: an aliasing regression — assert that after a multi-iterate `Newton` step,
  a body's `accum_wrench` still equals the external field input (i.e. `enqueue`
  never wrote through to the external seed).
