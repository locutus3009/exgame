# Accelerator — joint-kernel compute & dispatch architecture

Living design doc for the compute pipeline that evaluates joint force laws and
their Jacobians on the GPU: build-time kernel codegen (`crates/newton/build.rs`),
runtime dispatch (`Accelerator` in `crates/newton/src/accelerator/mod.rs`), the
Vulkan objects under it (`crates/rembrandt/src/lib.rs`), and the world storage
they share (`crates/aristotle/src/world.rs`). Consolidated here
so the design survives out of source-comment form. The inline markers
(**settled** / **weakly worked out** / **not yet touched**) grade the DESIGN's
maturity; for what is actually CODE vs. still prose, see **Implementation
status** immediately below.

Cross-refs in the source point back here rather than duplicating prose.

---

## Implementation status (2026-07-22)

**Landed:**
- **The integrator is async (b40cdee, "Make async!").** `Mechanism::step`,
  `ImplicitIntegrator::step_all`, the `Newton` solve and the explicit `bridge`
  are all `async fn`; the sync entry is `pollster::block_on` (tests/examples).
  This is the Part II foundation — the suspension point batched dispatch needs —
  realized end-to-end.
- **`dyn ImplicitIntegrator` → closed `enum` (77fc3a4).** The async trait method
  had forced `Pin<Box<dyn Future>>` boxing at the dispatch boundary. The scheme
  set is fixed (four variants), so it is now an inherent `async fn step_all` on
  `enum ImplicitIntegrator<T>` the mechanism awaits directly — no boxing,
  integrator stored inline. Evolves Part II; the prose predates it.
- **Per-island concurrency inside a mechanism.** `Mechanism::step` drives its
  islands with `futures::future::join_all` — while one island is parked, the
  others progress. First slice of "step many at once"; see the gap below.
- **`enqueue` is the CPU kernel-runner; integrator side accelerator-complete
  (2026-07-19).** Every integrator's force-law evaluation now goes THROUGH
  `Accelerator::enqueue`: it runs `edge.eval` (+ `Differential::jacobian` only
  when a `Some(slot)` destination is passed) and scatters outputs into World-backed,
  domain-owned slots — per-body `total_wrench` (the gather, accumulated) and the
  per-connection `JointEdge.jacobian` block. `Newton` drives it via a `dispatch`
  helper (PRE + seed + enqueue-per-edge); `residual`/`assemble_matrix` became pure
  slot-readers; the explicit bridge calls `enqueue(None)`. `Mechanism` owns an
  `Arc<Accelerator>`. Swapping this CPU body for a batched GPU dispatch writing the
  SAME slots touches no integrator. Storage contract + rationale:
  [`docs/architecture/records/2026-07-19-accelerator-storage-contract-design.md`](docs/architecture/records/2026-07-19-accelerator-storage-contract-design.md).
- **Kernel codegen aligned to `Differential` (Part I §1, Part V).** build.rs traces
  `jac_kernel` → Lua for the four joint types; the retraction is now `Motor::exp`
  (unit, no `factor`, `NIN 31→30`), pinned to CPU `Differential` by the build
  cross-oracle. The `Accelerator` builder loads that shader set.
- **GPU joint kernels run via GLSL backend** (`2baf542` commit, 2026‑07‑20). The worker thread now dispatches compiled GLSL shaders for each joint type; this replaces the previous Lua‑based backend. Joint kernel execution is fully GPU‑accelerated, while the simple‑sum test showcases the transport layer.
- **Gather (§6) runs THROUGH the batch (`f329747`, 2026-07-20).** Phase 3 is a
  `MessageInput::Gather` job per body — `total_wrench = external + Σ` incident
  connection wrenches, summed in list order on the worker. The per-body incidence
  list (`RigidBody::incident_terms: Arc<[GatherTerm]>`) is pre-baked by
  `bake_incidence(&mut bodies, &joints)` on topology change (`Islands` calls it;
  a direct `dispatch` caller in unit tests must too), so the step-time path is a
  cheap `Arc` clone — no per-Newton-iteration joint scan. Replaces the earlier
  async main-thread reduction.

- **The implicit solve is Newton–Schulz over blocked storage (2026-07-21).** The
  dense Gaussian elimination is gone. The system matrix is `m²` blocks of
  `[[T; 6]; 6]` in World storage; the linear solve is `X ← X(2I − AX)` on a
  per-island approximate inverse that persists between steps, so every step of
  the inner loop is a GEMM over disjoint output blocks. Design and measurements:
  [`docs/architecture/records/2026-07-21-newton-schulz-block-solver-design.md`](docs/architecture/records/2026-07-21-newton-schulz-block-solver-design.md)
  and its `findings/` companion.
- **The step is almost entirely dispatch work now.** Three per-body/per-block
  stages joined `Pre`/`Eval`/`Gather`: `BodyPost` (world mass block, residual and
  its self-scale), `AssembleBlock` (the matrix as a gather over connection
  Jacobians, with the midpoint factors applied as columns are read — the same
  shape as the wrench gather, one level up), and `Gemm`/`BlockMatVec`. Measured,
  the CPU-resident share of a step fell from 7.2% to 0.73%, which moves the
  Amdahl ceiling for a GPU move from 13.9× to ~137×.
- **Messages carry groups, not single items.** `GEMM_CHUNK = 32` output blocks
  (or bodies) per message. One item per message is the cleaner contract but a
  cell is ~2 µs of arithmetic against ~5 µs of channel + oneshot + rayon slot;
  measured 7.07 → 5.85 ms per step on the cloth demo. Cells stay disjoint across
  messages, so this changes only how work is carried.

- **World storage is GPU-resident (2026-07-22).** `RawMap`'s per-type payload is
  no longer a `Vec<T>`: it is a vulkano `Subbuffer<[T]>` allocated through
  `rembrandt::GpuAccelerator::allocate_buffer` with
  `MemoryTypeFilter::HOST_RANDOM_ACCESS`, one buffer per stored type, sized once
  at `World::DEFAULT_CAPACITY` (256 Ki elements). `aristotle` now depends on
  `rembrandt`. `ReadView::get_map()` hands out that buffer, and — the point of
  the change — `WorldKey::raw_index()` is directly the element index a shader
  addresses, so no separate host→device layout step exists. The type erasure that
  makes this work is `AnyVec`, now implemented for `Subbuffer<[T]>` as well as
  `Vec<T>`.
- **First real GPU dispatch runs end to end (2026-07-22): `simple_sum`.** The
  worker's batch path submits actual Vulkan compute work. Concretely:
  `AcceleratorBuilder::build` loads a GLSL 460 kernel (`mod shader_simple_sum`,
  via `vulkano_shaders::shader!`), builds its pipeline, allocates an index-table
  `Subbuffer` of `gpu_in_flight()` rows and builds the descriptor set ONCE —
  binding 0 is the world's flat scalar map, binding 1 the index table.
  `MessageInput::SimpleSum` jobs accumulate in `gpu_batch` and flush on full or
  idle, exactly like the Lua batch. `Shaders::dispatch_simple_sum` then fills the
  table with one `Triple { a, b, out_idx }` of `raw_index()`es per job, pushes the
  row count as a push constant, dispatches `ceil(count / LOCAL_SIZE_X)`
  workgroups and blocks on the fence; the oneshot notify stays sequential on the
  worker, off the compute path, as on the Lua path. Vulkan failures come back as
  `EvalError::Backend`, one submission's outcome for every job it carried.

  The push-constant row count is load-bearing, not a formality: the index table
  is reused across flushes, so rows past `count` still hold the previous batch's
  triples, whose slots may since have been freed and recycled. The kernel's
  `idx >= count` guard is what stops the tail workgroup writing through them.

  How many flushes a batch becomes is TIMING, not a property of the work: the
  flush fires on a full batch or on an idle `recv_timeout` tick, and with
  `gpu_in_flight()` far above the test's job count it is only ever the idle tick
  — i.e. a race between the producer and the debounce. The same 30 000 jobs split
  into two flushes on some runs and one on others. Consequence to be aware of
  when reading the test as coverage: `gpu_simple_sum` exercises the stale-row
  case only on the runs that happen to split; on a single-flush run the table
  holds nothing stale and the guard proves nothing.

  Scope, plainly: `simple_sum` adds two scalars. It is NOT a joint kernel and
  carries no PGA — it exercises the TRANSPORT (batching, index table, descriptor
  set, submit, fence) against real hardware. The joint kernels are still Lua.
  Covered by `accelerator::tests::gpu_simple_sum`.
- **The GPU path is gated to `T = f32` (2026-07-22).** The kernel declares
  `float[]`, so binding the world's scalar map is only sound for f32, and
  `AcceleratorBuilder::build` now asserts `TypeId::of::<T>() == TypeId::of::<f32>()`
  when a `World` is supplied — the same erased-scalar guard shape the Lua backend
  uses in `lua::t_to_f64`. Before the guard an `f64` world built the pipeline
  silently. Covered by `gpu_path_rejects_non_f32_world`.
- **`rembrandt` owns the Vulkan objects.** Instance selection, device, compute
  queue, memory allocator, descriptor-set allocator and command-buffer allocator
  live in `GpuAccelerator`, exposed as accessors (`device`, `queue`,
  `ds_allocator`, `cmd_allocator`, `allocate_buffer`, `build_pipeline`). It
  supplies the bricks; assembling and submitting a command buffer is the caller's
  job. `gpu_in_flight()` derives an in-flight-invocation estimate from AMD/NVIDIA
  device properties (fallback 64 Ki, hard cap 1 Mi) and is what sizes the batch.

**Where the time actually goes (perf, cloth demo, 6.0 ms/step, Lua backend):**

| | share |
|---|---|
| Lua kernels — table marshalling in/out | 24.8% |
| Lua kernels — interpretation | 22.6% |
| rayon / crossbeam scheduling | 16.5% |
| allocator | 7.5% |
| PGA on the CPU side | 5.0% |
| **mpsc send/recv** | **0.18%** |
| **oneshot / futures / join_all** | **0.13%** |

The channel is not the cost — this was measured because it was assumed to be.
Roughly half the Lua share is not computing anything: it is building an input
table per kernel call and taking an output table apart. That half disappears
from a change of representation alone, before any gain in execution speed.

**Now that the stages are on the GPU, the step is SUBMISSION-latency bound
(2026-07-27, RX 9060 XT / RADV).** A flush costs ~52 µs whether it carries one job
or thirty-six, so what matters is how many submissions a step makes, not how much
work rides them. Measured on `cloth_grid_stability` (8 bodies, 10 connections, 6
unknowns): **90 flushes and 876 jobs per step**, against ~51 logical barriers —
the rest is tearing, 31% of flushes carrying exactly ONE job. For contrast
`gpu_simple_sum`, which submits 6000 jobs with no await in between, packs them
into 3 flushes: the transport batches fine, the await structure is what splits it.

Four changes landed against that (same commit):
- **PRE was never dispatched for a body no connection names** — the `.shared()`
  futures are lazy and only phase 2 polled them, so a free body kept the
  `Motor::identity()` / zero-velocity its slots were allocated with and `BodyPost`
  formed the residual at that phantom state forever. `dispatch` now awaits the
  whole PRE map after phase 2. This is what `#[ignore]`d
  `newton_free_body_conserves_world_momentum_and_energy` ("does not converge");
  it now runs in 0.13 s.
- **The Newton iteration after an accepted line-search probe re-runs only the
  Jacobian** (`Accelerator::eval_jacobians`). The probe already ran the full
  dispatch at the very iterate the next iteration starts from, so PRE's inputs,
  the gather's output and `BodyPost`'s residual/scale/mass block are all still
  valid and its `tnorm` IS the next `rnorm`; only the derivative is missing. Four
  dispatch waves become one, 14 barriers per iteration become 11. Relies on the
  two kernels of a family agreeing on the connection value — measured bit-exact,
  pinned by `plain_and_jacobian_kernels_agree_on_the_connection_value`.
- **Rounds are baked round-major**, so `submit_rounds` stops regrouping and
  cloning `m²` `WorldKey`s on each of the five products an iteration runs.
- **The worker no longer spins.** `recv_timeout(1 ns)` with three folds over the
  kind map on every turn was ~25% of the process (the deadline's `clock_gettime`
  alone 5%); it now blocks when nothing is queued and carries the occupancy
  counters. The flush decision is unchanged.

Result on the cloth suite: 14.2 s → 4.9 s wall; `longer_steps…` 6.95 → 4.91 s
with CPU time 2.72 → 1.79 s, flushes 134 775 → 83 575, single-job flushes 31.3% →
9.3%. Trajectories unchanged.

**Then the benchmark changed and so did the answer (2026-07-27).** All of the
above was measured on a 2×4 curtain — 6 unknowns — where the step is bound by
submission latency. The characteristic that actually matters is how far the cloth
can be scaled UP, and there the picture is different: swept over 6/15/28/45
unknowns the median step grew as **m^1.9**, quadratically, while the arithmetic
of the block products is cubic. So the cost was not the work; it was the MESSAGE
COUNT. Per Newton iteration `assemble` submits `m²` cells and `gemm` `4m²` at the
warm budget, against roughly `36m` per-body messages — 86% of all traffic at 45
unknowns.

Three changes followed, in that order:

- **The round is the unit of the message, not the row.** A row-driven message
  carries a whole round and occupies a contiguous SPAN of the batch table;
  `ShaderSetup::setup` reports the invocations it filled and `check` walks the
  batch with a running base. Concurrent mechanisms stay transparent by
  construction — their rounds are separate messages that land next to each other
  and go out in the same dispatch — which `concurrent_mechanisms` pins by
  requiring three chains of DIFFERENT lengths sharing one accelerator to land
  bit-identically where the same three land alone.
- **The contraction measure is a kernel.** `‖I − A·X‖²` was `m²` `WorldKey::read`s
  on every Newton–Schulz iteration AND it sat between the two block products, so
  it stalled the pipeline as well as costing the reads. `functions::reduce_sq_term`
  folds it, one row per 126 blocks, and it now rides the same wave as the second
  product. Still on the host and still `m²`, for a later pass: `x_is_finite` and
  the block copies of `checkpoint`/`rollback`, which run once per span rather
  than twice per iteration.
- **§11: PRE and the connection kernels onto rows.** With the block stages fixed,
  these were the whole of the traffic (~1129 of 1137 messages per iteration,
  eight line-search lanes × one message per body and per connection). §10 came
  with it — `dt`, `warp` and the retraction are slot-backed now, which is what
  made the rows bakeable. Every kernel is row-driven, so `plain_shader!` has one
  arm, `MessagePayload` is `Rows` and nothing else, and the message path carries
  no scalar type at all.

| unknowns | before | round=message | +GPU reduce | +§11 |
|---|---|---|---|---|
| 6 | 3.70 | 3.55 | 3.71 | **2.97** |
| 15 | 8.42 | 6.06 | 7.10 | **5.76** |
| 28 | 15.48 | 9.26 | 9.29 | **7.49** |
| 45 | 41.91 | 20.78 | 17.73 | **13.76** |

Median ms/step. Growth exponent **1.9 → 1.24**, 3.0× at the largest grid that runs
today. `cloth_scaling` is the sweep, `#[ignore]`d because it is a measurement.

Two things the sweep says that are NOT about the transport. Convergence degrades
with size: at 6 unknowns no frame subdivides, at 45 unknowns 18% do, and the tail
(p95 41 ms against a 14 ms median) is entirely those. And there is a hard ceiling
from World's fixed 256 Ki slots per type — the `Row` storage goes as
`m²·(2⌈m/62⌉ + 3)`, which runs out somewhere around 160 unknowns.

**Placeholder / not wired:**
- **GPU joint kernels.** ✔ Implemented – GPU dispatch now uses WGSL shaders for each joint type, and the engine calls `shaders.rs` to execute them.
- **Cross‑mechanism epoch driver** ❌ *Not yet implemented* – the current design uses only a timer (`recv_timeout`) and lacks a high‑level executor.


- **Part III's lock split is LANDED (2026-07-21), second door renamed by the
  storage move (2026-07-22).** `RawMap` still carries a hand-written `unsafe impl
  Sync` with a recorded safety argument (this closes the Part V open item), and
  the outer `RwLock` still splits structural mutation (write guard) from
  "structure is stable" (read guard). The external API is untouched:
  `WorldKey::read` and `write` still take exclusive guards and remain safe under
  arbitrary concurrency.

  What changed with the GPU-resident payload: the `UnsafeCell` is gone — the
  interior mutability now belongs to the `Subbuffer` itself — and the second door
  is `ReadView::get_map() -> &Subbuffer<[T]>`, still addressed by
  `WorldKey::raw_index()`. Element access goes through vulkano's own host guards
  (`Subbuffer::read`/`write`, which report a conflict as `HostAccessError`)
  rather than a raw pointer. Source comments in `world.rs` still name the older
  `parked_slice_mut`/`UnsafeCell` door.

  It exists for a STRUCTURAL reason, not a performance one. A batch task reads
  sibling slots of the same type as the one it writes — `Gemm` reads `A`/`X` and
  writes `R`, `AssembleBlock` reads the mass block and writes the matrix,
  `BodyPost` reads momentum and the gathered wrench and writes residual and scale.
  Holding an exclusive guard across such a body is impossible: `RwLock` is not
  reentrant, so `write()` under a held read guard deadlocks. The lock is not
  expensive (measured 0.76%), it is the wrong SHAPE.

  Measured 6.03 → 5.88 ms per step, about 2.5%.

**Part IV — partly built.** Its premise landed: the per-type array IS in
GPU-host-shared memory and the accelerator addresses it in place (see "World
storage is GPU-resident"). What the prose describes and the code does NOT have:
the bespoke `GpuVec<T>` over raw `ash` (it is a vulkano `Subbuffer`), the
sub-allocated pool, growth/re-pointing (capacity is fixed at construction), and
the non-coherent + explicit flush/invalidate discipline (the buffer is
host-mapped, no barriers are issued).
(The retraction convention, Part V, is now RESOLVED — `Motor::exp`, pinned by the
cross-oracle.)

---

## Part I — Kernel codegen (`newton/build.rs`)

### 1. What is produced (today)
For each registered joint type (currently `CriticallyDampedWarped`,
`SimpleSpringDamper`, `PerpendicularDamperWarped`, `TorsionalDamperWarped`) the
build script traces `jac_kernel` — the two-body wrench law under 24-wide
first-order forward-AD (`Tangent<N24>`) — into a branch-enumerated IR (`viete`)
and emits one kernel. Today the target is **Lua** (an oracle backend, verified
against the f64 carrier at trace time); the real target is a GPU shader
(**wgsl → SPIR-V**). A `shaders_map.rs` table is generated for the crate to
`include!`. **One kernel = one joint TYPE**, reused for every INSTANCE via the
indirection of Part I §4–6.

### 2. Evaluation point — the math contract (**settled**)
The pose perturbation is a **pure differential (value 0)**. The finite pose lives
entirely in a runtime `base` Motor (8 even-grade components per body); the kernel
forms `pose = base ∘ Motor::exp(δ)` with δ-value = 0 (unit retraction, matching
`Differential`; NOT `Twist::exp`, whose half-angle ½ would halve the pose columns
— see Part V). Because the exp bivector's
standard part is then a hard zero, viete resolves exp's small-angle
`is_effective_zero` branch to its Taylor limit **at trace time** and prunes the
trig branch — shrinking every kernel **~71–76%** with no loss of generality (the
pruned branch is unreachable at δ=0). Mirrors `clifford::pga3::ad::Differential::seed`.

I/O layouts: see the **Kernel I/O layout** appendix.

### 3. One kernel, both integrator families (**settled**)
The connection kernel is integrator-**agnostic** (unit retraction). Families
differ in what they feed as `base`/`vel`, how much output they read, and — for
implicit — in the PRE/POST stages (§7) that bracket the shared core:
- **implicit** (Newton midpoint): `base` = midpoint pose (formed in PRE from
  vmid), `vel` = vmid; reads value + all 24 Jacobian columns; column dt-scaling
  is done in POST, not the kernel.
- **explicit** (Symplectic/Explicit/Lie Euler): `base` = current pose,
  `vel` = world velocity; reads value only; no PRE/POST.

### 4. Planned GPU codegen shape (**settled in outline**)
The compute graph does NOT change when we leave Lua. In the IR the leaves already
reference inputs/params by **index** (`Input(k)` / `Param(j)`) and roots are
positional outputs — so lowering to GPU is a **leaf/root-emission change only**:
- a leaf becomes an indexed load from a flat typed array,
  `in_array[base(workgroup) + k]`;
- outputs (wrench value, Jacobian, fatal) become indexed stores.

An entry shader keyed on `workgroup.x` picks the block (= one joint instance),
reads the indirection table(s) for which rows its poses/vels/params come from,
and scatters its outputs. The inner kernel then operates as e.g.
`wrench_out[wrench_id][slot] = …`. The flat scalar in/out layout is
**shader-internal and codegen-tunable** — authoritative in build.rs (appendix
here), and must NOT leak into the typed `enqueue` contract.

### 5. Typed batching & the three output streams (**settled in intent**)
At the `enqueue` boundary data is **typed**: poses are `Motor` rows, vels
`Twist`, wrenches `Wrench` — `WorldKey`s into the world's per-type dense storage
(Part IV). Whole rows are gathered from storage with a parallel indirection table
of which rows each block uses; work accumulates across joints until a dispatch is
worthwhile. Three output streams, each with its own addressing rule:
- **wrench VALUE** — per-body target, ACCUMULATED across joints (§6);
- **Jacobian** — per-connection block, plain scatter, no contention (consumed
  per-connection by the matrix assembler);
- **fatal** — per-block flag, host scans after the batch.

### 6. Wrench accumulation → segmented gather stage (**settled, pad-to-max**)
A body's wrench sums contributions from every joint touching it. We do NOT use
f32 atomics (not core in wgsl; non-associative ⇒ non-deterministic) and NOT a
sparse per-body list. Instead:
- each joint-block writes its two wrenches to a **dense per-connection** output
  array (connection i → slots 2i, 2i+1); deterministic, contention-free;
- a **separate compute stage** sums per body via a **segmented reduction** — a
  flat list of (body, wrench) pairs grouped by body (grouping from the
  indirection table), reduce-by-key. Variable degree = SEGMENT LENGTH, not a
  divergent branch. Fixed order ⇒ deterministic (the point of avoiding atomics);
- **divergence removal**: PAD every body's segment to a fixed length (max
  degree). Padding entries point at a reserved **zero record** (output slot 0
  kept permanently zero; real connection-ends numbered from 1). Zero is the
  additive identity ⇒ branchless, harmless.

The gather is O(connection-ends), tiny next to the ~45k-instr Jacobian compute,
so pad-to-max is fine for now (sort-by-degree to shrink padding is a known lever,
deferred until a profile shows a hub — measure first).

Pipeline:
```
IMPLICIT: [PRE per-body] → conn kernels → gather → [POST per-body]  ↺ (Newton iterate)
EXPLICIT:                  conn kernels → gather → apply value
```

### 7. Implicit-only PRE/POST stages (**settled in shape**)
Everything dt-dependent and Newton-specific lives in two per-body stages that
bracket the shared conn+gather core, so the kernel stays agnostic:
- **PRE** (before conn kernels): form each body's midpoint pose
  `base = snap_pose ∘ exp(½dt · vmid)` from the current vmid iterate — one Motor
  exp per body. These `base` Motors are the connection kernels' input.
- **POST** (after gather): assemble the residual `P_s^mid − P_s^n − ½dt·W`, the
  inertia mass-block, and **scale the Jacobian columns** — pose columns by
  −½dt·½dt, velocity columns by −½dt (mirrors `newton.rs::assemble_matrix`).

For implicit these two, around conn+gather, ARE the Newton-iteration loop body
(re-run per iterate with updated vmid). Explicit runs neither.

**Consequence — the kernel must use UNIT retraction.** Because POST applies the
full −½dt² to pose columns, the kernel must not bake any scaling into the
retraction: it uses `base ∘ Motor::exp(δ)`, matching `Differential::jacobian`
exactly (no exp-time, no `Twist::exp` half-angle). Resolved and pinned by the
cross-oracle (Part V, resolved).

---

## Part II — Async dispatch / executor (`integrator/mod.rs`)

### Why the whole integrator goes async  (**landed** — b40cdee)
The batched dispatch (§5) only pays off if work from many integrators
co-accumulates into one dispatch. Done synchronously, that forces splitting every
integrator by hand into "produce work → stop → resume" stages — a hand-rolled
state machine, and the architecture would be awful. Async gives that suspension
point for free: an integrator reads as straight-line code but yields at each
dispatch. So `Mechanism::step` and the integrators became **async** — done; the
closed-enum `ImplicitIntegrator` (Implementation status) awaits `step_all`
without boxing. The suspension point exists; what still awaits nothing real is
the accelerator itself (`enqueue` is a no-op — Implementation status).

### Model
- Every joint-block dispatch is `enqueue(...).await`. `.await` **suspends** this
  integrator; the cooperative scheduler resumes another — including on the same
  thread.
- The driver steps **all mechanisms of an epoch concurrently**, not one at a
  time. That is where batch breadth comes from: while mechanism A is parked on
  its dispatch, B..Z submit theirs. (**Partly built:** islands WITHIN one
  mechanism already interleave via `join_all`; the cross-mechanism epoch driver
  above `Mechanism` is not built — Implementation status.) Safe to interleave in
  any order because
  external fields are synchronous but **frozen to the epoch** — a step never
  depends on another mechanism's mid-step state, so traversal order is
  irrelevant.
- Work may be submitted from other threads AND the current one; **awaiting (not
  blocking)** is what lets the current thread keep contributing.

### Decisions
- **One accelerator (B).** A single `Accelerator` receives lightweight **index
  records** — where to read inputs, where to write outputs, which pipeline(s) the
  block participates in — not the payload. It owns batching and dispatch order.
- **Flush = heuristic only (A).** Total kernel count is unknowable ahead of time
  (implicit's Newton-iteration count is convergence-dependent), so no exact
  threshold. Natural backstop: **quiescence** — when no future is runnable
  (everyone parked on the accelerator), force-flush; a timer is a latency bound
  so a half-full batch still progresses.
- **Await per block (D).** Each block is awaited individually; the accelerator
  decides dispatch ORDER; integrators do NOT join their own joints. Batch fill
  comes from concurrent mechanisms, not from one integrator's joints. Tradeoff
  accepted: a single mechanism's mutually-independent joints (one Newton iterate,
  all at the same vmid) serialize across dispatch waves — per-mechanism latency —
  but throughput stays high because other mechanisms fill each wave.
- **Custom executor, not tokio/async-std.** Async here is core and load-bearing,
  so hand-roll it (small executor + waker wiring; a pollster-style `block_on` at
  the sync entry). Deliberate exception to "synchronous by default", not a
  reversal. **Interim state:** the sync entry is `pollster::block_on` (as
  intended) but concurrency currently rides `futures::join_all`, not a hand-rolled
  executor — fine while `enqueue` is a no-op, revisited when the accelerator owns
  batching/quiescence.
- **Lua first.** Lands BEFORE the shader migration; the Lua backend is driven
  through this same async dispatch (a "dispatch" just runs the Lua kernel per
  block), validating scheduler + batching without a GPU.
- **Implicit self-dependency.** A Newton iterate needs its blocks' results before
  assembling the matrix / next iterate, and PRE/POST are await points too — so an
  implicit integrator serialises on its own dispatch chain; the executor fills
  the gaps with other mechanisms.

---

## Part III — Result write-back & world storage locking

**Supersedes the earlier "copy inputs into the batch" sketch.** No copy, no
arena.

### Partitioned ownership ⇒ no copy
While an integrator instance is parked on `await`, it is waiting for its pair of
wrenches, and **nothing but the accelerator writes them**; the parked instance
does not read them either. So there is no data race on those slots — the
accelerator writes results **in place** into world storage, no copy, no arena.

### The lock split
The only real hazard is **structural** (a `Vec` realloc / slot reuse invalidating
the indices the accelerator holds), NOT data. So World's per-type `RwLock` is
repurposed (or split):
- **read guard** = "structure is stable (no realloc / reindex) — read AND write
  your own elements". Many dispatches and commits proceed in parallel.
- **write guard** = structural mutation only (insert → push/realloc, remove →
  free-list slot reuse). Rare; a **barrier** that drains in-flight compute.

Element data lives behind interior mutability (`UnsafeCell<T>`; `T: Pod/Copy`),
written under a structure-read guard. Indices survive `push` (append doesn't
shift), so the structural lock guards against Vec-header races and slot reuse, not
index arithmetic.

### Two doors for data writes — do NOT merge them
- **Ordinary host access** — `WorldKey::write`/`read`, commits: stays
  **explicitly serialized** (exclusive on data). This is the default and safe
  under arbitrary concurrency; its contract is unchanged.
- **Accelerator write-back** — a **separate `unsafe` path** (conceptually
  `write_parked(slot)`), taken only under the structural read guard, justified
  **only** by the hard invariant below. It is NOT a relaxation of `write()`.

The lock-free write-back is licensed by the GPU-await-parking regime, not by
anything true of general host code — so general host concurrency must remain
explicitly serialized. Merging the two (making `write()` itself a read-lock) is
the mistake to avoid.

### The soundness lynchpin: ≤1 writer per slot
Guaranteed by construction:
- dispatch writes **per-connection** outputs (§6) — disjoint slots;
- gather writes **per-body**, one thread per body — disjoint;
- commit writes a mechanism's own bodies' pose/momentum — disjoint;
- **epoch invariant**: a mechanism's structure is **frozen within an epoch**,
  changes only between epochs, so no body is shared across mechanisms mid-epoch.
  (This is what also makes the async traversal order-independent.) Structural,
  currently documented but **not enforced** — see open items.

---

## Part IV — Shared-memory storage: `GpuVec<T>`

> **As built (2026-07-22):** the premise below is real — the per-type array lives
> in GPU-host-shared memory and is written in place — but it is a vulkano
> `Subbuffer<[T]>` of fixed capacity, not the bespoke `GpuVec` over raw `ash`
> described here, and none of the pool / growth / non-coherent-flush machinery
> exists. See "Part IV — partly built" in Implementation status. The rest of this
> Part is design prose.

World's per-type dense array is allocated in **GPU-host-shared memory** so the
accelerator reads/writes in place with no host↔device shuttle. Implemented as a
**bespoke `GpuVec<T>`** — NOT `Vec<T, A>` with the `allocator_api` (that is
nightly-only; a bespoke type keeps us on stable and owns the mapping directly).

### Allocator / pool
One persistently-mapped `HOST_VISIBLE` `VkDeviceMemory` **pool**, sub-allocated
(Vulkan caps total allocations, so not one per Vec), alignment
`max(Rust Layout align, GPU min-align)`. The pool is `Arc`-shared; GpuVecs
sub-borrow ranges; drop order is GpuVecs before unmap. Raw `ash` (engine renderer
is raw Vulkan, not wgpu).

### Growth — the real hazard, closed by the epoch invariant
When a GpuVec grows, its backing moves → the **device address / handle changes** →
shaders that bound the old buffer are stale. But growth is a **structural** op, so
by the epoch invariant it happens **only between epochs** — precisely when
re-pointing GPU bindings is safe. With **Buffer Device Address** (fits raw ash),
"re-point" is just updating a pointer in the **indirection tables** we already
maintain, not descriptor-set churn:
```
grow (between epochs, under structural write lock)
  → new sub-range from pool → copy → release old → bump device address/generation
  → refresh device addresses in the indirection / BDA table
within an epoch: addresses stable, in-place writes, shaders access by BDA
```
`first_free` amortizes (realloc only when the free-list is empty); reserve
generously so growth is rare.

### Coherency — non-coherent + flush at dispatch boundary
`HOST_VISIBLE | (DEVICE_LOCAL if ReBAR / unified for true zero-copy)`,
**non-coherent** (coherent has too much overhead). Non-coherent needs barriers in
**both** directions, per dirty sub-range, at the dispatch boundary:
- host wrote an input → `vkFlushMappedMemoryRanges` **before** the GPU reads;
- GPU wrote an output → `vkInvalidateMappedMemoryRanges` **before** the host reads.

`GpuVec` (or the pool) owns flush/invalidate of its dirty ranges. In the
north-star (all local physics on GPU) these barriers collapse to the `fix <-> f32`
world boundary and commit/readback; on the transitional path (Lua backend,
hybrid) the host touches the data, so the discipline is needed from the start.

### North star
Move all local physics into the shaders, so nothing round-trips through host
storage mid-compute and only **`fix <-> f32`** crosses the world boundary (ties
to the i128-position / f32-dynamics split). Then the lock and flush questions
largely dissolve — the host only sees committed state.

---

## Part V — Open questions

**Kernel (Part I):**
- **Retraction convention — RESOLVED (2026-07-19).** The kernel retracts with
  `base ∘ Motor::exp(δ)` — the exact map `Differential::jacobian` uses — so it
  drops the `factor`/`half` input entirely (`NIN 31→30`) and there is no exp-time.
  The `−½dt²/−½dt` column scaling stays in newton's matrix assembler. The build
  cross-oracle (`crates/viete/tests/kernel_differential_oracle.rs`) traces the
  base+differential kernel and asserts value + all 24 columns equal a direct CPU
  `Differential::jacobian` at the same point. It immediately caught a **factor-of-2
  bug**: the kernel had used `Twist::exp` (which bakes the geometric half-angle ½)
  instead of `Motor::exp`, halving every pose column — precisely the CONVENTION
  divergence viete's own f64-carrier check cannot see. Same harness is the seed
  for the eventual GPU-vs-CPU conformance check.
- **Workgroup/thread mapping.** One workgroup per joint instance, threads over
  the 24-wide AD-gradient axis (aligns with the `vectorize_lanes` vec4 hint) —
  exact thread↔lane assignment and tiling is a sketch.
- **Indirection-table shape.** Several tables (pose-idx, vel-idx, param-idx,
  wrench-out-idx, jac-out-idx) vs one struct-of-indices per block: undecided;
  on-GPU encoding open.
- **Fatal semantics in wgsl/SPIR-V.** Indexed fatal array + host post-scan is the
  shape, but abort-batch vs mark-and-continue, per-block slot layout, and
  interaction with that block's wrench/Jacobian outputs — undecided. (Lua handles
  fatal implicitly; none today.)
- **Where POST scaling physically runs** — gather stage, a separate stage, or CPU
  during residual assembly — undecided.
- **Precision seam.** Kernel is f32 throughout, incl. base-pose Motor components.
  Given the i128-position / f32-dynamics split, whether the base pose's position
  part survives narrowing to an f32 shader input is unexamined (joint anchors are
  small local offsets, so probably fine).
- **Params marshalling.** Per-type, per-instance arrays; `n_params` varies per
  shader. Encoding/precision not worked out.
- **viete wgsl/SPIR-V backend.** Does not exist. Needs parameterized indexed
  input/output arrays driven by the indirection tables, and explicit fatal
  sections.

**Kernel marshalling — NOT a bridge to SPIR-V (measured and reasoned, 2026-07-21).**
The tempting move is to hand the Lua kernels an index table once and pass indices
per call, so they address a flat data array the way Part I §4's GPU leaves will.
It does not work, for two independent reasons.

*It buys nothing.* The boundary is already cheap: pre-sizing the argument tables
and switching to raw get/set measured neutral (6.02 → 6.07-6.11 ms, noise), and
mpsc/oneshot are 0.3% between them. The Lua cost is not at the boundary but
INSIDE the kernel — an 8k-entry SSA table plus thousands of vec4 helper calls,
each allocating a table, because PUC Lua has no SIMD and the SLP pass (correct
for SPIR-V) lowers to allocating helpers there. That is a property of the
stand-in, not a defect, and the pass stays as it is.

*It cannot be faithful.* PUC Lua's only cheaply indexable aggregate is a table.
Rust memory reaches Lua only as userdata behind `__index` (a call per element), a
string copy, or a table copy — there is no zero-copy view. So the payload is
always materialized on the Lua side, and the one property the scheme exists for —
addressing without copying — is exactly the one Lua cannot reproduce. A kernel
written as `data[base + k]` over userdata would have the right SHAPE and
per-access economics off by orders of magnitude, making any measurement there
worthless as a GPU predictor.

Nor can the dense storage be memcpy'd into a Lua array, which is the obvious next
thought once the payload is known to be a flat `Vec<T>`. A Lua table's array part
is `TValue[]` — a value plus a type tag, 16 bytes on 64-bit — not `double[]`.
Every element has to go through the API that writes the tag, which is precisely
what `luaH_*` costs. The only memcpy-shaped route is handing the payload over as
a Lua *string* and unpacking it in-language, which trades the insert for a
`string.unpack` per element and is unlikely to be better.

Consequence for the estimate: the ~25% of a step spent in Lua table work is not
work to be MOVED to the GPU, it is work that ceases to exist — `v[]` becomes
registers and vec4 becomes real vector ops. The addressing scheme gets designed
against SPIR-V directly; there is no intermediate rehearsal.

**Dispatch / executor (Part II):**
- Cross-thread submission path (queue + waker into the executor) and whether
  foreign threads run coroutines or only submit index records.
- **Determinism invariant to guard:** batch COMPOSITION must change only timing,
  never physics. The gather is order-independent (fixed-order segmented sum), so
  this should hold — a property to protect, not assume.
- Back-pressure when integrators outrun the dispatch drain; flush-policy
  parameters (timer period, size hint).
- Stage scheduling/synchronization: the N sequential per-type stages plus the
  gather stage — ordering, barriers, how the pipeline is driven.

**Storage / locking (Part III–IV):**
- **Enforcing the epoch invariant** (mechanism structure frozen within an epoch;
  no body shared across mechanisms). Currently documented only — the lock-free
  write-back is UB if it is violated. Enforce structurally or guard it.
- `Sync` for the `UnsafeCell<T>` storage — hand-written, with a recorded safety
  argument.
- `remove` under the structural write guard must be a barrier that waits for all
  held read guards (in-flight dispatches) to drain before reusing a slot.
- Exact `GpuVec<T>` / accelerator write-back API surface — TBD.

---

## Appendix — Kernel I/O layout (authoritative)

**Input** (`NIN = 30`). The pose perturbation is a pure differential (value 0 —
the finite pose lives in `base`), so pose-twist DOF carry NO input values; only
their 12 AD gradient axes exist. The AD gradient-axis layout is the FIXED newton
24-order (pose A 0..6, pose B 6..12, vel A 12..18, vel B 18..24) and is
independent of these value-input indices; velocity value j feeds gradient axis
12+j.

```
0..12   velocity values (vel A 0..6, vel B 6..12)  → gradient axes 12..24
12..20  base pose of body A: 8 even-grade Motor components
20..28  base pose of body B: 8 even-grade Motor components
28      dt
29      warp
```

The pose retraction is `base ∘ Motor::exp(δ)` — the **same** map
`Differential::jacobian` uses, so there is NO exp-"time"/`half` input (a `Twist::exp`
would bake the geometric half-angle ½ and halve every pose column — the
kernel-vs-`Differential` oracle pins this). The `−½dt²/−½dt` column scaling lives
in newton's matrix assembler, not the kernel (Part V, resolved).

**Output** (`NOUT_FULL = 300`) — 12 wrench components × 25 slots. Components in
order (body A's wrench, then body B's — force x,y,z then torque x,y,z each):

```
[0]=A.fx [1]=A.fy [2]=A.fz [3]=A.tx [4]=A.ty [5]=A.tz
[6]=B.fx [7]=B.fy [8]=B.fz [9]=B.tx [10]=B.ty [11]=B.tz
```

Component c occupies `out[c*25 .. c*25+25] = [value, ∂/∂axis0 … ∂/∂axis23]`, the
24 partials along the fixed gradient-axis order above. Explicit reads only the
value slot of each component (`out[c*25]`); implicit reads the whole 25-wide
group.
