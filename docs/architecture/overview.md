# Architecture overview

Start here for design. `CLAUDE.md` is the operational entry point
(intent, build, run); this file is the design index.

## North star

This is an experimental N-body gravity simulation with a live
visualization, evolving into a space game built on a multibody
dynamics substrate (clifford + newton) with a deterministic
authoritative integrator and secondary (non-authoritative)
prediction. The determinism contract scopes to integrated state
only ([open/determinism-boundary](./open/determinism-boundary.md));
downstream — rendering, analysis, UI — is free of fixed-point /
determinism constraints.

Active vessel modelling follows the Rigid Finite Element model
([decided/vessel-model-rfe](./decided/vessel-model-rfe.md)):
elastic-node joints with closed-form damped-oscillator integration;
kinematic articulation joints separately; no general constraint-solver
multibody, no LOD / on-rails, no controlled-vs-remote split.
Integration is per-subsystem with operator splitting; symplectic
ordering applies to gravitational COM only.

Momentum integrates in body-frame
([decided/frame-convention](./decided/frame-convention.md)); LieEuler
is the canonical integrator.

## Crates

Every workspace member is listed and nothing is listed that is not a workspace
member. Checked against `cargo metadata --no-deps --format-version 1` by
`tools/quality/check_docs.py`'s companion gate 14
([QUALITY_GATES.md](../QUALITY_GATES.md#14-entry-point-crate-table-consistency)),
which AR-0006 implements against `CLAUDE.md`; `CLAUDE.md` is the authoritative
table and this one must agree with it. Crates are named after people; the name
carries no information about the contents.

| Crate | Doc | Role |
| --- | --- | --- |
| `peano` | (no doc) | The type-level and algebraic foundation: Peano naturals (`Z`, `Succ<N>`, `Nat`), a cons-list `Vector<N, T>` whose length is a type, and the trait tower `AbelianGroup → Ring → Commutative → Scalar`. `bytemuck::Pod` on the storage so vectors upload to the GPU. |
| `clifford` | [crates/clifford](./crates/clifford.md) | `no_std` geometric algebra. `Mv<A, S>` over an algebra descriptor `A` and a `peano::Ring` `S`; the `pga3` facade adds `Motor`, `Screw<V, S>` with `Twist`/`Wrench` variance tags, `Point`, `Plane`, `Line`, `Direction`, and the `Dynamics`/`Differential` AD entry points. Forward-mode AD is `Tangent<GradN, R>`. Cut over from the legacy Multivector/jet stack 2026-06-25. |
| `functions` | (no doc) | The scalar-generic *bodies* of `newton`'s implicit-solver kernels — `pre`, `gather_term`, `matvec_term`, `gemm_term`, `reduce_sq_term`, `assemble_term`, `body_post` — written once over `peano::Scalar` so `newton/build.rs` can trace them into GLSL. |
| `joints` | (no doc) | The two-body force elements: `JointEdge<T>` binding two `WorldId`s to a `Joint<T>`, and the laws `AxialSpringDamper`, `PerpendicularDamperWarped`, `TorsionalDamperWarped`, each a pure world-frame function polymorphic in the scalar. |
| `rembrandt` | (no doc) | **The vulkano device layer, and live.** `GpuAccelerator` owns the `Device`, compute `Queue` and the allocators, allocates storage buffers, builds compute pipelines and estimates in-flight width; it re-exports `vulkano`, `vulkano_shaders` and `bytemuck`, so it is the single place their versions are pinned. `aristotle` and `newton` both depend on it, and `newton/build.rs` reaches it transitively — which is why this workspace cannot be built without a Vulkan device. |
| `aristotle` | (no doc) | The world-state layer: `World`, `WorldBuilder`, `WorldKey<T>`, `Epoch`/`EpochBuilder`, and `WorldId`. Storage is GPU-resident — one `vulkano::Subbuffer<[T]>` per registered type with a cached host mapping, a generational free-list and type erasure through `rembrandt::AnyVec`. This is where the `unsafe` lives. |
| `newton` | [crates/newton](./crates/newton.md) | Dynamics on clifford: `RigidBody`, `Inertia`, `Mechanism`, `Component`, `ForceField`, the gravity propagator, the explicit `ExplicitEuler` / `SymplecticEuler` / `LieEuler` and the coupled implicit `Newton` mid-point integrators. Owns the GPU accelerator front end (`src/accelerator/`) and the build-time GLSL codegen (`build.rs`). |
| `viete` | [crates/viete](./crates/viete.md) | The symbolic tracer: runs any `clifford::Scalar` computation over a symbolic scalar (`Sym`), enumerates every `is_effective_zero` branch into a decision-trie IR, emits GLSL or Lua and verifies it against the `f64` carrier. Front end for all generated shader code. Landed 2026-06-26. |
| `hitchcock` | (no doc) | Camera as a world object: a four-body self-orienting rig in its own `newton` `Mechanism`, reading gravity through an anchor as a sensor and never perturbing the simulation. WIP — geometry and stiffnesses still being tuned. |
| `melies` | (no doc) | Standalone wgpu + winit example host for `newton` / `hitchcock` examples. Examples only, not on the engine path; the only windowed code and the only wgpu in the workspace. |
| `borges` | (no doc) | Reserved scaffold. Default `cargo new` template, no dependencies, not implemented. |
| `gates` | (no doc) | Reserved scaffold. Default template; depends on `winit`, otherwise not implemented. |
| `ligeti` | (no doc) | Reserved scaffold. Default `cargo new` template, no dependencies, not implemented. |
| `vulkano-test` | (no doc) | Workspace member under `experiments/`, not `crates/`. A standalone vulkano compute binary for trying things outside the engine. |

**Crate-doc debt.** Governance below says a crate doc is added when a crate ships
a non-trivial public API. Five crates now qualify and have none: `newton`'s
dependencies `rembrandt` (246 lines) and `aristotle` (1103), plus `peano` (1423),
`joints` (1280) and `hitchcock` (1498). `functions` (245) is kernel bodies with no
API surface of its own, and the three scaffolds and `melies` do not qualify. Line
counts are `*.rs` under each crate's `src/` only, one method for all six — counting
tests as well would put `hitchcock` at 2002 and make the figures incomparable. This
is recorded as debt rather than silently marked `(no doc)`, because `(no doc)` in a
table reads as "nothing to say".

Removed: `demo` / `render` (GUI host) and `cordic` / `table-builder`
(fixed-point) — see [decided/legacy-crates](./decided/legacy-crates.md). There
is no GUI host at present.

## Subsystem documents

| Doc | Role |
| --- | --- |
| [ACCELERATOR.md](../../ACCELERATOR.md) | The GPU compute pipeline that evaluates joint force laws and their Jacobians: build-time GLSL codegen (`crates/newton/build.rs`), runtime dispatch (`crates/newton/src/accelerator/`), the Vulkan objects under it (`crates/rembrandt`) and the world storage they share (`crates/aristotle`). Authoritative for that subsystem. It stays at the repository root, where thirteen in-source comments already point at it (4 in `crates/aristotle/src/world.rs`, 6 in `crates/newton/build.rs`, 2 in `crates/newton/src/accelerator/mod.rs`, 1 in `crates/viete/tests/kernel_differential_oracle.rs`) — those back-references are the mechanism that kept it maintained, and moving the file would break them. |

`ACCELERATOR.md` keeps two axes separate that most documents collapse into one:
**design maturity** (`settled` / `weakly worked out` / `not yet touched`, marked
inline) and **implementation state** (landed / placeholder / not wired, in its
status section). Keep them separate when editing it. Collapsing them into a single
"status" would hide both a settled design that nothing implements and an
implemented stage whose design was never worked out, and both exist in it today.

## DECIDED

| Doc | One-line summary |
| --- | --- |
| [core-design-principle](./decided/core-design-principle.md) | Carry the small quantity, never form it by subtraction. |
| [integrator-purity](./decided/integrator-purity.md) | The integrator is a pure function of cheaply-clonable state; branch rule (actions break the branch); newton::Mechanism is the current realization. |
| [vessel-model-rfe](./decided/vessel-model-rfe.md) | RFE elastic-node + kinematic-articulation joint model; per-subsystem operator-splitting integration; warp is an emergent precision resource. |
| [contact-prediction-display](./decided/contact-prediction-display.md) | Contact is a branch-breaking re-base event; predictors carry a no-contact certificate; the map is a non-authoritative display overlay. |
| [frame-convention](./decided/frame-convention.md) | Body-frame momentum integration, with the four-pillar argument (anisotropic-I, L_COM conditioning, Lie-Poisson, cotransform source). |
| [no-reparenting](./decided/no-reparenting.md) | Single global coordinate system for life of session; no floating-origin, no chunked-world, no re-basing. Precision tactics from reparenting architectures don't apply. |
| [single-scalar-type](./decided/single-scalar-type.md) | All physical quantities flow through **one** scalar type uniformly — CPU substrate, GPU shader, snapshots. The specific type (f64 vs Fixed-point) is still open; the invariant is uniformity. |
| [legacy-crates](./decided/legacy-crates.md) | `render` + `demo` (first-iteration form) are frozen learning artefacts; not deleted, not on production path. Future graphics framework lives in new crates following the named convention (e.g. `spielberg` for camera). |
| [camera-gravity-sensor](./decided/camera-gravity-sensor.md) | The camera rig's anchor is a gravity *sensor*, not a gravitating body; its plumb-bob hang gives a warp-invariant ĝ as the "down" reference for attitude. |
| [near-zero-ad-singularities](./decided/near-zero-ad-singularities.md) | Geometric quantities smooth in value can have singular derivatives under forward AD; the norm is ε-softened, and the ½ of the GA commutator product lives in the semantic API, not in `commutator`. |
| [nilpotent-cap-truncation](./decided/nilpotent-cap-truncation.md) | `nilpotent_cap` drops product grades > cap; legal only for fully-nilpotent metrics (enforced in `blade_count`). Realized by `Jet<N>` flat multi-dual AD (cap=1). `cap` is single-source-of-truth: gp filter, inverse routing, and the Neumann-series inverse must all read it — drift makes the closed-form inverse silently wrong. |

## OPEN deliberations

| Doc | Status |
| --- | --- |
| [fixed-point-vs-f64](./open/fixed-point-vs-f64.md) | On pause; `f64` is the current scalar; full Fix integration deferred. |
| [determinism-boundary](./open/determinism-boundary.md) | Principle holds; implementation in flux pending fixed-point-vs-f64. |
| [state-vs-parameters](./open/state-vs-parameters.md) | Discipline still load-bearing — only integrated quantities are state. |
| [no-floating-origin](./open/no-floating-origin.md) | Closed by policy; integer differencing is exact. |
| [body-rotation-attitude](./open/body-rotation-attitude.md) | Structurally answered (PGA Motor); precision representation under f64 still open. |
| [close-encounters](./open/close-encounters.md) | `r → 0` regime; deferred and conditional; constrained non-breaking when undertaken. |
| [camera-attitude](./open/camera-attitude.md) | Deferred by the owner: sensing ĝ is settled, turning it into camera orientation is not. Perpendicular damper and the angular parts are to be prototyped by hand first. |
| [relativistic-layer](./open/relativistic-layer.md) | Future, undecided; renderer half split off to planned/relativistic-renderer. |

## Planned subsystems

| Doc | Role |
| --- | --- |
| [planned/rendering](./planned/rendering.md) | Universal rendering invariants; first-iteration wgpu backend frozen as lesson ([lessons/2026-05-24-rendering-backend-wgpu](./lessons/2026-05-24-rendering-backend-wgpu.md)); backend choice reopened for the next iteration. |
| [planned/relativistic-renderer](./planned/relativistic-renderer.md) | Two-view design; HDR + spectral; aberration; analytic starfield. |
| [planned/setting](./planned/setting.md) | Sketch of the eventual game setting (G-star + pseudo-Saturn + habitable moon + other large bodies); intent capture, not invariant. |
| [planned/setting-patera](./planned/setting-patera.md) | The long-form working draft behind that sketch: the Pater system, Terra, the epochs, and the open threads, with `[OPEN]` / `[TENT.]` markers on what is not settled. Intent capture, not invariant. |
| [planned/implicit-solver-energy-hessian](./planned/implicit-solver-energy-hessian.md) | Design report for an implicit step in which the user supplies only energy functionals and the kernel differentiates everything, geometry and force laws alike, into a mechanism Hessian. **Not what is implemented** — the shipped implicit path is penalty joints with `Differential` Jacobians and a Newton–Schulz block solve. |

## Lessons (retrospective archeology)

| Doc | Captures |
| --- | --- |
| [lessons/2026-05-24-render-demo-first-iteration](./lessons/2026-05-24-render-demo-first-iteration.md) | What was tried in the first `render` + `demo` iteration, what survived, what didn't, bug patterns encountered; both crates frozen as learning artefacts. Distilled principles live in user-level memory. |
| [lessons/2026-05-24-rendering-backend-wgpu](./lessons/2026-05-24-rendering-backend-wgpu.md) | The "wgpu as renderer backend" decision from the first iteration — formerly `decided/`, moved here once it failed the substrate test (WGSL has no f64). |
| [lessons/2026-05-24-render-uses-pga](./lessons/2026-05-24-render-uses-pga.md) | The "PGA throughout, glam only at GPU upload boundary" decision — formerly `decided/`, moved here once the seam compromise turned out to be the load-bearing flaw. |
| [lessons/2026-06-07-gp-optimization-frozen](./lessons/2026-06-07-gp-optimization-frozen.md) | The whole "improve the geometric product" effort, frozen 2026-06-07. No path won; one required giving up a strict invariant. Reopen only if `gp` becomes the bottleneck inside the implicit integrator. |
| [lessons/2026-06-07-gp-table-vs-compute](./lessons/2026-06-07-gp-table-vs-compute.md) | Hypothesis that direct computation could beat the `GP_TERMS` table — **refuted by measurement**. For `Jet` the table is the only fast path. |
| [lessons/2026-06-07-gp-sort-terms-by-k](./lessons/2026-06-07-gp-sort-terms-by-k.md) | Hypothesis that sorting `GP_TERMS` by output slot would fold repeated accumulations into a local reduction — **refuted by measurement**. |
| [lessons/2026-06-07-motor-exp-profiling-snapshot](./lessons/2026-06-07-motor-exp-profiling-snapshot.md) | Reference profiling snapshot for `Motor::exp` across AD towers, deliberately attached to no decision — the baseline any future change is compared against, with its method caveats stated. |

## Findings (measurement notes)

Empirical notes that a `decided/`, `open/` or `planned/` document rests on: what
was measured, on what code, and what the numbers ruled out. They are not
decisions and they are not lessons — they are the evidence a decision cites, kept
so that a claim can be re-checked rather than re-argued. Each names its source
file and the command that reproduces it where one exists.

| Doc | Measured |
| --- | --- |
| [findings/2026-05-30-camera-chain-dynamics-analysis](./findings/2026-05-30-camera-chain-dynamics-analysis.md) | The two-stage camera spring chain: four alternative formulations tried against the Family B characterisation, all rejected; `hitchcock` stays on the original `T6` form, and its `2·v_target/ω₀` steady-state drag-lag is a known trade-off. Backs [open/camera-attitude](./open/camera-attitude.md) and [decided/camera-gravity-sensor](./decided/camera-gravity-sensor.md). |
| [findings/2026-06-01-implicit-constraint-solver-prototype](./findings/2026-06-01-implicit-constraint-solver-prototype.md) | One backward-Euler step `M·a = f_ext + Jᵀλ` solved by local Gauss–Newton on the saddle system, as a standalone `f64` sketch. Conclusion: rigid constraints want a new `Solver(bodies, f_ext, graph, dt)` trait, **not** an `ImplicitIntegrator` extension. Backs [decided/integrator-purity](./decided/integrator-purity.md). |
| [findings/2026-07-21-newton-schulz-convergence](./findings/2026-07-21-newton-schulz-convergence.md) | Newton–Schulz on matrices harvested from the production `assemble_matrix`. Warm start reaches 1e-6 relative `dv` error in two to three iterations; `f32` carries the tolerance with an accuracy floor of 3e-7. The block-diagonal cold-start seed **diverges to `inf`** outside its contractive range — the design it was checked against was wrong on that point, which is why the note exists. Backs the implicit solver in [crates/newton](./crates/newton.md). |

## Records (migrated design specs)

Design specs written under the previous process, kept when this system replaced
it because something that survives cites them: a doc comment in live source, or
[ACCELERATOR.md](../../ACCELERATOR.md). They are the only part of the abandoned
`docs/superpowers/` corpus that was not deleted — three documents of seventy-eight.

The bucket is the claim, and this bucket's claim is weak on purpose: **a record is
the spec a shipped piece of code was written to, not an obligation on it.** Where a
record disagrees with the code, with a `decided/` document, or with `ACCELERATOR.md`,
those win. Each carries that statement in its own header. Nothing may be added here;
the bucket exists to hold what was cited, and it closes at three.

| Doc | Cited by | For |
| --- | --- | --- |
| [records/2026-06-01-camera-spring-pga-echo](./records/2026-06-01-camera-spring-pga-echo-design.md) | `crates/joints/src/axial_spring_damper/critically_damped_warped.rs` | Relative-velocity damping and the λ-extrapolation that cancel the cascade drag-lag in the camera spring chain. |
| [records/2026-07-19-accelerator-storage-contract](./records/2026-07-19-accelerator-storage-contract-design.md) | [ACCELERATOR.md](../../ACCELERATOR.md) | Why accelerator output lives in persistent, World-backed, domain-owned slots: the two aliasing bugs that produced the rule. |
| [records/2026-07-21-newton-schulz-block-solver](./records/2026-07-21-newton-schulz-block-solver-design.md) | [ACCELERATOR.md](../../ACCELERATOR.md) | The block-solver design and the alternatives it rejected with numbers. Its cold-start claim is refuted by [findings/2026-07-21-newton-schulz-convergence](./findings/2026-07-21-newton-schulz-convergence.md). |

## Governance

- **When to update a DECIDED doc.** Any change to an invariant
  requires an explicit Socratic-review discussion first (see
  [process/design-review-method](../process/design-review-method.md)).
  Mechanical updates (cross-ref refresh, typo) need no review.
- **When to add a new DECIDED file.** A deliberation moves from
  `open/` to `decided/` only after the owner explicitly says so in a
  Socratic session. Rename the file to a topic name (drop
  deliberation numbers).
- **When to add a new OPEN file.** Whenever a question arises that
  isn't covered. Cross-link it from related decided / open files.
- **When to add a new crate doc under `crates/`.** When a new
  workspace crate ships at least one non-trivial public API.
- **Resume after a break.** Read
  [process/resume-bookmark](../process/resume-bookmark.md) for what
  the last session decided and what the next thread is.
