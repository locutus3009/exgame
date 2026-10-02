# AGENTS.md / CLAUDE.md — agent entry point

`AGENTS.md` is a symlink to this file; both names lead here.

Experimental N-body gravity simulation evolving into a space game, built on a multibody-dynamics
engine over a geometric-algebra substrate. Rust workspace, `resolver = "3"`, edition 2024.
Headless: there is no game or visualization host. `melies` hosts wgpu examples, and that is the
only windowed code in the tree.

**Read [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) before changing anything.** Work here is
coordinated: one worker owns one task, one branch and one worktree, and edits only the paths its
task lists.

## Build-time requirement: a Vulkan compute device

`cargo build` and `cargo test` **require a working Vulkan compute device on the machine at build
time**, not merely at run time.

`crates/newton/build.rs` generates GLSL kernels. To do that it builds a symbolic world:
`build_world()` (`crates/newton/build.rs:541`) calls `aristotle::World::builder()`, which at
`crates/aristotle/src/world.rs:467` constructs `rembrandt::GpuAccelerator::new()`. That
constructor (`crates/rembrandt/src/lib.rs:91`) is a chain of `.expect()` calls —
`"no Vulkan library"`, `"failed to create instance"`, `"no devices"`, `"no GPU found"`,
`"no compute queue"`, `"failed to create device"` — with no fallback and no feature gate.

Without a device the build script panics and the build aborts with one of those messages, none
of which hints that a *build* needed a GPU. Which one depends on how the device is missing: no
loader at all gives `no Vulkan library`, while the container and hosted-runner case — loader
present, no ICD installed — gives `failed to create instance` at
`crates/rembrandt/src/lib.rs:100`. There is no CPU path. This is why hosted CI runners cannot
build this workspace, and it is a known constraint rather than an oversight.

## Workspace layout

Every crate below exists, and nothing exists that is not below — verified by hand against
`cargo metadata --no-deps` at the commit that wrote this table. **Nothing enforces it yet.**
Gate 14 in [docs/QUALITY_GATES.md](docs/QUALITY_GATES.md#14-entry-point-crate-table-consistency)
specifies the check that will; AR-0006 builds it. Until then, if you add or remove a crate, this
table is your responsibility. Crates are named after people; the name carries no information
about the contents.

| Crate | Role |
| --- | --- |
| `peano` | The type-level and algebraic foundation everything else stands on: Peano naturals (`Z`, `Succ<N>`, `Nat`), a cons-list `Vector<N, T>` whose length is a type, and the trait tower `AbelianGroup → Ring → Commutative → Scalar`. `bytemuck::Pod` on the storage so vectors upload to the GPU. Optional `fixed` feature gives a fixed-point `Scalar` carrier. |
| `clifford` | `no_std` geometric algebra. `Mv<A, S>` is a stratified multivector over an algebra descriptor `A` (a generator chain `Gen<SQ, Rest>` / `Nil`) and a `peano::Ring` `S`. Named algebras `Pga3`, `Complex`, `Quaternion`, `Dual`, `Cl30`. The `pga3` facade adds the physics types: `Motor`, `Screw<V, S>` with `Twist`/`Wrench` variance tags, `Point`, `Plane`, `Line`, `Direction`, and the `Dynamics`/`Differential` AD entry points. Forward-mode AD is `Tangent<GradN, R>`. |
| `functions` | The scalar-generic *bodies* of `newton`'s implicit-solver kernels — `pre`, `gather_term`, `matvec_term`, `gemm_term`, `reduce_sq_term`, `assemble_term`, `body_post`. Written once over `peano::Scalar` so `newton/build.rs` can trace them into GLSL instead of anyone hand-writing a shader. No state, no types beyond `Block<T> = [[T; 6]; 6]`. |
| `joints` | The two-body force elements: `JointEdge<T>` binding two `WorldId`s to a `Joint<T>`, and the concrete laws `AxialSpringDamper`, `PerpendicularDamperWarped`, `TorsionalDamperWarped`. Each is a pure world-frame function `(poses, velocities, epoch) -> [Wrench; 2]`, polymorphic in the scalar so the same source runs on `f64`, on `Tangent`, and through the symbolic tracer. |
| `rembrandt` | The vulkano device layer. `GpuAccelerator` owns the `Device`, compute `Queue` and the memory, descriptor-set and command-buffer allocators; it allocates storage buffers, builds compute pipelines, and estimates in-flight width. It also re-exports `vulkano`, `vulkano_shaders` and `bytemuck`, so it is the single place their versions are pinned. It deliberately does not record or submit command buffers — that is the caller's job. |
| `aristotle` | The world-state layer: `World`, `WorldBuilder`, `WorldKey<T>`, and `Epoch`/`EpochBuilder`. Storage is GPU-resident — one `vulkano::Subbuffer<[T]>` per registered type, with a cached host mapping, a generational free-list and type erasure through `rembrandt::AnyVec`. Also still the home of `WorldId`. This is where the `unsafe` lives. |
| `newton` | The dynamics engine: `RigidBody`, `Inertia`, `Mechanism`, `Component`, `ForceField`, gravity propagator, and the integrators — explicit `ExplicitEuler` / `SymplecticEuler` / `LieEuler`, and the coupled implicit `Newton` mid-point step. Also owns the GPU accelerator front end (`src/accelerator/`) and the build-time GLSL codegen (`build.rs`). |
| `viete` | The symbolic tracer. Runs any `clifford::Scalar` computation over the symbolic scalar `Sym`, enumerates every `is_effective_zero` branch into a decision-trie IR, and emits GLSL or Lua from it, verified against the `f64` carrier. It is the front end for all generated shader code. |
| `hitchcock` | Camera as a world object: a four-body self-orienting rig in its own `newton` `Mechanism`, wired by axial springs and torsional/transverse dampers, reading gravity through an anchor as a sensor and never perturbing the simulation. WIP — geometry and stiffnesses are still being tuned. |
| `melies` | Standalone wgpu + winit example host for `newton` / `hitchcock` examples. Examples only, not on the engine path. The only windowed code and the only wgpu in the workspace. |
| `borges` | Reserved scaffold. Default `cargo new` template, no dependencies, not implemented. |
| `gates` | Reserved scaffold. Default template; depends on `winit` but is otherwise not implemented. |
| `ligeti` | Reserved scaffold. Default `cargo new` template, no dependencies, not implemented. |
| `vulkano-test` | Workspace member under `experiments/`, not `crates/`. A standalone vulkano compute binary used to try things outside the engine. |

Removed earlier: the `demo` / `render` visualization host and the `cordic` / `table-builder`
fixed-point crates. See [docs/architecture/decided/legacy-crates.md](docs/architecture/decided/legacy-crates.md).

## Graphics and algebra vocabulary — get these right

Two things about this repository are routinely stated wrongly. Both are load-bearing.

**Graphics.** Compute is **vulkano** (0.35), in `rembrandt`, `newton` and `experiments/vulkano-test`.
Windowed rendering is **wgpu** (29), in `melies` only, and `melies` hosts examples rather than
the engine. **No first-party crate depends on `ash`** — no `Cargo.toml` in this workspace names
it and no source file imports it. `ash` appears in `Cargo.lock` only as a transitive dependency
of `vulkano`, `wgpu-hal` and `gpu-allocator`. Any text saying the renderer "will be raw
Vulkan/ash" is describing a plan that was not taken.

**Algebra.** `clifford` cut over from the legacy stack on 2026-06-25. The current vocabulary is
`Mv<A, S>` and `Tangent<GradN, R>`. The pre-cutover names `Multivector` and `Jet<N> = Cl(0,0,N)`
are **gone**; there is no `jet` module, no `blade.rs`. The identifiers `Jet1`, `Jet6`, `Jet12`,
`Jet24` survive only as type aliases for `Tangent<N, R>` (`crates/clifford/src/tangent.rs:309`).

## The accelerator

The dominant work of the last quarter, and the thing least visible from the crate table. It is
documented in **[ACCELERATOR.md](ACCELERATOR.md)**, a living subsystem document with separate
maturity markers for the design and for the code. Read it before touching anything under
`crates/newton/src/accelerator/`, `crates/newton/build.rs` or `crates/aristotle/src/world.rs`.

Three pieces:

- **Build-time GLSL codegen.** `crates/newton/build.rs` traces each joint law and each solver
  stage over `viete::Sym`, enumerates the branches, and writes one GLSL kernel per case into
  `OUT_DIR`. `crates/newton/src/accelerator/shaders.rs` then compiles them with
  `vulkano_shaders::shader!`. The rule is that a shader is *generated* from a function over
  `Scalar`, never hand-written; the single exception is `src/accelerator/simple_sum.glsl`.
- **Async dispatch.** `crates/newton/src/accelerator/mod.rs` runs a worker that batches messages
  into one command buffer per flush and dispatches the solver stages — evaluation of each joint
  type plain and Jacobian, `Pre`, `Gather`, `Gemm`, `AssembleBlock`, `BlockMatVec`,
  `BlockReduce`, `BlockCopy`, and the four `BodyPost` variants (`BodyPostDiagonal`,
  `BodyPostFull`, `BodyPostGatheredDiagonal`, `BodyPostGatheredFull`). The full list is
  `MessageKind::all()` in `crates/newton/src/accelerator/mod.rs`.
- **The tokio migration.** `Mechanism::step`, `ImplicitIntegrator::step_all`, the `Newton` solve
  and the explicit `bridge` are all `async fn`; `dyn ImplicitIntegrator` was replaced by a closed
  enum to avoid boxing futures; islands run concurrently through `futures::future::join_all`.
  Runtime use is thin — `tokio::sync::RwLock` in `Mechanism` and `Camera`, a multi-thread runtime
  in `melies`, and `#[tokio::test]` throughout. `#[async_trait]` is on `ForceField`, `Component`
  and `Example`, not on the accelerator, which is a concrete struct with inherent `async fn`s.

**The open risk.** The world storage is sound only under an "at most one writer per slot"
invariant (`crates/aristotle/src/world.rs:149`, `:196`, `:202`). It is guaranteed by construction
and by the epoch discipline, documented in ACCELERATOR.md, and **not enforced** by anything.
`crates/aristotle/src/` contains 17 uses of `unsafe` resting on it — 8 in `world.rs` and 9 in
`epoch.rs`. Do not add a second writer path to a slot without reading ACCELERATOR.md part III.

## Build, test, run

```sh
cargo build --workspace
cargo test --workspace     # the primary loop
```

Both need the Vulkan device described above. Do not launch `melies` examples from an automated
context — they are windowed and block. If one needs running, ask the owner.

Quality gates are specified in [docs/QUALITY_GATES.md](docs/QUALITY_GATES.md) but are **not yet
implemented**; [docs/QUALITY.md](docs/QUALITY.md) records exactly what is and is not enforced
today. Do not describe a gate as running until it does.

## Conventions

- Commit with `git commit -S -s`: signed, with a `Signed-off-by:` trailer matching the author
  character for character.
- Agent attribution: a standard Git `Co-Authored-By:` trailer, one line, last before the
  sign-off. The tool substitutes its own name and model; do not hardcode model identity.
- Never put an absolute home path, a credential, a private host or IP, or an agent session
  reference into a commit message, a task note or a tracked file.
- All comments, doc comments, string literals and documents in this tree are in English. Keep
  them that way.

## Pointers

- **Process:** [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — how work is claimed, verified,
  reviewed and integrated.
- **Quality:** [docs/QUALITY.md](docs/QUALITY.md), [docs/QUALITY_GATES.md](docs/QUALITY_GATES.md),
  [config/quality-tools.json](config/quality-tools.json).
- **Documentation map:** [docs/README.md](docs/README.md) — which documentation system is
  authoritative, and which is abandoned.
- **Design index:** [docs/architecture/overview.md](docs/architecture/overview.md), with its
  `decided/`, `open/`, `lessons/`, `planned/`, `crates/`, `findings/` and `records/` buckets and
  its governance rules. Its crate table now agrees with the one above; both were checked against
  `cargo metadata --no-deps`.
- **Accelerator:** [ACCELERATOR.md](ACCELERATOR.md).
- **Design-review protocol:** [docs/process/design-review-method.md](docs/process/design-review-method.md).
