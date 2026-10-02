# game-experiment

An experimental N-body gravity simulation evolving into an indie space game, built on a
multibody-dynamics engine over a geometric-algebra substrate.

The distinguishing idea is that the physics is written **once**, generically over a scalar type,
and then instantiated three ways: at `f64` on the CPU, at a forward-AD scalar to get exact
Jacobians for the implicit solver, and at a *symbolic* scalar that is traced and compiled into
GLSL compute shaders at build time. A joint force law is one function; the GPU kernel that
evaluates it is generated from that function rather than written alongside it.

Rust workspace, edition 2024, `resolver = "3"`. MIT licensed ([LICENSE](LICENSE)).

Status: **headless engine under active development.** There is no game, no renderer and no input
layer. Components are exercised through `cargo test` and through windowed examples.

## Build

**You need a working Vulkan compute device on the machine to build this repository**, not only
to run it.

`crates/newton/build.rs` generates the compute kernels at build time. Doing so constructs a
world (`build_world()`, `crates/newton/build.rs:541`), which calls
`aristotle::World::builder()`, which at `crates/aristotle/src/world.rs:467` constructs
`rembrandt::GpuAccelerator::new()` (`crates/rembrandt/src/lib.rs:91`) — a chain of six
`.expect()` calls with no fallback and no
feature gate: `no Vulkan library`, `failed to create instance`, `no devices`, `no GPU found`,
`no compute queue`, `failed to create device`.

Which one you get depends on how the device is missing, and none of them mentions a build. With
no loader at all it is `no Vulkan library`. In the case that actually matters — a container or a
hosted runner, where the loader is installed but no ICD is present — it is
`failed to create instance` at `crates/rembrandt/src/lib.rs:100`. Nothing in any of these
messages suggests that a *build* needed a GPU.

This is a real constraint, not an oversight: it is why hosted CI runners cannot build this
workspace, and it is recorded in [docs/QUALITY_GATES.md](docs/QUALITY_GATES.md#the-vulkan-precondition).

```sh
cargo build --workspace
```

The toolchain is pinned in [config/quality-tools.json](config/quality-tools.json) (currently
1.98.1). A `rust-toolchain.toml` is on its way in.

## Test

```sh
cargo test --workspace
```

This is the primary development loop and the same device requirement applies. Some tests are
`#[ignore]`d, including two that sit on unresolved `FIXME: deadlock?` sites in the implicit
solver.

Benchmarks build with fat LTO and a single codegen unit so that numbers are comparable; the
machine-specific `-C target-cpu=native` is deliberately not baked in:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench -p clifford
```

Windowed examples are hosted by `melies`. They block, and they are not for automated contexts.

## Workspace layout

A summary. [CLAUDE.md](CLAUDE.md) holds the authoritative crate table — it is the one gate 14
will check — so if the two ever disagree, believe `CLAUDE.md`.

| Crate | Role |
| --- | --- |
| `crates/peano` | Type-level naturals, length-typed vectors, and the `Ring`/`Scalar` trait tower everything else is generic over. |
| `crates/clifford` | Geometric algebra: `Mv<A, S>` over a generator chain, the `pga3` facade (`Motor`, `Twist`, `Wrench`, `Point`, `Line`, `Plane`), and forward-mode AD as `Tangent<N, R>`. |
| `crates/functions` | The scalar-generic bodies of the solver kernels, written once so they can be traced into shaders. |
| `crates/joints` | Two-body force elements — axial spring-damper, perpendicular and torsional dampers — as pure functions returning wrenches. |
| `crates/rembrandt` | The vulkano device layer: device, compute queue, allocators, buffers, pipelines. |
| `crates/aristotle` | World state: GPU-resident typed storage, generational keys, and the epoch coordinator. |
| `crates/newton` | The dynamics engine: rigid bodies, inertia, mechanisms, force fields, gravity, explicit and implicit integrators, the GPU accelerator front end, and the build-time GLSL codegen. |
| `crates/viete` | The symbolic tracer that turns a scalar-generic function into branch-enumerated IR and then into GLSL or Lua. |
| `crates/hitchcock` | Camera as a world object — a self-orienting four-body rig inside its own mechanism. |
| `crates/melies` | wgpu + winit host for examples. Examples only; not on the engine path. |
| `crates/borges`, `crates/gates`, `crates/ligeti` | Reserved scaffolds, not implemented. |
| `experiments/vulkano-test` | A standalone vulkano compute binary for trying things outside the engine. |

Compute is vulkano; the only wgpu is in `melies`. No first-party crate depends on `ash`.

Outside the crates, [`coordination/`](coordination/README.md) holds the development process's
own state: milestones, numbered tasks with their plans and evidence logs, and `handoffctl`, the
tool that writes them. [`coordination/HISTORY.md`](coordination/HISTORY.md) records what was
done before the project was published and why the task set restarted from empty.

## Where to read next

- [CLAUDE.md](CLAUDE.md) (also `AGENTS.md`) — the orientation entry point: what exists, the
  correct vocabulary, and the accelerator.
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — how work is claimed, verified, reviewed and
  integrated. Read it before changing anything.
- [docs/QUALITY.md](docs/QUALITY.md) — what is enforced and, just as importantly, what is not.
- [docs/README.md](docs/README.md) — which documentation system is authoritative and why.
- [CLAUDE.md](CLAUDE.md#the-accelerator) — the GPU subsystem and the invariants it enforces
  (its former design document was retired in M1; the code's module docs describe it).
- [docs/architecture/overview.md](docs/architecture/overview.md) — the design index.
- [coordination/README.md](coordination/README.md) — the task coordinator: milestones, tasks,
  leases and the transitions between them.

## Contributing

Development is agent-driven and coordinated through numbered tasks kept in
[`coordination/`](coordination/README.md). A worker, human or agent, claims a task under a lease,
records evidence as it goes, and submits the result; it never marks its own work done. Only an
independent reviewer can move a task to `done`. One worker owns one task, one branch and one
worktree at a time, and edits only the paths that task lists. Commits are signed and carry a
`Signed-off-by:` trailer matching the author exactly. See
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).
