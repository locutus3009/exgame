# Roadmap — M3 and beyond

**Status.** Planned / intent. Written at the close of M1 (2026-10-02), when M2 was opened. M2 is
governed by [its milestone document](../../../coordination/milestones/M2.md); everything here is a
proposal. A milestone becomes real only when it gets a document in `coordination/milestones/` and
its tasks get plans. The further down this page, the less detail, on purpose: the near milestones
are derived from design documents that already exist, the far ones from intent only. The roadmap
ends at the demo (M9); what follows it is planned with the owner once the demo exists.

## Rules every milestone follows

- **Each milestone ends with a performance review and optimisation task.** Profile the workloads
  the milestone added or changed, record where the time goes, fix what the profile names, and
  record the before/after numbers. It is a required task of the milestone, not a follow-up, and
  the milestone does not close without it. M2 carries it as AR-0209.
- **Build time is not a constraint.** Heavier build-time code generation, more specialised
  kernels, longer compile times are all acceptable when they buy runtime speed.
- **Runtime cost decides design questions.** A technique that costs a lot of runtime
  performance is out, however elegant: that is why the energy-Hessian implicit solver
  ([implicit-solver-energy-hessian](./implicit-solver-energy-hessian.md)) is not on this roadmap.
- **No relativistic layer.** [open/relativistic-layer](../open/relativistic-layer.md) and
  [relativistic-renderer](./relativistic-renderer.md) record a discussion, not a plan. Nothing
  here builds toward them.

## Where the project stands

The repository is an engine, not yet a game. M0 made quality claims checkable, M1 made the GPU
accelerator's storage and dispatch sound, and M2 checks that the physics is right and puts the
first flyable ship into a first version of the game's star system.

The game is set in the Pater system ([setting-patera](./setting-patera.md)): a young G6–G8V star,
the Saturn-class giant Pater at 0.85 AU with the tidally locked habitable moon Terra, the
semi-molten Eos, the cold super-Earth Chione, the super-Jupiter Frater, an ice giant, trojans,
the Ark at Pater's L5, and a K-dwarf companion at 100–200 AU. The story's present is the
beginning of the space age on Terra. The setting is deliberately neutral about genre.

## The scalar split, and one conflict that comes before everything else

**The scalar question is decided:** `f32` for local physics, fixed-point for the reference frames
themselves. The code is shaped for it but does not yet commit to it: kernels run in `f32` on
island-local poses, while an island's origin is a generic scalar `S` (`mechanism/island.rs`), and
only M1's AR-0107 test instantiates it as a fixed-point anchor (`Fix<I96F32>`). That test measured
the split working, with kernel inputs independent of world offset up to 10⁹. Making fixed-point
frames the production default, not a test instantiation, is M3's frame/local boundary task. Two documents still say otherwise and
are stale: [decided/single-scalar-type](../decided/single-scalar-type.md), which demands one type
everywhere, and [open/fixed-point-vs-f64](../open/fixed-point-vs-f64.md), which leans to `f64`.
Superseding both with a record of the split is part of M2's AR-0206. What remains to design
under the split is *where* a quantity crosses from frame to local, and that the crossing never
rounds an absolute position through `f32`, which is M2's AR-0204.

One decided invariant the code currently violates. It is not a bug, it is the result of building
the accelerator first, and it gets more expensive with every milestone that ignores it:

1. **A pure, forkable canonical state.** [decided/integrator-purity](../decided/integrator-purity.md)
   requires the authoritative integrator to be a pure function of a cheaply clonable state value,
   with branchable trajectories for prediction. The world is now a GPU-resident shared store
   written in place. Nothing can fork it today, so the prediction tier that
   [decided/contact-prediction-display](../decided/contact-prediction-display.md) describes has
   nowhere to run.

## M3 — A forkable state, and determinism

**Outcome.** The conflict above is resolved by decision and then by code, and the determinism
gate that M0 and M2 deferred becomes buildable. The scalar split makes the gate's scope precise:
the fixed-point frames are bit-reproducible by construction, so the gate has to pin the `f32`
local physics, whose GPU reductions must stay order-fixed (M1 already made the gather and the
batch composition order-independent).

This milestone starts with one decision only the owner can make, as a design review under
[design-review-method](../../process/design-review-method.md), recorded in `decided/`:

- **D1 — what is canonical.** Either the GPU World is the canonical state and forking means
  copying per-type buffers on the device (cheap at current sizes, and growth already exists), or
  the canonical state lives on the host as a plain value and the GPU World is a cache rebuilt from
  it. The first keeps M1's design; the second is closer to the letter of integrator-purity.

**Candidate tasks.**

| Candidate | What |
| --- | --- |
| Frame/local boundary | Make fixed-point the production island origin, and make the split a type: frame quantities are fixed-point, local ones `f32`, and the only conversion is the anchor-relative one, so an absolute position cannot reach `f32` by accident. |
| World fork | `World::fork` (or the D1 equivalent): a copy-on-branch snapshot whose stepping cannot touch the parent, with a test that a branch and its parent diverge only by the inputs applied to the branch. |
| Prediction tier | A secondary predictor that steps a fork forward, stops at predicted contact as contact-prediction-display requires, and never writes back. |
| Determinism gate | Bit-reproducible integrated state across runs and batch compositions, made a gate in `tools/quality/run-gates.sh`, scoped as [open/determinism-boundary](../open/determinism-boundary.md) says. |

## M4 — The whole Pater system

**Outcome.** The full system of setting-patera integrates for long spans, stays stable, and can be
time-warped, with every scale from Terra's 7-day orbit to the companion star's millennia handled
without breaking integrator purity.

M2's AR-0207 builds the first slice: the star, Pater, Terra and one more moon. M4 is the rest.

**Candidate tasks.**

| Candidate | What |
| --- | --- |
| Full body set | Every `[decided]` body of setting-patera with its decided parameters, and the `[OPEN]` ones flagged rather than invented: mini-Mercury, Eos, Pater with Terra, Rhea, Io, the small moons and the irregular sentinel, Chione, Frater with its retinue, the gatekeeper ice giant, the trojans, the companion star. |
| Multi-rate stepping | One step size cannot cover a 7-day moon and a century-scale companion. A hierarchical or multi-rate scheme that keeps the symplectic scope and integrator purity, with the reentry-class step reduction of [decided/vessel-model-rfe](../decided/vessel-model-rfe.md) as the only variable-step path. |
| Long-term stability | A gated soak: element bounds over at least 10⁴ years for the planets and 10³ Terra orbits for the moon system, including the Terra–inner-moon 3:2 resonance setting-patera relies on. |
| Lagrange points | The analytic L4/L5 checks M0 deferred, with the Ark at Pater's L5 as the case that matters. |
| Close encounters | Classify the L4-probe ejection in [open/close-encounters](../open/close-encounters.md) as physical or artefact, and land the step-reduction slot if it is needed. |
| Time warp | The warp control the vessel model presumes, capped by close encounters as decided. |

## M5 — A graphics pipeline beside the compute pipeline

**Outcome.** A real renderer that draws the simulation from its own GPU storage, on the same
device as the accelerator, under the invariants of [rendering](./rendering.md).

[planned/rendering](./rendering.md) reopened the backend choice after the first `wgpu` attempt
failed: the substrate scalar could not reach the shader. Under the scalar split the renderer does
what the kernels do, drawing camera-relative `f32` against fixed-point frames, so the constraint
that survives is that the substrate reaches the shader with raw SPIR-V access. That points at one
answer already in the tree:
**vulkano graphics on the device `rembrandt` already owns.** That shares buffers with compute with
no copy, so the renderer can read the World where the integrator wrote it. `melies` stays what it
is, an example host on `wgpu`; the game's renderer is a new crate.

**Candidate tasks.**

| Candidate | What |
| --- | --- |
| Backend decision | Record the vulkano-on-`rembrandt` choice (or its rejection) in `decided/`, with the UI framework that follows from it. |
| Device sharing | One device, a compute queue and a graphics queue, with synchronisation between a frame and a flush. The renderer reads; it never writes simulation state (invariant 1). |
| Tactical view | The sensor / tactical view first: bodies, orbits and predicted trajectories from the M3 prediction tier, with the contact marker contact-prediction-display requires. |
| Scale range | Rendering from a cockpit to the companion star: camera-relative `f32` against fixed-point frames, reversed or logarithmic depth. |
| HDR from day one | An HDR, linear-light pipeline: a young G star, a gas giant filling Terra's sky and moonlit nights span far more range than sRGB, and retrofitting HDR is a rewrite. |
| Camera | The `hitchcock` rig as the game camera, and the deferred threads of [open/camera-attitude](../open/camera-attitude.md). |

## M6 — A frame budget, enforced

**Outcome.** Measured budgets for a frame that includes both simulation and rendering, enforced by
criterion benchmarks on an agreed gate machine, so a regression fails a gate instead of being
found by the next milestone's review.

Every milestone already ends with a performance review; this one turns the reviews into enforced
budgets, and it comes after graphics because until a frame exists there is no budget to hold the
simulation to. M0 and M2 both deferred budgets for want of one. What is already measured: the
step is bound by submission latency (about 52 µs per flush, whatever it carries), and cloth step
time grows as m^1.24 after M1's message reshaping.

**Candidate tasks.** Replace the fence wait per flush with timeline semaphores so the worker does
not block; device-local storage for the hot types with an explicit host-transfer path where the
host still reads; workgroup sizing per kernel; the gravity propagator on the GPU if profiling says
it matters; and the gate machine and criterion budgets themselves.

## M7 — Vessels and flight

**Outcome.** Ships as [decided/vessel-model-rfe](../decided/vessel-model-rfe.md) describes them:
rigid finite elements with elastic nodes and kinematic joints, propellant and mass change,
maneuver planning on the M3 prediction tier, and contact as a branch-breaking re-base event. M2's
flyable ship is the seed; this is the version a game can be built on.

## M8 — Surfaces and atmospheres

**Outcome.** Terra has a surface to stand on and an atmosphere to climb through and fall back into,
because the demo starts on the ground. Terra's geography, climate and sky are richly specified in
setting-patera (tidally locked, near side and far side, a 7-day day); none of it has an engine
counterpart yet.

**Candidate tasks.** A surface model for Terra at the resolution a launch and a landing need
(terrain as data, contact through the decided contact operator); an atmosphere model for drag and
heating during ascent and reentry, with the reentry-class step reduction of vessel-model-rfe as
its only variable-step path; rendering the surface and the sky from the ground (Pater overhead, the
eclipse seasons); and the same for Rhea and Io at lower fidelity, since they are the first
destinations.

## M9 — The demo

**Outcome.** Something a person plays rather than tests, set where the story starts: the beginning
of Terra's space age. Launch from Terra's surface, reach orbit, reach Rhea or Io, and see the Ark
at L5 from afar. setting-patera leaves the genre open; the engine most naturally serves a sandbox
in the KSP tradition, and the demo is built so that choice can still be made after it.

The roadmap stops here. Economy, story, factions and everything after the demo are planned with
the owner once the demo exists.
