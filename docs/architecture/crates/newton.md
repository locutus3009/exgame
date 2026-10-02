# newton — dynamics on clifford

Multibody rigid-body dynamics implemented on top of the
[clifford](./clifford.md) PGA library. Extracted from clifford in
commit `5328757 Move Newton dynamics to separate crate`. Exercised through
`cargo test` (the earlier `demo` consumer was removed).

## Role

| Module | Type(s) | What it does |
| --- | --- | --- |
| `body.rs` | `RigidBody` | Pose (`Motor`), momentum (`Wrench`, body-frame), inertia + instantaneous dynamics only. |
| `inertia.rs` | `Inertia` | Twist → Wrench mapping. Diagonal or full symmetric tensor. |
| `joint.rs` | `Joint` | Penalty-method spring + damper. The RFE elastic-node realization. |
| `gravity.rs` | `GravityCharge`, `GravityPropagator`, `charge_of` | N-body gravity via charge propagation. |
| `integrator.rs` | `Integrator`, `ExplicitEuler`, `SymplecticEuler`, `LieEuler` | Time-stepping strategies. |
| `mechanism.rs` | `Mechanism`, `Entity`, `Behavior`, `ForceField`, `Inert`, `WorldId` | Owns bodies, joints, force fields; runs `step(dt)`; supports `split` / `merge`. |

## State ownership

`Mechanism` owns its bodies and joints by value. This is the current
realization of the integrator-purity invariant
([decided/integrator-purity](../decided/integrator-purity.md)). The
predecessor pattern (double-buffered `World` behind RAII guards in
old `demo/sim.rs`) is superseded; the principle (exactly-once flush,
no aliased state across the step boundary) carries forward; the
implementation does not.

`Mechanism::split` and `Mechanism::merge` are the topology operations
that close the merge/split body-identity residual previously listed
under [vessel-model-rfe](../decided/vessel-model-rfe.md).

## Frame

Momentum is integrated in body-frame; see
[decided/frame-convention](../decided/frame-convention.md). The
`world ↔ body` transformation happens during the force-aggregation
step of `Mechanism::step`, as a co-adjoint pullback on per-body
`Wrench` inputs (implemented as
`clifford::pga3::screw::Wrench::cotransform`).

## RigidBody

State (`pose: Motor`, `momentum: Wrench` in body-frame, `inertia`) +
**only instantaneous** dynamics: `velocity()`, `kinetic_energy()`,
`gyroscopic()`, `world_momentum()`. It contains no time step —
so it does not depend on `Transcendental` (see the note about `Scalar` in
[crates/clifford](./clifford.md)). This is the seam between "state" and
"evolution". It stays a pure **Copy-POD** — this matters for `Mechanism`
(storage by value in a slotmap, migration).

Gyroscopic term: `[P,V] = ½(PV − VP)` — the se(3) bracket,
the antisymmetric part of the geometric product of two bivectors, pure
grade 2. It carries no power, it sets the precession. Euler's equation in
body-frame: `Ṗ = W + [P,V]`.

## Inertia

The velocity → momentum mapping (`P = I·V`, `Twist → Wrench`).
Since Twist and Wrench are in the same basis (J is deferred to `power`),
inertia is a scaling of the `[T;3]` parts through the public screw
accessors; **no blades and no dual**. The internal representation
is a private `enum InertiaTensor`:

- `Diagonal([T;3])` — principal moments, component-wise.
- `Full([[T;3];3])` — a general symmetric tensor; `apply` = matvec,
  `apply_inverse` = a 3×3 solve via adjugate/determinant.

`principal_moments` for `Full` is still `todo!()` (these are the
eigenvalues, it needs `acos`/diagonalization; the dynamics does not require it).

## Joint (spring + damper)

Penalty method: the joint is a source of force, not a rigid constraint
(no DAEs). This is exactly the RFE elastic-node realization (see
[decided/vessel-model-rfe](../decided/vessel-model-rfe.md)).

Anchors are stored in the bodies' local coordinates. `wrenches(a, b)`
returns a pair of body-frame wrenches; the world forces are equal and opposite
(3rd law). `potential_energy(a, b) = ½·k·(L−L₀)²` — only the spring
term (the damper is dissipative and has no potential); it enters the total
energy of the system alongside the kinetic energy of the bodies.

The **PGA / Euclidean** division of labour (deliberate):

- **PGA**: transforming anchors (a point by a motor), rotating the force into
  body-frame (a direction by the inverse motor), the spatial twist
  `ξ = Ad_M(V) = M·V·M̃` for the anchor velocity via
  `v(p) = ξ_lin + ξ_ang × p`.
- **Euclidean**: distance, direction, spring/damper force magnitude,
  torque `r×F`. These are intrinsically metric quantities; in a degenerate
  metric, computing them through norms of PGA lines would require
  bulk/weight norms (not built, subtle).

## Integrators

The time-stepping strategy. An open set of schemes → a trait. The body provides
instantaneous dynamics; the integrator decides how to step.

| Variant | Property |
| --- | --- |
| `ExplicitEuler` | Position from old velocity. Drift ∝ dt·T on momentum and energy. Order 1. Comparison baseline. |
| `SymplecticEuler` | Velocity, then position from new velocity. Symplectic for canonical systems; on the (non-canonical) free-rotation Lie–Poisson system drift ∝ dt·T remains. Order 1. |
| `LieEuler` | Coadjoint-conserving: momentum updated by `δM̃ · P · δM` conjugation with the same δM used for the pose. Gyroscopic term emerges from the conjugation, not added separately. **World momentum exactly preserved (machine precision, any dt) under body-frame.** Energy bounded-band, no secular drift on the conservative part. Order 1. Canonical choice. |

`LieEuler` machine-precision-conserves the Casimir |L| under the
body-frame convention for any dt (see
[decided/frame-convention](../decided/frame-convention.md) §3.3). It
does NOT pointwise-conserve energy — energy oscillates — but does not
drift secularly on the conservative part. Confirmed by a run of a pair of
bodies with a spring (c=0) to t=40. With dampers the energy MUST decrease —
exact pointwise energy conservation is NOT a goal for our system;
Gonzalez/midpoint was deliberately not done.

## Mechanism

A multigraph of bodies: fully connected, with cycles, parallel joints
allowed, self-loops forbidden. It contains the body graph + joints + force
fields + an encapsulated integrator (statically, as a type parameter
`I: Integrator<T>`).

**Addressing:** outward — only `WorldId` (= usize), assigned by the
user, unique **globally by construction** (an invariant of the world,
not of an individual mechanism — `merge` relies on this), stable.
Inside — slotmap keys (`BodyKey`/`JointKey`), which do NOT LEAK
OUTWARD. A bidirectional mapping `world_to_key` (HashMap) +
`key_to_world` (SecondaryMap, for the reverse translation in the field phase).

**Storage:** bodies in `SlotMap<BodyKey, Entity>`, joints in
`SlotMap<JointKey, JointEdge>`. Removing a body is O(joints): a linear
pass without adjacency lists (a deliberate trade-off — removal is
rare, stepping is frequent).

**`Entity { body: RigidBody, behavior: Box<dyn Behavior> }`** —
the unit of migration. The behaviour (`Behavior::pre_step`, updates the body's
state before forces are computed: fuel, inertia, engine mode) lives
INSIDE `Entity`, so on migration (docking/undocking)
it moves TOGETHER with the body. `RigidBody` meanwhile stays Copy-POD —
the behaviour is NOT put inside the body (otherwise the body would lose
Copy/Clone/PartialEq, which are needed everywhere).

**`ForceField::accumulate(&self, bodies: &IndexMap<WorldId, Box<dyn Component>>, out: &IndexMap<WorldId, WorldKey<Wrench>>, epoch: &Epoch, origin: &Vector3<S>)`**
— an external force. It sees ALL bodies (N-body uses this for pairwise forces),
`&self` (read-only; internal mutation of the field goes through interior mutability).
`epoch: &Epoch<T>` carries the step (`epoch.dt()`) and identifies the tick for
broker reads; `origin` is the island's S-anchor. The contribution is written ADDITIVELY
into `out[id]` (the body's World-backed wrench slot). The mechanism does NOT know about
gravity specifically — it is one implementation of the trait.

**Two-phase protocol:** `accumulate` (all fields) → integration → `publish`
(caches per-body state into the BACK buffer; default no-op). `publish` replaced the old
`prepare`: publishing AFTER integration makes the coupling synchronous (no
structural lag-1) — this is how `GravityPropagator` publishes fresh charges.

**Field primitives:** `GravityPropagator` (N-body gravitation) and
`UniformField(accel)` — a constant world acceleration `F = m·a` (gravity
`[0,0,−g]`, a crosswind `[0,a,0]`; kinematic bodies with `mass()==0` get
zero automatically).

**Migration — only at the mechanism level.** The primitives
`attach`/`detach` (a single object) are PRIVATE. The public interface is
`split`/`merge`:

- **`split<J>(&[WorldId], integrator: J) -> Mechanism<T, J>`** —
  extract bodies into a new mechanism with an EXPLICITLY given integrator `J`
  (it may differ in type — "we landed, the planet has its own solver").
  The new mechanism has EMPTY fields (forces are set up anew). `self`
  keeps its integrator and fields. Edges are classified by the group
  boundary: internal ones MOVE with the bodies; boundary ones are BROKEN; external ones
  stay in `self`.

- **`merge<J>(other: Mechanism<T, J>, link: (Joint, WorldId, WorldId))`**
  — pour `other` into `self`, creating exactly one joint CROSSING the
  boundary. `self` keeps its integrator and fields; `other`'s integrator
  (of type `J`) and its fields are destroyed with `other`. Merging mechanisms with
  different integrators works WITHOUT `dyn`.

**The `step(dt)` loop — five phases, the order is essential** (all forces from
the state at the start of the step):

The step runs **PER ISLAND** (`Islands`), islands concurrently via `join_all`;
inside an island:
1. **pre-step:** `behavior.pre_step` of every body (sees the island's S-anchor).
2. **zero the island's wrench buffer.**
3. **fields:** `field.accumulate(bodies, out, epoch, origin)` — the ADDITIVE contribution
   of EXTERNAL forces in the WORLD frame. (Joints are NOT computed here — they are inside
   the integrator step; the world→body pullback also moved INSIDE the integrator, not here.)
4. **integrate:** `integrator.step_all(bodies, joints, wrenches, epoch)` — coupled
   implicit integration of the WHOLE island through `Accelerator` (PRE → conn kernels →
   gather; see `ACCELERATOR.md`). World-frame residual assembly
   ([decided/frame-convention](../decided/frame-convention.md)).
5. **re-anchor origin to the COM** (`Island::recompute_centroid`), then
   `field.publish` (BACK buffer) and publication of the `centroid`/`centroid_velocity`
   summary under their own locks.

Borrow cleanliness is achieved by separating phases: reading (3,4) and writing (5)
are separated in time, with the key-indexed wrench buffer between them.

**Connectivity is a diagnostic, NOT an invariant.** A mechanism is a logical
orchestration container (a shared wrench buffer, a shared integrator pass,
shared fields), NOT a physical entity: it has no coordinate system
of its own. A disconnected mechanism is a LEGAL state. The diagnostic
methods `components() -> Vec<Vec<WorldId>>` and `is_connected()`
let the user decide whether to cut a disconnected mechanism
— by feeding a component from `components()` into `split`. (`feb1374 Add
diagnostics for disconnected mechanism`.)

**Allocations in the step:** the buffers (`wrenches: SecondaryMap`, `field_out`,
`field_keys`) are struct fields and reuse their capacity without
reallocation. The hot path is allocation-free.

## Gravity propagator

`GravityCharge` / `GravityPropagator` / `charge_of` — N-body
gravity reified as a `ForceField`. Reads each body's mass parameter
(see [state-vs-parameters](../open/state-vs-parameters.md)),
computes pairwise charges, accumulates wrenches into the buffer. Freshly integrated
charges are cached via `ForceField::publish` (post-integrate BACK-buffer, so the
next step's `accumulate` reads charges matching the current positions — no lag-1).
The geometric-cull lesson
from the original `Q_a=ACCEL_MIN` discipline carries forward as a
*parameter-side* concern — it's not part of integrated state, so
cutoff choice doesn't enter the determinism budget.

## Testing philosophy

- **Invariants, not coordinates.** Energy conservation (`½⟨P,V⟩`),
  conservation of total momentum (3rd law), drift convergence ∝ dt
  (integrator order).
- **Qualitative laws where quantitative ones are out of place.** On the
  conservative part (c=0) the total energy does not drift
  (a bounded band to t=40); with a damper (c>0) the total energy
  decreases **monotonically** and tends to the rest energy. This encodes "does not
  decay without dissipation, decays with it".
- **Orchestration is checked by an end-to-end test.** `Mechanism` runs
  the same physics as the `Joint`/`Integrator` unit tests, but through `step`
  — plus orchestration-specific invariants: the `ForceField` path,
  migration via split/merge, a body's behaviour moving together with it.
- **Convergence distinguishes correctness from error.** The world
  angular momentum test: a drift ratio of ≈2 (first order) confirms the
  integrator and the gyro sign; a wrong sign would give ≈1. For `LieEuler` the same
  test shows machine precision.

## Open tails

In order of likely usefulness:

1. **N-body gravity as a `ForceField`** — the interface is already tailored
   for it (sees all bodies, pairwise forces, symmetry). It will give the final
   Kepler-orbit integration test: closure, conservation of
   orbital angular momentum under `LieEuler`, boundedness of the total energy.
2. **`principal_moments` for `Full Inertia`** — via `acos`
   (extend `Transcendental` in clifford), if diagnostics or step
   selection by eigenfrequencies is ever needed.
3. **Fixed-point support** — clifford already supports
   `Multivector<…, Fix>`; full integration chain through newton does not
   build (see
   [fixed-point-vs-f64](../open/fixed-point-vs-f64.md)).
4. **Higher order of accuracy** — if trajectory accuracy becomes
   the bottleneck: RKMK / composition schemes on top of the existing
   `Motor::exp` (in clifford). `LieEuler` already preserves angular momentum;
   this is a separate axis (accuracy of approximating the solution).
5. **Caching `view` in the field phase** — if the profile shows that
   allocating the slice of references is significant. It will require changing the
   `ForceField` signature.

## History of key decisions

- **`Inertia` is a mapping, not a multivector.** The old `Impulse /
  Inertia` as a geometric division was a category error.
- **`LieEuler` conserves angular momentum, but not "energy exactly".** Exact
  energy conservation for a dissipative system is the wrong goal.
- **Behaviour in `Entity`, not in `RigidBody`.** The hook must
  move with the body on migration. Inside `RigidBody` it would kill the
  body's Copy-POD semantics.
- **World ids outward, slotmap keys inside.** The user
  assigns stable meaningful ids; slotmap keys (with generations)
  are a detail.
- **The integrator is a property of the mechanism, NOT of the body.** `merge<J>`/`split<J>`
  accept a foreign integrator type `J` only in the signature — it does not
  penetrate inside, so merging/splitting mechanisms with different
  integrators works without `dyn`. (The approach "an integrator on every body
  via `dyn`" was rejected as overcomplication.)
- **Migration only via mechanisms.** `attach`/`detach` (a single object)
  are private; `split`/`merge` are public.
- **Connectivity is a diagnostic, not an invariant.** A disconnected graph is physically
  correct and legal. We do NOT add a constraint check.
- **`ForceField::publish` — a two-phase protocol** (`cc70858`, formerly `prepare`).
  The field caches per-body state in the `publish` phase AFTER integration
  (BACK buffer, swapped outside), so that the next step's `accumulate` reads state
  corresponding to the CURRENT positions — without structural lag-1; this is how the
  gravity propagator publishes charges.
- **Body-frame momentum.** After three-stage debugging (`a664c22` →
  `4321437` → `ba1ae0d`) body-frame was confirmed; details and the
  four-point argument are in
  [decided/frame-convention](../decided/frame-convention.md).

## Cross-references

- [crates/clifford](./clifford.md) — the GA substrate.
- [decided/frame-convention](../decided/frame-convention.md) —
  body-frame momentum, the load-bearing design choice.
- [decided/vessel-model-rfe](../decided/vessel-model-rfe.md) — the
  RFE model that `Joint` implements.
- [decided/integrator-purity](../decided/integrator-purity.md) — the
  state-ownership invariant `Mechanism` realizes.
- [open/state-vs-parameters](../open/state-vs-parameters.md) —
  `ForceField` / `GravityPropagator` honour the parameter / state
  split.
- [open/fixed-point-vs-f64](../open/fixed-point-vs-f64.md) — full
  Fix integration through newton is open.
