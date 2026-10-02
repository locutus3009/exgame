# Implicit constrained-dynamics step — prototype findings

**Date.** 2026-06-01. **Status.** Findings note. Prototype only; **no
engine change, no migration.** The only live edit on `main` is the
stale-comment fix to `crates/newton/src/integrator.rs:142` (commit `9702215`).

**Artifact.** [`crates/newton/tests/implicit_constraint_prototype.rs`](../../../crates/newton/tests/implicit_constraint_prototype.rs)
— a self-contained f64 sketch (no clifford / no traits / no real
`ForceField`) of one backward-Euler step `M·a = f_ext + Jᵀλ`, solved
implicitly by a local Gauss–Newton on the saddle system. Three passing
tests: rigid-rods-stay-rigid (pinned double pendulum), CoM-free-fall
(Jᵀ internal-force structure), empty-graph-collapses-to-per-body
(degenerate dispatch).

**Origin.** De-risking artifact agreed across the integrator/field design
thread — written *before* touching any signature, to make the shape of a
phase-5 solver fall out of working code rather than be declared. Validated
by external review (see §5 for the corrections that review surfaced).

## 1. The cut that was validated

`M·a = f_ext + Jᵀλ`. External forces `f_ext` are the known right-hand
side, **frozen at step start** (the prototype's `external_forces` stands
in for phase-4 `ForceField::accumulate` plus any penalty-joint wrench).
`J` is the rigid-constraint Jacobian from topology; `λ` the multipliers
the solver finds implicitly. Constraints are evaluated on the **trial
state** inside the Newton loop; `f_ext` is not.

This dissolves the lag-1 / implicit tension we were stuck on: the lag-1
gravity snapshot lives entirely in `f_ext`, which never re-enters the
Newton loop. External explicit, constraints implicit, phases do not
cross. Not a compromise — a phase separation.

Validated plumbing: saddle KKT with correct blocks (`M`, `−dt²Jᵀ` top
row, `J` bottom row), Newton convergence on the trial state, the
CoM-free-fall invariant (Σ Jᵀλ = 0 from equal-and-opposite constraint
forces), exact rigid bilateral satisfaction, and the degenerate dispatch.

## 2. Main finding — gather → solve → scatter

The sharper statement (refining "get_index earns its keep"): the solver
works in a **flat dense buffer** (`q`, `v` as one `Vec`) and never touches
body entities inside the loop. Phase 5's real geometry is:

> **gather** (all bodies → flat arrays) → **solve** (pure dense linear
> algebra on the copy) → **scatter** (write back, one pass).

Two consequences close long-open questions:

- **Container layers are distinct.** The solver wants a flat buffer
  *inside*; identity keying is needed only at the **gather/scatter
  boundary**, to translate the `WorldId`-keyed constraint endpoints (the
  graph references bodies by identity) into dense Jacobian rows. The
  requirement is "ordered dense buffer + `WorldId→row` map" — `IndexMap`
  packages both (`get_index_of` ↔ `get_index`), but `(Vec,
  HashMap<WorldId,usize>)` is equivalent. This reconciles the
  supervisor's "flat Vec + separate map" with the IndexMap-superset
  claim: both true, at different layers. Not a hot-loop concern.

- **`get_disjoint_mut` vs `split_at_mut` is a non-question.** Two-phase
  structure means simultaneous `&mut` to two bodies is never needed:
  scatter writes each body exactly once, `values_mut()` suffices.
  Disjoint access returns only under a PGS-style scheme (mutate body
  pairs *during* iteration) — which a global KKT-on-flat-state is not.
  The prototype *structurally eliminated* the question, did not merely
  avoid it.

## 3. Signature conclusion (unchanged decision)

With rigid constraints, phase 5 needs `(bodies, f_ext, graph, dt)` —
strictly more than today's `step_all(bodies, wrenches, epoch)`, which
never sees topology. So the object stops being an *integrator* (time-step
strategy) and becomes a *solver* (constraint resolution + integration).

→ This points to a **new `Solver` trait**, with the per-body explicit
schemes as the empty-graph degenerate case — **not** an extension of
`ImplicitIntegrator`. The `Vec→IndexMap` question we started from is the
wrong axis; the signature change is larger and lives at a different layer.

**But there is still no engine consumer.** Joints are penalty
(`joint.rs:12`, "without solving a DAE"); no contact, no articulated solver.
Therefore: **do not migrate.** Pull the trigger only when a real
rigid-joint / contact feature lands. The prototype makes the shape known
*in advance* so that trigger is cheap, and names the two heavy risks
(§4) before they are hit.

## 4. What the prototype does NOT validate — two named risks

These are the hard parts; leaving them unstated would falsely imply the
path is validated.

- **Risk 1 — PGA constraint Jacobian.** Here bodies are planar point
  masses, `q=[x,y]`, the scalar-constraint Jacobian is trivial. In the
  engine a body is a 6-DoF `Motor`/`Wrench`; a rigid constraint's `J` is
  a map `Twist → scalar` and `Jᵀλ` is a `Wrench` (co-twist). Building the
  constraint Jacobian in PGA and applying `Jᵀλ` as a co-twist is the
  *core* of proposal C ("Jacobian Twist→Wrench") and is exactly what the
  prototype abstracts away. **Unvalidated; needs its own f64-on-clifford
  prototype.** Risk #1.

  Sharper framing for that prototype: this is **not** "rewrite the
  Jacobian for 6 DoF" — it is "the constraint becomes a map between
  tangent spaces." The pose variation is not a `δq` but a left/right-
  invariant shift `δξ ∈ se(3)` (a derivative along `log Motor`); the
  Jacobian row is `∂C/∂ξ` with tangent `Twist`, and `Jᵀλ` assembles into
  a `Wrench` through the *same* duality already working in
  `cotransform` / Ad*. So the pass/fail criterion is concrete: **does the
  PGA constraint Jacobian land on the existing Ad/Ad* machinery in
  `body.rs`** (the machinery already under test)? If it does, that is a
  strong signal the approach is intact. If it demands a separate
  convention, that separation *is* the true cost of proposal C.

- **Risk 2 — bilateral vs contact.** Rigid rods here are *bilateral*
  (equality `C=0`). A contact is *unilateral* (`λ≥0`, complementarity:
  pushes, never pulls). A dense equality KKT solve does not cover it —
  contact needs LCP / active-set, not one linear solve. "Constraints/
  contacts" said with a slash are different animals; the prototype
  honestly closes only the first. If the real goal is contacts,
  complementarity is **unvalidated.** Risk #2.

## 5. Backward Euler is dissipative — a conscious base choice

The fast-path (free bodies) is symplectic Euler. The **solver-path is
backward Euler**: L-stable but numerically **dissipative** — it bleeds
energy. For stiff contact that is usually *desirable* (kills spurious
oscillation). Two caveats to hold consciously:

- **The scheme is island-dependent.** A free body integrates symplectic;
  the moment it joins a constraint island it switches to backward Euler
  and starts damping *all* its DoF, including unconstrained ones (the
  pendulum swing in test 1 slowly decays — the test does not catch it,
  it checks only rod rigidity and that the tip fell). A body's long-term
  energy behaviour **toggles** as it enters/leaves islands. Fine for a
  game; not for orbital accuracy coexisting with constraints.

- **The symplectic-with-constraints answer is RATTLE/SHAKE**, which ties
  back to proposal B (variational/symplectic). It is **not** a drop-in:
  RATTLE changes the solver shape to gather → solve-position →
  **project-velocity** (a second solve enforcing `J·v=0`) → scatter. That
  velocity projection is a second linear KKT solve, so RATTLE costs
  **exactly one extra KKT-resolve per step** over backward Euler — not an
  argument against it, just the explicit price of conservativeness to know
  before the "dissipative contact vs symplectic constraint" fork. If a
  *conservative* constraint solver is ever wanted, backward Euler is the
  wrong base. Same discipline as the `f_ext`-staleness trade: know it,
  don't stumble into it.

## 6. Prototype simplifications (not production)

- **Gauss–Newton** drops the geometric-stiffness term `−dt²·∂(Jᵀλ)/∂q`.
  Negligible here (≤6 iters, moderate dt) and it never affects final
  constraint satisfaction — but it grows with `λ` (stiffness/forcing); a
  tight contact at large dt may need the full term for convergence. Do
  not generalise "the term is small" to the stiff regime.
- **Linear solve** is partial-pivot Gauss. The KKT is symmetric
  *indefinite* (zero bottom-right block); production wants LDLᵀ
  (Bunch–Kaufman) or a Schur-complement reduction.
- **CoM-free-fall invariant** (test 2) holds only for internal `Dist`
  constraints. A `Coord` pin is an *external* ground reaction — with a
  pin the CoM would not fall at `g`. Test 2 correctly excludes the pin;
  this is a boundary of the invariant, not a general property.

## 7. Decision

Unchanged from the thread. Engine stays frozen. The comment fix
(`9702215`) is the only live edit. Findings recorded next to the
prototype; the migration trigger is deferred to a real joint/contact
consumer, at which point the solver shape (§2–3) is known and the two
risks (§4) are already named. When that thread resumes, Risk 1 (the PGA
constraint Jacobian on `clifford` types) is the next prototype, before
any `Solver` trait is written.
