# Vessel model & integration architecture (DECIDED)

Resolved via Socratic design review (deliberation 10 in the
pre-refactor `CLAUDE.md`). The owner's background (BMSTU SM1; PhD in
mathematical modelling of solar-radiation-pressure on large flexible
space structures) is why these map directly onto Rigid Finite Element
/ floating-frame practice — engage at research-peer level here, do not
re-derive fundamentals.

- **No general constraint-solver multibody.** MSC Adams / full
  Euler-multibody jointed simulation is **rejected** — too costly, and
  community practice (Kerbal Joint Reinforcement) confirms players
  *remove* such DOF, not integrate them better. Same status as the
  symplectic-ordering exclusion.
- **Vessel = Rigid Finite Element (RFE) model.** A few rigid
  super-bodies with precomputed mass properties (including
  configuration-parameterized, e.g. inertia `I(L)` for deployment),
  connected by a small explicit joint set. **It degenerates to a
  single rigid body when there are no compliant elements** — a simple
  rocket is one rigid body, zero overhead. Consequence: the active
  vessel and a distant object are the *same model at different element
  counts*; active⇄background is a fidelity-of-discretization change,
  **not a model swap** — there is no KSP on-rails seam / kraken.
- **Joint taxonomy (DECIDED).** Two categorically distinct kinds, the
  floating-frame reference-motion vs elastic-deformation split:
  - **Elastic nodes** — small *linear* relative motion, stiff
    spring+damper, advanced by the **exact closed-form damped-linear-
    oscillator propagator** (unconditionally stable at any step,
    deterministic in fixed-point, zero substep, operator-splitting-
    compatible with the symplectic gravity core) (implemented as
    `newton::Joint`, penalty method with spring + damper; see
    [crates/newton](../crates/newton.md)). This *is* the core
    "carry the small quantity" principle: the joint deflection is the
    first-class integrated small quantity, never formed by subtracting
    large rigid configurations.
  - **Kinematic (articulation) joints** — large prescribed/servo
    motion modelled **linearly** (commanded ramp / constant rate),
    *not* a stiff spring, analytically propagated.
  - The single element that is *simultaneously* large-articulating and
    elastically compliant is **rejected** — decomposed into rotators +
    elastic nodes. **No nonlinear continuum FEM/ANCF. No fixed-basis
    global linear modal** (articulation breaks linearity).
- **Articulation back-reaction.** A **constant-rate symmetric rotor**
  (artificial-gravity torus about its symmetry axis): constant inertia
  contribution, constant momentum bias → pure linear kinematic DOF,
  **no transient, no warp penalty** — the common case. A **transient
  asymmetric slew** (manipulator) carries a nonlinear inertia
  back-reaction onto carrier attitude (the free-floating-manipulator /
  Shuttle-RMS attitude-disturbance problem); it is a rare scripted
  event and is **allowed to slow warp** (coupled small-step during the
  slew), like reentry.
- **Integrator is per-subsystem split (operator splitting), not
  global.** The symplectic semi-implicit Euler invariant is **scoped
  to the gravitational COM translation only** (long-term orbit
  stability). Non-conservative perturbations (continuum/low-Kn drag,
  SRP) and the about-COM (attitude/structural) subsystem use their own
  integrators. Explicit translational(COM) / rotational(about-COM)
  decomposition.
- **Integration class is a per-body attribute**, fixed between
  branch-breaking topology events. *Near-conservative* bodies
  (planets, low-A/m craft): symplectic, large steps. *Perturbation-
  dominated* bodies (solar sails, high A/m): non-symplectic, the
  dominant secular force is non-gravitational. A stow→deploy event
  flips a body's class (it is branch-breaking).
- **Non-gravitational forces are deliberately low-fidelity
  (scope-out).** Cannonball / A-m SRP; simplified mass-to-area
  aerodynamics. **No YORP, no SR-stabilized fine attitude coupling
  (no JWST-grade), no shape-coupled secular SRP** (even though that is
  the owner's thesis domain — explicitly out of scope). Atmosphere =
  **per-planet altitude cutoff** (KSP-style, possibly higher); above
  it no drag / no long-term perturbation. The cutoff is a
  deterministic state-derived threshold that drops a sub-threshold
  term *exactly* — same principle as the gravity precision cutoff.
- **Uniform simulation — no remote/local vessel distinction, no
  on-rails, no attention-dependent physics.** One force law for every
  body; the fidelity tier is a **deterministic state-derived
  predicate** (altitude vs cutoff, A/m, proximity), *never* control
  status. "Under control" only injects branch-breaking control inputs;
  it never changes the force law. (Principia's defining choice — the
  thing that buys the determinism contract; dissolves the
  controlled⇄remote seam entirely.)
- **Time-warp is an emergent, deterministic precision resource**, not
  a free global slider. Achievable warp is a function of the bodies
  present: near-conservative bodies take huge symplectic steps; linear
  elastic nodes use the exact propagator at any step; only
  stiff-nonlinear / perturbation-dominated regimes (reentry/aero
  burn), transient asymmetric articulation slews, close encounters,
  and branch-breaking config events (inertia recompute) force small
  steps and thereby cap/slow warp. Warp auto-reduces where fidelity
  demands it. **No LOD / freezing / variable-step escape hatch** (all
  are model-switch or determinism-breaking, already rejected).

## Residual considerations (still OPEN under this resolved model)

- **RFE discretization as a topology parameter.** Element count is
  simultaneously the mode-fidelity dial, the stiffness spectrum, the
  fork-state size, and must stay consistent across branch-breaking
  topology events. Who authors it and whether it may vary along the
  timeline is unresolved.
- **Continuous reconfiguration vs the branch model.** A slow
  continuous deployment / truss extension is a structure change that
  is *not* an instantaneous branch-breaking action — it occurs *within*
  an action-free interval. Partially handled (allowed to slow warp;
  inertia coupling carried for asymmetric transients), but its
  interaction with branchable-history / basis-evolution is still a
  consideration.

(The merge/split body-identity bookkeeping residual previously listed
here has been **closed** by `Mechanism::split` / `Mechanism::merge` in
the `newton` crate; see [crates/newton](../crates/newton.md).)

## Cross-references

- [frame-convention](./frame-convention.md) — body-frame momentum is
  the reason anisotropic `I(L)` is cheap in body-frame and expensive
  in world-frame, making this decision load-bearing in the long run.
- [integrator-purity](./integrator-purity.md) — operator-splitting
  scope.
- [crates/newton](../crates/newton.md) — current implementation:
  `Joint` (penalty-method elastic node), `Mechanism::split`/`merge`,
  per-body integrators.
- [open/close-encounters](../open/close-encounters.md) — the
  reentry-class step-reduction slot referenced under time-warp lives
  here.
