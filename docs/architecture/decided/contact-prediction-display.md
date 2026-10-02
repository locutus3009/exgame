# Contact, prediction & trajectory display (DECIDED)

Resolved via Socratic design review, continuing from the vessel
model. Concerns collision/contact and the prediction/display tiers.

- **Contact is a branch-breaking dynamical re-base event.** This
  *extends the Branch rule* from "only player/scripted actions break
  the branch" to also include **dynamically-detected contact**. A
  predicted collision is the dynamical twin of a committed action:
  secondary predictors integrate *only up to* predicted contact and
  **stop** — they never predict *through* it. Contact is resolved by
  the **authoritative core only**, and is permitted to slow warp
  (judged small in practice). You can be shown *that* and *when* you
  will hit, never *what happens after*, in the planning view.
- **The contact resolver is a decoupled state→state operator** (the
  impact map of event-driven non-smooth dynamics) applied at the
  re-base point — alter linear/angular velocities, spawn debris, merge
  bodies — **not woven into the integrator**. Contact is the
  macroscopic face of [close-encounters](../open/close-encounters.md)
  (`r→0`, non-smooth frontier), *not* "a similar thing" to the
  smooth/linear vessel subsystems.
- **Predictors carry a conservative no-contact certificate, not a
  contact prediction.** Bounding volumes are the region *outside*
  which prediction is definitely valid (broad-phase culling /
  conjunction screening). A speculative *branch* terminates when
  bounding volumes intersect; inside the overlap the single
  authoritative-rooted prediction still runs, but **branching
  (speculative parallelism) is suppressed**.
- **Branching is purely a performance optimization** — a speculative
  precompute cache so a warp-change / vessel-switch is instantly
  responsive (CPU speculative-execution / prefetch, matching the
  project's own branch-prediction analogy). It is **not** a
  planning/epistemic search. General principle: **branching value is
  proportional to the action-free horizon**; action-dense regimes get
  no speculative parallelism *by construction*, independent of
  contact.
- **Docking / terminal proximity has no prediction.** The open-loop
  action-free predictor has no horizon there because proximity ops is
  the most action-dense regime (a near-continuous stream of
  branch-breaking inputs). It is flown **manually, closed-loop, on a
  relative-motion (Clohessy–Wiltshire) sensor view** — the prior-art
  universal (KSP/Orbiter/Principia all do this). A *closed-loop
  guidance-aware* predictor (predicting under an assumed guidance law)
  is a different object, **deferred** to game design and tied to the
  epistemic/sensor-view question.
- **The map/trajectory display is a non-authoritative display
  overlay** (see [open/determinism-boundary](../open/determinism-boundary.md)).
  Osculating conics where a dominant primary exists (prior decision);
  they **degenerate at libration points** (an L4/L5 colony) — there
  the *same full model* drives the display, never a cheaper model.
  KSP's lying-conic failure does **not** transfer: there the divergent
  thing *was the state*; here it is display only, so display
  divergence is harmless by construction.
- **The predictor is configurable** — true (bit-exact, same-dt,
  action-free, contact-terminated) / coarse (fast, larger dt,
  divergent) / dispersion tube — all realised as **pure hard re-forks
  from authoritative state every ΔT, retaining the true past points**.
  No state sync, no blend/correction logic (a pure fork of a canonical
  value; zero aliased running-predictor state — consistent with
  [integrator-purity](./integrator-purity.md)). Coarse-mode map jitter
  under warp is accepted cosmetic cost. The bit-exact same-dt fork
  *is* the speculative precompute buffer (one artifact serves
  precompute and display).

## Residual considerations (OPEN)

- **Contact-resolution *method*** — soft-contact vs impulse vs
  complementarity (LCP), plasticity/crumple, bounce-vs-capture-vs-
  fragment, and that solver's fixed-point determinism. Deferred.
- **Epistemic-status rendering & the uncertainty tube.** A tube whose
  width is *dynamically meaningful* cannot come from either predictor
  (the bit-exact fork has *zero* dynamical uncertainty by
  construction; the coarse one is wrong-but-unquantified). It requires
  an explicit **epistemic/sensor-uncertainty tier** (initial-state
  covariance + process/sensor model, propagated by STM or ensemble) —
  a new object distinct from both predictors, and a hook into the
  deferred *perfect-knowledge vs navigation-uncertainty* parameter
  ([open/determinism-boundary](../open/determinism-boundary.md)).
  "Tube tightens as career progresses" = orbit-determination accuracy
  as a progression currency. All **deferred to game-design stage** at
  the owner's call. Note: drawing the forward arc as a dispersion tube
  also makes the hard-refork discontinuity *semantically invisible*
  (a corrected point inside its own predicted spread is consistent,
  not a jump) — honesty and the snap-fix are the same solution.
