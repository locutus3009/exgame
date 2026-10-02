# Integrator purity (DECIDED)

The authoritative integrator **MUST be a pure function of a
cheaply-clonable canonical state value**: no hidden globals, no
internal mutable/cached state that aliases between the authoritative
timeline and a prediction, no shared RNG. The canonical world is a
plain forkable value type; the trajectory/history must be **branchable**
(provisional futures hanging off the authoritative past, overwritten by
truth as time advances). This is what makes parallel secondary
predictions, replan-often, prediction-as-progression, and the deferred
perfect-knowledge-vs-uncertainty choice all *possible later*. It is
**not deferrable**: forfeiting purity once cannot be retrofitted. Treat
any introduction of hidden/aliased state into the integrator as a
regression, the same as reordering the symplectic update.

## Branch rule (DECIDED): only dynamics is branch-simulated; actions break the branch

A secondary prediction integrates *dynamics only* forward from a
snapshot. Any player or scripted action (a burn, staging, docking, a
commanded attitude change — any input) is a **branch-breaking event**:
it invalidates the speculative branch exactly like a CPU branch
misprediction flushes the pipeline. The model is literally CPU branch
prediction — speculate the action-free dynamical future; on any
committed action, discard the speculative branch and re-snapshot /
re-predict from the new authoritative state. Prediction is only ever
valid over the action-free interval; a committed action is the
explicit re-base point.

Contact (dynamically-detected, not player-initiated) is also a
branch-breaking re-base event; see
[contact-prediction-display](./contact-prediction-display.md).

## Symplectic ordering, where it applies

Symplectic semi-implicit Euler ordering (velocity, then position from
the *new* velocity) is intentional **for the gravitational COM
translation only** — long-term orbit stability rests on it. Do not
reorder or "tidy" it into explicit Euler in the gravity core; doing so
leaks energy and destroys orbit stability. Non-conservative
perturbations and about-COM (attitude / structural) subsystems use
their own integrators (operator splitting); see
[vessel-model-rfe](./vessel-model-rfe.md).

## Current realization: `newton::Mechanism`

`newton::Mechanism` owns its bodies and joints by value; this is the
current realization of the invariant. The predecessor pattern —
double-buffered `World` behind RAII guards in old `demo/sim.rs`
(landed in commit `81b5026`) — was superseded by the migration to
`newton::Mechanism` (commit `ca91c7a` onward). The *principle*
(exactly-once flush, no aliased state across the integrator boundary)
carries forward; the *implementation* does not. Do not reintroduce the
old guard pattern.

`Mechanism::split` and `Mechanism::merge` are the topology operations
that close what was previously listed as the merge/split
body-identity-bookkeeping residual. See
[crates/newton](../crates/newton.md).

## Cross-references

- [frame-convention](./frame-convention.md) — body-frame state
  ownership is a precondition for the purity invariant under the
  current dynamics layer.
- [vessel-model-rfe](./vessel-model-rfe.md) — operator-splitting
  scope: symplectic ordering is gravity-COM only.
- [open/close-encounters](../open/close-encounters.md) — any
  close-encounter mechanism must respect the purity invariant; no
  integrator-internal escape hatch.
