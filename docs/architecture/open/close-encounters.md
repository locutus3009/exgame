# Close encounters (`r → 0`) — OPEN

**Original CLAUDE.md deliberation: 8.**

The real frontier: a dynamics problem (softening / regularization),
orthogonal to representation. *Macroscopic contact/collision is the
same frontier* — its architecture is DECIDED (see
[contact-prediction-display](../decided/contact-prediction-display.md));
only the contact-resolution **method** there remains open.

**Now has a reproducing case:** the Earth–Moon L4 probe ejecting via a
*non-contacting* deep lunar flyby.

## Status (from the cutoff thread)

The current gravity integrator is **retained unchanged for now**;
better proximity handling is deferred, conditional ("if even needed"),
must precede the `Q_a`/Lyapunov tuning when undertaken, and is
constrained to the **non-breaking reentry-class step-reduction slot**
(no integrator swap / no canonical-state regularization).

The reentry-class mechanism is the time-warp invariant's "close
encounters force small steps and thereby cap/slow warp" path — see
[decided/vessel-model-rfe](../decided/vessel-model-rfe.md). It
preserves [integrator-purity](../decided/integrator-purity.md) and is
*not* the forbidden variable-step escape hatch. It explicitly must
**not** be an authoritative-integrator swap or a coordinate
regularization of canonical state (those would break the
symplectic-scope and integrator-purity invariants). The
contact-DECIDED re-base/impact operator covers *contacting* events
only; a *non-contacting deep flyby* (the actual L4 ejection) has no
impact-operator slot — its only non-breaking path is the
reentry-class step-reduction.

The L4 ejection itself is **unverified** as physical vs. `r→0`
artifact; classify it by qualitative outcome at matched epoch under
`dt`/precision refinement, not pointwise.

## Constraints under the current code (newton::Mechanism)

The two constraints surfaced when the old `demo/sim.rs` had a
double-buffered `World` are **partially moot** — that pattern is
superseded by `newton::Mechanism`. Re-expressed in newton terms:

1. Any new close-encounter mechanism must be a `Mechanism::step`
   precondition or postcondition (deterministic, state-derived,
   observable at the warp-cost level), not an integrator-internal
   escape hatch.
2. It must not introduce hidden state across the step boundary
   (`Mechanism` owns its state by value; no aliased running state).

The reentry-class step-reduction slot under the time-warp invariant
(see [vessel-model-rfe](../decided/vessel-model-rfe.md)) is the
sanctioned non-breaking path; an authoritative-integrator swap or a
coordinate regularization of canonical state is rejected (would break
the symplectic-scope and integrator-purity invariants).

## Cross-references

- [decided/integrator-purity](../decided/integrator-purity.md) —
  purity invariant constrains any close-encounter mechanism.
- [decided/frame-convention](../decided/frame-convention.md) —
  orthogonal; close-encounter step-reduction does not interact with
  body-frame or LieEuler structure.
- [decided/vessel-model-rfe](../decided/vessel-model-rfe.md) — the
  reentry-class step-reduction slot.
- [decided/contact-prediction-display](../decided/contact-prediction-display.md) —
  the macroscopic-contact face of the same frontier.
