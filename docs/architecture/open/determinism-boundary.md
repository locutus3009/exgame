# Determinism boundary — OPEN

**Status.** Re-opened on 2026-05-24, pending
[fixed-point-vs-f64](./fixed-point-vs-f64.md).

## Principle (unchanged)

The determinism contract covers only the integrated simulation state
(position, velocity, the driving acceleration, proper time).
Everything downstream — rendering, UI, HUD/sensor views, analysis,
logging — reads that state and may freely use `f32` / `f64`.
Non-determinism downstream is harmless because it never feeds back into
the state.

**One authoritative integrator; prediction is secondary.** There is
exactly one true integrator producing the canonical, reproducible
timeline — "exact" means *authoritative and bit-reproducible*, not
error-free. All player-facing prediction is a secondary,
non-authoritative integration that never writes back to canonical
state. Whether a secondary run is seeded from the true state
(perfect knowledge) or from an estimate (navigation uncertainty) is a
**deferred game-design parameter, not an architecture decision** — the
substrate must support both.

## What's in flux

What scalar type defines the boundary, and whether platform-determinism
of an `f64` integration is a buildable contract for this project's
scale. Until [fixed-point-vs-f64](./fixed-point-vs-f64.md) is resolved
the *implementation* of the boundary is undefined; the *principle*
holds.

## Cross-references

- [integrator-purity](../decided/integrator-purity.md) — the
  state-ownership half of the contract is settled and survives any
  representation choice.
- [state-vs-parameters](./state-vs-parameters.md) — corollary: only
  what is integrated is deterministic state; parameters and outputs
  are not.
