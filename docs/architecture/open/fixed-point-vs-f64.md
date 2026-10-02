# Fixed-point vs `f64` — OPEN

**Original deliberations:** 1, 2, 3, 4 in the pre-refactor `CLAUDE.md`.
The fixed-point thread previously resolved (single `Fix=I88F40`,
`distance` via `strict_hypot`, geometric `Q_a=ACCEL_MIN`,
predictability-horizon criterion) is **re-opened** as of 2026-05-24.

**Re-scoped 2026-05-24** by
[decided/single-scalar-type](../decided/single-scalar-type.md):
whichever type wins this deliberation, it must be **the same one
type throughout** the project (CPU substrate, GPU shader, every
boundary). The decided invariant is uniformity; this file remains
the venue for the f64-vs-Fixed type choice itself.

## Status

On pause. `demo` runs on `f64`. `clifford` supports `Multivector<…, Fix>`
across its multivector / motor / metric layers (proof-of-concept tests
via `I88F40` in `clifford/src/lib.rs`), but the full integration chain
(multivector → motor → integrator → mechanism) did not come together
cleanly because of trait-resolution friction in the `fixed` crate.

## Current likely landing

`type Value = f64` becomes the project's scalar type. Fixed-point
retained in `clifford` as an instantiable scalar for callers that want
it; not the default. Deterministic across platforms is then an
`f64`-discipline question (avoid platform-varying `libm`, no unordered
reductions), not a representation question. See
[determinism-boundary](./determinism-boundary.md).

## What re-opening it later would look like

Solving the `fixed`-crate trait-resolution issues; or replacing the
`fixed` dependency with a hand-rolled fixed-point that matches
`clifford`'s `Algebraic` / `Scalar` / `Transcendental` traits cleanly.

## What was carried over from the prior resolution

- The `strict_hypot`-based distance lesson and the "never form the small
  quantity by subtraction" principle survive intact in
  [core-design-principle](../decided/core-design-principle.md).
- The `Q_a` geometric-cull lesson survives in spirit (parameters are not
  state); see [state-vs-parameters](./state-vs-parameters.md).
- What got reopened is the scalar-representation choice itself.
