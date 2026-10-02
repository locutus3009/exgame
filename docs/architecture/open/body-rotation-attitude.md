# Body rotation / attitude representation — OPEN

**Original CLAUDE.md deliberation: 7.**

## Status

Structurally answered. Orientation is the even-grade versor of the PGA
motor (`clifford::pga3::Motor`), integrated via `Motor::exp` of a
body-frame `Twist`. Angular momentum lives in body-frame as part of
`RigidBody::momentum` (see
[decided/frame-convention](../decided/frame-convention.md)).

What remains open:

1. **Precision representation for the rotational component** under the
   current `f64` scalar — small-angle / log-rotation representations,
   secular-drift mitigation under LieEuler. The "carry the small
   quantity" principle applies to angular state too: large angular
   intervals integrated as composed small `Motor::exp(B·dt)` factors
   rather than re-derived from a large total angle.
2. **Any future fixed-point return** (depends on
   [fixed-point-vs-f64](./fixed-point-vs-f64.md)). Under fixed-point,
   the `Motor::exp` argument's representation and the choice of
   re-normalisation frequency become bigger questions.

## Cross-references

- [crates/clifford](../crates/clifford.md) — Motor / Twist / Wrench
  definitions.
- [crates/newton](../crates/newton.md) — RigidBody and integrator usage.
- [decided/frame-convention](../decided/frame-convention.md) — chosen
  frame.
- [decided/core-design-principle](../decided/core-design-principle.md) —
  the "carry the small quantity" lens applied to angular state.
