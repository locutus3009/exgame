# No reparenting — single global coordinate system

**Status.** DECIDED 2026-05-24.

## Decision

All world positions are expressed in **one** global coordinate
system, used uniformly across the project. No floating-origin
escape hatch, no chunked-world tiling, no re-basing of coordinates
during simulation. The single origin is permanent for the
lifetime of a session.

## Rationale

The motivation is **architectural clarity and substrate uniformity**:
chunked / floating-origin / re-parenting schemes introduce a class
of "which chunk am I in?" / "is this position absolute or
chunk-relative?" boilerplate that infects every cross-subsystem
boundary (sim ↔ render ↔ UI ↔ networking ↔ persistence). The
overhead is paid forever, in every newly-added subsystem.

Eliminating reparenting requires a scalar type with sufficient
precision to express the full game scale in a single origin —
see [single-scalar-type](./single-scalar-type.md).

## Implications

- The scalar type **must carry enough precision** to express
  positional values across the full game-world extent (Saturn-system
  scale for [planned/setting](../planned/setting.md): ~1.5×10¹² m at
  the outer planets, sub-metre resolution on planetary surfaces ⇒ ~13
  decimal digits, which `f64`'s ~16 digits comfortably covers).
- Precision tactics from re-parenting architectures (camera-relative
  rendering, chunk-local physics, etc.) **do not apply** to us in
  general — see
  [[precision-is-architectural-not-optimization]] memory.
- The renderer's view-matrix composition must keep precision in
  the scalar type all the way to the GPU shader; this is why we
  need [single-scalar-type](./single-scalar-type.md) to be
  load-bearing in the shader too.
- Determinism: integer-differencing-is-exact (the original rationale
  for fixed-point + integer coordinates) loses its absolute footing
  under `f64`, but the practical equivalent is "platform-deterministic
  `f64` arithmetic discipline" — see
  [open/determinism-boundary](../open/determinism-boundary.md).

## Prior art and contrast

- Outerra, SpaceEngine, KSP+Principia, Star Citizen — all reparent
  via chunked origin, floating origin, or hybrid. Their precision
  tactics work because reparenting keeps local coordinates small
  enough for `f32`. Our choice is the opposite trade.
- REBOUND (N-body integrator) uses `f64` for absolute positions
  across solar-system scale without reparenting. Validates the
  precision side; REBOUND has no renderer, so it doesn't address the
  graphics side.

## Cross-references

- [single-scalar-type](./single-scalar-type.md) — the precision
  invariant that makes no-reparenting feasible.
- [core-design-principle](./core-design-principle.md) — carry the
  small quantity; reparenting would be exactly the kind of "subtract
  to make small" hack this principle rejects.
- [open/no-floating-origin](../open/no-floating-origin.md) —
  precursor deliberation, **superseded** by this invariant.
- [open/determinism-boundary](../open/determinism-boundary.md) — the
  question of how to make `f64` arithmetic platform-deterministic
  remains open; that is the implementation question, not the
  policy question.
