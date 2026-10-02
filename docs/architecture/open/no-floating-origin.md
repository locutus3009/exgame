# No floating origin / re-basing — SUPERSEDED

**Status.** **Superseded 2026-05-24** by
[decided/no-reparenting](../decided/no-reparenting.md). The original
"closed by policy" stance has been promoted to a hard architectural
invariant generalised across all reparenting variants (floating
origin, chunked-world, re-basing) under one rule.

This file is preserved as archeology — it captures the original
fixed-point-centric motivation. The new decided/ file generalises
the invariant to any scalar type satisfying the precision
constraint, decoupling the rule from its fixed-point origin.

---

## Original content (archeology)

All-integer / deterministic representation was the goal. Integer
position differencing is exact — catastrophic cancellation is a
floating-point problem, not an integer problem. No floating-origin
escape hatch is introduced; no rebasing of coordinates is performed
during simulation.

*Caveat (now resolved by the decided/ file).* The fixed-point
representation that motivated this policy was itself in flux
([fixed-point-vs-f64](./fixed-point-vs-f64.md)). If the project
lands on `f64` as the scalar type, the no-floating-origin rule
loses its "integer differencing is exact" backing — the new
[decided/no-reparenting](../decided/no-reparenting.md) replaces
that backing with "the scalar type carries enough precision in
single origin" and pushes the determinism question into
[open/determinism-boundary](./determinism-boundary.md).

*Original CLAUDE.md deliberation: 6.*
