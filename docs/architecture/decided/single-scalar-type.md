# Single scalar type throughout

**Status.** DECIDED 2026-05-24.

## Decision

All physical quantities (positions, velocities, momenta, time,
trajectories, view transforms, shader uniforms) flow through **one**
scalar type, uniformly across every subsystem and every boundary:
CPU substrate (`clifford`, `newton`, `cordic` instantiations), demo
host, future graphics framework, GPU shaders, snapshots, UI
read-back. **No type-conversion at architectural boundaries** is
permitted — only at the final "to display" step (framebuffer pixel
format, etc.) where the GPU itself constrains representation.

The *specific* choice of which type is the load-bearing invariant
is **open**. Current candidate is `f64`. Alternative still on the
table: a Fixed-point type (e.g. `I88F40` via `clifford`'s `Fix`
instantiation backed by the `cordic` crate). The hard invariant is
**uniformity**, not the specific type.

## Rationale

A single uniform type eliminates conversion-at-boundary as a class
of bug:

- **No precision loss at seams.** Every subsystem boundary that
  converts (e.g. CPU `f64` → GPU `f32`) is a precision sink; the
  precision lost there can't be recovered downstream.
- **No representation-mismatch bugs.** When sim runs in `f64` and
  render expects `f32`, "is this position in `f64` or `f32`?"
  becomes a question developers have to answer everywhere. With one
  type, that question vanishes.
- **Substrate-throughout becomes feasible.** If the project's
  geometric substrate (`clifford::pga3` Motor/Twist/Point) flows
  end-to-end, the scalar parameter of those types must flow with it.

This invariant is downstream of and **necessary for**:

- [no-reparenting](./no-reparenting.md) — single global coordinate
  system requires a scalar type with enough precision for the full
  game extent in one origin.
- [[pga-throughout-includes-shader]] memory — the PGA substrate
  reaching the GPU shader requires the scalar type to reach the
  shader too.

## Constraints the chosen type must satisfy

- **Sufficient precision** for the full game-world extent in a
  single origin (Saturn-system scale, sub-metre resolution).
  `f64` gives ~16 digits, easily covers 12 orders of magnitude.
- **Available in the shader language** of the chosen backend. WGSL
  has no `f64`; SPIR-V `Float64` capability is available on most
  consumer hardware. This narrows the backend choice — see
  [[math-substrate-precedes-gpu-api]] memory.
- **Compatible with `clifford`'s `Algebraic` / `Scalar` /
  `Transcendental` traits**. `f64` works out of the box; Fixed-point
  works via `cordic` for transcendentals.
- **Deterministic across platforms** to the extent the project needs
  it. `f64` requires discipline (avoid platform-varying `libm`,
  bound reductions); fixed-point is deterministic by construction.
  See [open/determinism-boundary](../open/determinism-boundary.md).

## What this does NOT decide

- The specific type (`f64` vs `Fixed-point` vs something else). That
  remains an open deliberation; current state in
  [open/fixed-point-vs-f64](../open/fixed-point-vs-f64.md).
- The precision discipline within the chosen type (no-`libm`
  conditioning, reduction order, etc.). See
  [open/determinism-boundary](../open/determinism-boundary.md).

## Cross-references

- [no-reparenting](./no-reparenting.md) — together with this
  invariant defines the precision architecture.
- [core-design-principle](./core-design-principle.md) — uniformity
  of representation extends the "carry the small quantity"
  principle to a "carry everything in one type" stance.
- [open/fixed-point-vs-f64](../open/fixed-point-vs-f64.md) — the
  specific-type-choice deliberation, now scoped by this invariant
  (whatever wins, it must be the SAME type everywhere).
- [open/determinism-boundary](../open/determinism-boundary.md) —
  determinism is downstream of type choice.
