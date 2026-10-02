# Legacy crates — `render` and `demo` are learning artefacts

**Status.** DECIDED 2026-05-24. **Update 2026-06-06:** the frozen crates
(`render`, `demo`, and the fixed-point `cordic` / `table-builder`) have since
been **removed from the workspace** — there is no GUI host at present. The
lessons in `lessons/` are preserved; the crate sources are gone. The decision
below is kept as the historical record of why they were frozen first.

## Decision

The workspace crates `render` and `demo`, in the form they reached
at the end of the 2026-05-24 first-iteration cycle, were **frozen
learning artefacts** (and have since been removed, 2026-06-06 — see Status):

- Originally not deleted from the workspace; removed 2026-06-06 once their
  lessons were captured.
- Not on the production path going forward.
- Not iterated for production purposes — only ad-hoc tweaks
  (typos, lint cleanup) are acceptable; substantive engineering
  work goes into the new graphics framework.
- Their decisions are preserved in `lessons/` (see
  [lessons/2026-05-24-render-demo-first-iteration](../lessons/2026-05-24-render-demo-first-iteration.md),
  [lessons/2026-05-24-rendering-backend-wgpu](../lessons/2026-05-24-rendering-backend-wgpu.md),
  [lessons/2026-05-24-render-uses-pga](../lessons/2026-05-24-render-uses-pga.md)).

The future graphics framework will be built in **new crates**
following the project's named-crate convention (e.g. the planned
camera crate `spielberg`). It will start fresh with constraints
derived from the lessons: substrate must reach shader; one scalar
type throughout (see [single-scalar-type](./single-scalar-type.md));
single global coordinate system (see
[no-reparenting](./no-reparenting.md)); raw SPIR-V access (backend
choice itself remains open — see [planned/rendering](../planned/rendering.md)).

## Rationale

The first-iteration `render` + `demo` work hit fundamental
constraint mismatches (WGSL has no `f64`, so PGA-throughout-with-
single-scalar-type couldn't reach the shader; the GPU-upload seam
forced compromises that calcified up into camera design). Further
iteration on the same substrate cannot fix the root cause — only
restart on a different backend / substrate can. See
[lessons/2026-05-24-render-demo-first-iteration](../lessons/2026-05-24-render-demo-first-iteration.md)
for the empirical narrative.

Keeping the crates rather than deleting them serves three purposes:

- **Reference for what didn't work.** "We tried that, here's why it
  didn't" — directly inspectable in code, not just prose.
- **Reference for what did work.** Patterns that survived (snapshot
  bridge structure, HDR pipeline + tonemap, Hillaire 2020 LUT
  layout, immediate-mode UI gating, per-axis camera physics
  invariants) are copyable into the new framework when relevant.
- **Lesson for future agents and the owner.** Reading
  `docs/architecture/lessons/` files with the original code
  alongside is a richer educational artefact than prose alone.

## What "legacy" means in practice

- `cargo test --workspace` must continue to pass — the crates stay
  buildable.
- The crates are not extended with new features. They are not
  refactored to track changes in `clifford` / `newton` unless those
  changes break compilation, in which case minimal porting
  (signature adjustments) is acceptable.
- The new graphics framework does not depend on `render` /
  `demo` — they are independent code paths.
- When the new framework reaches feature parity for the next demo
  scenario, the legacy `demo` can stop being the main launchable;
  whether to remove it from the workspace at that point is a future
  decision (not pre-decided here).

## Cross-references

- [lessons/2026-05-24-render-demo-first-iteration](../lessons/2026-05-24-render-demo-first-iteration.md)
  — main narrative.
- [lessons/2026-05-24-rendering-backend-wgpu](../lessons/2026-05-24-rendering-backend-wgpu.md)
  — wgpu choice, frozen.
- [lessons/2026-05-24-render-uses-pga](../lessons/2026-05-24-render-uses-pga.md)
  — PGA-with-glam-seam choice, frozen.
- [planned/rendering](../planned/rendering.md) — universal renderer
  invariants survive; backend choice reopened.
- [single-scalar-type](./single-scalar-type.md),
  [no-reparenting](./no-reparenting.md) — invariants the next
  iteration must respect.
