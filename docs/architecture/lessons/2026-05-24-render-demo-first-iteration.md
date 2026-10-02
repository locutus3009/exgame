# Render + demo first iteration — lessons

**Status.** Lessons-only artefact. Captures what was learned from the
2026-05-24 brainstorm → spec → plan → implementation cycle for the
`render` and `demo` crates. The crates themselves are frozen as
learning artefacts (not deleted, not on the production path); a
future iteration will build the graphics framework from scratch with
the lessons applied. Future architectural decisions (ash backend,
PGA-in-shader, etc.) belong to that future iteration's design
documents, not here.

**Date.** 2026-05-24 brainstorm → 2026-05-24 implementation freeze.

## What was attempted

A first-slice renderer for the Earth-Moon demo: 3D camera (free
orbit), atmospheric scattering (Hillaire 2020) on Earth, trajectory
rendering, time-warp UI. Architecture decisions made up-front:

- `wgpu` as backend ([lessons/2026-05-24-rendering-backend-wgpu](./2026-05-24-rendering-backend-wgpu.md),
  originally `decided/rendering-backend.md`)
- PGA `Motor`/`Twist`/`Point` throughout scene/camera/frame logic,
  `glam` Mat4/Vec3 **only** at GPU upload boundary
  ([lessons/2026-05-24-render-uses-pga](./2026-05-24-render-uses-pga.md),
  originally `decided/render-uses-pga.md`)
- raw `wgpu`, no higher-level framework
- WGSL shaders (compiled from string sources)

Implementation reached: assets crate skeleton, render crate with
geometry / camera / frame / snapshot / bodies / sprites / trails /
atmosphere (Hillaire 2020 — transmittance + multi-scatter + sky-view +
aerial-perspective + sky_render) / tonemap / egui passes, demo crate
wired around winit. 142 tests pass.

Atmospheric scattering visually does NOT work as designed: jitter at
close approach, atmospheric effect invisible from typical viewing
distances, sun-disc stability issues. Root causes section below.

## Decisions made — survived / didn't

| Decision | Outcome |
|---|---|
| `wgpu` backend | **Did not survive.** Future iteration uses `ash`. WGSL/wgpu can't host f64 PGA substrate in shader. |
| PGA throughout scene/camera/frame | **Survived in spirit, broke in detail.** Orbit-camera was forced off Motor+Twist onto spherical (yaw/pitch/distance) because the seam at GPU upload prevented carrying the camera composition end-to-end. |
| Snapshot read-only bridge from Mechanism | **Survived.** Read-only one-way flow from `newton::Mechanism` to renderer is clean and the right pattern. |
| HDR pipeline + ACES tonemap | **Survived as principle**; specific RGBA16Float intermediate buffer + ACES filmic + sRGB conversion is correct architecture for any future graphics path. |
| Hillaire 2020 atmospheric scattering | **Survived as model choice**, didn't survive in implementation — see bug patterns below. |
| egui for UI | **Survived for first-iteration UI** if we stay on wgpu-or-similar; if future framework is raw ash + custom shaders, egui-wgpu binding will need a swap to egui-ash or a custom immediate-mode UI. |
| `f64` scalar throughout the CPU substrate | **Survived and strengthened.** Future invariant: f64 *everywhere* including shaders (SPIR-V `Float64` capability). |
| Single-origin world coordinates (no reparenting) | **Strengthened to invariant.** Driving constraint behind needing f64-everywhere. |
| `demo` + `render` as overlapping responsibility | **Did not survive.** See lesson on demo/engine conflation below. |

## Bug patterns encountered

These are tactical bug categories worth remembering as classes, not
just instances.

### (E) Layered shader bugs hide each other

The atmospheric scattering subsystem went through three compounding
bugs, each masking the next:

1. **WGSL implicit bind-group layout dropped unused bindings.**
   `aerial_persp.wgsl` declared `multi_scatter_lut` at @binding(3)
   but never referenced it in the shader body. wgpu's WGSL frontend
   stripped it from the auto-derived layout. The Rust bind-group
   descriptor sent all 6 entries → runtime validation error
   "descriptor has 6, layout has 5". Symptom: panic on first frame.
   Fix: actually use the binding (compute multi-scatter contribution
   per Hillaire 2020); also masked a real correctness gap.

2. **Depth threshold `>= 0.999999` matched distant-body pixels.**
   `sky_render.wgsl` used `depth >= 0.999999` to mean "this pixel is
   sky, not body". With far=1e13 and sun mesh at ~1.5e11 m, the
   sun's depth value was ≈ 0.9999993 — strictly above threshold,
   so sky_render overwrote sun-mesh pixels with sky_view LUT
   contribution (mostly black at orbital altitudes). Symptom: sun
   invisible. Fix: strict `depth >= 1.0` (depth buffer clears to
   exactly 1.0, body pixels are strictly < 1.0 under `Less` depth
   test).

3. **`length(p)` assumed planet at world origin.** Both
   `sky_view.wgsl` and `aerial_persp.wgsl` computed altitude as
   `length(eye) - planet_radius` and `length(p) - planet_radius`,
   silently assuming the atmospheric body sits at world origin.
   Earth is at heliocentric ~1.5e11 m, so altitude was always
   ~1.5e11 — way above atmosphere → LUTs returned nothing. Fix:
   pass `planet_center_world` uniform, work in planet-relative
   coordinates inside the shader.

Each fix unmasked the next. Without ground-truth visual reference
(reference rendering, validated test scene), debugging was
guess-and-fix.

**Mitigation for future iteration:**
- Explicit bind-group-layout validation pass (not relying on
  WGSL's auto-derivation) — disappears naturally with raw SPIR-V
  where bind groups are written manually.
- Spec-level invariants for shader-side code: "every declared
  binding must be used", "depth threshold for sky distinction must
  match exactly the clear value and depth test convention",
  "altitude computation must explicitly carry the planetary body's
  reference position".
- Reference renders or test fixtures for atmospheric scattering
  (compare against Bruneton or Hillaire's original output).

### (H) WGSL implicit layout drop as a bug class

WGSL silently strips unused @binding declarations from auto-derived
bind-group layouts. This is a category-of-bug similar to "struct
padding mismatch": invisible at compile time, runtime validation
catches it. With raw SPIR-V the layout is written explicitly so the
category disappears. Worth remembering as a wgpu-specific gotcha for
the duration any wgpu code remains in the tree (frozen render crate).

## Why `render` and `demo` are frozen learning artefacts

Both crates are explicitly NOT on the production path going forward.
Reasons not to delete them:

- They are working code — useful as "before / after" comparison when
  the new framework reaches feature parity.
- They contain correct implementations of patterns that DO survive
  (snapshot bridge, HDR pipeline, render-pass composition,
  Hillaire 2020 LUT structure) — copyable as reference.
- They contain instructive examples of patterns that DIDN'T survive
  — the spherical-camera departure from
  [2026-05-24-render-uses-pga](./2026-05-24-render-uses-pga.md)
  is exactly the kind of "boundary compromise propagating into
  architecture" that the lesson is about.

Reasons not to keep iterating:

- Fundamental constraint mismatch with future direction (PGA must
  reach shader, requires raw SPIR-V, requires f64 in shader, etc.).
- Further fixes to atmospheric scattering, jitter, precision-at-
  close-approach all run into the f32-Mat4-at-shader bottleneck;
  patches do not address root cause.

Future iteration starts fresh with constraints derived from the
lessons.

## Memory entries that distil these lessons

Durable principles extracted to user-level memory (apply across all
future sessions):

- `pga-throughout-includes-shader` — substrate must reach GPU shader.
- `math-substrate-precedes-gpu-api` — pick scalar + substrate first.
- `precision-is-architectural-not-optimization` — different
  precision architectures have non-transferable tactics; prior art
  only applies when invariants match.
- `demo-vs-engine-conflation-cost` — explicit tagging required.

## What's NOT decided here

This file does NOT formalize:

- The future graphics framework's backend (ash is strongly preferred
  but not yet a `decided/` invariant — future Socratic session
  formalises).
- The future graphics framework's PGA-in-shader implementation
  strategy (apply-only with hand-coded SPIR-V primitives vs
  compile-clifford-subset-to-SPIR-V; both are open).
- When and how `render` and `demo` get removed from the workspace
  (could be soon, could persist as `legacy/` reference for years).

The first-iteration's `decided/rendering-backend.md` (wgpu) and
`decided/render-uses-pga.md` (with the orbit-camera departure note)
have been **moved to this `lessons/` directory** alongside this
file — they were decisions that did not survive, so they are
archeology, not active invariants. Revision is a future session's
work.

## Cross-references

- [2026-05-24-rendering-backend-wgpu](./2026-05-24-rendering-backend-wgpu.md)
  — the first-iteration "wgpu as backend" decision, now a frozen
  lesson; explains the substrate-mismatch that broke it.
- [2026-05-24-render-uses-pga](./2026-05-24-render-uses-pga.md)
  — the first-iteration "PGA throughout, glam at GPU upload" decision,
  now a frozen lesson; the seam compromise was the load-bearing
  flaw.
- [planned/rendering](../planned/rendering.md) — universal renderer
  invariants survive most of the iteration; backend choice reopened.
- [planned/setting](../planned/setting.md) — eventual game setting,
  drives what the new framework eventually has to deliver.
- `docs/superpowers/specs/2026-05-24-render-crate-first-slice-design.md`
  — the spec that drove this iteration, and
  `docs/superpowers/plans/2026-05-24-render-crate-first-slice.md`, the
  plan that executed it. Both were deleted with the abandoned
  `docs/superpowers/` corpus and remain in git history. Resolution of
  what they claimed follows from this lesson file.
