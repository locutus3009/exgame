# Rendering backend — wgpu (frozen lesson)

**Status.** Decision made 2026-05-24 during the first render/demo
brainstorm. **Did not survive the iteration** — see
[2026-05-24-render-demo-first-iteration](./2026-05-24-render-demo-first-iteration.md)
for why. Preserved as lesson artefact, not active invariant. The
crate `render` (workspace member) was built against this decision
and is also frozen as a learning artefact. Future graphics
framework's backend is open and will be decided in a future
Socratic session, almost certainly NOT wgpu.

**Originally captured as** `decided/rendering-backend.md`; moved to
`lessons/` once it became clear the choice did not hold up against
the substrate requirements (f64 + PGA-in-shader). Content below is
the original document as written, for archeology.

---

## Decision

`wgpu` is the rendering backend, used at the **raw level** —
the renderer crate owns its own scene representation, its own
asset and scripting systems, and its own scheduler, and bridges
directly to `newton::Mechanism` for physics state. No higher-level
framework (Bevy, rend3, kajiya, etc.) sits between application
code and `wgpu`. Cultural fit: the project hand-builds its own
substrate everywhere (`clifford`, `newton`, `cordic`, custom
fixed-point); the renderer crate follows the same pattern.

## Rationale

The raw-Vulkan motivators originally flagged in
[planned/rendering](../planned/rendering.md) were:

- Hardware ray tracing for gravitational lensing (relativistic).
- Mesh shaders for the analytic relativistic starfield.
- Exotic HDR surface formats.

All three live on the relativistic-renderer track, which is deferred
to a distant horizon per the same brainstorm session. None of the
non-relativistic effects in scope — atmospheric scattering, ring
scattering, volumetric storm clouds, planetary surface BRDF,
heightmap-based terrain LOD — requires hardware RT, hardware
tessellation, or mesh shaders in production-grade real-time
renderers. The established prior art (Bruneton 2008, Hillaire 2020,
Schneider/Nubis, Outerra-style CDLOD) is fragment + compute +
raymarching against precomputed LUTs and density fields, all of
which wgpu covers natively.

Pipeline-overhead concerns are non-binding at our draw-call scale
(tens to low hundreds; wgpu's validation tax becomes visible at
~10 000+).

## What this does NOT decide

- UI framework choice (`egui` is the natural fit on wgpu, but the
  explicit decision is deferred to the spec).
- Atmospheric model (Bruneton precomputed vs Hillaire dynamic vs
  Nishita single-scattering).
- HDR pipeline contract (linear-HDR + tonemap vs sRGB straight).
- Modularity axis of the renderer crate.
- Scene representation (retained vs immediate vs hardcoded).
- Crate boundary (`crates/rendering` vs in-`demo`).

## Documented ceilings of wgpu

For future reference, the techniques wgpu does NOT expose natively:
hardware tessellation shaders, mesh shaders, hardware ray tracing
(`VK_KHR_ray_tracing`), exotic descriptor-indexing patterns,
BAR-mapped device memory, sparse residency. None bind for any
in-scope effect.

## Escape hatch

If a future requirement binds a wgpu ceiling, `wgpu-hal::api::Vulkan`
exposes raw `ash::Device` and raw `vk::Image` / `vk::Buffer` handles
for the one binding pass (the "Hybrid" option from
[planned/rendering](../planned/rendering.md)'s backend section). This
is reliable on Linux + Vulkan backend; cross-platform requires
per-backend escape code (`wgpu-hal::api::Metal` on macOS,
DX12-or-forced-Vulkan on Windows). Migration cost: ~200-500 lines of
bridge code per escape pass.

## Cross-references

- [2026-05-24-render-demo-first-iteration](./2026-05-24-render-demo-first-iteration.md)
  — narrative + bug-pattern archeology + why this decision is now
  a frozen lesson, not an active invariant.
- [2026-05-24-render-uses-pga](./2026-05-24-render-uses-pga.md) —
  the PGA-throughout-with-glam-seam decision, also frozen lesson;
  the wgpu choice forced the seam by lacking f64 in WGSL.
- [planned/rendering](../planned/rendering.md) — backend choice
  reopened, points back here for the wgpu attempt's archeology.
- [planned/relativistic-renderer](../planned/relativistic-renderer.md)
  — when this track re-activates, re-evaluate: the deferred
  raw-Vulkan motivators (hardware RT, mesh shaders) are real on that
  track.
- [open/determinism-boundary](../open/determinism-boundary.md) —
  renderer is downstream of integrated state; backend choice does
  not touch the determinism contract.
