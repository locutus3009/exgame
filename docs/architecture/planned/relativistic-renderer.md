# Relativistic renderer — planned subsystem

**Status.** Planned. Renderer half of the relativistic-layer thread
(physics half: [open/relativistic-layer](../open/relativistic-layer.md)).
This is intentionally a hard problem and is not on a near-term
schedule.

## Two-view design

Two views, both physically grounded:

1. **Physically correct relativistic *window*.** Spectacle and
   navigation hazard. Aberration, Doppler shift, the relativistic
   headlight effect (forward brightening, aft darkness). Applies
   local-frame light transport on top of the scene.

2. **De-relativized base-frame *sensor / tactical* view.** What the
   player actually flies by. Renders from base-frame state directly.

Same problem, two hats. Maps onto the base ↔ local frame
architecture from [open/relativistic-layer](../open/relativistic-layer.md):
the sensor view renders from base-frame state; the window applies
local-frame light transport on top.

The two-view design is **likely mandatory for playability** — a
purely relativistic window is unfly-able, a purely de-relativized
view forfeits the spectacle and the navigation-hazard signaling.

## Aberration breaks linear rasterization

Straight edges curve in the window view. Linear rasterization
pipelines do not handle this without per-pixel work. The
high-value tractable target is the **analytic relativistic
starfield** — per-star direction transformation + blackbody-temperature
Doppler shift + `D⁴` beaming — rendered by ray-casting the unit
sphere, not by rasterizing meshes.

HDR rendering and spectral handling are **non-negotiable** for a
"useful" relativistic visualization; clipping to sRGB at the end of
the pipeline forfeits the physical content (the headlight-effect
brightening near c is many decades of dynamic range, and the
Doppler shift across a wide blackbody is intrinsically spectral).

Design deliberately; do not bolt onto the current `minifb`
renderer. The aim is a *useful game*, not a primitive-shapes demo.

## No covariant scalar gravity

Nordström-style scalar covariant gravity is observationally falsified
(no light bending, wrong perihelion sign). Don't pursue. 1PN scalar
add-ons (`dτ/dt = 1 − v²/2c² + V/c²`) are local-frame corrections,
not new gravity, and live on the physics side
([open/relativistic-layer](../open/relativistic-layer.md)).

## Renderer is downstream of state

The renderer reads from state; it never writes back. This is the
[determinism boundary](../open/determinism-boundary.md) applied to
the relativistic renderer specifically. Aberration, Doppler,
headlight — all of them are read-only transforms on what the sensor
view shows; the simulation state is unchanged.

## Cross-references

- [open/relativistic-layer](../open/relativistic-layer.md) — physics
  half (rapidity-carry, 1PN add-ons, EIH gating).
- [planned/rendering](./rendering.md) — universal invariants the
  relativistic renderer must satisfy (especially backend choice
  with raw Vulkan for HDR / ray-tracing / mesh shaders).
- [open/determinism-boundary](../open/determinism-boundary.md) —
  renderer is downstream of state.
