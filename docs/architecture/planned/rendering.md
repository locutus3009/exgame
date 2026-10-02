# Rendering — planned subsystem

**Status.** Planned. The only rendering code in the workspace today is
`demo/src/render.rs` (an 800×800 `minifb` window with trails and a
3×5 bitmap-font HUD). This document captures the **universal
invariants** that apply to any rendering layer — current `minifb`
baseline or a future GPU backend — and flags the
raw-Vulkan-access constraint for the eventual backend choice. When a
dedicated rendering crate ships, this file migrates to
`crates/rendering.md`.

## Role

Realize the visualization layer: sim → screen for orbital and vessel
state. Currently a 2D earth-centered view (Earth–Moon + L4 probe).
Eventually a 3D scene supporting the two-view design from
[relativistic-renderer](./relativistic-renderer.md): a physically
correct relativistic *window* and a de-relativized base-frame
*sensor/tactical* view.

## Invariants (universal — apply to any backend)

These are adapted from vitte's `.claude/architecture/planned/rendering.md`
and `.../ui-shell.md`. Each is applicable to the current `minifb`
baseline; each survives a backend change.

1. **The renderer never mutates simulation state.** One-way snapshot
   flow from `newton::Mechanism` to the renderer. This is the
   renderer-side specialization of the determinism boundary
   ([open/determinism-boundary](../open/determinism-boundary.md)).

2. **View-only state is renderer-owned.** Camera pose, zoom, hover
   highlight, UI widget state, frame toggles like `ROTATING_FRAME`,
   trail buffers — none of it round-trips through the sim. The
   current `Trail` ring buffer in `demo/src/render.rs` is the right
   model.

3. **Render passes own their GPU resources.** No god-object holding
   the whole pipeline state. *Cautionary:* vitte's post-mortem on a
   prior renderer cites a 1882-line `App` struct with 52 `Option<_>`
   fields and a manual teardown order commented "to avoid segfault."
   Don't recreate that.

4. **No speculative command-enum variants.** Every renderer command
   has a consumer at land time; no unimplemented placeholders, no
   silent-drop default arm.

5. **Per-frame hot paths do not `.unwrap()` external input.** A stray
   event must not kill the render loop.

6. **UI is a pure consumer of simulation state.** Player inputs
   become commands (when commands are eventually a thing); the UI
   never directly mutates world state.

7. **UI capture / focus gates game input.** Pointer hover and
   keyboard focus on UI widgets are consumed at the UI boundary and
   never forwarded to the sim. (Framework-agnostic principle; the
   `egui::Context::wants_pointer_input()` / `wants_keyboard_input()`
   idiom is one realization.)

8. **Heavy read-only sim → UI data flows on a dedicated subscription
   channel** when the main loop grows past the current
   single-thread model — not on the main event queue.

9. **Per-axis physics camera with key-state API.** Accel / decel /
   max velocity per axis; opposite-push cancellation
   (`North + South → Neutral`); critical-distance auto-decel using
   `v²/(2·decel)`. The push API is **key-state** (`set_pressed(dir,
   bool)`), not impulse-per-frame; otherwise the controller requires
   every-frame re-pushes.

10. **Immediate-mode UI preferred, post-construction widget-ID
    back-patching rejected.** A UI framework where a widget's event
    identity is only available after a post-construction traversal
    (the Linebender `masonry` / `vello` / `parley` family) is
    rejected. Event identity must be available at widget
    construction — the game-state key (a body ID, a button label, an
    action enum) *is* the identity. `egui` exhibits this property;
    so does any reasonable in-house immediate-mode layer.

## Current baseline (`demo/src/render.rs`, minifb)

- 800×800 window via `minifb`.
- Framebuffer + drawing primitives.
- `world_to_screen` transform; `synodic_rotate` for the Earth–Moon
  co-rotating frame.
- `Trail` ring buffer per body, sampled in the active frame.
- 3×5 bitmap font with `format_sim_time` for the HUD.
- Conventions: earth-centered; screen X = +global X; screen Y =
  +global Y; XY plane only (Z ignored); trails recorded in the
  active frame at sample time.

This baseline satisfies invariants 1, 2, 4, 5, 6 today. Invariants 3,
7, 8, 9, 10 do not yet have non-trivial test surface but are stated
upfront so they survive any backend swap.

## Backend choice — reopened

First-iteration choice was `wgpu`, see
[lessons/2026-05-24-rendering-backend-wgpu](../lessons/2026-05-24-rendering-backend-wgpu.md).
That decision **did not survive the iteration**: WGSL has no f64
type, so the chosen substrate (PGA Motor + Twist + f64) couldn't
reach the shader, forcing a seam compromise that calcified upward
into camera design. The first-iteration `render` and `demo` crates
are frozen learning artefacts ([lessons/2026-05-24-render-demo-first-iteration](../lessons/2026-05-24-render-demo-first-iteration.md)).

The backend choice is now **reopened** for a future Socratic
session under stronger constraints: substrate must reach shader;
scalar must be f64 throughout; raw access to SPIR-V is required.
Almost certainly NOT wgpu next time.

UI framework choice follows from the backend; deferred until the
new backend lands.

## Cross-references

- [open/determinism-boundary](../open/determinism-boundary.md) —
  renderer-side specialization.
- [planned/relativistic-renderer](./relativistic-renderer.md) —
  two-view design, HDR, aberration.
