# Render uses PGA throughout — glam only at GPU boundary (frozen lesson)

**Status.** Decision made 2026-05-24 during the first render/demo
brainstorm; orbit-camera departure note added later in the same
iteration. **Did not survive** — the GPU-boundary compromise turned
out to be the load-bearing flaw, not the camera framing. The
correct generalisation (per
[[pga-throughout-includes-shader]]) is *PGA throughout including
shader*; the orbit-camera departure was a downstream symptom of the
boundary seam, not an architectural choice to formalise. See
[2026-05-24-render-demo-first-iteration](./2026-05-24-render-demo-first-iteration.md)
for the narrative.

Preserved as lesson artefact, not active invariant. Future graphics
framework will keep "PGA throughout" as a stronger invariant *that
extends into the shader*, not stopping at GPU upload.

**Originally captured as** `decided/render-uses-pga.md`; moved to
`lessons/` once the seam-compromise turned out to be the actual
mistake. Content below is the original document, including the
orbit-camera departure note that was added near the end of the
iteration — both preserved for archeology.

---

## Decision

The `render` crate operates in `clifford::pga3` types throughout —
`Motor`, `Twist`, `Point` (and `Wrench` if ever needed, though
unlikely for rendering). Plain-3D types (`glam::Vec3`, `glam::Mat4`,
`glam::Quat`) appear ONLY at the point of writing uniform / vertex
/ instance data to GPU buffers, never in scene, camera, frame, or
animation logic.

## Rationale

- The physics substrate (`clifford`, `newton`) already speaks PGA.
  The snapshot boundary becomes trivial — no conversion, no
  impedance mismatch, no risk of "two representations slightly
  diverging".
- *(Originally) camera state is naturally a `Motor`, velocity a
  `Twist`, integration is `pose = velocity.exp(dt) * pose`. This
  framing was reversed for the **orbit camera** in commit `04e604e`
  after the first-slice UX revealed that 6-DOF Twist integration
  applied screws in the world frame (left-composition) rather than
  the camera-local frame, breaking WASD / dolly / focus-snap feel.
  The orbit camera now uses explicit spherical state (yaw, pitch,
  distance, focus_body_idx); the view Motor is constructed from this
  state on demand via `glam::Mat4::look_at_rh` composed with the
  active frame's `Motor`. Scene, frame, snapshot, trails — all stay
  in PGA. Free-flying or vessel-attached cameras added later may
  reasonably revisit the Motor+Twist substrate; orbit-camera UX
  specifically does not benefit from it.*
- Reference frames (Inertial, Synodic, BodyLocked, BodyCoRotating)
  each produce a `Motor`; world↔frame conversion is Motor
  composition / inverse. The frame enum's
  `fn motor(&self, snapshot: &Snapshot) -> Motor` is the entire
  abstraction.
- Trail samples store as `Point`s; conversion to `Vec3<f32>` happens
  only inside the trail GPU upload pass.
- "Carry the small quantity, never form it by subtraction" —
  [core design principle](../decided/core-design-principle.md) — extends
  naturally: rapidity, body-frame momentum, AND camera deltas all
  live as their Lie-algebra elements (`Twist`s) instead of decomposed
  Euler triples. Camera state is one more substrate where PGA pays
  off.
- Avoids the "two-paradigm" risk the owner explicitly flagged for
  async ([[code-hygiene-deps-and-async]]): same principle, different
  axis — PGA throughout, no mixed PGA + Euler representations.

## What "GPU boundary" means concretely

A small `render::gpu_bridge` module exposes the conversion helpers:

- `motor_to_mat4(m: Motor<f64>) -> Mat4` — view / model / projection-
  composed matrices written to uniform buffers.
- `point_to_vec3(p: Point<f64>) -> Vec3` — positions written to
  vertex / instance buffers and uniform fields.
- `twist_to_vec6(t: Twist<f64>) -> [f32; 6]` — if a twist ever
  needs to reach a shader (rare; debug visualization at most).
- f32 conversion lives in these helpers (PGA is f64 in scene; GPU is
  f32 in uniforms).

These functions are the ONLY place `glam` types appear. Scene,
camera, frame, snapshot, trails, animation code — all PGA.

## What stays in `glam`

- The `Mat4` / `Vec3` / `Quat` types themselves — for GPU buffer
  layout (bytemuck-friendly column-major float, well-tested SIMD).
- The dep is justified as **peripheral infrastructure** per
  [[engine-and-game-not-universal-engine]]'s "core vs peripheral"
  dependency principle: the core (transforms, composition) stays
  ours via PGA; only the bytemuck-shaped GPU layout types are
  outsourced.

## What this does NOT decide

- Whether the renderer keeps f64 or downgrades to f32 internally
  (PGA generic over scalar; this file says scene-side is PGA but
  the scalar choice is open — current pose is `f64`, GPU is `f32`,
  so f64 in scene + f32-cast at upload is the obvious answer).
- Whether `glam` is the right GPU-side library (could swap to
  another bytemuck-friendly Vec/Mat library; immaterial to this
  invariant).

## Cross-references

- [2026-05-24-render-demo-first-iteration](./2026-05-24-render-demo-first-iteration.md)
  — narrative + why this decision is now a frozen lesson, not
  an active invariant.
- [2026-05-24-rendering-backend-wgpu](./2026-05-24-rendering-backend-wgpu.md)
  — the wgpu backend choice this PGA-with-glam-seam decision
  silently depended on (WGSL has no f64, so PGA couldn't reach the
  shader; the seam was forced by backend, not by design).
- [decided/frame-convention](../decided/frame-convention.md) — body-frame
  momentum integration in physics, still active.
- [decided/core-design-principle](../decided/core-design-principle.md)
  — carry the small quantity; future "PGA-throughout-including-shader"
  invariant extends this principle to camera deltas, frame
  transforms, AND shader-side composition.
