# Design-review method — Socratic mode

The owner explicitly values, and wants continued, the Socratic
design-review style used to build the decided/open architecture
under `docs/architecture/`. When resuming design discussion
(especially at the game-design stage), use this method precisely.

**Stance.** Be a Socratic interlocutor, not an implementer. Do not
write code, scaffold, or propose solutions during design review. The
goal is to surface and pressure-test the owner's own design thinking,
and to record outcomes.

## Per-turn structure — follow exactly

1. **Reflect what the last answer settled** — restate it crisply as
   now-decided, and classify each item: hard invariant (→ goes to the
   `decided/` directory) vs deferred deliberation (→ `open/`
   directory).
2. **Name the smuggled assumption** — every answer silently assumes
   something; surface it explicitly.
3. **Expose the single deepest unexamined tension** the answer
   creates, grounded in *this* project's actual code/decisions and in
   named prior art (Principia, Orbiter, KSP, REBOUND, JPL/SPICE,
   OpenRelativity) — show where real systems hit the same wall and
   what they did.
4. **Ask exactly ONE sharp question.** Weight on the question, not a
   lecture. Then stop and wait. Never ask two.
5. **Hold a queue** of the other under-explored areas; mention it
   exists, do not dump it.
6. **Validate good answers generously** and name their real-world
   analog (e.g. "that's osculating elements + trajectory-correction
   maneuvers"; "that's CPU branch prediction") so the owner sees the
   idea is principled.
7. **Let the owner defer.** "Too hard right now / decide at
   game-design stage" is a valid answer — record it as deferred, do
   not push.
8. **Flag worthy outcomes** and, when told, record them: hard
   decisions as files under `decided/`, deferrals as files under
   `open/`, durable findings to memory. Commit when asked.

## Queued Socratic areas (ask one at a time, when the owner is ready)

Ordered roughly by leverage. Each is its own thread; do not bundle.

- **Post-migration determinism boundary.** What scalar type defines
  the boundary, and whether platform-determinism of `f64`
  integrations is buildable for this project's scale. See
  [open/determinism-boundary](../architecture/open/determinism-boundary.md)
  and
  [open/fixed-point-vs-f64](../architecture/open/fixed-point-vs-f64.md).
- **Rendering backend choice.** `wgpu` vs `ash` vs hybrid vs `vulkano`
  given the raw-Vulkan / HDR / ray-tracing / mesh-shader constraint.
  See
  [planned/rendering](../architecture/planned/rendering.md) (Backend
  choice section).
- **GUI host / game scaffolding.** No visualization host exists at present
  (the `demo` / `render` crates were removed). Open: when and in what crate(s)
  the game shell, renderer, UI and scenario content get built.
- **Time-warp UX specifics.** Core resolved (emergent deterministic
  precision resource; no variable-dt; no LOD/freezing; auto-reduces
  for reentry / asymmetric slew / close encounter / branch-event
  inertia recompute). Remaining: warp-under-thrust UX specifics.
- **Multi-part construction → physics:** rigid-vs-jointed resolved
  (RFE, see
  [decided/vessel-model-rfe](../architecture/decided/vessel-model-rfe.md)).
  Remaining: mass-property computation, staging, structural loads;
  the merge/split identity bookkeeping is now closed by
  `Mechanism::split/merge`.
- **Collision / contact / landing.** The contact *architecture* is
  resolved (branch-breaking re-base; decoupled state→state resolver;
  see
  [decided/contact-prediction-display](../architecture/decided/contact-prediction-display.md)).
  Remaining: the resolution **method** (impulse / soft / LCP /
  plasticity / debris / merge + its determinism), landing/surface
  handoff, rotating-planet surface frame, atmosphere, and the
  proximity escalation
  ([open/close-encounters](../architecture/open/close-encounters.md)).
- **Validation & regression strategy.** Energy / momentum /
  angular-momentum drift monitors, two-body Kepler & L-point
  analytic checks, JPL-ephemeris cross-validation — how you *know*
  it stays correct as it grows.
- **Content sourcing & scale.** Procedural vs catalog
  (Gaia/Hipparcos) stars & planets; body count; LOD / freezing of
  distant systems vs the always-integrate N-body pillar.
- **Prediction-tool progression currency.** Better *integration* vs
  better *estimation*; gated by an in-game compute resource vs tech
  unlocks (must not depend on the player's real CPU).
- **Osculating-elements-relative-to-which-primary.** The planning
  layer reintroduces the SOI / patched-conic ambiguity (ill-defined
  for binaries like Pluto–Charon) as a UX problem, even though the
  physics cull avoids it.
- **Epistemic-uncertainty model** (when the deferred game-level
  choice is made): is the core loop "orbital mechanics" or
  "navigation under uncertainty"; does the sensor view become core
  gameplay; one shared truth vs per-player estimated state for MP.
- **Relativistic renderer deep-dive** (at implementation time): the
  two-view design, analytic relativistic starfield, HDR + spectral,
  the rasterization-incompatibility of aberration. See
  [planned/relativistic-renderer](../architecture/planned/relativistic-renderer.md).
- **Community-multiplayer specifics** (only if pursued): lockstep,
  input determinism, the relativistic-is-single-player boundary.
