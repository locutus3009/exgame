# Game setting — sketch

**Status.** Planned / intent. Captured during the 2026-05-24
brainstorm session for the planet-rendering spec. Not yet
pressure-tested into a `decided/` invariant; recorded here so future
renderer, content, and physics decisions stay coherent with the
owner's intent.

## Star system sketch

- **Primary star.** G-class, slightly sub-solar luminosity.
- **Habitable zone.** A Saturn-class gas giant — the "pseudo-Saturn".
- **Habitable world.** A moon of the pseudo-Saturn, mass and radius
  somewhere between Mars and Earth.
- **Other moons.** Several additional moons of the pseudo-Saturn
  around the habitable one (sizes / compositions / atmospheres to
  be designed).
- **Other planets.** Several additional large planets in the system
  (physically consistent configuration still to be designed).

The whole layout still needs a pass for physical consistency:
planet masses, semi-major axes, eccentricities, mutual long-term
stability, moon-resonance structure of the pseudo-Saturn,
habitable-zone heat budget under the slightly dimmer primary. None
of this is fixed yet.

## Visual-rendering implications

The setting drives a non-trivial real-time visual surface:

- **Atmospheric scattering** for the habitable moon and any other
  atmosphered body (Bruneton / Hillaire / Nishita class).
- **Ring scattering and shadowing** for the pseudo-Saturn: rings
  cast on the planet's cloud tops, on the moons, and self-shadow.
  Hapke's BRDF extends to ring-particle phase functions in the
  literature.
- **Volumetric storm clouds** at Great-Red-Spot scale on the
  pseudo-Saturn: 3D-noise / curl-noise density fields with
  raymarched single-scattering plus a cheap multi-scatter
  approximation (Schneider/Nubis class).
- **Surface BRDF** for airless moons (Hapke's regolith model in its
  proper home).
- **Multi-body shadow casting** between large objects (gas giant
  eclipsing its moons; rings shadowing the planet and its moons).

## First-slice scope

The first-slice renderer spec narrows this to a single body —
**Earth in the existing Earth-Moon demo** — so the modular renderer
skeleton can be validated end-to-end (one atmosphere model, one
camera mode, one scene API, one tonemap, one trajectory pipeline)
before multiplying bodies, effects, or scales. The setting recorded
above informs the modularity axis: the skeleton must extend to
"Saturn + rings + moons + storms" without architectural rewrite.

## Open questions (downstream of this sketch)

- Concrete mass and orbital parameters for a stable
  configuration.
- Whether the player's vantage is fixed (cockpit / vessel-locked),
  free (orbital cinematics), or both.
- Day-night cycle, obliquity, surface-rotation modelling on
  atmosphered bodies (currently zero of this lives in
  [crates/newton](../crates/newton.md); the
  [open/body-rotation-attitude](../open/body-rotation-attitude.md)
  thread covers the kinematic side).
- Whether the rings are a passive visual element or are simulated
  (ring-particle self-gravity, shepherd-moon resonances are inside
  N-body reach but expensive).

## Cross-references

- [planned/rendering](./rendering.md) — universal renderer
  invariants the setting must respect.
- [lessons/2026-05-24-rendering-backend-wgpu](../lessons/2026-05-24-rendering-backend-wgpu.md)
  — first-iteration wgpu choice (frozen lesson); future backend
  for this setting is reopened.
- [open/body-rotation-attitude](../open/body-rotation-attitude.md)
  — planetary rotation / obliquity for atmosphered bodies.
