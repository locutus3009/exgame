# Relativistic layer — OPEN

**Original CLAUDE.md deliberation: 9.**

Goal: Newtonian planetary dynamics *and* subluminal relativistic player
ships, without numerical relativity.

## Approach: two-layer split with one-way weak coupling

- **Gravity layer.** Instantaneous Newtonian
  `V(r) = -G Σ mᵢ/|r-rᵢ|` in the privileged/barycentric base frame;
  planets stay Newtonian since their `(v/c)² ~ 1e-8`.
- **Kinematics layer.** Ships/test particles as Minkowski 4-vectors;
  `dp/dt = -m∇V + F_thrust`, `p = γmv`; proper-time accumulator
  `dτ = dt/γ`.

This is honestly *frame-dependent "Newtonian gravity + SR kinematics"*,
not covariant gravity — a deliberate choice.

## Representation: carry the small correction

- Represent ship 4-velocity by **rapidity** (`v = c·tanh φ`): boosts =
  rapidity addition, base→local = rapidity subtraction, and it avoids
  the `1-β²` near-cancellation (the
  [core design principle](../decided/core-design-principle.md): carry
  the small correction — `γ-1`, `V/c²`, rapidity — as a first-class
  quantity, never via subtraction of near-equal numbers).

## Optional modular 1PN scalar add-ons

- Optional `dτ/dt = 1 - v²/2c² + V/c²` for gravitational time dilation
  without NR.
- Full **EIH** 1PN N-body is possible but is an *orthogonal
  gravity-accuracy* upgrade: implicit (iterate accelerations) and its
  `1/c²` terms (`~1e-8` of the Newtonian term) are the sharpest
  stressor of the acceleration precision budget (i128 territory).
  Gate it.

## What is rejected

- **No Lorentz-invariant scalar-gravity shortcut exists** (Nordström
  is observationally falsified — no light bending, wrong perihelion
  sign). Covariant gravity is necessarily tensor; do not pursue a
  scalar covariant model.

## Renderer side

The relativistic-rendering aspects (aberration, Doppler, headlight,
two-view design, HDR / spectral, analytic relativistic starfield)
are split off into
[planned/relativistic-renderer](../planned/relativistic-renderer.md).
This file covers the physics half only.

*Original CLAUDE.md deliberation: 9.*
