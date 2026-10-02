# Camera attitude — OPEN (deferred)

The `hitchcock` camera's gravity-direction *sensing* is settled (see
[decided/camera-gravity-sensor](../decided/camera-gravity-sensor.md)):
the anchor yields a warp-invariant ĝ. What turns ĝ into the camera's
orientation is deliberately deferred — the owner wants to prototype the
angular parts by hand, on top of the now-correct spring + scaled-gravity
perturbation.

## Deferred threads

1. **Perpendicular damper.** Only the radial closing rate is damped today
   (inside `linear_wrench`). A perpendicular damper is needed once ĝ
   slews (moving/transitioning attractors) — it damps the relative
   perpendicular velocity, with no perpendicular *spring* (the
   standoff-0 central spring already restores all directions). **Needed
   for the anchor AND for the real camera + intermediate bodies**, not
   just the anchor (owner note, 2026-06-01). Warp scaling to watch:
   warp-invariant *alignment time in frames* wants the damping coefficient
   ∝ 1/warp (same as the linear spring's `2ω₀`); the readout should be a
   *direction*, not a magnitude.

2. **ĝ → Motor coupling.** ĝ is currently a position/direction (anchor
   offset). How it becomes the camera's attitude versor: kinematic TRIAD
   readout (view axis + ĝ) applied as an impulse, vs a physical torque.
   Not chosen.

3. **Rotation-impulse target.** Main-camera reorientation is by *instant
   impulses* — to the real camera (snappy, jarring) or to the intermediate
   (smoothed through the real←interm cascade, with lag). Symmetry with the
   linear cascade suggests intermediate; undecided.

4. **Roll degeneracy.** When ĝ ∥ view axis (looking straight up/down the
   gravity vector), the two-vector basis collapses and roll about the view
   axis is unconstrained. Graceful-degradation rule needed (likely: hold
   last attitude via the damper, as in the null-g case).

## Null-gravity / barycenter / L-point

Resolved as graceful degradation, NOT a fallback reference: |g|→0 ⇒ no
hang ⇒ nothing to build ⇒ the damper holds the last attitude. No
secondary reference (which would re-import the SOI/primary choice the
physics deliberately avoids).

## Cross-references

- [decided/camera-gravity-sensor](../decided/camera-gravity-sensor.md)
- [open/body-rotation-attitude](./body-rotation-attitude.md) — attitude
  *representation* (Motor/Twist); this note is the camera attitude
  *control law*.
