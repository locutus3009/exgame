# Camera gravity sensor (DECIDED)

How the `hitchcock` camera derives a local "down" reference, and the
two invariants that govern it. Landed 2026-06-01 (newton
`force_on_probe`; `CameraField` refactor).

## The anchor is a gravity *sensor*, not a gravitating body

The camera rig is three bodies — real (camera), intermediate, anchor —
sprung in a cascade real←interm←target. Only the **anchor** carries an
orientation role: an isotropic spring pulls it to the target centroid
(standoff 0), and it feels gravity. Its displacement from the target is
a plumb-bob hang along the local gravity direction ĝ; that direction is
the "down" reference for attitude (paired later with the view axis —
two-vector / TRIAD).

The restoring term for *all* directions is the standoff-0 central spring
(`f = −k·r`, isotropic) — so no separate perpendicular *spring* is
needed; only a perpendicular *damper* will be (deferred, see
[open/camera-attitude](../open/camera-attitude.md)). For a static ĝ the
displacement is purely radial and the spring's own radial damping settles
it cleanly; the perpendicular damper only earns its keep once ĝ slews.

## Invariant 1 — anchor gravity is scaled by 1/warp²

The camera spring stiffness is softened as `c0/warp²` (in
`linear_wrench`) so retargeting time *in frames* is warp-invariant. Under
a **sustained external force** the equilibrium is a force balance, not a
setpoint: δ = g/ω² with ω² = c0/warp², i.e. δ ∝ warp². Fed raw gravity,
the anchor's hang offset grows as warp² — at warp 1e9 it is ~light-years,
sampling a different field entirely. (The existing linear tests are blind
to this: they run G=0.)

**Decision:** the gravity read by the anchor is scaled by **1/warp²**,
the same knob `c0` carries. Then δ = g·(1+√c0·base)²/c0 — warp-invariant
in both magnitude (~mm) and direction (= ĝ). This keeps the spring law
uniform everywhere and preserves the (deferred) perpendicular-damper
analysis; the alternative (unscaled `c0` for the anchor) gives the same δ
but reopens that analysis. The anchor is an instrument, not gameplay
physics, so attenuating its gravity coupling is legitimate.

## Invariant 2 — gravity is an *input* to `CameraField`, not a peer force field

The category was wrong before: registering the anchor in the N-body
`GravityPropagator` and adding gravity as a peer `ForceField` modeled a
*sensor reading* as *physical participation*. The camera is a kinematic
follower; gravity is never a force it participates in — it is an external
field the anchor *reads*. (Analogy: a star tracker reads a reference; it
is not in the force/torque budget like a reaction wheel.)

**Decision:**
- The anchor is **not registered** with the propagator and does **not**
  gravitate the world (observer must not perturb the simulation).
- There is **no gravity `ForceField`** in the camera mechanism; its only
  field is `CameraFieldArc`.
- The `GravityPropagator` is a **field-level dependency** (one world ⇒
  one gravity): injected at `CameraField::new(gravity)`, stored OUTSIDE
  the `cameras` lock (immutable `Arc`, read-only on the hot path),
  cloned into each camera. It is bound for the field's life; not
  abstracted to a swappable/per-camera source (YAGNI; see
  [engine-and-game](engine-and-game-not-universal-engine in memory)).
- The propagator gained a **probe query** `force_on_probe(&GravityCharge)
  → Wrench` (and a `charge_of(&RigidBody)` convenience): sums pairwise
  force on an *unregistered* probe charge against the lag-1 front
  snapshot. `CameraField` queries it at the anchor's position each step,
  scales by 1/warp², and adds it to the anchor wrench. This read sits in
  the same lag-1 frame as the existing `target.centroid()` read
  ([epoch-lag](../decided/../decided)).

The probe primitive is reusable (trajectory prediction wants the same
"field at a point").

## Cross-references

- [open/camera-attitude](../open/camera-attitude.md) — everything
  deferred: perpendicular damper (for camera + intermediate too),
  ĝ→Motor coupling, rotation-impulse target, roll degeneracy when
  ĝ ∥ view axis.
- [decided/integrator-purity](./integrator-purity.md) — the camera is a
  separate mechanism with its own field; gravity-sensor reads do not
  touch the authoritative gravity core.
- `hitchcock/src/camera.rs`, `hitchcock/src/linear.rs`,
  `newton/src/gravity.rs` (`force_on_probe`).
