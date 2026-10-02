# Near-zero AD singularities: ε-softened norm + where the ½ lives

Decided 2026-06-25, after the camera example panicked under the cut-over
substrate (see [resume-bookmark](../../process/resume-bookmark.md)).

## Problem

Under forward-AD (`Tangent`), geometric quantities that are smooth in value can
have **singular derivatives** at degenerate configurations. The camera rig builds
the implicit-Newton Jacobian by differentiating the joint wrenches; at the
linearization point two anchors coincided, so the line length `d = √(Σx²)` was
`0`, and its derivative `x/d` is the classic `0/0` (gradient of a Euclidean norm
at the origin). The new substrate's `Tangent::sqrt_explicit` computed the
derivative with `try_recip().unwrap()` → **panic**.

## Two distinct singularity classes (do not conflate)

- **Removable** (a finite limit exists): `sin(l)/l → 1`, `cos(√u)`, the Study
  functions in `Motor::exp`. A Taylor polynomial *is* the function near zero —
  this is the `Motor` representation-select pattern.
- **Essential** (→ ±∞, no limit): `sqrt'(a)=1/(2√a)`, `ln'(a)=1/a`, the `atan2`
  gradient, and `x/‖x‖` (norm gradient / unit direction at the origin). **No
  Taylor can make these finite.** Only *regularization* can.

## Decisions

1. **`commutator` stays pure `a*b − b*a` (no ½).** The GA commutator-product ½
   (the se(3) Lie bracket / point-velocity normalization) lives in the
   twist/wrench **semantic API** — `Twist::velocity_at`, `Wrench::bracket` — as a
   `HALF` associated const, the same place `Twist::exp` carries the rotor half.
   Legacy folded the ½ into `commutator`; the new split keeps the raw bracket
   honest. If a commutator-derived physical quantity is 2× wrong, the ½ belongs in
   the semantic op, never in `commutator`.

2. **Raw transcendental derivatives divide, they do not panic.** `Tangent`
   `sqrt`/`ln`/`atan2` use plain `/` (legacy `Jet` parity): at a singular point the
   gradient is NaN/inf, not a panic — so a downstream guard can discard it.

3. **The real near-zero fix is the ε-softened norm**, not a guard on NaN.
   `Mv::soft_norm(ε)` / `Line::soft_weight_norm(ε)` = `√(‖·‖² + ε²)`: the ε floor
   keeps the radicand `≥ ε² > 0`, so the AD gradient `(∂s/∂xᵢ)/(2√(s+ε²))` is finite
   **through** the origin (→ 0 as `‖·‖ → 0`; the numerator vanishes too). The point
   is that the *linearization* stopped degenerating, not "softening in general" —
   the Newton row stays finite through the coincident config the solver passes
   through. **Closed form only — no Taylor branch.** Unlike `Motor::sinc_sq` (a
   genuine removable `0/0`), the ε floor already makes `√(s+ε²)` finite in value AND
   all derivatives (denominator `≥ ε`), so there's nothing to continue; a series
   would only recover the sub-ε value increment `s/2ε`, below f64 precision relative
   to ε. (An earlier version carried the series; it was removed as dead weight.)
   Cost: an `O(ε²)` deviation (e.g. spring force at rest `~k·ε²/2L₀`) — accepted;
   test tolerances reflect that floor.

   **Precondition: positive-definite norm context (`s ≥ 0`).** `soft_norm` is a
   generic `Mv` method; on an indefinite signature (Minkowski) a timelike element
   has `s < −ε²`, so `s + ε²` < 0 and `sqrt` returns NaN *silently*. Current callers
   use it only on the Euclidean direction part of a `Line` (sig +++). Documented in
   the method; no debug-assert (would ripple a `Real: PartialOrd` bound into every
   joint for a debug-only check).

   **ε is a geometric length, kept off the solver residual band.** `soft_norm`
   plateaus at ε, so any constraint that enters the residual through a distance can't
   converge below the ε scale. ε must be sourced from geometry (smallest meaningful
   separation), not from the Newton tolerance; with residual ~1e-13 an
   ε ~ 1e-6·(geometry) sits safely above the band.

4. **Joints regularize instead of guarding.** The perpendicular + axial dampers
   dropped their value-based `is_effective_zero` coincidence guards and use
   `soft_weight_norm`. At coincidence the line direction → 0, so the force → 0
   smoothly (C¹) — the same "no force" the guard gave, but differentiable, so the
   implicit-Newton Jacobian stays bounded.

5. **ε is a proper parameter, not a magic constant.**
   `newton::joint::default_softening()` is a `const fn` floor default (requires
   `#![feature(const_ops)]` — accepted, on track for stabilization). Each
   distance-using joint carries a `softening` field
   (`PerpendicularDamperWarped::new(damping, softening)`; `AxialSpringDamper`
   builder `.softening(ε)`); `RigBuilder` threads it; the camera derives it
   scene-scaled as `r_object / 1000`.

## Open

- Wire `RigidBody::effective_size` (the gravity sphere radius, see
  [gravity-softening / close-encounters](../open/close-encounters.md)) into the rig
  softening in place of `r_object / 1000`, so one length governs both gravity
  softening and joint regularization.
