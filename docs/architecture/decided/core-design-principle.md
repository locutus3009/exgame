# Core design principle — carry the small quantity, never subtract for it

The recurring failure mode across every layer of this project is
**catastrophic cancellation**: forming a small, dynamically meaningful
quantity by subtracting two near-equal large numbers. It appeared as
the `I102F26` precision floor (force residuals lost under the
Sun-mass-driven quantum), and it reappears at every level discussed
below — position differencing over light-year coordinates, the
acceleration cull threshold, relativistic `γ` from `1-β²`, clock rates
near a potential.

The unifying fix, applied everywhere: **represent the small correction
as a first-class quantity in its own right; never derive it by
subtraction.** Concretely:

- **Encke / relative formulation** for orbital deviations.
- **Acceleration quantum `Q_a`** for the gravity cutoff (the small
  threshold is the design parameter, never a difference of two large
  forces).
- **Rapidity** (not `γ`) for boosts. Boost composition becomes addition;
  base ↔ local is subtraction of rapidities, not of nearly-equal `γ`s.
- **`γ-1` and `V/c²` carried directly** for kinetic energy and
  gravitational clock rates.
- **Angular state in body-frame.** `L_COM = L_o − r × p_lin` is
  catastrophic cancellation when `L_o` is much larger than `L_COM`
  (Earth in heliocentric coordinates: `L_o ~ 2.66e40`, physical
  `L_COM ≈ 0`). Body-frame momentum carries the small quantity
  directly via the cotransform of force on ingest; the
  large-large-minus is never formed. See
  [frame-convention](./frame-convention.md).
- **`distance` via `strict_hypot`, never `distance²`.** The squared
  intermediate is the catastrophic intermediate; the linear-magnitude
  distance is computed directly. This is the overflow-flavoured form
  of the same principle: never materialise a catastrophic intermediate.
  (Now exercised in `clifford::pga3::Motor` and friends; the original
  fixed-point context is in flux —
  [open/fixed-point-vs-f64](../open/fixed-point-vs-f64.md).)

Any new layer must be checked against this principle before
implementation.
