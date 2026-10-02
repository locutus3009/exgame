# Frame convention — body-frame (DECIDED)

## Decision

Linear and angular momentum are integrated in the **body-frame** of each
rigid body. `newton::integrator::LieEuler` is the canonical integrator under
this convention. The `world ↔ body` transformation happens at exactly one
boundary per body per step, inside `newton::Mechanism::step` during force
aggregation.

## History

Three commits across 2026-05-23 and 2026-05-24 — `a664c22 Switch to world
frame` (2026-05-23) → `4321437 Fix world frame` (2026-05-23) →
`ba1ae0d Restore usge of body-frame` (2026-05-24). The world-frame attempt
was abandoned after the conditioning argument (§3.2 below) was found to
dominate.

## Why body-frame wins

Four arguments in order of weight:

### 1. Anisotropic inertia is a body-property

`I_body` is attached to the body's mass distribution. In body-frame, `I⁻¹`
is a diagonal division per step (principal axes). In world-frame, you carry
`I = R · I_body · Rᵀ` and rotate the tensor every step — or move momentum
to body and back, equivalent cost. For the vessel model with
deployment-dependent configuration-parameterized `I(L)` (see
[vessel-model-rfe](./vessel-model-rfe.md)), body-frame is the only cheap
option. This is the load-bearing argument under future work.

### 2. Conditioning: world-frame `L_COM = L_o − r × p_lin` is catastrophic cancellation

For Earth in heliocentric coordinates, `L_o ~ r · p ~ 2.66e40` while the
physical `L_COM ≈ 0`. Subtracting two large near-equal numbers loses
significant digits on every step.

Body-frame rotates the force by the pose; rotation preserves norm; no
precision loss. This is the project's
[core design principle](./core-design-principle.md) applied to angular
state — never form the small quantity by subtraction. Currently invisible
(scene spins are zero so the magnitude in question is small) and becomes
load-bearing the moment rotating bodies far from the origin enter the
simulation.

### 3. Lie-Poisson conservation

In body-frame, the LieEuler conjugation `δM̃ · P · δM` cancels structurally,
so the `lie_conserves_world_momentum_to_machine_precision` test holds for
any `dt`. The world-frame analogue smears this structure and would need an
implicit (midpoint / RATTLE) scheme to recover.

*Caveat.* This is conservation of the Casimir `|L|`, not of energy;
LieEuler's secular energy drift remains and is orthogonal to the
frame-convention question.

### 4. Cotransform source is CM-displacement, not rotation

The Plücker torque `τ_o = r × F` comes from the body's CM offset from the
world origin. It cannot be removed; at best it migrates between "ingest
force" and "extract velocity". In body-frame, it happens on ingest
(operating on force, the small quantity, in its natural representation).
In world-frame, it appears on velocity extract, where it produces the
cancellation in §2.

## Implementation

`RigidBody::momentum` (in `newton/src/body.rs`) is a body-frame `Wrench`.
`Mechanism::step` (in `newton/src/mechanism.rs`) performs the
`world ↔ body` transformation during the force-aggregation step — a
co-adjoint pullback on per-body wrench inputs. The co-adjoint pullback on
`Wrench` is implemented in `clifford/src/pga3/screw.rs` as `cotransform`.

## Cross-references

- [integrator-purity](./integrator-purity.md) — body-frame state ownership
  underpins the integrator-purity invariant.
- [vessel-model-rfe](./vessel-model-rfe.md) — anisotropic `I(L)` for
  deployment-dependent vessels is what makes the body-frame choice
  load-bearing in the long run.
- [core-design-principle](./core-design-principle.md) — body-frame is the
  angular-state instance of the same principle (never form the small
  quantity by subtraction).
- [open/close-encounters](../open/close-encounters.md) — orthogonal; its
  step-reduction mechanism interacts with neither body-frame nor LieEuler
  structure.
