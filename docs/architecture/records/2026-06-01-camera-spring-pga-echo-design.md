# Camera linear spring: PGA echo of the scalar prototype

> **Migrated design record.** This document was written under the previous
> process and lived in `docs/superpowers/specs/`, which was deleted. It was kept
> because a live doc comment in `crates/joints/src/axial_spring_damper/critically_damped_warped.rs`
> cites it as the explanation for the moving-target handling in production. It is history, not
> obligation: its `Status:` line records what was true when it was written, and
> where it disagrees with the code or with [ACCELERATOR.md](../../../ACCELERATOR.md),
> they win.

Date: 2026-06-01
Status: design, awaiting review

## Context

The 1-D scalar separation test (`hitchcock/src/linear.rs`,
`simulate_movement`) was reworked (commit `bd9be32`) to compute the camera
spring on the **signed separation** (the radius-vector) rather than on a
signed radius, and now drives the cascade with two ingredients:

- **relative-velocity damping** — each spring damps the closing rate
  `ḋ = unit·(vₐ − v_b)`, not the body's absolute velocity;
- **λ-extrapolation** — the intermediate body is driven through a virtual
  predictive point `λ = 2·p_interm − p_real` that tracks the target, with
  `a_interm = (a_λ − a_real)/2` (the minus is intentional — it leaves the
  desired overshoot).

The production force law has **diverged** from this prototype:

| aspect | scalar test (current) | production (`calculate_linear_force` + `accumulate`) |
|---|---|---|
| separation / `d ≥ 0` | yes | already yes (`pa.join(pb)` → `unit`, `weight_norm`) |
| velocity damping | **relative** (`v_real − v_interm`) | **absolute** (`unit.power(&a.0.velocity())`) |
| intermediate cascade | **λ-extrapolation** | naive (intermediate tracks target directly) |
| `c0_min` / `linear_tolerance` | none | present (stiffness floor ∝ `v_target²`) |
| integrator | symplectic Euler (manual) | LieEuler |

The docstring in `calculate_linear_force` records *why* production uses
absolute damping + a `c0_min` clamp: the naive relative cascade diverges
under the discrete scheme (per-step gain ≈ 0.18 at `ω₀dt = 0.105`, so a
downstream body chronically lags an aggressive upstream transient and
overshoots). The scalar prototype recovered stability of the relative form
specifically through λ-extrapolation. Bringing production in line therefore
means porting **both** the relative damping and the λ cascade.

## Goal

Make `calculate_linear_force` + `accumulate` a faithful PGA echo of the
scalar prototype, and drop the now-obsolete `linear_tolerance` machinery.

Non-goal: angular tracking, anchor update, zoom/`effective_size`,
2-D flyby demo — all remain existing TODOs, untouched here.

## Design

### 1. Core spring becomes a pure wrench helper

Refactor `calculate_linear_force` into a pure function that returns the
wrench and writes no slot, applies no `c0_min` clamp:

```
fn linear_wrench(pa: Point, va: Twist, pb: Point, vb: Twist,
                 standoff: T, c0: T, dt: T) -> Wrench
    line = pa.join(pb)
    d    = line.weight_norm()
    if d.is_zero() { return Wrench::ZERO }
    unit = Wrench::from_line(&line) * (1/d)
    ḋ    = unit.power(&(va − vb))                 // RELATIVE (was a.0.velocity())
    f    = camera_spring_one_way_acceleration(c0, d, ḋ, standoff, dt)
    unit * f
```

Stiffness: `c0 = self.linear / warp²` directly (no speed-dependent floor).

### 2. `accumulate` orchestrates the λ cascade

For each live camera:

```
(p_real,   v_real)   from real body
(p_interm, v_interm) from intermediate body
(p_target, v_target) from target.centroid() / target.centroid_velocity()

w_real = linear_wrench(p_real, v_real, p_interm, v_interm, standoff=0,        c0, dt)

p_λ = 2·p_interm − p_real          // strictly via multivectors, see §4
v_λ = 2·v_interm − v_real          // Twist arithmetic (Add/Sub/Mul<T>)
w_λ = linear_wrench(p_λ, v_λ, p_target, v_target, standoff=r_object, c0, dt)

w_interm = (w_λ − w_real) * ½       // minus = intentional overshoot

out[real_slot]   += w_real
out[interm_slot] += w_interm
```

All camera masses are `m_camera = 1`, so wrench ≈ acceleration and the
linear combination `(w_λ − w_real)/2` is the same identity as in the scalar
test. λ-damping is now relative to the target's actual `v_target`, which is
the correct moving-target generalisation (the scalar test pins the target,
so `v_target = 0` there).

### 3. Remove `linear_tolerance` (breaking API)

- Drop the field from `Camera`.
- Drop the parameter from `Camera::new` and `CameraField::add_camera`:
  `add_camera(target, r_object, linear, gravity)`.
- Delete the `c0_min` / `four` / `vb_abs2` block in the spring.
- Update every call site (the camera.rs test helpers `make_camera`,
  `two_cameras_coexist`, `simulate_movement_pga3`).

### 4. λ strictly via multivectors

`Point` is a wrapper over a `PGA3` multivector with homogeneous weight
`E123`. Both `p_interm` and `p_real` are finite points of weight 1
(`Point::new` sets weight 1; a unit motor `transform` preserves it), so

```
let two = T::ONE + T::ONE;
let p_λ = Point::from_multivector(p_interm.as_multivector() * two
                                  - p_real.as_multivector());
```

has weight `2·1 − 1 = 1` and Euclidean coords `2·c_interm − c_real` — the
intended affine reflection of real through intermediate, computed without
ever dropping to `[x,y,z]` and without a PGA point-in-point reflection.
`v_λ = v_interm * two − v_real` uses `Twist`'s component arithmetic.

### 5. Remove `BodyOrRaw`

With both spring operands reduced to raw `(Point, Twist)` pairs extracted in
`accumulate`, the `BodyOrRaw` enum is no longer needed and is removed.

### 6. Cosmetic

- Deduplicate `m_camera` (currently defined in both `camera.rs` and
  `linear.rs`) — keep one `pub(crate)` definition.
- Resolve `// TODO: refactor this function, split into reasonable functions`
  on `accumulate`, since it is rewritten here.

## Risk & verification (practice-first)

The central uncertainty: the relative form previously diverged under
LieEuler. λ-extrapolation restored stability under **symplectic** Euler in
the scalar test; whether it also holds under **LieEuler** (production
integrator) is unknown and is decided empirically by Family B over the full
`warp × initial-speed × scale` sweep.

If Family B diverges under LieEuler despite λ, that opens a separate
fork — a dedicated camera integrator (hinted at in the old docstring) —
which is **out of scope** for this task and would be its own design.

## Test plan

Family B (`simulate_movement_pga3` / `pga3_separation_characterization` in
`camera.rs`) is the lead test and becomes an **exact echo** of the scalar
test:

- overshoot guards removed (matching `linear.rs`);
- `C0 = 1600`;
- `delta` tuned to the actual measured `std/mean` spread (not the loose
  `2.0` characterization value);
- the warp×speed×scale grid is already identical to `linear.rs`; the two
  tests should now report near-identical frame counts. A direct
  cross-check assertion (scalar vs PGA frame counts agree within a small
  tolerance for a few representative `(x0, v0, warp)`) is optional and
  added only if cheap.

Family A construction tests are updated mechanically for the dropped
`linear_tolerance` argument.

Gate: `cargo test --workspace` green before commit.

## Out of scope

Angular/normal tracking, anchor dynamics, zoom via `effective_size`, the
2-D flyby demo, and any dedicated-integrator work.
