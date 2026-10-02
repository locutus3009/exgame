# Newton–Schulz convergence on real Newton systems

Date: 2026-07-21
Source: `crates/newton/src/integrator/implicit/newton/prototype.rs`
Reproduce: `cargo test --release -p newton prototype -- --nocapture`

Matrices are harvested from the production `assemble_matrix` on two 1 kg masses
joined by an `AxialSpringDamper` (rest 2.0, damping 5.0) with ±50 m/s transverse
speeds, so the spring axis swings appreciably within the step.

## Summary

- **Warm start: validated, with room to spare.** Two to three iterations reach a
  `dv` relative error of 1e-6 at every stiffness tested, even under a `dt`
  perturbation far larger than one step's worth.
- **f32: carries the tolerance.** The accuracy floor is 3e-7 on `dv`, three
  orders below the 1e-4 target.
- **Cold start: the spec is wrong.** Outside its contractive range the
  block-diagonal seed does not converge slowly — it **diverges to `inf`**, and
  additional iterations make it worse. See "Design gap" below.

## Cold start (block-diagonal seed `X₀ = 𝕀⁻¹`)

| dt | k | ‖I − A·X₀‖ | iterations to `dv` < 1e-3 |
|---|---|---|---|
| 1/60 | 1e2 | 0.138 | 2 (→ 5e-16 by iter 4) |
| 1/60 | 1e3 | 0.378 | 3 (→ 7e-16 by iter 6) |
| 1/60 | 5e4 | **14.2** | never — `inf` by iter 8 |
| 1/30 | 1e2 | 0.496 | 3 (→ 3e-16 by iter 5) |
| 1/30 | 1e3 | **1.58** | never — `inf` by iter 11 |
| 1/30 | 5e4 | **63.4** | never — `inf` by iter 7 |
| 0.1  | 1e2 | **6.25** | never |
| 0.5  | any | — | never |

Convergence is quadratic where it happens, exactly as predicted: the residual
goes 1.4e-1 → 1.5e-2 → 1.9e-4 → 2.8e-8 → 1.5e-15.

The contraction boundary agrees with theory. For two 1 kg masses on a spring the
reduced mass is ½, so `ω = √(2k)`; at k = 5e4, `ω ≈ 316 rad/s` and
`(dt²/4)·ω² ≈ 6.9` at dt = 1/60, against 14.2 observed — the remainder is the
damping term `(dt/2)·c`, which the prediction omits. The predicted boundary
`dt < 2/ω` holds.

## Warm start (converged `X` carried onto a perturbed matrix)

`X` is seeded from the exact inverse of `A` — standing in for "`X` as it was at
the end of the previous step" — then `A` is perturbed by scaling `dt`. One
step's worth of change is well under 1%; 5% is deliberately pessimistic.

Iterations to `dv` relative error < 1e-3:

| k | drift 0.1% | drift 1% | drift 5% |
|---|---|---|---|
| 1e2 | 1 | 1 | 1 |
| 1e3 | 1 | 1 | 1 |
| 5e4 | 1 | 1 | 2 |

Iterations to < 1e-6, the stricter reading:

| k | drift 0.1% | drift 1% | drift 5% |
|---|---|---|---|
| 1e2 | 2 | 2 | 2 |
| 1e3 | 2 | 2 | 3 |
| 5e4 | 2 | 2 | 3 |

Stiffness barely matters on the warm path. This is the design's central bet and
it pays.

## f32 floor

Measured on the warm path (1% drift), every iterate rounded through f32:

| k | `‖I − A·X‖` floor | `dv` relative error floor | iterations to reach it |
|---|---|---|---|
| 1e2 | 3.1e-7 | 6.3e-8 | 2 |
| 1e3 | 1.7e-7 | 3.3e-8 | 2 |
| 5e4 | 4.6e-7 | 2.7e-7 | 3 |

Verdict: **f32 carries the 1e-4 relative tolerance**, with roughly three orders
of margin. No design change on this axis.

## Design gap: the cold start

The spec states that past the contractive limit "the seed alone is not enough and
convergence comes from the budget and from the warm start." The first half is
right, the second is not. Newton–Schulz squares its residual every iteration:
below one that is quadratic convergence, above one it is quadratic divergence.
`ns_seed_iterations` therefore does nothing useful in the stiff regime — the very
regime that justifies an implicit integrator — and spending more of it is
strictly worse than spending less.

The warm path is unaffected: once `X` is good it stays good. The gap is confined
to the first solve on a freshly allocated cache, which happens once per island
lifetime — but it happens on the first frame of every stiff scene, including
`cloth_grid`.

Whatever fills the gap must also decide what the solver does when it detects
non-contraction, since a diverged `X` yields a non-finite `dv` and the current
`newton_solve` reaches the line search through comparisons that are all false for
NaN. That path is untested and must be settled explicitly rather than assumed.

## The fallback seed and the shape of the guard

`X₀ = Aᵀ/(‖A‖₁·‖A‖_∞)` converges everywhere measured. Iterations to `dv` < 1e-3:

| dt | k=1e2 | k=1e3 | k=5e4 |
|---|---|---|---|
| 1/60 | 9 | 9 | 6 |
| 1/30 | 10 | 9 | 4 |
| 0.5 | 18 | 18 | 18 |

Stiffness barely matters; the step size does. **`ns_seed_iterations = 24`** covers
the measured range with margin, and where it does not fit, dt subdivision is the
designed fallback.

**The guard cannot be a threshold on `ρ = ‖I − A·X‖_F`.** The classical guarantee
for the transpose seed bounds the *spectral radius* below one, not the Frobenius
norm — and at n = 12, `‖I‖_F` alone is 3.46. Measured, the transpose seed starts
at `ρ₀ ≈ 2.8–3.3` and converges monotonically. A `ρ ≥ 1` test would reseed on
every pass, forever.

Divergence is instead detected by monitoring growth. Trajectories the monitor
sees:

```
dt=1/60 k=1e2  mass-block  1.38e-1 → 1.48e-2 → 1.86e-4 → 2.84e-8 → 1.53e-15 → 9.33e-16 → 6.24e-16 → 7.81e-16
dt=1/60 k=5e4  mass-block  1.42e1 → 1.95e2 → 3.81e4 → 1.45e9 → 2.11e18 → … → inf
dt=1/60 k=5e4  transpose   3.30e0 → 3.25e0 → 3.19e0 → 3.08e0 → 2.93e0 → … → 4.20e-2
dt=0.5  k=5e4  transpose   3.18e0 → 3.16e0 → 3.15e0 → … → 2.78e0
```

A bare `ρ_after ≥ ρ_before` false-fires on the first row, where a converged
iterate bounces on the machine-precision floor by about 1.25×. Divergence is a
squaring and grows by 13.7× on the first pass. The predicate

```
fire  ⟺  !(ρ_after < 2 · ρ_before)
```

fires on exactly the two diverging rows and on neither converging one, verified
by `sweep_residual_trajectories_for_the_guard`. The negated `<` is deliberate:
NaN loses every comparison, so a diverged iterate fires the guard instead of
slipping past it.

## Chosen defaults

- `ns_iterations` = **2**. The warm table needs 2 at worst. A third was carried
  as margin and later dropped on a demo measurement: 5.94 → 5.53 ms per step,
  with the final position differing by 6e-8 out of 4.3e-3 — 1.4e-5 relative, an
  order inside the integrator's own 1e-4 tolerance. Frame-time spikes and a 30 s
  step both pass at 2.
- `ns_seed_iterations` = **24**. Covers the transpose seed's worst measured cost
  (18, at `dt = 0.5`) with margin.
