# Camera chain dynamics — analysis and findings

**Date.** 2026-05-30. **Status.** Findings note; no code change to camera
on main. The architectural infrastructure (`ImplicitIntegrator` trait,
`Mechanism` refactor, `clifford::Transcendental::exp_explicit`) shipped
in commit `e856fa2`. Camera-side implementation work stayed in the
`implicit-integrator-camera` worktree as design exploration.

**Related artifacts.**

- Spec and plan: `docs/superpowers/specs/2026-05-30-implicit-integrator-camera-design.md`
  and `docs/superpowers/plans/2026-05-30-implicit-integrator-camera.md`. Both were
  part of the abandoned spec corpus deleted from the working tree; they remain in
  git history at the commit before that deletion.
- Worktree branch: `implicit-integrator-camera` (kept as draft).

## 1. The problem

`hitchcock::CameraField` operates a two-stage spring chain for camera
smoothing: real ⟵ intermediate ⟵ target. Intermediate filters the
target's motion; real filters intermediate. The chain is critically
damped with shared `ω₀ = √(c0/m_camera)` and was supposed to give
C³-smooth camera trajectories during retargeting.

Empirically, in the original `T6` implementation (absolute damping on
each body's own radial velocity, with a sign-corrected `wa = unit · f`),
the chain settled cleanly across the full Family B characterisation
(scales 11 m ⋯ 100 ly, warps 10⁰⋯10⁹), but accepted a steady-state
drag-lag of `2·v_target/ω₀` when the target moves. The owner wanted to
test whether a more sophisticated dynamics could eliminate the drag-lag
without losing stability at extreme scales.

## 2. What was tried

| Approach | Result | Why |
| -------- | ------ | --- |
| **A. Absolute damping (T6 original)** | ✓ all scales, ✓ all warps | Each body damps its own radial velocity. Spring formula is implicit-Euler Padé; per-body it is unconditionally stable. Drawback: steady-state lag `2·v_target/ω₀` when target moves. |
| **B. Relative damping** (target velocity subtracted from body velocity in the formula) | ✗ overshoot at any scale > 10 km | The per-step downstream tracking authority saturates at `dt·(2ω₀+ω₀²dt)/(1+ω₀dt)² ≈ 0.18` of `a_target` at the test's `ω₀·dt = 0.105`. Real chronically lags intermediate's transient, accumulates kinetic energy, overshoots. |
| **C. Relative damping + feed-forward** (sequential implicit Euler with chain `a_target`) | ✗ same overshoot pattern | The feed-forward formula `a = (−ω₀²u − (2ω₀+ω₀²dt)v̇ + dt(2ω₀+ω₀²dt)·a_target) / (1+ω₀dt)²` is the implicit-Euler-correct chain step. The under-tracking coefficient `dt·(2ω₀+ω₀²dt)/(1+ω₀dt)²` is intrinsic to the discrete symplectic scheme at finite `dt`. |
| **D. Analytic propagator, real chases target directly (no chain)** | ✓ all scales, ✓ all warps, including 100 ly stretch goal | The closed-form solution `u(t) = (u₀ + (v₀+ω₀u₀)·t)·exp(−ω₀·t)` of the 1-D critically-damped oscillator is unconditionally stable on any `dt`. No discretisation gain loss. Trade-off: intermediate body becomes redundant; chain C³-smoothing is lost. |
| **E. Coupled-analytic chain** (exact closed form for the 4×4 chain ODE) | ✗ overshoot to negative `x_real` at ≈5·time_constants | The chain ODE has a 4-fold pole at `−ω₀`. Its closed-form `G(t)` (lower-left block of `exp(M·t)`) has cubic polynomial in `t·ω₀`: `G[0,0] = ω₀²t²/2 − ω₀³t³/6`. The cubic term turns positive `b(t)` (real-above-intermediate) negative around `t = 3/ω₀` and large negative around `t = 5/ω₀`. Real physically crosses through the intermediate during settling and continues past the origin. **This is a continuous-time property of the chain**, not a numerical artefact. |

## 3. Why E ruled the whole class out

The textbook intuition "critically damped means no overshoot" applies to
single-degree-of-freedom systems. A two-stage critically-damped chain
has a 4-fold pole; its step response includes a `t³·exp(−ω₀t)` term
that crosses zero. For a step input from far away (large `u_inter₀`),
real builds up enough kinetic energy during intermediate's transient
that, by the time intermediate settles near its equilibrium, real
overshoots through it and onward past the origin. The exact closed
form reproduces this faithfully — there is no `dt` small enough to
avoid it, because it is the right answer.

This rules out the chain-with-critical-damping architecture for the
"large step → no overshoot" requirement. The chain can satisfy that
requirement only by either (a) over-damping (ζ > 1 — slower response)
or (b) replacing the second stage with a higher-order filter (3rd or
4th order critically damped on real alone — single body, multiple
internal states, no chain), or (c) a different smoothing strategy
entirely (Kalman-like predictor, PID, etc.).

## 4. The infrastructure that did ship

The architectural plumbing built in the worktree is general-purpose and
is worth keeping even though the camera-specific consumer is deferred:

- `newton::ImplicitIntegrator<T>` — a coupled-bodies integrator trait
  with three per-type impls bridging the existing `Integrator<T>`. This
  opens the way for any future stiff or coupled solver (contacts,
  articulated bodies, soft body, cloth) without further `newton` API
  churn.
- `newton::Mechanism::step` phase 5 now calls `step_all` once on the
  whole body set, instead of looping per body. Behaviour is identical
  for callers using `SymplecticEuler` / `ExplicitEuler` / `LieEuler`.
- `clifford::Transcendental::exp_explicit` — needed for analytic
  propagators of any linear ODE (camera, chain, contact penalty,
  whatever lands next).

These changes are on `main` in commit `e856fa2`.

## 5. What the worktree retains

The `implicit-integrator-camera` branch keeps the full exploration
record: every approach (B, C, D, E above) implemented and committed
against a working Family B test, with the test results documented in
the commit messages. If the camera dynamics work resumes, that branch
is the starting point — pick the dynamics decision (e.g. "ship D and
delete intermediate" or "build a 3rd-order single-body filter") and
re-spec from there.

## 6. Current camera state on main (unchanged)

`hitchcock::CameraField` continues to use the original `T6` formulation:
absolute damping, sign-corrected `wa = unit · f`, two-stage chain. It
passes Family B across the full warp sweep and scale range (11 m ⋯
100 ly). The steady-state drag-lag of `2·v_target/ω₀` is a known
trade-off, not a bug — it shows only when the target is moving with
sustained velocity, which the existing tests do not exercise.

Camera dynamics evolution is deferred per the owner's directive. The
infrastructure above is the foundation when it resumes.
