# Implicit joint solver: differentiable functional → mechanism Hessian

A design report. It records the architecture of the implicit step for penalty joints based on
automatic differentiation of energy functionals through the nilpotent
Clifford algebra (`clifford` crate, dual `Jet` towers).

The principle we arrived at: **the user specifies only the energy functionals
(the elastic potential and the Rayleigh dissipation); the core differentiates EVERYTHING itself** — both
the geometry and the laws themselves — in dual passes, and assembles the mechanism Hessian.
No derivatives "by hand".

---

## 1. State and bases

A mechanism is `N_bodies` rigid bodies. The full phase state of each body is
**12 DOF**: 6 positional + 6 velocity. These are two DIFFERENT tangent spaces,
varied independently.

**Position** is a motor `M_i` (an element of SE(3)). The perturbation lives in the algebra se(3):
`M_i(ε) = M_i ∘ exp(ε·B)`, where `B` runs over the 6 basis screws of se(3)
(3 rotations `e23/e13/e12` + 3 translations `e01/e02/e03`). The basis of the positional axes is
these 6 screws. Position enters the geometry **through `exp` (nonlinearly)**.

**Velocity** is a twist `ξ_i` (a bivector of se(3), itself a vector of the same
6-dimensional space). The perturbation is `ξ_i + η`, where `η` tags the same 6 screws
**directly**. Velocity enters **linearly — no `exp` needed**.

The key difference: the positional seed goes through `Motor::exp` in the dual (expensive),
the velocity seed is a direct tag in the twist (cheap).

The global basis of the mechanism: `12·N_bodies` axes (6 pos + 6 vel per body).

---

## 2. Two functionals on two arguments

A joint is specified by TWO closures, generic over the dual scalar `T: Scalar`:

```
potential(d: T)   -> T     // elastic deformation potential Π(d)
dissipation(ḋ: T) -> T     // Rayleigh dissipation function R(ḋ)
```

This is ALL the user writes. Hooke's law (`Π = ½k(d−L₀)²`) is a special case;
the core does NOT assume a specific law. Any smooth function will do — AD
differentiates it automatically as part of the pass.

Arguments:

- **`d = d(q)`** — the deformation (the distance between the anchors),
  `d = ‖anchor_a(M_a) ∨ anchor_b(M_b)‖` (the weight of the join line). A function of positions ONLY.
- **`ḋ = ḋ(q, ξ)`** — the deformation rate, `ḋ = power(unit(q), ξ_a − ξ_b)`, where
  `unit(q)` is the unit direction wrench of the line of action. A function of positions AND
  velocities. (Frame: the twists are pushed to the world through `Ad_pose` before pairing.)

The potential depends only on `q`; Rayleigh on both. Hence the block structure
of the Jacobian.

---

## 3. What is differentiated and which blocks result

The Jacobian of the implicit system (derivatives of the generalized force with respect to what is updated
implicitly — positions and velocities):

```
                 ∂/∂q                 ∂/∂ξ
            ┌──────────────────┬──────────────────┐
   from Π   │  ∂²Π/∂q²         │       0          │   Π does not depend on ξ
(elasticity)│  STIFFNESS K     │                  │
            ├──────────────────┼──────────────────┤
   from R   │  ∂²R/∂ξ∂q        │  ∂²R/∂ξ²         │   R depends on both
 (Rayleigh) │  CROSS TERM      │  DAMPING C       │
            └──────────────────┴──────────────────┘
```

- **`∂²Π/∂q²`** — the stiffness matrix (position×position). AD through the chain
  `ε → exp → anchor → d → Π`.
- **`∂²R/∂ξ²`** — the damping matrix (velocity×velocity). Since `ḋ`
  is linear in `ξ`, `∇_ξ ḋ = unit`, and `∂²R/∂ξ² ∝ R''(ḋ)·unit⊗unit` — the outer square
  of the line direction.
- **`∂²R/∂ξ∂q`** — the cross term: the damping force changes when the position shifts
  (through `unit(q)`). Usually small, sometimes dropped, but genuinely present.

The local block of a joint between two bodies is **24×24** (12 pos + 12 vel), structured
(the top-right is zero).

The tower `Jet<12, Jet<12, f64>>` (inner level = first derivatives along 12
axes, outer = directions for the second ones) gives the full 12×12 block in a single pass.
Compressed storage: `PGA3<Jet12<Jet12>>` = 2704 f64 = ~21 KiB (under a naive dense
2^N it would be ~2 GiB).

---

## 4. Differentiable functional — why AD rather than hand-written derivatives

The principle "specify the energy — the core computes everything" means: the law `E(d)` is given as
an ordinary function over the dual scalar, and AD carries the derivative **through it
AND through the geometry** in a single pass. The composition
`ε,η → exp/twist → d,ḋ → Π,R` is differentiable as a whole, because every link
(including `Motor::exp` via Study numbers, and the law itself) is defined generically over `T`.

**The chain-rule decomposition is correct, but as an OPTIMIZATION, not as a way of
differentiating.** The identity
`∂²Π/∂q² = Π''·∇d⊗∇d + Π'·∇²d`
has been verified: it matches a direct AD run to machine precision. But "attaching
Π', Π'' by hand" violates the principle (the user must not take derivatives of the
law). Therefore in the core the law is differentiated automatically; the decomposition
makes sense only where it gives a measurable gain (see below — for simple laws
it does not).

---

## 5. Full Newton versus Gauss-Newton: indefiniteness

Empirical result (demo on real PGA geometry): **the full Hessian of the
deformation energy is NOT positive definite away from the rest length.** Measurements gave
`∂²Π = −5.52` for a stretched joint; at other points `−10.85`, `−14.31`.

The reason, from the decomposition `∂²Π/∂q² = Π''·∇d⊗∇d + Π'·∇²d`:
- the first term `Π''·∇d⊗∇d` (= `k·∇d²` for Hooke) is an outer square, **always ≥ 0**;
- the second term `Π'·∇²d` (= `k(d−L₀)·∇²d`) under stretching and concave geometry
  (`∇²d < 0`) **is negative and dominates** → the Hessian goes negative.

**Consequence for Newton-Raphson.** N-R as root finding is universal, but in
minimization (`∇E = 0`, which is what the implicit step is) it is safe ONLY with an SPD Hessian:
- `H > 0` → the step `−H⁻¹∇E` goes down (towards the minimum) — what we want;
- `H < 0` → the step goes UP (towards the maximum) — the solver stretches the joint further,
  diverges;
- `H` indefinite → steps towards a saddle / jumps.

Pure N-R on the physical energy is unsafe — the energy does not guarantee SPD.

**Gauss-Newton fixes this structurally.** For a least-squares energy (and a penalty joint
is exactly that: `r = d − L₀`) GN drops the second term, leaving
`H_GN = Π''·∇d⊗∇d` — **always SPD**. So the GN step always descends, in any
configuration. Near the rest length (`d ≈ L₀`, which is where a taut joint holds the system)
the dropped term `~(d−L₀) → 0`, so there GN is also exact. GN is not a compromise
for speed but **a regularization for the stability of the descent direction**.

---

## 6. Performance: what was measured

The demo "three ways to get ∂²Π/∂q²" (2000 iterations, tower `Jet12<Jet12>`,
LTO + codegen-units=1):

| method | ns/iter | rel. |
|---|---:|---:|
| (1) brute force — full Π through the tower (potential in AD) | ~11.7 M | 1.00× |
| (2) decomposition — d through the tower, potential analytically | ~10.9 M | 0.93× |
| (3) Gauss-Newton — FLAT Jet12 (∇d only) | ~0.61 M | 0.05× |

Conclusions:

- **(1) == (2) exactly** (the chain rule is an identity). The decomposition is correct.
- **The decomposition by itself gives NO benefit** (1.07×) — for simple laws
  the potential is trivial in the dual, the geometry dominates (`exp` + `join` + `norm`),
  identical in both. "Moving the potential out of AD" is a lever only for expensive laws
  (nonlinear ones with transcendental terms); for simple ones do not count on it.
- **The real gain is Gauss-Newton (19×)**: a flat `Jet12` (13 f64, first
  variation) instead of the tower `Jet12<Jet12>` (169 f64, second). Both cheaper and SPD.
- `∇d` (the Jacobian of the distance) is the force direction, `unit = Wrench::from_line/d`,
  which is already computed in `wrenches`. The GN Hessian = `k·∇d⊗∇d` = the outer square
  of the existing force direction. The implicit step is assembled from the same geometry
  as the explicit force.

(Earlier in the same work: two multiplicative bugs gave 1000× and 57× — `blade_count`
in a loop condition vs an associated constant, and an `is_effective_zero` guard in the hot
`gp` on dense AD data. Transcendental optimization of `exp` is a dead end (~7%);
the dominant cost is PGA `gp` convolutions, 93%.)

---

## 7. Assembling the mechanism Hessian

The energy is additive: `E_total = Σ E_joint`. Therefore the global matrices are sums
of local blocks scattered by body indices (FEM assembly).

**Global blocks** (size `12·N_bodies` along each argument):
- **K** — stiffness, `Σ ∂²Π/∂q²` (scatter of the positional blocks).
- **C** — damping, `Σ ∂²R/∂ξ²` (scatter of the velocity blocks).
- the cross term `Σ ∂²R/∂ξ∂q`.
- **M** — the mass matrix (body inertia), on the velocity DOF.

**Scatter.** A joint between bodies `p` and `q` writes its local block into positions
`(p,p), (p,q), (q,p), (q,q)` of the global sparse matrix. Bodies without common
joints give zero blocks → the matrix is sparse and block-structured.

**Free term.** The `K` external forces (gravity, aerodynamics) are **NOT varied**
over the step (they are treated as constant) → they go DIRECTLY into the free term as generalized forces
along the se(3) axes of the bodies, NOT into the Hessian (their derivative with respect to the state = 0 by
assumption). `∇E_total` (the scatter of the joint gradients) goes there too.

**Step structure** (schematic; the exact form depends on the integrator —
Lie-Euler etc.):

```
[ M + dt·C + dt²·K ] · Δ = −(∇E_total) − F_ext + (inertial terms)
```

---

## 8. Iteration scheme (damped Newton)

The matrix is inherently **ill-conditioned**: taut penalty joints (large `k`) give
large stiffness eigenvalues, soft DOF give small ones; the ratio grows with
spring stiffness. This is a built-in property of the penalty method. It is treated by regularization
(the symptom) and by the implicit scheme + a preconditioner (the root cause).

**Regularization — Levenberg-Marquardt:** we solve `(J + λ·D)·Δ = −r`.
- `D = diag(J)` (diagonal scaling) is BETTER than `D = I` when the blocks
  have different scales — and we have `K`, `C`, `M` in one matrix with different scales;
  `λI` suppresses everything equally, `λ·diag` — proportionally.
- `λ` is **adaptive**, not a constant: the step improved the residual → decrease `λ` (trust
  Newton), made it worse → increase `λ` and REDO the step. This also fixes the indefiniteness
  (`J + λD` becomes SPD for a large enough `λ`).

**Residual — weighted norms (the plural matters):** the components
have different dimensions (positions in m/rad, velocities in m/s, rad/s). A bare Euclidean norm
adds up incomparable things. What is needed:
- separate criteria `‖residual_q‖ < tol_q` AND `‖residual_ξ‖ < tol_ξ`, or
  weighting by masses/stiffnesses;
- a relative criterion `‖rₙ‖/‖r₀‖ < ε` (not only an absolute one).

**Loop:**
1. Assemble `J = M + dt·C + dt²·K` (local 24×24 blocks via AD on the tower →
   scatter into the sparse global matrix) + the free term (`∇E`, `F_ext`).
2. Regularize `J + λ·diag(J)`, `λ` adaptive.
3. Solve `(J + λD)·Δ = −r` (sparse Cholesky or CG; for CG a
   preconditioner is MANDATORY; block-diagonal — inverting the diagonal
   12×12 blocks — is cheap and effective).
4. Update `q ← q ∘ exp(Δ_q)`, `ξ ← ξ + Δ_ξ`. Line-search truncation if the
   residual grew (even a correct direction can overshoot).
5. Has the residual (weighted, separate, relative) converged? → stop; otherwise → step 1
   with adaptation of `λ`. An iteration limit + fallback to the best one (do not hang the frame).
6. **Warm start** of the next frame: pose extrapolation (`2qₙ − qₙ₋₁`) — cuts
   iterations from ~5 to 1-2 (×3-5). Requires `Motor::log` (the inverse of `exp`) for
   extrapolation in se(3). The main speed lever at the solver level.

---

## 9. Where the load is

Two centres of gravity; which dominates depends on the scale of the mechanism:

- **Few joints, thick towers** → `exp` + block assembly dominate. Inside:
  `Motor::exp` in building the dual poses + the geometry of `d` through it (~0.44 ms/exp
  on the tower; 93% is PGA `gp` convolutions). Velocity (Rayleigh) is cheaper: `ḋ` is linear in `ξ`,
  no `exp` needed.
- **Many joints** → solving the linear system
  `12·N_bodies × 12·N_bodies` (sparse) dominates. It scales with the number of bodies, and on
  large mechanisms overtakes assembly. The preconditioner is critical (poor
  conditioning will kill CG convergence).

Structural levers in `gp` (not exhausted): materializing `const GP_TERMS` (~2×),
sparsity of bivector operands in `exp` (~2×). At the solver level: warm start
(×3-5), Gauss-Newton (flat vs tower, an order of magnitude), rayon-by-constraint (×N_cores),
GPU for thousands of joints (cloth scale).

---

## 10. Summary — separation of responsibilities

**The user specifies:** two functionals, generic over `T: Scalar` —
`potential(d)` and `dissipation(ḋ)`. Smooth, arbitrary. Nothing more.

**The core does by itself:**
- seeding along the 12 pos axes (through `exp`) + 12 vel axes (directly), a tower for
  second derivatives;
- the geometry `d(q)`, `unit(q)`, `ḋ(q,ξ)`;
- `Π = potential(d)`, `R = dissipation(ḋ)` — AD differentiates the laws themselves through
  the composition;
- the local blocks `∂²Π/∂q²`, `∂²R/∂ξ²`, `∂²R/∂ξ∂q`, gradients;
- scatter into the global sparse system; external forces → the free term;
- damped Newton (Levenberg-Marquardt, adaptive `λ`, weighted residual,
  line search, warm start).

**Open decisions for the implementation:**
- the main path — Gauss-Newton (flat `Jet12`, SPD, cheap) or the full tower
  Hessian with LM regularization (honest, but expensive and indefinite — a fallback);
- `Motor::log` for warm start (the inverse of `exp` via the Study form);
- the linear solver preconditioner (block-diagonal);
- parametrization of `d` for a real joint between two bodies (the current demo is a special case
  of a single rotation; all 6 DOF per body are needed for a representative `∇²d`).
