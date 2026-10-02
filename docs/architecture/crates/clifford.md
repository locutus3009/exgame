# clifford — pure geometric-algebra library

> **⚠ Substrate cut over 2026-06-25.** The crate is now the former `clifford::new`
> architecture, promoted to the crate root; the legacy `Multivector<N, M, T>` /
> `jet` / `CliffordSpec` stack described in much of this doc has been **deleted**.
> The core is now `Mv<const N, L: BladeLaw, R: Ring>` (the law lives in `BladeLaw`
> via `Gen`/`Tensor`, Koszul-signed; no `CliffordSpec`/cap machinery) and AD is the
> **coefficient ring** `Tangent<N, R>` (nested for higher order), not a degenerate
> metric. Trait chain `AbelianGroup→MulMonoid→Ring→{Commutative,Invertible}→Scalar`
> + `StandardPart` + `Lift`. The `pga3` facade (Motor/Twist/Wrench/objects/
> Differential/Dynamics) is unchanged in spirit. Sections below that say
> `Multivector`/`Jet`/`CliffordSpec` are historical — read them for the *principles*,
> not the current type names. New near-zero/convention decisions:
> [decided/near-zero-ad-singularities](../decided/near-zero-ad-singularities.md).

Geometric algebra crate consumed by [newton](./newton.md). Two levels: a generic
`Mv<const N, L: BladeLaw, R: Ring>` core (any blade law / scalar ring) and a
non-polymorphic `pga3` facade nailed to Cl(3,0,1) for 3D rigid-body
kinematics.

The dynamics layer (RigidBody, Inertia, Joint, Mechanism, ForceField,
Integrator) was extracted into the `newton` crate in commit
`5328757 Move Newton dynamics to separate crate`; see
[newton](./newton.md).

---

## Principles

They held throughout development and should be guarded as the crate
grows:

1. **Type carries a law, not a rename of `PGA3`.** `Motor`, `Twist`,
   `Wrench`, `Inertia` (and their newton-side consumers) exist not as
   wrappers around a multivector but because each carries an invariant
   or transformation law it protects. If a new type doesn't carry a
   law — it's probably an unnecessary wrapper.

2. **Conventions are locked inside and geometrically tested.** Point
   embedding signs, right-handedness of rotations, half-angle, sandwich
   order, power pairing — all implementation details. They don't leak
   outward: the public API operates on `[T;3]`, geometric concepts, and
   world ids. Each convention is locked by a test that reads
   geometrically ("+X→+Y around +Z"), without mentioning blades.

3. **Invariants instead of coordinate checks.** Tests check conserved
   quantities (energy, momentum), order of convergence, dissipation
   monotonicity — not "a coordinate grew". This caught two mistakes:
   wrong sign on `e12` and a false assumption of symplecticity on the
   Lie–Poisson system.

---

## Layer map

```
src/
├── metric.rs          CliffordSpec (+ nilpotent_cap), Euclidean, Signature<P,Q>, blade_count
├── math.rs            Algebraic, Scalar, Transcendental (AD-carrying) traits
├── jet.rs             JetMetric, Jet<N,T>=Cl(0,0,N) — multi-dual AD
├── trivial.rs         (helper)
├── multivector/       raw Clifford algebra (generic over N, M, T)
│   ├── mod.rs         Multivector<N,M,T>, Blade<N,MASK>, dense, nnz
│   ├── common.rs      reorder_sign, blade_factor
│   ├── gp.rs          geometric product (sparse GP_TERMS, cap filter)
│   ├── reverse.rs     reverse (X̃)
│   ├── norm.rs        norm_squared, norm, normalize
│   ├── inverse.rs     try_inverse (scalar fast path / Neumann series / Bareiss)
│   └── dual.rs        Poincaré complement: dual/undual, outer (∧), regressive (∨)
└── pga3/              specifically Cl(3,0,1)
    ├── mod.rs         Pga3Metric, type PGA3<T>, public facade
    ├── blades.rs      basis blades SCALAR, E0..E0123
    ├── objects.rs     Point, Plane, Direction, Line
    ├── motor.rs       Motor (SE(3) versor); exp via smooth sinc_sq/cos_sq/dsinc_sq
    └── screw.rs       Twist, Wrench, power, cotransform
```

Layer boundary: `multivector` knows nothing about physics or SE(3) — it is
metric-independent algebra. `pga3` builds a semantic layer on top of it,
in which every type has a law. `pga3` is deliberately NOT polymorphic over
N/M — it is always Cl(3,0,1); all the generic machinery lives in `multivector`.

---

## Core: `multivector`

`Multivector<const N: usize, M: const CliffordSpec, T>` — an array
`[T; blade_count::<N,M>()]` of coefficients indexed by the blade's
**storage slot**. `blade_count` is the PHYSICAL, cap-aware size (= `storage_len`):
for non-truncating metrics it is `= 2^N` (slot ≡ mask), for truncating ones (Jet)
only the live blades. The logical mask space `2^N` is `algebra_dim`; mask↔slot is
`pos`/`mask_at`. `CliffordSpec` (formerly `Metric`, renamed in `d077c90` — it
carries the metric and `nilpotent_cap`) sets the squares of the axes
(`square::<N>(i) -> i8`); through const generics `blade_count` makes the type
non-existent for an invalid signature, and the truncation invariant lives there
too (cap < 1<<N ⇒ the metric is nilpotent everywhere, see
[nilpotent-cap-truncation](../decided/nilpotent-cap-truncation.md)).
`CliffordSpec` is a `const trait`: the metric is read at compile time.

`Blade<N, MASK>` is a ZST index: `mv[E123]` resolves to `mv.c[MASK]` at compile
time through `const Index`. It is correct **only for non-truncating metrics**
(mask ≡ slot); the impl is gated by `assert_full` (a compile-time error on a
truncating metric), so Blade indexing applies to PGA/Euclidean/Signature, but not to Jet.

Key for everything else:

- **`reverse()`** — `(-1)^(g(g-1)/2)` by grade. The basis of structural
  inversion of versors (for a unit motor `M⁻¹ = M̃`, with no numerical
  division).
- **`dual.rs`** — the Poincaré complement, **metric-independent** (it does not
  touch `blade_factor`), so it works in a degenerate metric, where the
  metric dual `A·I⁻¹` breaks. From it: `outer` (∧) and
  `regressive` (∨). In the plane-based convention (1-vector = plane) these are
  **meet** and **join**.
- **`gp` is sparse.** The geometric product stores not a dense table but a
  list of non-zero triples `GP_TERMS: [(si,sj,f,sr); nnz]` (storage slots),
  computed at compile time (`generate_gp_terms`). The generator iterates over
  storage OPERANDS (`storage_len²`, not `2^N×2^N`) and bakes in the `pos()` slot
  of the result; grade truncation is built into the const filter (a triple survives
  only if `popcount(result) ≤ M::nilpotent_cap`). One predicate serves both `nnz` and
  `GP_TERMS`; runtime `gp()` walks the precomputed table. For non-truncating metrics
  slot ≡ mask ⇒ the table is bit-for-bit as with direct indexing; for `Jet` (cap=1)
  grade≥2 is structurally absent. (Storage iteration also matters for const eval:
  the full `2^N×2^N` at N=12 is 16.7M iterations, beyond the limit.)
- **`try_inverse` has three branches** (`inverse.rs`): (1) scalar fast path —
  if only grade 0 is non-zero, the inverse is trivial; (2) truncated
  nilpotent metric (`cap < 1<<N`) — a closed-form **truncated Neumann series**
  `a⁻¹ = (1/a₀)·Σ_{j=0}^{cap}(−m)^j` (the left-multiplication matrix is
  singular here, so Bareiss is unusable); (3) full algebra — general **Bareiss**
  (fraction-free) on the regular representation, O((2ⁿ)³), NOT used in the hot
  path for motors. All three branches and the gp filter read the same
  `M::nilpotent_cap` — a divergence would silently break the series (see the decided doc).

---

## Numeric layer: `math.rs`

Trait layers that gate special functions, so that requiring them is
explicit in bounds and so that fixed point can be plugged in separately:

- `Algebraic` — full arithmetic + `EffectiveZero` + `from_u32`
  (binary doubling). The base for all of the multivector algebra.
- `Transcendental` — **the AD-carrying trait**, not just sin/cos. Methods
  with the `_explicit` suffix (`exp_explicit`/`ln_explicit`/`sqrt_explicit`/
  `sin`/`cos`/`powf`/`powi`) are the points where `Jet` implements the chain rule
  element-wise. Plus the associated type `Real` and `value(&self) -> Real`:
  the projection onto the real part (for f64 — itself, for `Jet<…,T>` —
  recursively down to the base) — it is NOT differentiated, it is needed only for
  **branch selection** (the threshold in `sinc_sq`, the sign in `norm`). On f64/f32
  the `_explicit` methods are thin wrappers over `num_traits::Float`; on `Jet` —
  derivative formulas.
- `Scalar = Transcendental + Algebraic` — an alias trait combining both.
  Introduced to avoid dragging a dozen bounds through motor/screw and consumers
  in newton. (Previously `Transcendental: Algebraic` as a supertrait — now both
  are independent and stitched together in `Scalar`.)

The default `Algebraic`/`Transcendental` is a blanket impl via
`num_traits::Float` (f64/f32). It is not implemented for `fixed` types →
a separate impl on top of CORDIC (the `cordic` crate is removed — the fixed-point
path is frozen, see [fixed-point-vs-f64](../open/fixed-point-vs-f64.md)). `Scalar`
is implemented by a blanket impl for any suitable `T` — including `Jet<N,T>`,
so `Jet` itself works as a scalar under `PGA3<…>` (hence `PGA3<Jet6>`).

**Important about `Scalar`:** it is placed ONLY on impl blocks that
transitively call `exp`/sin/cos (motor::exp, twist::exp, and consumers
in newton). Blocks of pure algebra/access (the motor's group operations —
`identity`/`compose`/`inverse`/`transform`) stay on MINIMAL
bounds. `Scalar` is broader than they need, and would drag in a superfluous
dependency on `Transcendental` — for example, it would strip newton::RigidBody of
the property "instantaneous dynamics without transcendental functions".

---

## Automatic differentiation: `jet`

AD here is **not a separate subsystem, but the same multivector machinery in a
degenerate metric.** `Jet<N,T> = Multivector<N, JetMetric, T>` = Cl(0,0,N):
N generators εᵢ, each εᵢ²=0. Dual numbers `a+b·ε` are exactly Cl(0,0,1).
The chain rule falls out of the geometric product by itself; gp, inversion,
and all operations are reused as is.

- **cap=1 → flat multi-dual.** `JetMetric::nilpotent_cap = 1`: grade≥2
  **structurally does not exist** (storage is compressed to the live blades, rather
  than "present, but zeroed"). Layout: `c[0]` is the value, `c[i+1]` is ∂/∂xᵢ; the N
  axes lie contiguously in slots `1..=N` via `axis_blade`/`pos` (`storage_len = 1+N`).
  Sizes: `Jet6`=7 f64, `Jet12`=13, `PGA3<Jet12<Jet12>>`=2704 f64 ≈ 21 KiB (under dense
  2^N it was ~2 GiB). The mechanism and the invariant are in
  [nilpotent-cap-truncation](../decided/nilpotent-cap-truncation.md).
- **Two axes of use.** *Flat* `Jet<N>` (N axes on one level) — the
  full gradient in a single pass (seed εᵢ into axis i, ∂f/∂xᵢ from `c[axis_blade(i)]`).
  *Tower* `Jet<1, Jet<1, …>>` — higher/mixed derivatives in layers; the compositions
  `Jet1<Jet6>`, `Jet6<Jet6>` give a Hessian-vector product and the full Hessian;
  `Jet12<Jet12>` — the full 12×12 Hessian of the joint energy of two bodies
  (6+6 DOF se(3)).

### Endpoint: energy formulation of the implicit integrator

Why all this: run the SE(3) motor algebra through a dual scalar and obtain
the dynamics derivatives for the implicit integrator (implicit midpoint) of dynamic
bodies. **Decision (accepted): the formulation is energy-based — we seek not the force
Jacobian but the energy Hessian.** The scalar energy E is computed on `PGA3<Jet<1, Jet<6, T>>>`:

- the inner `Jet<6>` → the gradient ∇E along the six se(3) axes = generalized force;
- the outer `Jet<1>` → seeds a direction v, extracts ∇²E·v —
  the Hessian-vector product, the tangent stiffness for the Newton
  step (six passes v=e₁…e₆ → the full 6×6 Hessian).

The Hessian of a scalar energy, NOT the Jacobian of the vector wrench: it is symmetric by
construction (Schwarz) → a symmetric stiffness matrix, a better-conditioned solve,
and the scheme stays variational. Directly differentiating the wrench (`PGA3<Jet<6>>`,
vector-valued) does not guarantee symmetry — hence the move to energy.

This required rewriting **`Motor::exp`** to be differentiable (see below):
the old `√(−s)` branch with division by the rotation length is non-differentiable at zero.
The tests `pga3::motor::jet1_tests` exercise flat `PGA3<Jet6>`, the tower
`PGA3<Jet1<Jet1>>`, the target Hv construction `PGA3<Jet1<Jet6>>` and the full
`PGA3<Jet6<Jet6>>`.

Connection with newton: this is exactly Risk-1 "PGA constraint Jacobian" that
[resume-bookmark](../../process/resume-bookmark.md) pointed to — now in energy
form (a Hessian) rather than as a wrench Jacobian.

## Geometry: `pga3::objects`

The only place where the sign convention of point dualization lives.
Outside — only `(x,y,z)`.

Point embedding (standard PGA):
`P = e₁₂₃ + x·e₀₃₂ + y·e₀₁₃ + z·e₀₂₁`. In the canonical storage order
`e₀₃₂ = −e₀₂₃`, `e₀₂₁ = −e₀₁₂` — hence the minus signs on x and z. This is the sign
of an axis permutation, not an arbitrary choice.

| Type        | Grade | What it carries                          |
|-------------|-------|------------------------------------------|
| `Point`     | 3     | a point, `coords()` divides by the weight (projectively invariant) |
| `Plane`     | 1     | `a·x+b·y+c·z+d=0`                         |
| `Direction` | 3     | an ideal point (weight 0), a displacement vector |
| `Line`      | 2     | opaque for now; `Point::join`, `Plane::meet` |

---

## Kinematics: `motor` and `screw`

### Motor

An even SE(3) versor. The sandwich convention is the canonical
**`X' = M·X·M̃`** (matches the PGA literature: Gunn, bivector.net,
Klein). Inversion is structural: `inverse() = reverse()` (valid for a
unit versor).

Group operations (`identity`/`compose`/`inverse`/`transform`/`Mul`)
— on MINIMAL bounds (pure algebra, sin/cos not needed).
`exp`/`normalize` — on `Scalar`.

**`exp`** — closed form via Study numbers, **rewritten to be
differentiable** (`473dc74`). The bivector `B` gives `B² = s + p·I`
(I = e₀₁₂₃, I²=0). Parametrization through `u = −s = l²` **without extracting `l`**:
`exp(B) = [cos_sq(u) + (p/2)·sinc_sq(u)·I] + [sinc_sq(u)·B − p·dsinc_sq(u)·(I·B)]`,
where `cos_sq/sinc_sq/dsinc_sq` are smoothly continued even functions of `u`
(a series near zero, closed form away from it; the branch is selected by `u.value()`, so
that the derivative flows through the selected branch). **There is NO branch on the
magnitude of the angle**: the former `l = √(−s)`, the division by `l` and the separate
translational branch `1+B` are non-differentiable at zero and broke the `Jet` scalar —
they were removed. `B²` and `I·B` are taken from the ready-made `gp` (signs from the
tested product); only the scalar coefficients are done by hand. This is exactly the point
that makes passing `PGA3<Jet<…>>` through the motor algebra correct.

`Motor::exp(B)` computes **literally** exp(B), with no hidden
factor of ½. The half (`B = S/2` for the sandwich) is added on the
caller's side — in `Twist::exp`.

### Twist / Wrench

Both are bivectors (grade 2) in **the same basis**. The difference is the law, not the
data: `Twist` is a velocity (contravariant), `Wrench` is a force/momentum
(co-screw), related via the duality `J`. The sign convention (right hand) is
locked in `new`:

| Phys. component | Slot | Sign (in `new`) |
|-----------------|------|----------------|
| `linear[i]`  → e₀ᵢ | E01,E02,E03 | `−vᵢ` |
| `angular[0]` (ωₓ) | E23 | `−ωₓ` |
| `angular[1]` (ω_y) | E13 | `+ω_y` (since the right-handed triple has e₃₁ = −e₁₃) |
| `angular[2]` (ω_z) | E12 | `−ω_z` |

The signs were derived by hand through the sandwich and checked by directional tests:
`+Y→+Z` around X, `+Z→+X` around Y, `+X→+Y` around Z (right-hand
rule).

**`power(W, V) = ⟨W ∧ J(V)⟩₄ = f·v + τ·ω`.** Via the dual, NOT the
metric `⟨W̃·V⟩₀`: the latter zeroes the linear part on the degenerate
axis (e₀ᵢ² = 0). This is the foundation of the energy tests.

**`Wrench::cotransform`** — the coadjoint pullback of a wrench through a motor.
This is the place where, in [frame-convention](../decided/frame-convention.md),
the world↔body conversion happens per body per step. The implementation is
a sandwich product with `M̃` on the left and `M` on the right.

---

## Testing philosophy (for the GA level)

- **Invariants, not coordinates.** Energy conservation (`½⟨P,V⟩`),
  conservation of total momentum (3rd law), drift convergence ∝ dt
  (integrator order).
- **Conventions — via geometrically readable tests.** No blades in
  asserts.
- **Integers where possible** (algebra, duality) — exact comparisons
  without eps.
- **Tuning "screws" are marked.** Where a sign/magnitude was derived with
  risk — the test says explicitly what to flip on failure.

---

## History of key decisions (GA level)

- **`exp` via Study numbers, not Taylor `1+B·dt/2`.** The old
  approximation did not produce a unit versor → required numerical
  inversion in `transform`.
- **`transform` via `reverse`, not `try_inverse`.** For a versor
  `M⁻¹ = M̃`; Bareiss O((2ⁿ)³) is not needed in the hot path.
- **`power` via the dual, not `⟨W̃·V⟩₀`.** The metric product
  zeroes the linear part on `e₀ᵢ²=0` — we would lose half the energy.
- **co/contravariance NOT via sandwich order.** The old `M̃XM` vs
  `MXM̃` is the same thing twice; the real twist↔wrench duality is
  via `J`.

### Dual algebra / AD (June 2026, `78c473c`→HEAD)

- **Dual numbers → Cl(0,0,N), not a separate type.** Started with a
  standalone `Jet1` (`a+b·ε`, its own arithmetic; `c1d8390`). Then realized:
  it is exactly Cl(0,0,1) — a degenerate metric on the already existing
  `Multivector`. Introduced `JetMetric`, generalized to Cl(0,0,N) (`27e1451`,
  `b53e365`). Payoff: gp/inversion/operations are reused, the chain rule comes
  for free. AD stopped being a parallel subsystem.
- **`nilpotent_cap` (grade truncation) was born here.** The full Cl(0,0,N) stores
  2^N and computes the mixed εᵢεⱼ; a gradient needs only first order →
  cap=1 cuts grade≥2 in gp (`c1f5d54`). The invariant (truncation is legitimate only in
  an everywhere-nilpotent metric) was moved out into
  [decided](../decided/nilpotent-cap-truncation.md).
- **Truncation broke inversion → Neumann series.** The left-multiplication matrix under
  cap truncation is singular; added a closed nilpotent form, routed
  by the same cap.
- **`Metric`→`CliffordSpec` (`d077c90`).** The spec now carries both the metric and the cap;
  it became a `const trait`.
- **gp: dense table → sparse `GP_TERMS` (`ae77543`)** with a cap filter
  built in at const eval.
- **storage compressed to the live blades (2026-06-06, `502d8e1`…`32cd3c7`).**
  `blade_count` became cap-aware (`storage_len = Σ_{k≤cap_grade} C(N,k)`),
  `algebra_dim` carries the logical 2^N; `axis_blade`→slot via `pos`; `reverse`/
  `norm`/the `GP_TERMS` generator are indexed by storage; `dual`/`outer`/`Index<Blade>`
  are gated by `assert_full` to non-truncating metrics. `PGA3<Jet6<Jet6>>` 512 KB→6.1 KB,
  `PGA3<Jet12<Jet12>>` ~2 GiB→21 KiB; the 64 MB spawned threads were removed.
  Non-truncating metrics are bit-for-bit. The spec/plan —
  `docs/superpowers/{specs,plans}/2026-06-06-jet-storage-compression*` — were deleted
  together with the abandoned `docs/superpowers/` corpus; available in git history.
- **`Transcendental` became the AD-carrying trait** (`_explicit` + `value`/`Real`),
  and `Motor::exp` became differentiable (`473dc74`). The endpoint is the energy
  Hessian via `PGA3<Jet<1,Jet<6,T>>>` (see the AD section above).

## Open tails (GA level)

- **`principal_moments` for `Full Inertia`** (in newton) — via `acos`
  (extend `Transcendental` here), if diagnostics or step selection by
  eigenfrequencies is ever needed.
- **Fixed-point `exp`** — an impl of `Transcendental` for `I88F40` via
  CORDIC, when deterministic motor arithmetic in fixed point is
  needed (the CORDIC crate is removed for now). See
  [fixed-point-vs-f64](../open/fixed-point-vs-f64.md).
- **Higher order of accuracy** — if trajectory accuracy becomes
  the bottleneck: RKMK / composition schemes on top of the existing
  `exp`. This belongs in the consumer (newton), not in clifford itself.

## What lives in newton instead

Dynamics (RigidBody, Inertia, Joint, Mechanism, ForceField,
Integrator with SymplecticEuler / ExplicitEuler / LieEuler, gravity
propagator) was extracted into the `newton` crate in commit
`5328757 Move Newton dynamics to separate crate`. See
[newton](./newton.md).
