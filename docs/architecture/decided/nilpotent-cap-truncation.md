# Nilpotent grade-cap truncation (SUPERSEDED 2026-06-25)

> **Obsolete after the `clifford::new` cutover.** AD is no longer a degenerate
> Clifford algebra (`Jet<N> = Cl(0,0,N)`); it's the coefficient **ring**
> `Tangent<N, R>` (first-order, `εᵢεⱼ=0` by construction, nested for higher order),
> which has no grade-cap and no `CliffordSpec`. The `nilpotent_cap` / `CliffordSpec`
> machinery below was **deleted** with the legacy stack. Kept for the reasoning
> (why first-order truncation needed a closed-form Neumann inverse). See
> [near-zero-ad-singularities](./near-zero-ad-singularities.md) and the
> [resume-bookmark](../../process/resume-bookmark.md).

`CliffordSpec::nilpotent_cap::<N>()` is the **maximum retained grade of a
product**. Default is `1<<N` (no truncation). A value `< 1<<N` truncates the
geometric product: any term landing on a blade of grade `> cap` is dropped.

**Legality invariant (enforced).** Truncation is a two-sided ideal — and thus a
well-defined quotient algebra — **only when every generator squares to 0**
(`square::<N>(i) == 0 ∀ i`). For metrics with `square ≠ 0`, contractions lower
grade, so "drop grade > cap" is not closed and would be silently wrong.
`blade_count` asserts this at compile time: `cap < 1<<N` ⇒ metric must be fully
nilpotent. Corollary downstream: `cap < 1<<N` is a *sufficient* witness that the
algebra is fully nilpotent, used directly as the branch condition in
`try_inverse`.

**The one realized user:** `Jet<N,T> = Cl(0,0,N)` with `cap = 1` — flat
multi-dual numbers `ℝ[ε₁..ε_N]/(εᵢεⱼ=0)` for one-pass AD gradients. **Storage is
compressed to live blades only** (2026-06-06): `storage_len = Σ_{k≤cap_grade}
C(N,k)` with `cap_grade = min(nilpotent_cap, N)`. A grade > `cap_grade` blade has
**no slot at all** — it does not exist structurally (not a stored zero). For
`Jet<N>` that is `1 + N` slots: value + N first-order axes packed consecutively
in `1..=N` via `pos()`/`axis_blade` (NOT power-of-two `1<<i`). So `Jet6` = 7 f64,
`PGA3<Jet12<Jet12>>` = 16·13·13 = 2704 f64 ≈ 21 KiB — under the old dense `2^N`
it would have been ~2 GiB (unrepresentable). See auto-memory
`jet-dense-2n-storage-accepted` and spec
`docs/superpowers/specs/2026-06-06-jet-storage-compression-design.md` — deleted with the abandoned `docs/superpowers/` corpus; recoverable from git history.

**Why it exists / endpoint.** The cap mechanism is what makes `Jet` cheap enough
to push the whole SE(3) motor algebra through a dual scalar for the implicit
(midpoint) integrator of dynamic bodies. The formulation is **energy-based
(decided): compute the Hessian of a scalar energy, not the Jacobian of the
wrench.** Energy `E` runs on `PGA3<Jet<1, Jet<6, T>>>` — the inner `Jet<6>`
yields ∇E (the six se(3) axes = generalized force), the outer `Jet<1>` seeds a
direction and yields ∇²E·v (a Hessian-vector product = tangent stiffness for the
Newton step; six passes give the full 6×6 Hessian). A scalar-energy Hessian is
symmetric by construction (Schwarz) → symmetric stiffness, better-conditioned
solve, variational structure; differentiating the vector wrench (`PGA3<Jet<6>>`)
would not guarantee that. This is why `Motor::exp` was rewritten branch-free over
`u = l²` (smooth `sinc_sq/cos_sq/dsinc_sq`, no `√` / `÷l`): the old
`l = √(−s)` branch is non-differentiable at zero and broke the dual scalar.
See [crates/clifford](../crates/clifford.md#automatic-differentiation-jet).

## Single-source-of-truth invariant for `cap` (DO NOT LET DRIFT)

`cap` is read in places that must agree — **all derived from the one
`M::nilpotent_cap`**:
- **storage layout** — `cap_grade = min(nilpotent_cap, N)` drives `storage_len`,
  `pos` (mask→slot), `mask_at` (slot→mask), `blade_count` (physical size). A blade
  is stored iff `popcount(mask) ≤ cap_grade`;
- `nnz` / `generate_gp_terms` — which product terms survive (gp filter
  `popcount(k) ≤ nilpotent_cap`); the table iterates storage operands and bakes
  `pos()`-slots, so the gp filter and the storage layout must agree;
- `try_inverse` — the branch that routes truncated metrics to the closed form;
- `try_inverse_nilpotent` — the truncated **Neumann series** `a⁻¹ = (1/a₀)·Σ_{j=0}^{cap}(−m)^j`.

The inverse's correctness *depends on the gp filter cutting grade > cap*: `m` is
nilpotent with `m^(cap+1)=0` **only because** `m * s` (a gp) discards the
over-cap terms. All three read the same `M::nilpotent_cap`, so they are
consistent by construction. **If the gp truncation predicate ever diverges from
this `cap`** (different threshold, different predicate), the series stops
converging to the true inverse and small-`cap` tests may not catch it. Keep the
cutoff defined once, in `nilpotent_cap`, and route everything through it.

The full-matrix Bareiss inverse (`try_inverse_default`) is **only** valid for the
untruncated case (`cap == 1<<N`, e.g. PGA/Euclidean): under truncation the
left-multiplication matrix annihilates every grade > cap basis blade and is
singular.
