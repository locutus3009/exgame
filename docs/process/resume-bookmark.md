# Resume bookmark

Where design work stopped and what to pick up next. The
[governance rules](../architecture/overview.md#governance) send a reader here
after a break, so this file stays; what changed is what it is allowed to claim.

## Current — where to resume

**This file is no longer the record of where work stopped.** That record now
lives in [`coordination/`](../../coordination/README.md), in numbered tasks with owners, leases,
plans and recorded evidence. An artifact there has an identity outside the
session that produced it, which is the property this file never had: it was
written at the end of a session, by that session, and nothing obliged the next
one to update it. It went ten weeks without an edit while the code moved
underneath it, and its last section still named a next thread that has since
shipped in a different form.

To resume design work:

1. Read the coordination task list (`coordination/STATUS.md`) for what is claimed, in
   progress and blocked. That is the authoritative answer to "where did work
   stop".
2. Pick a thread from [open/](../architecture/open/) — the live design questions
   — or from the queued areas in
   [design-review-method](./design-review-method.md).
3. Read [ACCELERATOR.md](../../ACCELERATOR.md) before touching the accelerator.
   It is the living document for the subsystem that absorbed most of the work
   after the archive below was written.

## Archive — design sessions to 2026-06-26

> **Historical record, not current.** Everything below was last edited
> 2026-06-26 and describes the tree as it was then. It is kept because it is the
> only narrative record of the `clifford` cutover and the `viete` landing, and
> because the reasoning in it is still worth reading. It is **not** authoritative
> about the present, and at least these claims in it are now false:
>
> - **`rembrandt` is not a scaffold.** It is the live vulkano device layer that
>   `aristotle` and `newton` both depend on, and the reason this workspace needs
>   a Vulkan device at *build* time. See [CLAUDE.md](../../CLAUDE.md) and the
>   [crate table](../architecture/overview.md#crates).
> - **The "next thread: SPIR-V backend" shipped in a different form.** Kernel
>   codegen goes through `viete` to **GLSL**, generated at build time by
>   `crates/newton/build.rs`, not to SPIR-V emitted from the trie.
> - **The paths it names have moved.** `.claude/architecture/` is now
>   `docs/architecture/`, `.claude/process/` is `docs/process/`, and the
>   `docs/superpowers/` specs and plans it cites were deleted; all of them remain
>   in git history.
>
> Where it disagrees with the code, with `CLAUDE.md`, with `ACCELERATOR.md` or
> with a `decided/` document, they win.

### What just landed (as of 2026-06-26)

- **`viete` — symbolic PGA3 tracer + Lua backend, NEW top-level crate, merged to
  `main` (2026-06-26).** Traces any `clifford::Scalar` computation over a pure
  symbolic scalar `Sym` (a `Handle`, no f64 shadow), enumerating **every**
  `is_effective_zero` branch (full 2^k, worklist + trail replay) into a raw
  decision-trie IR over an arena (`Vec<Block>` + `BlockId`); emits Lua (whole trie
  with runtime `if math.abs(cond)<ε` conditions) and runs it in-process via `mlua`,
  verified against the same computation on `f64`. **Headline acceptance:** traced
  `Motor::exp` → Lua → executed matches `Motor::<f64>::exp` within 1e-9 at both a
  non-trivial (closed-form arms) and a near-seam (Taylor arms) input; `Motor::exp`
  yields exactly 8 leaves (3 representation-selects), confirming the design count.
  **Key decisions:** honest `try_recip` (forks via `is_effective_zero`, never lies
  "always Some") + fatality caught per-path by `catch_unwind` → `Term::Fatal` leaf,
  classified `DivByZero`/`NonInvertible`/`Unexpected` (the last surfaces tracer
  bugs); value-dedup constant pool is the **load-bearing** trie-merge invariant;
  per-path slot ids reset → deterministic prefixes. Public API: opaque
  `Tracer::builder()` → `tracer.trace(|inp| …) -> Trace<N,K>` with `emit_lua()`
  (raw Lua), `run_lua()` (optional default-on `lua` feature gating `mlua`),
  `pretty()` (trie dump), `leaf_count()`/`fatal_leaves()`. The `MyScalar` prototype
  that lived in `rembrandt`'s working tree is superseded (rembrandt back to
  scaffold). 13 unit + 2 acceptance tests; whole workspace green; clippy clean.
  Spec/plan: `docs/superpowers/{specs,plans}/2026-06-26-{sym-pga3-trace-design,viete-symbolic-tracer}.md`
  (deleted with the abandoned corpus; in git history);
  crate doc [crates/viete](../architecture/crates/viete.md).
  → **Next thread: SPIR-V backend** emitted from the same trie (verified against the
  already-proven Lua emitter — two backends from one IR must agree), then the GPU
  integration. CSE/hash-consing, DCE, and tracing matrix inverse (`bareiss_inverse`)
  are explicitly out of scope until then.

- **`clifford::new` CUTOVER — `new/` IS `clifford` now; legacy GA stack deleted,
  consumers migrated, merged to `main` (2026-06-25).** The owner chose to cut over
  *before* the planned `&ops`/perf work (accepting possible perf loss for the
  structural win). `new/` was promoted to the crate root; legacy
  `multivector`/`jet`/`metric`/`math`/`pga3`/`trivial` deleted; `newton` +
  `hitchcock` migrated to the new vocabulary; `generic_const_exprs` /
  `generic_const_items` / `inherent_associated_types` feature gates dropped (the
  legacy `dof_blocks` const-expr was the last user). Whole workspace + `melies`
  green. **Consumer remap:** `PrimitiveScalar`→`Scalar + StandardPart`,
  `ReRecursive`→`StandardPart` (`.re_recursive()`→`.standard_part()`),
  `Uplift`/`uplift_array`→`Lift`, `Algebraic`→`Ring` (but `inertia` divides →
  `Scalar`), `Transcendental`→`Scalar`; `Differential::<24,_>`→`<2,24,_>` (two
  const params); `T::from_u32`→free `from_u32`; `is_effective_zero` on a generic
  scalar routed via `.standard_part()`. The GA value-type methods (`conjugate`,
  `cotransform`, `compose`, `exp`, `Twist::new` refs, …) were UNCHANGED.
  Spec/plan: `docs/superpowers/{specs,plans}/2026-06-25-clifford-new-cutover*`
  (deleted with the abandoned corpus; in git history).

  **Conventions that landed (some were owner corrections mid-session):**
  (1) `commutator` is the pure `a*b − b*a` (NO ½) by explicit choice; the GA
  commutator-product ½ lives in the twist/wrench SEMANTIC API (`Twist::velocity_at`,
  `Wrench::bracket`) as `HALF` consts — same place `Twist::exp`'s rotor-½ lives.
  (2) `AddAssign`/`SubAssign` are now `const` supertraits of `AbelianGroup`;
  container impls mutate IN PLACE (perf), never `*self = *self + rhs`.
  (3) Reflexive `Lift<T>` is required where `Dynamics::eval` runs at the value level
  (`S=T`); threaded through Newton/bridge/Mechanism/JointEdge/CameraField.
  (4) Value-type parity completed: `Mv: PartialEq` (hand-written, ignores the
  phantom `BladeLaw`), objects `Add`+`Mul<R>`, `Motor::rotation_part`.
  (5) **Near-zero AD singularities** (camera example panicked at `sqrt(0)` under AD):
  `Tangent` `sqrt`/`ln`/`atan2` derivatives divide (legacy parity, no panic); the
  real fix is the ε-softened norm `Mv::soft_norm` / `Line::soft_weight_norm` =
  `√(‖·‖²+ε²)` (finite gradient through coincidence; closed form — the ε floor alone suffices, no Taylor branch). The
  joints replaced their value-based coincidence guards with it. ε is a proper
  parameter: `newton::joint::default_softening()` (const fn; needed `const_ops`),
  joint `softening` field / builder `.softening()`, camera derives it scene-scaled
  as `r_object/1000`. See [decided/near-zero-ad-singularities](../architecture/decided/near-zero-ad-singularities.md).
  → **Next thread, in order: (1) `&ops` ref-overloads (still pending — see below);
  (2) perf on the `Jet12<Jet12>` tower; (3) wire `RigidBody::effective_size` into
  the rig softening in place of `r_object/1000`.** Re-run the camera example end-to-end.

- **`clifford::new` — full functional-parity rebuild of the GA substrate,
  merged to `main` (2026-06-22, ~33 commits + a const pass).** A from-scratch
  re-architecture in `clifford/src/new/`, replacing the *idea* that AD is a
  degenerate Clifford algebra (`Jet<N,T> = Cl(0,0,N)`, see the older entry below)
  with the **boson/fermion = ring/algebra split**: `Mv<L: BladeLaw, R: Ring>` is
  the multivector and AD is the **coefficient ring** `Tangent<N, R>` (not a
  degenerate metric). `BladeLaw` derives the product from `Gen`/`Tensor`
  generators via the Koszul sign (no `CliffordSpec`/cap machinery); higher-order
  AD comes from *nesting* `Tangent`. Trait chain
  `AbelianGroup→MulMonoid→Ring→{Commutative,Invertible}→Scalar` + `StandardPart`
  (the algebraically-correct successor to `re_recursive`). Full parity: gp (with
  the degenerate `s==0` fix), reverse, dual/undual, wedge/meet, norm, Bareiss
  `inverse_algebra`, the ℂ/ℍ/Minkowski identities, Twist/Motor/Wrench/objects, and
  the forward-AD `Differential`/`Dynamics` validated against finite differences on
  the two-body 24-axis Jacobian. **256 crate tests green** (106 `new::` + 150
  legacy). A const-arithmetic pass then hoisted `Motor`'s Study constants to
  compile-time literals (rodata-confirmed). Two oracle-decided sign iterations
  (as the spec predicted): rotation needs `Twist::exp`'s ½, and the translational
  `e0i` embed sign is flipped vs legacy (new Koszul order) — consistently across
  Twist+Wrench. A ported test caught a pre-existing `ln'` bug (`1/ln a` → `1/a`).
  Spec/plan: `docs/superpowers/{specs,plans}/2026-06-22-clifford-new-parity-*.md`
  (deleted with the abandoned corpus; in git history).
  → **Legacy `multivector`/`jet`/`pga3` are UNTOUCHED and still in place** —
  `new/` is a parallel, proven substrate, not yet a cutover. **Next thread, in
  order: (1) `&ops` ref-overloads — BEFORE perf measuring (else you measure a
  by-value copy artifact); (2) perf comparison on the target `Jet12<Jet12>` tower
  (hot point is `Tangent::mul`, not the gp loop); (3) cutover of
  newton/hitchcock/melies + delete legacy — LAST, on clean numbers.** `&ops`
  aliasing watch: `gp(&x,&x)` is everywhere (`B²`/`M·M̃`), `rotor_is_unit` is the
  sentinel.

- **Dual algebra / AD on clifford (`78c473c`→HEAD, 2026-06-05/06).** Built
  `Jet<N,T> = Cl(0,0,N)` — AD as a degenerate Clifford algebra reusing the
  `Multivector` core, not a separate system. Journey: standalone `Jet1`
  dual numbers → realized it's Cl(0,0,1) → generalized to Cl(0,0,N). This
  spawned `nilpotent_cap` grade-truncation (cap=1 flat multi-dual; legality
  invariant in `blade_count`; closed-form Neumann inverse since truncation
  makes the matrix inverse singular), the `Metric`→`CliffordSpec` rename
  (now `const trait`, carries cap), a sparse `GP_TERMS` gp, and
  `Transcendental` reworked into the AD-carrier (`_explicit` + `value`/`Real`).
  `Motor::exp` rewritten **differentiable** (branch-free over `u=l²`, smooth
  `sinc_sq/cos_sq/dsinc_sq`, no `√`/`÷l`). Validated through
  `PGA3<Jet6>` … `PGA3<Jet6<Jet6>>` (`pga3::motor::jet1_tests`). Docs:
  [crates/clifford](../architecture/crates/clifford.md#automatic-differentiation-jet),
  [decided/nilpotent-cap-truncation](../architecture/decided/nilpotent-cap-truncation.md).
  → **Decision: the implicit-integrator formulation is energy-based — seek the
  Hessian of a scalar energy, not the Jacobian of the wrench**, via
  `PGA3<Jet<1, Jet<6, T>>>` (inner Jet6 = ∇E force, outer Jet1 = ∇²E·v
  stiffness). Symmetric-by-Schwarz stiffness, variational, better-conditioned
  Newton solve. This *is* Risk-1 from the constraint-solver thread, now in
  energy form.
- Camera gravity-sensor thread (2026-06-01). Designed + implemented how
  the camera anchor derives a "down" reference: it is a gravity *sensor*
  (plumb-bob hang along ĝ), not a gravitating body. Two invariants now in
  [decided/camera-gravity-sensor](../architecture/decided/camera-gravity-sensor.md):
  (1) anchor gravity scaled **1/warp²** (else the warp²-soft spring hangs
  the anchor light-years out at high warp); (2) gravity is an **input** to
  `CameraField`, not a peer `ForceField` — anchor unregistered, no gravity
  field in the camera mechanism, propagator injected at
  `CameraField::new` and stored outside the lock. Added newton
  `GravityPropagator::force_on_probe` (point-field probe over the lag-1
  front snapshot; reusable for trajectory prediction) + `charge_of`
  method. Deferred angular work (perp damper for camera+interm too,
  ĝ→Motor, impulse target, roll degeneracy) in
  [open/camera-attitude](../architecture/open/camera-attitude.md) — owner
  will prototype the damper by hand on top of the correct spring+gravity
  perturbation.
- Integrator/field design thread (`9702215` → `05ad8a0` → `ac26862`).
  Reviewed the `ImplicitIntegrator` batch interface against actual code:
  its coupled-solve consumer (camera) had already moved to a `ForceField`
  (`CameraFieldArc`), leaving three thin explicit bridges and a stale
  doc comment — fixed (`9702215`). Resolved the central fork: the only
  code-grounded implicit need is per-body energy conservation, which fits
  `Integrator::step`, NOT `step_all`. A real coupled `step_all` consumer
  (rigid constraints / contacts) does not exist — joints are penalty.
  → **Do not migrate the `ImplicitIntegrator` signature.** De-risking
  prototype + findings landed instead
  ([findings/2026-06-01-implicit-constraint-solver-prototype](../architecture/findings/2026-06-01-implicit-constraint-solver-prototype.md),
  `crates/newton/tests/implicit_constraint_prototype.rs`). Key conclusion: with
  rigid constraints, phase 5 wants a new `Solver(bodies, f_ext, graph,
  dt)` — strictly more than `step_all` — so a **new trait, not an
  `ImplicitIntegrator` extension**. Two heavy risks named, not closed:
  PGA constraint Jacobian (Twist→Wrench) and contact complementarity.
- Camera spring + mechanism polish on `main` (`ed1bc7d` → `7072fb4`):
  `ForceField` moved off positional slices onto `IndexMap<WorldId, …>`
  (identity keying); camera `linear_wrench` takes `warp` directly.
- `clifford` / `newton` split (commits `45cd20b` → `5328757`)
  implementing the vessel-model-rfe DECIDED design.
- `Mechanism::split` / `merge` + disconnected-mechanism diagnostics
  (`7ed726c`, `feb1374`) — closes the merge/split body-identity
  residual.
- `demo` mid-migration onto `newton::Mechanism` (`ca91c7a` onward);
  `Body` in `demo/src/sim.rs` is constructor sugar.
- Body-frame DECIDED with four-pillar argument
  ([decided/frame-convention](../architecture/decided/frame-convention.md));
  world-frame attempt abandoned after conditioning argument
  (`a664c22` → `4321437` → `ba1ae0d`).
- Fixed-point representation and determinism boundary back in
  [open/](../architecture/open/) — `f64` is the current scalar;
  full Fix integration paused on `fixed`-crate trait friction.
- The 690-line monolith `CLAUDE.md` was split into
  `.claude/architecture/{crates,decided,open,planned}/` +
  `.claude/process/` per the spec
  `docs/superpowers/specs/2026-05-24-claude-md-refactor-design.md` (both the
  spec and those paths are gone: the spec was deleted with the abandoned
  corpus, and AR-0010 moved the trees to `docs/architecture/` and
  `docs/process/`).

### Next thread (owner's choice)

- **If the constraint-solver / implicit-integrator thread resumes:** the
  clifford-side AD substrate for Risk 1 now exists (`Jet`, differentiable
  `Motor::exp`, `PGA3<Jet<1,Jet<6,T>>>` Hessian-vector validated). The
  formulation decision is made — **energy Hessian, not wrench Jacobian**.
  Next prototype is the newton-side wiring: express a body/joint energy `E`
  as a scalar on the `Jet` stack, read ∇E (force) + ∇²E·v (stiffness), and
  decide where it lands relative to the existing Ad/Ad* machinery in
  `body.rs` — **before** any `Solver` trait is written. Trigger is still a
  real joint/contact feature — engine otherwise frozen.
- Otherwise pick one from [open/](../architecture/open/)
  (`fixed-point-vs-f64`, `determinism-boundary`, `body-rotation-attitude`,
  `close-encounters`, `relativistic-layer`) or a queued area in
  [design-review-method](./design-review-method.md).
