# Newton–Schulz block solver for the implicit integrator

> **Migrated design record.** This document was written under the previous
> process and lived in `docs/superpowers/specs/`, which was deleted. It was kept
> because [ACCELERATOR.md](../../../ACCELERATOR.md) cites it for the block-solver design and
> its rejected alternatives. Its cold-start claim was refuted by measurement — see
> [findings/2026-07-21-newton-schulz-convergence](../findings/2026-07-21-newton-schulz-convergence.md). It is history, not
> obligation: its `Status:` line records what was true when it was written, and
> where it disagrees with the code or with [ACCELERATOR.md](../../../ACCELERATOR.md),
> they win.

Date: 2026-07-21
Status: implemented (2026-07-21)
Related: [`/ACCELERATOR.md`](../../../ACCELERATOR.md) (Parts I, III–V),
[`2026-07-19-accelerator-storage-contract-design.md`](2026-07-19-accelerator-storage-contract-design.md)

## Problem

`Newton::solve_slae` is a dense Gaussian elimination with partial pivoting over
`Vec<Vec<T>>`. It runs once per Newton iteration (up to 32) per dt-subdivision
span, and `assemble_matrix` reallocates the whole `n × n` nest each time.

Gaussian elimination is inherently sequential: column `k` cannot start before
column `k-1` is eliminated, and the pivot search is a reduction across rows. It
is the one part of the step that cannot be handed to the accelerator, and it sits
directly in the way of the project's north star — all local physics running as
parallel kernels over World-backed storage.

## Goal

Replace the linear solve with an iteration built exclusively from matrix
multiplication (plus, for the cold start, block-diagonal inversion), so that the
whole inner loop of the implicit integrator becomes accelerator work with a
uniform, dependency-free task shape.

### Non-goals

- Sparse or fill-truncated approximate inverses.
- Strassen/Laderman fast multiplication (rejected below, with numbers).
- A batching policy or scheduler for the new task type. Batching heuristics are
  per-task-type and belong to a later, separate piece of work; the existing
  worker TODO already marks that seam.
- GPU dispatch itself.
- Any change to `aristotle`'s storage.

## Method: Newton–Schulz as a preconditioner generator

Order-2 Newton–Schulz, iterated on a persistent approximate inverse `X ≈ A⁻¹`:

```
R  ← A · X                 GEMM
X' ← X · (2I − R)          GEMM, ping-pong X ↔ X'
```

so that `I − A·X' = (I − A·X)²` — quadratic convergence whenever `‖I − A·X‖ < 1`.
After the budgeted iterations, the Newton correction is a block matvec
`dv ← X · rhs`.

`2I − R` needs no separate pass: while reading `R[k][j]`, the task adds `2I` when
`k == j`. Ping-pong buffering is required because GEMM 2 reads `X` while writing
`X'`.

**No convergence guarantee is computed, checked, or required, and there is no
fallback path.** `dv` is a search direction, not a solve. Quality is the job of
machinery that already exists in `newton_solve`: backtracking line search on the
nonlinear residual, and dt subdivision when no step size reduces it. The feedback
has the right sign — subdividing dt shrinks `(dt²/4)K`, moves `A` toward the
inertia block, and makes Newton–Schulz converge faster. The solver's arithmetic
cost is therefore a compile-time constant, which is exactly what a batched
dispatch of uniform tasks needs.

### Matrix and seed

With `half = dt/2`, `C = −∂W/∂V`, `K = −∂W/∂θ`, the assembled matrix is

```
A = 𝕀 + (dt/2)·C + (dt²/4)·K
```

where `𝕀` is the world inertia mass block (block-diagonal). The seed is the
**scaled transpose**

```
X₀ = Aᵀ / (‖A‖₁ · ‖A‖_∞)
```

chosen for one property: it converges for ANY nonsingular `A`, with no condition
on the step or the conditioning. Measured cost: 9 iterations at `dt = 1/60`, 18 at
`dt = 0.5`, roughly independent of stiffness. `ns_seed_iterations = 24`.

In the blocked layout its ingredients stay in the dispatch model: the transpose of
block `(i, j)` is the transpose of block `(j, i)`, a per-block operation, and the
two norms are absolute row and column sums, reductions over blocks.

**Why not the mass-block inverse.** `X₀ = 𝕀⁻¹` is the obvious choice — four times
cheaper, since `A → 𝕀` as `dt → 0`, and it needs no numerical inversion at all
(`Inertia::apply_inverse` gives it in closed form). It was implemented and then
removed. Its contraction condition works out to

```
‖I − A·𝕀⁻¹‖ = ‖[(dt/2)·C + (dt²/4)·K]·𝕀⁻¹‖ < 1   ⟺   dt < 2/ω
```

— exactly the limit an EXPLICIT scheme is stable under. Past it the iteration does
not converge slowly, it diverges quadratically: measured `‖R₀‖ = 14.2` at
`dt = 1/60`, `k = 5e4`, reaching `inf` by the eighth pass, and no budget rescues
it. An implicit integrator exists to escape that limit, so making its linear solve
depend on the same limit is incoherent. Full tables:
[`findings/2026-07-21-newton-schulz-convergence.md`](../findings/2026-07-21-newton-schulz-convergence.md).

### The divergence guard, and what it is NOT for

A guard still watches for growth, but only over a **stale warm hint**. The
unconditional-convergence guarantee belongs to the seed, not to an `X` carried
onto a changed `A`: subdivision moves the stiffness term by four and a frame-time
spike by more, and an inherited `X` can land outside the convergence domain. An
attempt to drop the guard entirely was caught by
`stiff_nonlinear_step_matches_a_fine_reference`.

Its remedy is to return to the seed that always works, so it cannot fire twice,
and correctness does not rest on it — unlike the earlier design, where the guard
stood between "works" and "explodes" on the cold path.

**The guard is a monitor, not a threshold**, and this is not stylistic. The
obvious test — reseed when `ρ ≥ 1` — is wrong: the textbook guarantee for the
transpose seed bounds the *spectral radius*, not the Frobenius norm, and at
`n = 12` even `‖I‖_F` is 3.46. Measured, the transpose seed starts at `ρ ≈ 2.8`
and converges monotonically, so a threshold test would reseed on every pass
forever. What distinguishes divergence is that it *grows*, and by a factor:

```
fire  ⟺  !(ρ_after < 4 · ρ_before)      // measures are squared, so 2× becomes 4×
```

Divergence is a squaring — `14.2 → 195` is 13.7× — while a converged iterate
bouncing on the machine-precision floor moves by about 1.25×, which a bare
`ρ_after ≥ ρ_before` would misread as divergence. The predicate is a negated `<`
so that NaN, which loses every comparison, fires the guard rather than slipping
past the case it exists to catch.

### Warm start is load-bearing (measured, 2026-07-21)

Between physics steps `A` changes slowly, so the persistent `X` is already nearly
`A⁻¹` and a budget of 2–3 suffices; from a cold seed the same accuracy needs
roughly `2·log₂ κ` iterations. Persistence is not an optimization here — a cold
start every step would make the method uncompetitive.

Measured and confirmed: from a converged `X` carried onto a matrix perturbed by
5% in `dt` — far more than one step's worth — two iterations reach a `dv`
relative error of 1e-3 and three reach 1e-6, at every stiffness from `k = 1e2` to
`5e4`. Stiffness barely registers on the warm path. `ns_iterations = 3`.

f32 was measured on the same path and carries the tolerance comfortably: the `dv`
error floor is 3e-7 against a 1e-4 target.

Consequently `X` is **a hint, never a correctness input**. A stale `X`, one from a
different `dt`, or one from a different configuration, cannot produce a wrong
answer — at worst the line search rejects the step and dt subdivides. The only
invalidation trigger is a change of dimension.

There is no separate "recompute vs. freeze the matrix" decision: one persistent
`X` continuously chases the current `A`, and freezing/rebuilding are just the
extreme values of the budget.

## Free parameters

`Newton` becomes a named-field struct:

```rust
pub struct Newton<T: …> {
    ns_iterations: usize,       // warm budget, per Newton iteration
    ns_seed_iterations: usize,  // budget when the cache is empty
    inner: RwLock<NewtonInner<T>>,
}
```

Three fields — below the project's builder threshold, so a plain constructor plus
setters is enough.

The two budgets are split because the cold case is genuinely different: on the
first step, and after any topology change that drops the cache, `X₀ = 𝕀⁻¹` is not
contractive for a stiff island, so a warm-sized budget would produce a poor `dv`,
fail the line search and subdivide dt for the first frames. Defaults for both come
from the phase 0 measurements, not from this document.

## Data layout

The matrix is stored blocked, 6×6 per body pair — `dim = 6·|order|` is
block-structured by construction (row block = body, column block = body).

| object | type | home |
|---|---|---|
| `A`, `X`, `X'`, `R` | `Arc<[WorldKey<[[T; 6]; 6]>]>`, `m²` entries | World; key tables in `NewtonCache` |
| `rhs` | `WorldKey<Wrench<T>>` per body | already exists |
| `dv` | `WorldKey<Twist<T>>` per body | already exists |

`[[T; 6]; 6]` is `Pod` for `T: Pod`, so this needs **no new storage type**. A
message carries `Arc` clones of the key tables — a refcount bump, exactly the
pattern `MessageInput::Gather` already uses for `incident_terms`. Nothing is
marshalled or reassembled: the matrix is already blocked and stored blocked.

Block access is addressed by index. `WorldKey::read` taking a read guard is the
CPU-side form of that access; on GPU the task receives indices and reads without
synchronization for the lifetime of the warp (Part III/IV).

`Vec<Vec<T>>` disappears. Assembly writes 6×6 blocks directly: a connection's
Jacobian `[[Wrench<T>; 24]; 2]` decomposes into blocks naturally, and the mass
block fills the diagonal.

All four matrices use the same **dense** `m²` key table; sparsity is exploited in
the arithmetic, not in the storage. `A` is structurally block-sparse — diagonal
plus one block per connection end — so GEMM 1 sums only over
`k ∈ {i} ∪ neighbours(i)`, reusing the same incidence structure `bake_incidence`
already produces, and never touches the blocks it knows are zero. `R = A·X` is
dense, so GEMM 2 sums over all `k`.

## Dispatch

A new `MessageInput` variant. The unit of work is one output block:

```
out[i][j] = Σ_k a[i][k] · b[k][j]
```

Block multiplication is an identity, not an approximation — it is ordinary matrix
multiplication over the ring of 6×6 matrices, so the total arithmetic is
unchanged: `m²` tasks × `m` terms × 216 multiplications = `216 m³`, and
`n³ = (6m)³ = 216 m³`. The ring is **non-commutative**: the
factor order `a[i][k]·b[k][j]` is fixed and must not be swapped.

Dependency structure:

- reads block-row `i` of `a` and block-column `j` of `b`, shared and read-only;
- writes `out[i][j]` and nothing else, and no other task writes it;
- never reads `out`.

So within one GEMM all `m²` cells are independent — the same disjointness
invariant the existing per-connection dispatch relies on.

**Cells are grouped into messages** (`GEMM_CHUNK = 32`). One cell per message is
the cleanest contract, but a cell is only ~2 µs of arithmetic against ~5 µs of
channel send, oneshot and rayon scheduling. Measured step time against chunk size
on the cloth demo (`m = 15`, 225 cells): `1 → 7.07 ms, 4 → 6.40, 8 → 6.09,
16 → 5.93, 32 → 5.85, 64 → 5.83`. Flat past 16 — parallelism is not the constraint
here, per-message overhead is. Grouping changes only how work is carried: cells
stay disjoint across messages and the reduction order inside a cell is unchanged,
so neither the result nor its determinism moves. The `Σ_k` reduction is
sequential *inside* one task, in fixed order, so determinism matches the gather
stage (ACCELERATOR.md §6). Ordering exists only *between* GEMMs: two barriers per
Newton–Schulz iteration.

This is consistent with the current dispatch granularity, which already sends one
message per body (`Pre`), per connection (`Eval`) and per body again (`Gather`).
Whether the resulting message count wants coarsening is a batching question,
deferred.

## Persistence

`Island` owns a `NewtonCache<T>`; `step_all` receives `&mut` to it.

This placement is forced by the borrow structure rather than chosen: `Mechanism::step`
drives islands with `islands.iter_mut()` and `island.with_mut()`, giving disjoint
`&mut` per island with no lock, while the integrator itself is shared across all
islands concurrently (`&*integrator` under `join_all`). Putting the cache in
`Newton` would need a lock on the hot path plus a stable island key — and there is
none, since `Islands` is a `Vec` indexed positionally and `repartition` reshuffles
it. In `Island` the lifetime is automatic: repartition rebuilds the island and the
cache dies with it.

Cache contents:

- `order: IndexMap<WorldId, usize>` and the block sparsity structure of `A` —
  these stop being rebuilt every step, closing the existing TODO at the bottom of
  `newton.rs`;
- the key tables for `A`, `X`, `X'`, `R`;
- the dimension `m`, the sole invalidation trigger.

`spans` moves out of `NewtonInner` into the per-island cache as well: it is a
per-island diagnostic, and sharing it across concurrently stepped islands is a
pre-existing race. It stays **internal** — the `eprintln!` trace and nothing
else; see the removal of `last_spans` below.

## Found during implementation

Two defects in the surrounding step control, both invisible while the linear
solve was exact and both fatal once it became approximate:

1. **Subdivision gave up early.** It continued only while it kept improving
   `best` (`best < parent_best`). That is sound for an exact solve, where `best`
   falls with every halving; an approximate solve bottoms out at its own accuracy
   floor, so the condition degenerated into "give up at depth 1–2" and a
   non-converged step was committed. A non-converged implicit-midpoint step does
   not conserve energy — it injects it, since the `2·P_mid − P_n` reconstruction
   doubles the solver's error. Observed as `1.25 → 2.66 → 14.5 → 250 → 692 →
   19106 → NaN`. The gate is removed; subdivision now runs until it converges.

2. **The line search wrote through its own iterate.** `WorldKey::clone` clones an
   `Arc`, so cloning the iterate map and writing through the clone writes the
   original. Trials were destructive and never rolled back: eight halvings
   accumulated `dv·(1 + ½ + ¼ + …)` instead of testing `dv·α`, and the
   "best seen" snapshot aliased the live iterate. Newton effectively never
   converged. Fixing it cut the demo step from 26.7 ms to 7.07 ms — the same
   class of defect the storage-contract spec records.

A backward-Euler reconstruction for non-converged spans was written to bound the
error doubling, then removed: it measured bit-identical at `dt = 30`, 100 and
1000 s, so it guards nothing observable.

## Removals

`solve_slae` is deleted outright — no production path, no test-only retention.
The degenerate-column skip goes with it: the mass block makes `A` nonsingular for
dynamic bodies structurally, and kinematic bodies are already excluded from
`order`.

The linear-solve oracle becomes a dev-dependency (`nalgebra`, pure Rust, no system
BLAS; version resolved with `cargo add`, not written by hand), used **only** under
`[dev-dependencies]`.

`Newton::last_spans` is deleted too. It has exactly one consumer in the whole
repository — an assertion in `stiff_nonlinear_step_subdivides_and_stays_finite`;
nothing in production reads it. A public accessor that exists only to let a test
observe solver internals is a sign the test asserts the wrong thing:
subdivision is the *mechanism* by which the integrator survives a stiff step, not
the property worth guarding — and this redesign changes that mechanism, since the
budget, the line search and the subdivision now interact differently. The test is
rewritten against a reference trajectory instead (see Tests). The `spans` counter
survives as an internal diagnostic only.

Nothing is captured from the Gauss path as a baseline number. Whether the new
solver costs more is a question of cost, not correctness, and belongs in
`cargo bench` (the repository's own precedent: `Instant` loops lie, the libtest
harness is the truth).

## Rejected: Strassen / Laderman on the 6×6 block

`6 = 2·3`, and the matrix-multiplication tensor is multiplicative under Kronecker
product, so rank multiplies: Strassen's 7 for 2×2 nested with Laderman's 23 for
3×3 gives 6×6 in 161 multiplications instead of 216. The rank count is correct;
the trade is not, because we pay for *operations*, not multiplications:

| | mults | adds | issued ops |
|---|---|---|---|
| naive 6×6 | 216 | 180 | **216 FMA** (adds fuse) |
| Strassen outer (2 over 3) | 161 | 7·98 + 18·9 = 848 | 1009 |
| Laderman outer (3 over 2) | 161 | 23·18 + 98·4 = 806 | 967 |

55 multiplications saved for ~640 additions, and the fast variants' pre/post-
combination is pure addition with nothing to fuse into: 967 issued operations
against 216 fused ones, 4.5× worse. Strassen is also not componentwise backward
stable, and f32 accuracy is already the scarcer resource here.

Rank saving pays only when a "scalar" is expensive relative to adding scalars.
That becomes true one level up — on the `m × m` block matrix over the 6×6 ring,
where a "multiply" is 216 ops and an "add" is 36. One Strassen level at `m = 16`
saves 12.5% of block multiplies for roughly 5% addition overhead, netting ~7%.
That 7% is declined deliberately: it is bought by destroying the uniform,
dependency-free `m²`-task shape — recursion, intermediate temporaries, input
pre-combination, output post-combination, extra memory passes — which is the
entire reason dense Newton–Schulz was chosen. cuBLAS declines it for the same
reason.

## Phase 0 — prototype (first)

A standalone f64 prototype answers the numerical questions before any storage or
dispatch is moved, following the `hitchcock/distance.rs` pattern: free functions,
parameter sweep, explicit anti-pathology guards.

It must run on **real** matrices, not synthesized ones: it calls the existing
`assemble_matrix` on real mechanisms and harvests `A`. This is only possible while
the current assembly and Gauss are still in place — an independent reason the
prototype comes first.

Sources of `A`: the camera rig, a cloth patch, and
`stiff_nonlinear_step_subdivides_and_stays_finite`, each at several `dt`.

Measurements:

1. `‖I − A·X‖_F` versus iteration budget 1..12 from the `𝕀⁻¹` seed, across the
   harvested matrices and their conditioning range.
2. Error of `dv` against `nalgebra`'s exact solve, per budget — the quantity that
   actually matters, since `dv` only needs to be a descent direction.
3. Warm start: perturb `A` by roughly one step's worth of change, re-run at budget
   1..3 from the previous `X`, confirm the residual stays low.
4. The same sweeps in f32, to locate the accuracy floor (`~n·ε·κ`) and decide
   whether f32 carries the target tolerance at κ ≈ 10⁴.

Outputs: defaults for `ns_iterations` and `ns_seed_iterations`; the `dt·ω` beyond
which a fixed budget stops yielding a descent direction; a verdict on f32. If f32
does not carry, that changes the design, not a parameter — which is the point of
measuring first.

## Phase 1 — port

1. Block storage and blocked assembly (`A` as 6×6 blocks in World; sparsity
   structure and `order` into `NewtonCache`).
2. `NewtonCache` on `Island`, `&mut` threaded through `step_all`.
3. The GEMM `MessageInput` variant and its CPU worker body.
4. Newton–Schulz loop and block matvec replacing `solve_slae`; `solve_slae` and
   the `Vec<Vec<T>>` assembly deleted.
5. `Newton` restructured with the two budget fields.

## Tests

1. **Block GEMM against flat GEMM** on random matrices — no external crate, a
   reference triple loop in the test. Catches index transposition and, above all,
   swapped factors: the block ring is non-commutative, and a flipped
   `b[k][j]·a[i][k]` passes every dimension check.
2. **Solve oracle**: Newton–Schulz at a generous budget reproduces `nalgebra`'s
   solution of `A·dv = rhs` to tolerance.
3. **Stiff step against a reference trajectory.** Replaces
   `stiff_nonlinear_step_subdivides_and_stays_finite`, whose assertion
   (`last_spans() > 1`) pinned the mechanism rather than the result. The same
   initial condition is integrated by an independent scheme — `LieEuler` at a
   small step — and the single large stiff step must match that reference within
   tolerance. For the existing case (`k = 5e4`, `m = 1`, so `ω ≈ 316 rad/s`) a
   reference at `dt = 1e-4` is stable with wide margin; 5000 steps is nothing in
   release. This asserts the physical property subdivision exists to deliver
   ("the large stiff step is accurate") and survives any change of solver
   internals. `is_finite()` alone would not do: it passes a solver that failed to
   subdivide and committed a garbage best-effort step that happened to stay
   finite.
4. **Budget sweep**: the physics tests
   (`kinematic_body_does_not_stall_dynamic_neighbour`, the stiff reference test
   above, `newton_free_body_conserves_world_momentum_and_energy`) parameterized
   over `ns_iterations ∈ {1, 2, 3, 5, 8}` — where accuracy against the reference
   breaks down.
5. **Warm start holds**: over a run of steps, `‖I − A·X‖` does not grow after the
   first few.
6. **No visible glitch after a topology change**: following a `detach` that drops
   the cache, the trajectory keeps matching the reference — a cold cache must not
   produce an observable physical jolt. How many steps the cache internally takes
   to recover is not observable and is not asserted.
7. **Mass-block DOF ordering** (`newton_mass_block_is_diagonal_for_diagonal_inertia`)
   survives the move to blocked assembly — it is the existing guard against a
   silent permutation between `Dof::ALL`, `Twist::basis`, `wrench_rows` and the
   `dv` unpacking, and blocking adds a new opportunity for exactly that bug.

All tests run in release (`cargo test --release`), per the repository rule.

## Deferred

- Batching policy for the GEMM task type, and per-task-type dispatch heuristics.
- GPU dispatch: index-addressed block reads, workgroup/tile mapping (the 6×6
  granularity may want re-tiling for GPU, which changes the dispatch internals but
  not the storage contract).
- Contiguous block runs in World (`GpuVec`, Part IV) in place of a table of
  per-block keys.
- Higher-order (p ≥ 3) Newton–Schulz. Marginally better GEMMs-per-digit (~6%)
  and fewer barriers per unit convergence; measurable once the budget parameter
  exists.
