# `gp` optimization — effort FROZEN (frozen lesson)

**Status.** On 2026-06-07 the series of "let's improve the geometric product" experiments
was closed. **We stay with the current table-driven `gp` implementation** (`main`). No path
yielded a gain; one of them required sacrificing a strict invariant. The effort is
frozen — reopen it only if we hit `gp` from inside the implicit
integrator (and even that is unlikely — see below). All branches are preserved as archaeology,
and are NOT merged into `main`.

Brings together:
[2026-06-07-gp-table-vs-compute](./2026-06-07-gp-table-vs-compute.md),
[2026-06-07-gp-sort-terms-by-k](./2026-06-07-gp-sort-terms-by-k.md).
Related: [decided/nilpotent-cap-truncation](../decided/nilpotent-cap-truncation.md),
[decided/core-design-principle](../decided/core-design-principle.md).

## Main conclusion (the principle)

**A strict invariant (`no_std`) matters more than complicating the structure and losing `const`.**
The path that was most promising in concept (explicit `MulAdd`/FMA, branch `gp-muladd`) required:
dropping `no_std` from the crate, moving `num-traits` to `std` (otherwise `MulAdd` for f64 goes
to the **software fma** of `libm`, see below), and making the whole `gp`/`Mul`/`Div`/`try_inverse`
path **non-`const`** (num_traits `MulAdd` is not const). That is a cascade of complications for an
unconfirmed gain — rejected at the level of principle, never taken as far as a measurement.
The core-purity invariant (no_std, const substrate) is worth more than hypothetical
percentages.

## What was tried, and with what numbers

1. **A table-free path for Jet (cap=1) + a generalized one** (branch `gp-path-select`):
   REFUTED — 6.5× / 75% slower. The cap1 loop gets unrolled → the autovectorizer
   gathers along strided axes + spills. Details in a separate lesson.
2. **Sorting `GP_TERMS` by output slot `k`** (branch `gp-sort-by-k`):
   REFUTED — the instruction count is bit-for-bit the same, zero effect. The loop is
   data-driven, `k` is a runtime value → the compiler does not specialize, the layout is invisible.
3. **`std` instead of `libm` (+ dropping no_std)** (branch `std-feature`): NEUTRAL —
   the `coupling_exp` instruction stream is identical (FP/256b/loads/stores bit-for-bit),
   wall-clock within noise. `Motor::exp` goes through a branch-free polynomial in `u=l²`
   and does not call transcendentals (`sin/cos/√`), so the libm-vs-std route does not matter.
4. **Explicit `MulAdd`/FMA** (branch `gp-muladd`): not completed — rejected on the
   no_std principle (see above). A trap worth noting: `num-traits` with the `libm` feature
   (without `std`) compiles `f64::mul_add` to the **software** `libm::fma`
   (`float.rs:2144`), NOT to the hardware `vfmadd`; the hardware one only comes through `std`
   (`Self::mul_add` → `llvm.fma`). That is, "just enable MulAdd with libm" would have given
   a regression, not a speedup.

## Why there is nothing to squeeze out (profile of `gp_into`, perf)

The current table-driven `gp_into` is **a single pass over a table built once**,
and it already hits the right limits:
- **memory-movement-bound**: ~85% of retired instructions are load/store, 2.5–3 loads per
  FLOP; IPC ≈ 2.1–2.3 (≈ the ceiling of Kaby Lake's two load ports);
- the arithmetic is almost entirely scalar f64 (256-bit lanes — 0/19/26% of FLOPs across
  the tower levels): the 13-wide `Jet` is an odd shape, the autovectorizer does not take it;
- cache/branches are clean: data is L1/L2-resident, branch-miss ≤0.4%, no DRAM/faults.

## The only real lever (for later, NOT now)

Working with pointers/references to cut per-term loads of operands/the table
and to pack the wide `T` into 256-bit lanes (13-wide product → AVX packed) —
hits both scarce resources at once (load count + FP scalarity). Estimated gain
~**10–20%**, at the cost of noticeable complexity. Not justified at the current stage.

## When to reopen

Only if **the implicit (midpoint) integrator for dynamic bodies actually hits
`gp`** as a hot spot. The hunch: even there most of the time will go not into `gp` but
into the linear algebra of the Newton step (Cholesky/factorization of the symmetric stiffness, etc.),
so we measure THERE first, and only then — if `gp` surfaces — unfreeze the branches.
