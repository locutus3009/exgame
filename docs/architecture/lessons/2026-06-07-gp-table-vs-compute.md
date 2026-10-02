# GP: table vs direct computation — hypothesis REFUTED (frozen lesson)

**Status.** The hypothesis from the note `docs/GP_TABLE_VS_COMPUTE.md` (sketched
2026-06-06) was tested with a prototype on 2026-06-07 and **did not survive measurement**.
Conclusion: for `Jet` the `GP_TERMS` table does NOT get in the way — on the contrary, it is
the only fast path. We keep **only the table-driven `gp`** (as it was on `main`). The
prototype is preserved on the branch `gp-path-select` (commits `45cfac1`, `dec8d06`) as an
archaeological artifact, and is NOT merged into `main`. Below: what was assumed, what was
measured, and why the assumptions failed. The original note is appended at the end for archaeology.

Related: [decided/nilpotent-cap-truncation](../decided/nilpotent-cap-truncation.md),
auto-memory `motor-exp-benchmark`, `const-fn-in-loop-not-const-folded`.

---

## What the note assumed

The economics of `gp` supposedly flip between a full metric (PGA) and a
nilpotent one (Jet):

- PGA: the blade sign is expensive (a popcount fold over transpositions) → a cached table
  pays off (**this was confirmed and stays**);
- Jet (cap=1): the sign is trivial (+1), while the table is "large, sparse and materialized
  on inlining (≈16% of the profile)" → **direct computation from masks would supposedly be
  cheaper, and the table gets in the way**.

The plan was: a three-way compile-time selector in `gp` — table / strict cap=1 formula
(`out₀=a₀b₀; out[εᵢ]=a₀bᵢ+aᵢb₀`) / generalized table-free computation,
with a default derived from `nilpotent_cap` and an explicit boolean override in `CliffordSpec`.

## What was measured (RUSTFLAGS=native, `taskset -c 2`)

Microbenchmark of a single `gp` on the target `Jet12<Jet12>` (T = `Jet12<f64>`):

| path    | ns/iter |
|---------|---------|
| table   | **174** |
| cap=1   | 1 142   |
| general | 1 199   |

End-to-end `coupling_exp_jet12_jet12` (`Motor::exp` on `PGA3<Jet12<Jet12>>`):

| default                 | ns/iter |
|-------------------------|---------|
| `main` (table everywhere) | **411 000** |
| branch (Jet → cap=1)    | 721 000 |

I.e. the "optimization" makes the target workload **~75% SLOWER**. All three paths
give bitwise-identical results (the `path_equiv` test on the branch) — the failure is purely
one of performance, not correctness.

## Why the assumptions failed (root cause — from the asm)

Both monomorphizations were disassembled (`--emit asm`, Intel syntax) for
`Jet12<Jet12>`:

- **Table path** (~53 lines of asm): a tight loop of 25 iterations
  (`cmp rax,25; jne`). The indices `(i,j,k)` are loaded from a const table at runtime,
  so LLVM **keeps it as a loop** and vectorizes the *inner contiguous*
  `Jet12<f64>` product 4 f64 at a time (`vmulpd ymm`/`vaddpd ymm`). No gather, no
  stack frame.

- **cap=1 path** (~747 lines of asm): the bounded per-axis loop `while s < 13`
  is **fully unrolled**, after which the autovectorizer tries to
  vectorize *along the axis* — and the axes in `[Jet12<f64>; 13]` lie at a **stride of
  104 bytes**. Result: **51× `vgatherqpd`** (gathers), heaps of
  `vpcmpeqd/vmovhpd/vpextrq/vextractf128` and a **stack frame `sub rsp, 1736`**
  (a massive spill) + 6 callee-saved pushes. This gather-over-stride is what kills the path.

**Verdict in one line:** a table expressed as *data* gives the autovectorizer no
temptation to gather along strided axes; an *explicitly unrolled* per-axis loop
does. The source of the table's speed is not a "cheap sign", but the fact that a loop driven
by a length/indices from an array is kept as a loop by LLVM, which vectorizes the contiguous
inner product. The "~16% materialization of `GP_TERMS`" in the old profile is
cheap L1 reads of 8-byte triplets, not a bottleneck.

## What remains useful from this

- Confirmed: the table is the right path for **both** axes of the tower (both PGA and Jet).
- Methodologically: hypotheses of the form "the compiler will fold/simplify" must be checked
  with a measurement + asm BEFORE committing to a design (exactly as the note itself asked in
  item #1). Here the intuition "branchless is cheaper" turned out to be the opposite of the
  real codegen.
- For the future (if a table-free path is ever needed, e.g. for
  GPU/SPIR-V, which we are NOT doing now): write it so that it does NOT provoke
  autovectorization along the strided outer axis — for example, keep the outer
  traversal a data-driven loop rather than unrolling over a fixed N.

---

## Appendix: the original note (archaeology, as written on 2026-06-06)

> # GP: table vs direct computation — choice by metric
>
> A reminder note. Optimizing `gp` (the geometric product): for some
> metrics the cached `GP_TERMS` table is optimal, for others — direct computation
> from blade bitmasks. The choice is a property of the metric, made at compile time.
>
> ## Gist
>
> The economics of `gp` flip between a full metric and a nilpotent one:
>
> - **PGA3 (full Cl(3,0,1), cap ≥ N):** the sign of a blade product is expensive — a popcount
>   fold over transpositions to bring it to canonical order. The table caches
>   `(i, j, sign, k)`, is read from hot L1, and also skips pairs known to be zero
>   (they are simply absent from the table). **The table is optimal — confirmed by metrics.**
>
> - **Jet (nilpotent, εᵢ²=0, cap=1, truncation):** the sign is TRIVIAL. The product of
>   blades is degenerately simple:
>   - `a & b ≠ 0` → 0  (repeating any εᵢ zeroes it, εᵢ²=0) — a simple intersection check;
>   - `a & b = 0` and `popcount(a|b) ≤ cap` → ±ε_{a|b}, at cap=1 one factor is grade-0,
>     there are no permutations, the sign is +1.
>   No popcount fold over transpositions — a simple counter (intersection + cap check).
>   Plus the table for Jet is large and gets materialized (`const GP_TERMS` is copied on inlining
>   — ~16% observed in the profile). **Computation is cheaper — the table gets in the way.**  ← REFUTED
>
> The reason for the flip: nilpotency kills the permutation fuss over the sign.
> Expensive sign → the cache pays off (PGA). Cheap sign + a large sparse table →
> the cache is unnecessary and even harmful (Jet).  ← the last sentence is WRONG (see measurement/asm above)
>
> ## Path-selection criterion
>
> Derived from already existing constants, WITHOUT a separate flag:
>
> ```
> M::NILPOTENT_CAP < N   →   truncating metric (nilpotent by construction)
>                             → branchless computation from masks
> otherwise              →   GP_TERMS table
> ```
>
> A single source of truth (cap), no extra flag to keep consistent.
> (If there are more metrics with non-trivial economics — switch to an explicit
> `const USE_GP_TABLE: bool` in CliffordSpec.)
>
> ## Blade bit arithmetic (the common source that the table caches)
>
> A blade = a bitmask of generators. The product:
> - resulting blade: `k = a ^ b` (XOR of masks: shared generators cancel, the rest combine);
> - zeroing of degenerate ones (e0²=0 in PGA / εᵢ²=0 in Jet): the mask `a & b & DEGENERATE`;
> - sign (PGA, expensive): a popcount fold over pairs (i from a, j from b) with i>j — the parity of the number of transpositions;
> - sign (Jet cap=1, cheap): trivial (a grade-0 factor).
>
> The sign is folded in by MULTIPLICATION, not branching: `out.c[k] += sign * prod` (sign = ±1.0).
> The `GP_TERMS` table is the cached result of this bit arithmetic.
> Cache it (PGA: recomputation is expensive) or recompute it (Jet: recomputation is cheap) — the metric decides.
>
> ## CRITICAL: the implementation must fold at compile time
>
> Path selection — through an ASSOCIATED CONSTANT in the condition, NOT through a method:
>
> ```rust
> // GOOD: both operands are compile-time const → the if folds, the dead branch is dropped,
> // there is NO runtime branching, each metric is monomorphized into its single path.
> if M::NILPOTENT_CAP < N {
>     // direct computation from masks (Jet)
> } else {
>     // GP_TERMS table (PGA)
> }
> ```
>
> ```rust
> // BAD: a method/function in the condition goes to RUNTIME (will not fold).
> if some_method() { ... }   // a runtime branch in the hot gp — exactly what we are avoiding
> ```
>
> This project has already stepped on this rake: `blade_count` in a loop condition was not hoisted →
> a 1000× slowdown. The same vigilance here: a const in the condition — good, a method — bad.
> (In the prototype the selector was lifted into monomorphization associated constants — it folded
>  correctly; what failed was not the dispatch but the compute path itself.)
>
> **MUST be checked in MIR/asm** that the branch is gone: for `PGA3<f64>` only
> the table branch remains, for `Jet12<f64>` — only the compute branch, with no runtime `if`.
> Do not trust that it folded — look at the generated code.  ← it was precisely the asm that refuted the hypothesis
>
> ## Towers pick it up automatically (per-metric)
>
> `PGA3<Jet12<Jet12>>`: the outer level is Pga3Metric (cap≥N → table), the inner ones are
> JetMetric (cap<N → computation). Each level of `gp` looks at ITS OWN metric and picks
> its own path. The tower recursion assembles the mix by itself, no global decision is needed.
>
> ## CPU/GPU — a separate axis (for the future, for SPIR-V)
>
> Branchless computation from masks is also valuable on the GPU: no table in memory (latency),
> no warp divergence (all threads execute the same instructions on different data), register-only.
> But for CPU-PGA3 the table wins (measured) — the popcount fold is more expensive than an L1 read.
> So branchless for PGA is a potential GPU path (to be measured on SPIR-V), NOT a CPU
> improvement. For Jet computation wins on BOTH (cheap sign). The switch in the spec
> catches both axes: a property of the metric + (if needed) a property of the target.
> ← "for Jet computation wins" is WRONG on CPU (a ×6.5 loss); the GPU axis was not tested
>
> ## Status
>
> - [x] measure the Jet path both ways (table vs masks) on Jet12<Jet12> — DONE, hypothesis refuted
> - [x] implement the `if M::NILPOTENT_CAP < N` branching in gp_into — DONE in the prototype (branch gp-path-select)
> - [x] check in MIR/asm — DONE, the asm is what showed gather-over-stride
> - [ ] (far off) GPU/SPIR-V: branchless from masks for PGA — NOT doing it (out of scope)
