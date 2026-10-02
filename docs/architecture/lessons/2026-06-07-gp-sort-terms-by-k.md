# GP: sorting GP_TERMS by output slot `k` — hypothesis REFUTED (frozen lesson)

**Status.** A separate experiment on 2026-06-07 (after
[2026-06-07-gp-table-vs-compute](./2026-06-07-gp-table-vs-compute.md), but
independent of it). Hypothesis: group the terms of the `GP_TERMS` table by output slot
`k` at compile time, so that repeated `out.c[k] += …` become adjacent and
fold into a local reduction (one store at the end) instead of scattered
load-modify-stores of the same `out.c[k]`. The prototype on the branch `gp-sort-by-k`
(commit `442b73f`) **did not survive measurement**. We keep the table in generation order
(`si` outer, `sj` inner), as on `main`. The branch is preserved, and is NOT merged
into `main`.

Related: [2026-06-07-gp-table-vs-compute](./2026-06-07-gp-table-vs-compute.md),
[decided/nilpotent-cap-truncation](../decided/nilpotent-cap-truncation.md),
auto-memory `motor-exp-benchmark`.

## The hypothesis (why it sounded reasonable)

In the tower `PGA3<Jet12<Jet12>>` each `out.c[k]` of the outer PGA level is a whole
`Jet12<Jet12>` (169 f64 ≈ 1.3 KiB). If the triples come in arbitrary order of `k`, each
term hits its own large output slot → many load-modify-stores of wide
objects. Grouping by `k` (a stable insertion sort in the const generator,
**the body of `gp_into` is untouched, only the data layout changes**) was supposed to give
a local accumulator per `k` and one final store — "scatter into out" →
"reductions into a register". The cost — only the ordering in `generate_gp_terms`.

## What was measured (perf on the bench binary, `taskset -c 2`, RUSTFLAGS=native)

`coupling_exp_jet12_jet12`:

| metric                  | main (no sorting)     | branch (sorted by k)    |
|-------------------------|-----------------------|-------------------------|
| ns/iter                 | 434 502               | 445 572 (within noise)  |
| **instructions**        | **29.7058 G**         | **29.7057 G** (identical) |
| L1-dcache-loads         | 19.1959 G             | 19.1959 G               |
| L1 load-misses          | 263.5 M (1.37%)       | 273.9 M (1.43%)         |
| LLC references          | 87.1 M                | 70.6 M                  |
| LLC misses              | 0.90 M                | 1.23 M                  |

Correctness: all 127 tests green (the sum does not depend on the order of the terms).

## Why the hypothesis failed (root cause)

**The decisive number — the count of executed instructions matches bit for bit**
(29.7057 G in both). Permuting the const table does NOT change the instruction stream:
the same loads, the same stores, the same count. L1/LLC and wall-clock are within
run-to-run noise (the branch is even slightly slower).

The "local accumulator, one store" mechanism requires the compiler to **merge**
adjacent operations with the same `k`. For that the loop must be **unrolled, and `k`
must be a compile-time constant on every iteration**. But `gp_into` reads `k` from the table
at **runtime**, and LLVM keeps it as a data-driven loop (it does not unroll — the body is
a wide product of `T`). With a runtime `k` the compiler cannot prove
`out.c[k_i] == out.c[k_{i+1}]`, so it does a load-modify-store of `out.c[k]` on
**every** iteration regardless of the order. The data layout is invisible to the
optimizer.

On top of that, the "scatter" was not a bottleneck in the first place: the CPU store buffer
already forwards repeated stores to the same address — there was nothing to recover, which is
why neither instructions nor cache misses moved.

## General conclusion (together with the previous lesson)

Both 2026-06-07 experiments point to the same thing: **the compiler keeps the data-driven
`gp_into` loop as a loop and NEVER specializes it on `k`/`i`/`j`** (the body is
too large to unroll). Therefore neither a table-free rewrite nor
reordering the table changes what gets executed. The table-as-data
works precisely because the optimizer sees it as opaque runtime data.
The practical heuristic conclusion: **optimizations that rely on loop unrolling +
const-folding of indices will not work in `gp_into` as long as the body is heavy** — check
by measurement (instruction count is a quick detector of "codegen did not change").
