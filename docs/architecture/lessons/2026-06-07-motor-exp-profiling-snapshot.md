# Motor::exp — profiling snapshot across towers (reference, no conclusion)

**Status.** Pure measurements from 2026-06-07, NOT tied to a decision — a reference
performance snapshot of `Motor::exp` on different towers. Recorded "as is"
for the future (a baseline for comparison, if anything changes). For the decisions on `gp`
optimization see separately: [2026-06-07-gp-optimization-frozen](./2026-06-07-gp-optimization-frozen.md).

**Test bench.** Intel i7-7700HQ (Kaby Lake, AVX2, no AVX-512), 4 programmable
counters. `main` (table-driven `gp`, no_std + num-traits/libm). Build:
`RUSTFLAGS="-C target-cpu=native" cargo bench`, pinned with `taskset -c 2`, frequency under
load ~3.39 GHz (turbo). Timings — temporary `#[bench]`es (since reverted), perf — on
the bench binary.

**Method caveat.** libtest picks the iteration count itself for each run, and perf
was recorded in separate runs per event group (≤4 hw counters, no
multiplexing, except the cache group at ~83%). Therefore ABSOLUTE counters across groups
are not comparable — only RATIOS within a single run are valid (IPC, share of
vector FLOPs, share of mem operations) and `ns/iter` (libtest gives it per iteration).

## Wall-clock (`ns/iter`, 3 runs)

| tower | f64/component (total) | near-zero (Taylor series) | rotated θ=π/3 (closed branch) |
|---|--:|--:|--:|
| `PGA3<Jet12<Jet12>>` (Hessian) | 169 (2704 ≈ 21 KiB) | ~430 µs (411–480) | ~455 µs (440–493) |
| `PGA3<Jet6<Jet6>>`             | 49 (784 ≈ 6 KiB)    | ~44 µs (42.5–44.9) | ~46 µs (43–47) |
| `PGA3<Jet12<f64>>` (gradient)  | 13 (208 ≈ 1.6 KiB)  | ~15.6 µs (15.3–16.2) | ~21 µs (20.6–22.5) |

## Perf profile (near-zero, ratios within a run)

| metric | `J12<J12>` | `J6<J6>` | `J12<f64>` |
|---|--:|--:|--:|
| IPC | ~2.1–2.2 | ~2.1–2.27 | ~2.15 |
| FP scalar : 128b : 256b (instr.) | 5.52G : ~0 : 0.48G | 2.32G : 2.23G : 2.61G | 8.74G : ~0 : 0.49G |
| **share of vector FLOPs** | **~26%** | **~87%** | **~18%** |
| **mem operations, % of instructions** | **~83%** | **~54%** | **~83%** |
| loads : stores | 3.6 | 2.43 | 3.9 |
| L1 load-miss rate | ~1.37% | ~1.9% | ~0.056% |
| L2 / L3 / DRAM | negligible (L1/L2-resident) | negligible | negligible |
| branch-miss | ~0.4% | ~0.73% | ~1.48% |
| page-faults (one-off) / major / ctx-sw / migr | 156 / 0 / 0 / 0 | 146 / 0 / 0 / 0 | 130 / 0 / 0 / 0 |

## Perf profile (rotated, closed branch) — where measured

`PGA3<Jet12<f64>>` rotated: IPC ~2.39, share of vector FLOPs ~18%, mem operations
~58% of instructions (lower than near-zero's 83%), L1-miss ~0.068%, branch-miss ~1.30%.
I.e. the closed branch (√ + sin/cos + division of the dual) shifts the profile from
memory-bound to compute (transcendentals/division are register arithmetic),
hence +35% in time, but a higher IPC. For short towers this shift is noticeable; for
deep ones (`J12<J12>`, `J6<J6>`) the near-zero/rotated difference is small (gp dominates).

## Observation (a fact, not a decision)

Vectorization tracks the WIDTH OF THE LEAF `Jet`, not the depth of the tower and not the size
of a component: `Jet6` (6 axes) packs into AVX (6 = a 128-bit + a 256-bit lane) → ~87%
vector FLOPs; `Jet12` (12 axes) stays scalar (~18–26%) in BOTH towers,
including the short one with the smallest components (1.6 KiB) — so the issue is not
register pressure, but that LLVM's SLP/cost model does not take the 12-wide
axis loop. Memory meanwhile is practically ideal everywhere (L1/L2-resident, no
faults). This is exactly the foothold that the frozen "pointers/
references + explicit 256-bit packing" work would target, if `gp` ever becomes a bottleneck.

## For reference: per-call profile of `gp_into` (isolated, fixed N)

A single `gp_into` call (all triples are always traversed, no early-out ⇒
input-independent), `main`:

| type | time | cycles | instr | IPC | FP scalar | FP 256b | loads | stores | mem % | L1-miss |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `Jet12<f64>`        | 27.4 ns | 93 | 212 | 2.28 | 50 | 0 | 153 | 28 | 85% | ~0 |
| `Jet12<Jet12>`      | 840 ns | 2 839 | 6 114 | 2.15 | 1 275 | 75 | 4 228 | 1 028 | 86% | ~0 |
| `PGA3<Jet12<Jet12>>`| 175 µs | 598 055 | 1 306 693 | 2.19 | 247 296 | 21 888 | 858 989 | 240 173 | 84% | 0.87% |
