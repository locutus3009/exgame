# Game Experiment current coordination state

This file is generated. Read `README.md`, then use `tools/handoffctl snapshot`.
Never edit this file directly.

## In Progress

| Priority | Task | Summary | Next action | Owner |
| --- | --- | --- | --- | --- |
| P1 | [AR-0105](tasks/AR-0105.md): Fix the two FIXME: deadlock? reproducers in the implicit solver | dimension_change_reallocates_and_clears_the_hint and topology_change_causes_no_visible_jolt are #[ignore]d with 'FIXME: deadlock?'. Suspected: a WorldKey dropped while world.write::<T>() is held. | Claim; run both ignored tests with a timeout to confirm the hang, then confirm or refute the suspected cause before changing code. | worker-ar0105 |
| P3 | [AR-0107](tasks/AR-0107.md): Precision seam: measure f32 base poses far from the origin | The shaders bind Motor<f32> base poses directly; ACCELERATOR.md Part V calls the precision of that narrowing unexamined. Islands re-anchor at their centre of mass; measure whether that suffices. | Claim; write a test that steps the same mechanism at the origin and translated far away and compares relative motion. | worker-ar0107 |

## Open

| Priority | Task | Summary | Next action | Owner |
| --- | --- | --- | --- | --- |
| P0 | [AR-0101](tasks/AR-0101.md): World capacity: refuse overflow and grow between epochs | RawMap::insert has no capacity check, so a write past World's fixed 256 Ki slots is UB on the mapped path. Make overflow impossible and let a per-type buffer grow, re-binding descriptor sets. | Claim; add the capacity check and a failing growth test first, then implement growth with a per-map generation the accelerator re-binds on. | - |
| P0 | [AR-0104](tasks/AR-0104.md): Accelerator worker: failure path, bounded submission, flush policy | A Vulkan error panics the worker (dispatch(..).unwrap()), so callers only see WorkerGone; the channel is unbounded; every flush prints to stderr; batch composition is never varied in tests. | Claim; write the failing test for a forced submission error first. | - |

## Planned

| Priority | Task | Summary | Next action | Owner |
| --- | --- | --- | --- | --- |
| P0 | [AR-0102](tasks/AR-0102.md): Enforce one writer per slot per batch; repair the unsafe safety record | The world storage's 17 unsafe uses rest on 'at most one writer per slot', documented but unchecked. Check it on every GPU batch and bring every SAFETY comment in aristotle up to date. | Claim after AR-0101 lands; locate where each kind's output columns are known in shaders.rs and add the write-set check with a planted-collision test. | - |
| P1 | [AR-0103](tasks/AR-0103.md): Cross-mechanism epoch driver with frozen structure and quiescence flush | Nothing above Mechanism steps all mechanisms of an epoch; callers hand-roll join_all, structure can change mid-epoch, and the worker flushes on a 1 ns idle tick rather than on quiescence. | Claim after AR-0102 and AR-0104 land; design the driver API in the plan's evidence first, then implement and migrate callers. | - |
| P1 | [AR-0108](tasks/AR-0108.md): Retire ACCELERATOR.md and repoint every reference | With M1's implementation done, delete ACCELERATOR.md. Move the one authoritative piece (kernel I/O layout) next to the code that defines it and repoint every doc, tool and source comment that cites it. | Claim after every other M1 AR is done; regenerate the reference list with git grep, since the implementation ARs will have moved lines. | - |

## Done

| Priority | Task | Summary | Next action | Owner |
| --- | --- | --- | --- | --- |
| P2 | [AR-0106](tasks/AR-0106.md): Fatal semantics: size and count fatal slots from the trace | Kernels write fatal operands into a fixed float fatal[20] (build.rs FIXME), and n_fatals per kind is hand-entered in shaders.rs. Derive both from the trace and pin the mark-and-continue semantics. | Claim; read build.rs fatal_map and the shaders.rs check table, then make build.rs emit the per-kind fatal count. | - |
