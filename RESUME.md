# RESUME — state at the end of the 2026-10-02 session

Written when the session's three-hour timer fired. Nothing was running at shutdown: no worker or
reviewer agent, no claimed task, no open pull request. The canonical checkout is on `main`,
clean, and `coordination/tools/handoffctl doctor --live` reports consistent state.

## What happened

- **M1, "Accelerator completion", is complete.** All eight ARs are `done`, each reviewed at its
  exact head and merged through its own pull request. `ACCELERATOR.md` is deleted.
  `cargo test --workspace --release` on `main`: 415 passed, 0 failed.
- **M2, "Physics you can trust, and a first place to fly", is open.** It has nine ARs, all
  `planned`, and none started.
- **The roadmap after M2** is [docs/architecture/planned/roadmap.md](docs/architecture/planned/roadmap.md):
  - M3: a forkable state and the determinism gate.
  - M4: the whole Pater system.
  - M5: a graphics pipeline beside compute.
  - M6: an enforced frame budget.
  - M7: vessels.
  - M8: surfaces and atmospheres.
  - M9: the demo, where it stops.
  - Every milestone ends with a performance review.

## M1 — merged

| AR | What | PR |
| --- | --- | --- |
| AR-0106 | Fatal-slot counts generated from the trace; mark-and-continue per message | #2 |
| AR-0101 | World storage grows instead of writing past its buffer; descriptor sets rebind on growth | #3 |
| AR-0102 | One writer per slot enforced per GPU flush; every `unsafe` in `aristotle` has a SAFETY comment | #4 |
| AR-0104 | Flush failures return `EvalError::Backend` and the worker survives; bounded submission | #5 |
| AR-0103 | `newton::Driver`: epoch-wide concurrent stepping, structure frozen per epoch, quiescence flush | #6 |
| AR-0105 | The two `FIXME: deadlock?` tests fixed and running | #7 |
| AR-0107 | `precision_seam.rs`: f32 kernels with fixed-point frames, measured to 10⁹ | #8 |
| AR-0108 | `ACCELERATOR.md` retired; kernel I/O layout moved into `crates/newton/build.rs` | #9 |

Planning pull requests:
- #1: opened M1.
- #10: opened M2.
- #11: the roadmap.
- #12: the owner's direction for the roadmap.

**Incident, recorded in AR-0103's review.** PR #5 merged a stale head (`c5087de`), not the
reviewed one (`3d8f750`). The push default pointed at a remote that has since been removed, so the
worker's last commits never reached `origin`. `main`'s newton lib tests did not compile until
PR #6, which also supersedes the lost fix. Since then:
- every push names its target, as in `git push origin <branch>:<branch>`;
- every merge first checks that the PR head equals the commit that was verified.

## M2 — what is left (all `planned`)

Proposed order, one task at a time:

| Order | AR | What | Depends on |
| --- | --- | --- | --- |
| 1 | AR-0203 | Kepler orbit checks | — |
| 2 | AR-0201 | GPU-versus-CPU oracle for every generated kernel | — |
| 3 | AR-0207 | First Pater system, using setting-patera's decided values | AR-0203 |
| 4 | AR-0204 | Place a body at fixed-point frame precision | — |
| 5 | AR-0208 | Flyable ship around Terra: headless transfer test + `melies` example | AR-0207, AR-0204 |
| 6 | AR-0202 | Conservation tests per integrator | — |
| 7 | AR-0205 | BodyPost fatal coverage, typed `EvalError::Fatal` | AR-0201 |
| 8 | AR-0209 | Performance review and optimisation of everything M2 added | all implementation ARs |
| 9 | AR-0206 | Documentation drift, including superseding the stale scalar documents | all of the above |

## Leftovers

- **Worktree `coordination/workspaces/AR-0104`**, on branch `feature/ar-0104-worker-failures` at
  `3d8f750`.
  - It holds two commits that never reached `origin`; PR #6 superseded them.
  - The worktree is clean. Remove it with `git worktree remove coordination/workspaces/AR-0104`
    and delete the branch when convenient.
- **Merged feature branches remain on `origin`**, one per PR above. They are safe to delete.
- **Follow-ups found during M1 and not yet covered by an AR:**
  - Real driver-side submit and device-lost errors are untested; failures are injected before
    submit.
  - The quiescence flush has only been exercised on single-threaded test runtimes.
  - Nothing prevents dropping a `WorldKey` under its own type's write guard in other crates;
    AR-0105 fixed the one known site.

## Decisions on record

These come from the owner. They are also in the roadmap, and AR-0206 brings the older documents
into line.

- **Scalar split.** `f32` for local physics, fixed-point for reference frames.
  `decided/single-scalar-type.md` and `open/fixed-point-vs-f64.md` are stale.
- **Excluded.** No relativistic layer, which was only a discussion. No energy-Hessian solver,
  because of its runtime cost.
- **Every milestone ends with a performance review and optimisation task.** Build time is not a
  constraint.
- **The roadmap stops at the demo.** Economy and story are planned with the owner afterwards.

## How to resume

1. Bring the canonical checkout up to date and confirm the state:

   ```sh
   git switch main && git pull --ff-only origin main
   coordination/tools/handoffctl reconcile --commit --push
   coordination/tools/handoffctl doctor --live
   ```

2. Start M2 only when the owner says so, one worker at a time.
3. Run each task through the loop in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md):
   - promote, claim, create a worktree under `coordination/workspaces/AR-NNNN`;
   - implement, push with `git push origin <branch>:<branch>`, open a PR, submit;
   - review at the exact head, checking that the PR head equals the verified commit;
   - merge, then `handoffctl review --status done`.
4. Run tests in release only: `cargo test --workspace --release`. Never launch `melies` examples
   from automation.
5. When every M2 AR is done, run `coordination/tools/handoffctl complete-milestone M2`, then open M3
   from the roadmap.
