# Coordinator setup

Requires Python 3.12+, Git, the GitHub CLI and the pinned uv quality environment. Every command
below runs from `coordination/` unless it says otherwise.

## Where state lives

Coordination state is versioned **inside the product repository**, under `coordination/`, and
every transaction commits straight to `main`. Until this record's restart (see
[HISTORY.md](HISTORY.md)) it lived in a separate repository; the tool, its rules and its tests
are unchanged in substance, and three things changed because state now shares a repository with
the product:

- **One checkout writes state: the canonical one, on `main`.** Every worker worktree carries its
  own copy of `coordination/tools/handoffctl`, but a transaction run from there would take that
  worktree's own lock and commit task state onto a feature branch. `handoffctl` therefore refuses
  to commit or replicate from a linked worktree, or from any branch other than `main`. Workers
  always invoke the canonical checkout's copy, by its path, from wherever they are working.
- **A state commit contains only state.** The canonical checkout is also a product checkout, so
  its index may hold staged product changes. `handoffctl` commits exactly the files its
  transaction wrote, never the rest of the index.
- **Product pull requests move `main` too.** Replication refuses a non-fast-forward push, so
  after a pull request is merged on GitHub the next replication reports that the replica
  diverged. Bring the canonical checkout up to date with `git pull --rebase origin main` (the
  unpushed state commits are re-signed by the rebase only if `commit.gpgsign` is set, so set it)
  and run `tools/handoffctl reconcile --commit --push`.

State commits go to `main` directly, not through a pull request. They touch only
`coordination/` and are made by the tool under its lock after the full `validate()`, which is
the review a pull request would otherwise stand in for. Product changes still land only through
a pull request — see [../docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md).

## Runtime configuration

Create `coordination/.runtime/config.json` (directory mode 0700, file mode 0600 — it is
gitignored) with:

```json
{
  "github_repository": "OWNER/REPOSITORY",
  "push_enabled": true
}
```

The product checkout whose worktrees are inventoried is, by default, the one holding
`coordination/`. `projects_root` and `product_worktree` may still be given to point elsewhere.

`handoffctl` commits with `git commit -S -s`, so configure a signing key and an identity whose
address matches the `Signed-off-by` trailer exactly — the DCO gate compares them character for
character. Commits must be signed with a key listed in
[`../config/allowed-keys.asc`](../config/allowed-keys.asc); for this project that is the GPG key
`25DB2FB39C1190A9`:

```sh
git config user.name  "YOUR NAME"
git config user.email "YOUR ADDRESS"
git config user.signingkey 25DB2FB39C1190A9
git config commit.gpgsign true
```

`gh auth login` (or `GH_TOKEN`) is required before any live command: `reconcile`, `snapshot`
and `doctor --live` call `gh pr list` and `gh run list`. The offline half — `promote`, `claim`,
`update`, `release`, `render-status` and plain `doctor` — never touches GitHub.

The product must have an `origin/main` before live reconciliation. Then run, from the canonical
checkout on `main`, `tools/handoffctl reconcile --commit --push`, `snapshot`, and
`doctor --live`.

## Worker workspaces

Every worker gets its own Git worktree of the product repository, and they all live under
`coordination/workspaces/`, which is gitignored:

    coordination/workspaces/AR-0101      # branch feature/some-topic
    coordination/workspaces/AR-0102      # branch feature/another-topic

The directory name **is** the AR identifier, and a task's `worktree_key` matches it, so the live
worktree inventory can be read against the task graph without a translation table. A change to
the coordinator itself is product work like any other: it gets a worktree here and lands through a
pull request. The canonical checkout stays where it is and stays on `main`.

A workspace is deleted as soon as its task is integrated. They are working space, not a record --
the record is the task, its evidence log and the merged commits.

**Never relocate a live worktree.** A task already under way keeps whatever path it started with
until it finishes, and is then deleted like any other. Moving one breaks a running worker outright,
and even when nothing is running it leaves the worktree's `.venv` with shebangs pointing at the old
path -- so native binaries such as `ruff` keep working while every Python entry point fails to spawn.
That is a half-broken toolchain, which is worse than a cleanly broken one, because a partial gate run
can be mistaken for a passing one. If a move has already happened, recreate the environment with
`uv sync` rather than working around it. The convention applies to new workspaces only.

`privacy_errors()` skips this directory. Without that it would walk entire product checkouts on
every `doctor` run, and a failure there would block every transition rather than only reporting a
problem.

## Asking a submitted task for more work

A task in `in_review` has no owner, so a worker cannot re-enter it: `claim` refuses because the task
is not open, `resume` refuses because it is not blocked, `update` and `submit` refuse because there
is no owner, and `review` refuses because a worker may not review its own submission.

So a submitted task is **returned to open in the same action as the request for more work**, never
after it. Otherwise the worker is stranded by construction and the coordinator has to be asked to
undo it.

This was learned twice. Both workers stopped at the self-review guard rather than working around it,
which is that guard doing its job in the one direction it was not designed for -- against the
coordinator.

## Scratch space

Hermetic gate runs, scratch clones and fixture trees go on a large volume, never on a small
`/tmp`: a single gate run is several gigabytes, and two concurrent ones can fill a small
filesystem. The runner reads `TMPDIR`, so point it at a directory with room to spare:

    TMPDIR=/path/to/large/volume ./tools/quality/run-gates.sh <revision>

That command runs from the repository root, not from `coordination/`.

## Validation

```sh
uv sync --locked --only-group quality
uv run ruff format --check tools tests
uv run ruff check --no-fix tools tests
uv run mypy tools/handoffctl.py tools/status_renderer.py tests
uv run coverage run --branch -m unittest discover -s tests -p 'test_*.py'
uv run coverage report --fail-under=95
uv run python tests/validate_schema.py
tools/handoffctl render-status --check
tools/handoffctl doctor
tools/handoffctl check-commits --base origin/main --head HEAD
```

## Recovering an abandoned claim

A worker that dies before it submits leaves its claim behind. Long-running agent workers die
for ordinary reasons -- rate limits, timeouts, crashes -- so this is a durability property
rather than an incident, and `doctor` is meant to report it:

    ERROR: AR-0007: expired claim

That report is the lease working. What used to follow was not. `validate()` runs over every
task inside every transaction and refuses on any error, so one lapsed claim refused **every**
mutation, including `release` and `heartbeat`, the only two transitions that could have
cleared one. The coordinator could not repair the coordinator, and the only recovery on record
was a human editing task files outside the tool. The red check also propagated: the hosted
check runs `doctor`, so an unrelated merge failed on state it had never touched.

The recovery is now a transition of its own:

    tools/handoffctl expire AR-NNNN --expected-revision REVISION \
        --cleared-by WHO --note "why the worker is gone"

`doctor` prints this form itself whenever it reports an expired claim, so a red check names
the way out of itself.

What it does, and what it deliberately does not do:

- It returns the task to `open` and clears `owner` and `claim_expires`, so the work goes back
  in the queue and any worker may claim it again. It names no other destination: reopening the
  queue is the whole repair, and every other status has a transition that already governs it.
- It is **not owner authenticated**. The owner of an abandoned claim is by definition unable
  to call anything, and a transition that accepted their name would invite a caller to assert
  an identity nobody holds. `--cleared-by` and `--note` are both required and both non-empty,
  and the appended evidence entry records who cleared the claim, whose claim it was, when the
  lease ended, and why -- so a repair is a record and never a silent edit.
- It **refuses a claim that has not lapsed**, so it can never take work from a live worker.
  The condition it accepts is exactly the one `doctor` reports; a task whose expiry is absent
  or unreadable is a different fault and is refused with `does not hold an expired claim`.
- It is the **only** transition whose pre-commit validation tolerates a standing error, and it
  tolerates only messages that are *exactly* an expired claim -- matched whole, never as a
  substring of some other fault. Every other transition stays fenced by the full check, and
  `expire` itself is still refused by any other fault: a corrupt task, a broken dependency
  graph, a milestone fault, a privacy leak, or a claim error of a different kind. A lease that
  lapses part-way through the transaction is not excused either, because the tolerated set is
  fixed before the write. One thing it is *not* refused by is a stale `CURRENT.md` or
  `STATUS.md` -- but neither is any other transition, because `mutate` regenerates both from
  the tasks immediately before it validates.
- **Each abandoned claim needs its own `expire`, and they do not block each other.** The
  tolerated set is *every* standing expired claim, not only the one being cleared, so three
  lapsed leases are cleared by three calls in any order and each one succeeds. What keeps
  refusing meanwhile is every *other* transition, until the last claim is gone. That is the
  fence, not a failure.

`--cleared-by`, like `--reviewer` and `--owner`, is self-asserted. It makes an unattributed
repair impossible to do by accident and impossible to do without writing a false name into an
append-only log; it is not authentication, and does not pretend to be.

An expired claim is still an error, still reported by `doctor`, and still red on the hosted
check. The only thing that changed is that the process can now clear it from inside itself.

### One non-expiry fault blocks the recovery of every abandoned claim

This is the case to recognise mid-incident, because the deadlock comes back whole.

Suppose three tasks are `in_progress` and one of them records no `claim_expires` at all. A
hand edit is the usual way that happens -- and a hand edit is exactly what the one recorded
prior recovery used, so this state is reachable from the previous incident's own repair.
`doctor` calls the third task `active without claim`, which is a different message from an
expired claim and therefore outside the tolerance. The result:

- `expire` on either of the two genuinely abandoned tasks is refused -- **on the third task**,
  not on the task named on the command line.
- `expire` on the third task is refused with `does not hold an expired claim`, because its
  claim never lapsed; it was never written.

Nothing inside the tool clears any of the three. **Fix the non-expiry fault first** -- by hand
if that is what it takes -- and only then run `expire` for each abandoned claim. Reading the
whole of `doctor`'s output rather than its first line is what tells you which case you are in:
if every reported error ends in `expired claim`, the `expire` calls above are enough; if any
does not, that one comes first.

## Commit trailers: no session reference

The product's message-privacy check (gate 12) rejects a `Claude-Session:` trailer as a private
agent session reference, and the coordinator's own `check-commits` does the same. The
harness may ask an agent to add one; **the repository policy wins.** Keep `Co-Authored-By`, drop
the session URL. This was settled when the process was set up -- attribution yes, session URL no --
and the gate enforces it.

A worker found this the expensive way before the restart (AR-0016): its first commit failed
gate 12 on exactly this. Every agent committing here will hit it.

## `next_action` is the next action, not a summary

`maxLength` is 300 and it is not negotiable: the field tells the next person what to do, and the
evidence log is where the reasoning goes. Before the restart, workers repeatedly wrote a review
summary there -- AR-0007 reached 587 characters and AR-0009 540, which put `tests/validate_schema.py` in the red on
`main` while `doctor` still passed, because the transaction did not yet enforce the published schema.
AR-0015 closes that gap, so an over-long `next_action` now fails `doctor` as well.

That combination is a trap worth naming. A task in `in_review` has no owner, so `update` refuses for
want of one and `claim` refuses because the task is not open -- and once the transaction enforces the
schema, every other transition is fenced by the same `validate()`. It is the expired-claim deadlock
in a new shape. **Shorten `next_action` before submitting, not after.**
