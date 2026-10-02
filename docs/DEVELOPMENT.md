# Development process

This is the canonical process for working on `game-experiment`. It is not advice. A change
that did not go through it is not integrable, however good the code is.

Coordination state — tasks, plans, milestones, leases and evidence — lives in this repository,
under [`coordination/`](../coordination/README.md), and is written only by its tool,
`coordination/tools/handoffctl`, run from the **canonical checkout on `main`**. A worker's own
worktree carries a copy of the tool, but `handoffctl` refuses to write state from a linked
worktree or from any other branch; workers call the canonical checkout's copy by its path. In the
commands below, `CANONICAL` stands for that checkout. Its absolute location is local
configuration and never belongs in a commit message, a task note or a tracked file. Setup, and
the reasons for each of these rules, are in [coordination/SETUP.md](../coordination/SETUP.md).

The task set restarted from empty when the project was published. AR numbers below AR-0100 cited
in this and other documents are the retired M0 series, recorded in
[coordination/HISTORY.md](../coordination/HISTORY.md).

Read this document with [QUALITY.md](QUALITY.md) (what is actually enforced, and what is only
promised) and [QUALITY_GATES.md](QUALITY_GATES.md) (the gate list itself).

## The unit of work

The unit of work is an **AR** — one numbered task file under `coordination/tasks/`, with a
plan beside it in `coordination/plans/`. An AR names its own outcome, its dependencies, the exact set of paths it owns,
its branch and its worktree.

**One worker owns one AR, one branch and one worktree at a time.** Not two ARs, not one AR on
two branches, not two workers on one AR. This is the rule that makes the ownership tables
meaningful: if it holds, two ARs whose path sets are disjoint can run in parallel without any
further negotiation, and if it does not hold, no amount of care downstream recovers the
guarantee.

A worker edits only the paths its AR lists. Paths it does not own are read-only to it, even
when the fix is obvious and one line long, and even when the other AR has not started.

### Discovering work that is not yours

You will find defects outside your paths. That is normal and it is worth reporting. What you
must not do is widen your own scope to absorb them. There are exactly two correct moves:

- If the discovery **blocks** your AR, release it as `blocked` with a note naming the external
  dependency precisely, and stop.
- If the discovery does **not** block you, finish your AR and ask the coordinator to open a new
  AR for the discovery. Record what you found, where, and what proves it, in your evidence note.

Scope creep is the failure mode this process exists to prevent, and it is the one that always
looks like diligence at the time.

## The loop

### 1. Read before claiming

Read the AR, its plan, the milestone it belongs to, and this document. Check that every
dependency the AR names is `done`. Only an `open` task with satisfied dependencies may be
claimed; `planned` work becomes `open` through a reviewed coordinator promotion, not by a
worker deciding it is ready.

### 2. Claim, with a lease

```sh
CANONICAL/coordination/tools/handoffctl claim AR-NNNN --owner WORKER_ID --lease-minutes 180
```

The claim is a lease, not a lock forever. It records an owner and an expiry. An expired lease
means the coordinator may hand the work to someone else — so an expired lease also means *you*
must reconcile before touching anything, because the tree may have moved.

Create the worktree and branch the AR names, from the canonical checkout, at
`coordination/workspaces/AR-NNNN` (gitignored). Never work directly in the canonical checkout,
and never leave a worktree on `main`: the canonical checkout stays on `main` because it is the
one place state transactions run.

### 3. Heartbeat

```sh
CANONICAL/coordination/tools/handoffctl heartbeat AR-NNNN --owner WORKER_ID --lease-minutes 180
```

Heartbeat before the lease expires, not after. A silently expired lease on work that is in fact
progressing is indistinguishable, from outside, from an abandoned worker.

### 4. Run commands through the wrapper

```sh
CANONICAL/coordination/tools/handoffctl run --owner WORKER_ID AR-NNNN -- COMMAND ARGUMENTS
```

The wrapper records the command by argv digest together with its exit code, against the task.
This is what turns "I checked" into evidence. Any command whose result you intend to rely on —
a verification, a build, a check, a generator — goes through the wrapper. Exploration does not
need to.

The wrapper records the *digest* of the argv, not the argv itself, and it does not record
output. Both are deliberate: coordination state is public, and command lines and logs are the
usual way a private path or a credential escapes into it.

### 5. Record evidence as you go

```sh
CANONICAL/coordination/tools/handoffctl update AR-NNNN --owner WORKER_ID \
    --expected-revision N --note "what happened, what it proves, what is still unknown"
```

`--expected-revision` is a compare-and-swap: re-read `task_revision` from the task's front
matter immediately before every update, and let a rejection mean what it means — someone else
changed the task, so re-read it rather than retrying with a forced value.

Publish a note after every material result, every failure, and every change of next action. An
evidence log that only contains successes is a log that was written at the end from memory.

What belongs in a note: tool versions, commands by argv digest with exit codes, the immutable
commit examined, the observed outcome, **the negative cases**, and the limitations that remain.

What must never appear in a note, a commit message or a tracked file: an absolute home path, a
credential, a private IP or hostname, raw command output, a prompt transcript, or an agent
session reference.

Be clear about how much of that is actually checked, because it is less than it looks:

- **Nothing is hooked into `git push`.** There is no pre-push hook. Every check below is
  something a person or a gate runs.
- **`handoffctl check-commits` is a message-privacy scan only.** It runs against this
  repository, so it does examine product commits, but it checks no signature and no sign-off.
  `tools/quality/check_commits.py` (gate 12) is the full check.
- **Its pattern list is partial.** It catches Linux absolute home paths (a leading `/home`
  segment), Windows user paths, `password`/`token`/`secret`/`api_key` assignments, private-key
  blocks, agent session references and session-like UUIDs, and IP addresses in `10.*` and `127.*`
  only. It has **no hostname pattern**, and it does not cover the two other private ranges,
  `172.16/12` and `192.168/16`.

  (Those patterns are described here rather than quoted verbatim. The coordinator splits its own
  home-path regex across a concatenation so that its source does not match itself; a document
  that spells the literal out would be flagged by any repo-wide privacy scan ported from it.)
- **The product has no equivalent yet.** Gate 12 in [QUALITY_GATES.md](QUALITY_GATES.md)
  specifies one, with the wider IP coverage, and AR-0006 built it as `tools/quality/privacy.py`.

So for notes, for product commit messages, and for the gaps in the pattern list, the enforcement
is you. Treat the tooling as a backstop against the mistakes it happens to know about, not as
permission to stop reading what you wrote.

### 6. Independent review

**The person who wrote the change does not certify it.** A gate run by the implementer proves
that the implementer can make the gates pass, which is not the question. The gate runner is
invoked by a reviewer, against an exact immutable commit, from a hermetic worktree — see
[QUALITY_GATES.md](QUALITY_GATES.md).

The reviewer re-checks factual claims against named source files rather than accepting the
summary. For a documentation change this is the whole review; for a code change it is still
most of it.

### 7. Integration by pull request

Product changes land through a pull request against `main`, reviewed at an exact head. Not by
pushing to `main`, not by a fast-forward, not by a merge of a stale branch.

The one exception is coordination state. `handoffctl` commits each transaction straight to
`main` as a signed, signed-off `chore(state):` commit touching only `coordination/`, after its own
full validation under its lock. A change to the coordinator's *code*, schemas or tests is product
work and goes through a pull request like any other.

```sh
git commit -S -s
```

Every commit is signed and carries a `Signed-off-by:` trailer whose name and address match the
commit author character for character. Co-authorship trailers go last, immediately before the
sign-off block. No session URL, ever.

Never force-push, rebase or amend anything already pushed. If a pushed commit is wrong, add the
commit that fixes it; the history is evidence and rewriting it destroys the evidence.

Shared surfaces — `Cargo.toml`, the workspace lint policy, the tool manifest, the task schema —
are coordinator-owned integration points. Two ARs that both need one do not both edit it; they
serialize through the coordinator. A failed required gate preempts further feature work.

### 8. Submit, and stop

When the work is finished, you do not decide that it is done. You hand it over:

```sh
CANONICAL/coordination/tools/handoffctl submit AR-NNNN --owner WORKER_ID --note "..."
```

`submit` names no destination status, and that is deliberate: it makes self-certification
unreachable rather than merely forbidden. It moves the task to `in_review`, records you as
`submitted_by`, and **clears your claim**. Clearing the claim is also what withdraws your
authority — every owner-authenticated transition is refused afterwards, whether you present your
worker id or the now-empty owner field.

Make the note a real completion summary. It is the first thing a reviewer reads: the exact head,
what the acceptance criteria did, the negative cases, and — most valuable of all — what you are
unsure of. On this project the things that got fixed were, repeatedly, the things a worker wrote
down as unverified rather than the things it was confident about.

**Set `next_action` before you submit, not after.** Once the claim is cleared you cannot correct
it, and a stale next action describes work that is already finished.

Then stop. Holding a live claim while idle is not patience, it is an abandoned worker as far as
the health check can tell — and if the lease expires it reports an expired claim against work that
is complete.

### 9. Review — the coordinator's decision

```sh
CANONICAL/coordination/tools/handoffctl review AR-NNNN --reviewer WHO --expected-revision N \
    --status done --note "..."
```

`review` is the only transition out of `in_review` and **the only route to `done`**. It refuses a
reviewer who is the submitter. `release` will not name `done` at all, in the function or in the
subcommand's choices, so no worker can record its own work as accepted.

A worker still uses `release`, but only to put work down rather than to finish it:

- `open` — paused, with the exact next action written down so the next worker does not
  rediscover it.
- `blocked` — a *named* external dependency, not a vague one.

`planned`, `future`, `cancelled` and `superseded` are coordinator decisions: the first two demote
work that should not have been open, the last two retire it. A worker that thinks its AR belongs
in one of those says so in a note and lets the coordinator decide.

Never leave a stopped worker marked `in_progress`. A status vocabulary without a terminal state
was one of the seven failure modes of the previous process; `done` means done, it is reached only
through review, and nothing reopens it silently.

## Release of the product

There is no product release yet. When there is, it is a tag on `main` at a commit whose full
gate run passed, with the run's structured report recorded against the AR that cut it. Until
the gate runner exists (AR-0006) there is nothing to record, and this section deliberately
promises nothing further.

## Recovering from an interruption

Reconcile before repeating anything. After a crash, a timeout or an expired lease, the durable
state, the Git refs, the worktrees and any open pull request can all disagree.

```sh
CANONICAL/coordination/tools/handoffctl reconcile --commit --push
CANONICAL/coordination/tools/handoffctl doctor --live
```

An expired lease permits investigation. It does not authorize repeating an ambiguous external
effect — a push, a pull request, a release — until you have established whether it already
happened. Resolve concurrent work by reading and merging, never by a forced update or a
destructive checkout.
