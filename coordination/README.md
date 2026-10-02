# Game Experiment coordination

Tasks, plans, milestones and durable worker state for Game Experiment, kept in the product
repository itself. Process: [../docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md). Setup and the
in-tree rules — one canonical checkout on `main` writes state, and a state commit contains only
state: [SETUP.md](SETUP.md).

The task set **restarted from empty** when the project was published. What was done before,
milestone M0 and AR-0001 to AR-0018, is recorded in [HISTORY.md](HISTORY.md); new work starts at
milestone M1 and AR-0101.

Start with `STATUS.md` for the complete inventory and dependency graph, then use `CURRENT.md`
for the compact operational queue, then read the selected AR and its plan. Both are generated:
`tools/handoffctl render-status` writes `STATUS.md`, and every transaction rewrites both.

- `milestones/`: one file per milestone — outcome, falsifiable exit criteria, explicit out-of-scope.
- `tasks/`: strict JSON front matter plus an append-only evidence log.
- `plans/`: scope, dependencies, ownership, acceptance criteria and evidence obligations.
- `schema/`: the versioned task and milestone schemas, and the single statement of every
  mechanical bound. `validate()` reads them rather than restating them, so a transition
  refuses at the write what a later gate would refuse afterwards. The support net is total over
  the *vocabulary and shape* of those schemas — an unimplemented keyword, a non-object subschema
  where draft 2020-12 permits a boolean, or a bound whose value the checker cannot read is
  reported as an error rather than silently ignored — and not over semantics, for which the
  evidence is the differential fuzz. PROVENANCE.md states the bound and what it leaves unproved.
  `tests/validate_schema.py` remains a separate gate that the unit suite does not run: it
  checks the schemas themselves against draft 2020-12 with a real validator and exercises
  protocol states the repository does not happen to be in. Run both.
- `tools/handoffctl`: locked claims, revisions, leases, reconciliation and replication.
- `tests/`: coordinator fault, race and recovery tests, plus
  `fuzz_schema_parity.py`, a differential fuzz of the transaction gate against a real
  draft 2020-12 validator, and `attack_schema_support.py`, which mutates the published schemas
  at every position the reader walks and demands a loud refusal or agreement with that
  validator. Neither is in the gate sequence; run both when the schemas or the enforcement
  change.
- [PROVENANCE.md](PROVENANCE.md): the reuse boundary for the adapted coordinator.
- [SETUP.md](SETUP.md): local configuration and the validation sequence.

Never edit generated views directly. Private configuration, captures, credentials, raw command
output and transcripts stay outside Git.

Run `tools/handoffctl check-commits --base origin/main --head HEAD` to reject private references
in the messages of newly introduced commits, and
`tools/handoffctl render-status --check` to verify `STATUS.md` matches every task. A plain
`tools/handoffctl render-status` performs an offline deterministic refresh; claim, update,
promote, release and reconcile transactions refresh it automatically under the coordinator lock.

## Milestones

The two digits after `AR-` are the milestone series: every `AR-00NN` belongs to `M0`,
every `AR-01NN` to `M1`, and so on. `milestones/MN.md` governs series `NN` zero padded, and
carries JSON front matter (`schema_version`, `id`, `label`, `status`) followed by a
`# MN — Label` heading and the three mandatory sections `## Outcome`, `## Exit criteria` and
`## Out of scope`. The exit criteria must be falsifiable by a command; a claim without a
recorded exit code does not count.

`STATUS.md` labels each series subgraph with its milestone label, read from these documents.
The renderer itself does no file I/O: `handoffctl` reads the milestone documents and passes
the series-to-label map in, so rendering stays deterministic for a given input and every
label is escaped before it reaches the diagram.

`doctor` rejects a milestone document that breaks the contract, an AR whose series has no
milestone document, and two documents governing one series.

Record a milestone complete with:

    tools/handoffctl complete-milestone MN

The transition refuses an unknown milestone, one already complete, one whose document breaks
the contract, and one whose series still holds an AR that is neither `done`, `cancelled` nor
`superseded`. It shares the coordinator lock, validation, signed DCO commit and replication
with every other transaction. A failure before the commit rolls the document back; a failure
of replication afterwards does not, because the signed local commit is already durable —
reconcile and retry the push, exactly as for a task transition.

That terminal-status set is the **floor**, not the bar. A milestone document may demand more
of itself — M0's exit criterion 7 requires every AR in its series to be `done` — so closing
over anything short of `done` is never silent: `complete-milestone` names each such AR on
standard output, in the commit message, and in an appended entry in the milestone document
itself. `doctor` enforces only the floor; meeting the document's own criteria is the
coordinator's judgement, made against a record that cannot be quietly omitted.

A milestone's `status` is not yet surfaced in `STATUS.md` or `CURRENT.md`; the milestone
document is the only place it is recorded. That is a known gap to close before M1.

## When a worker goes away

`submit` clears the claim, so a worker that finishes cleanly no longer strands a lease. What
remains is the worker that dies *before* submitting, and its claim outlives it. `doctor`
reports that as an expired claim, which is the lease doing its job.

The defect was what happened next. `validate()` runs over every task inside every transaction,
so one lapsed claim refused every mutation -- `release` and `heartbeat` included, the only two
transitions that could have cleared it. The coordinator could not repair the coordinator.

    tools/handoffctl expire AR-NNNN --expected-revision REVISION \
        --cleared-by WHO --note "why the worker is gone"

`expire` returns the task to `open`, clears the claim, and records in the evidence log who
cleared it, whose claim it was, when the lease ended and why. It is not owner authenticated,
because the owner of an abandoned claim is by definition gone and no caller should be invited
to present that absence as a name. It refuses a claim that has not lapsed, so it can never
take work from a live worker, and it refuses anything but the exact condition `doctor` reports
as an expired claim.

It is also the **only** transition whose pre-commit validation tolerates a standing error, and
only ever a message that is *exactly* an expired claim, matched whole and never as a
substring, and that was already present before it ran. Every other transition remains fenced
by the full check, and `expire` itself is still refused by a corrupt task, a broken dependency
graph, a milestone fault, a privacy leak or a claim error of any other kind — and by a lease
that lapses part-way through its own transaction, because the tolerated set is fixed before
the write. It is *not* refused by a stale `CURRENT.md` or `STATUS.md`, but neither is any
other transition: `mutate` regenerates both from the tasks immediately before it validates.
An expired claim remains an error and remains red; what changed is that the process can now
clear it from inside itself, and `doctor` prints the command that does so beneath the report.

`--cleared-by`, like `--reviewer` and `--owner`, is self-asserted. It makes an unattributed
repair impossible to do by accident and impossible to do without writing a false name into an
append-only log; it is not authentication, and does not pretend to be.

## Descriptions must not outlive their evidence

A task body is a description followed by an append-only evidence log, whose entries
`handoffctl` writes as `- <ISO timestamp>: ...`. Once **any** evidence is recorded, `doctor`
rejects a description that still tells a reader work has not started. The trigger is the
recorded evidence rather than the status, so a task nobody ever worked on may truthfully say
so at any status, including `cancelled` and `superseded`, while a task with a history must
describe itself honestly whatever became of it.

Only the description is scanned. The evidence log is immutable by policy, so a rule that its
entries could trip would eventually become unsatisfiable — including for the very AR that
introduced the rule, whose evidence necessarily describes it.

The check is a **boilerplate detector, not a semantic gate**. It catches the wordings this
repository has actually used and nothing more: `"Implementation hasn't started."`,
`"Nothing has been implemented yet."` and `"Work has not commenced."` all pass it. Its
purpose is to stop a specific, observed drift — a template sentence surviving into a task
that has visibly moved on — not to judge arbitrary claims about progress.

### How the check can be switched off

Both of its failure modes are fail-open — the rule goes quiet rather than shouting — so they
are stated here rather than left for a reader to discover.

**Evidence the boundary cannot read.** An entry begins at column zero with `- `, an ISO
timestamp with a literal `T`, then `: `. Evidence written in any other shape is not evidence
as far as the rule is concerned, and would exempt the task wholesale. This one **fails
closed**: a body carrying a line that a reader would take for an entry — a leading `- ` and
a `YYYY-MM-DD` date — but no entry the strict pattern can read is reported as
`evidence log is not in the recorded entry format`. A task with no evidence at all is still
exempt, because that is the truthful case the design exists to permit.

**A description that contains an entry-shaped line.** The description ends at the first
entry, so a line inside it that is *shaped* like one truncates the description there and
everything below goes unscanned. This **cannot be closed**: an appended entry and a
hand-written sample of one are the same bytes, and no parser can separate them. It is the
same self-reference that made the rule's first version unsatisfiable, and the case most
likely to arise is a task description that documents this very format.

The mitigation is a writing rule, and the reason the near-miss cases above are pinned by
tests: **indent or fence any sample entry you put in a description.** `  - 2026-…` does not
begin at column zero, so it neither truncates the description nor is mistaken for evidence.

## Opening dependency-ready work

After reviewing dependencies and path ownership, the coordinator promotes a planned AR with:

    tools/handoffctl promote AR-NNNN --expected-revision REVISION --note "dependencies verified"

Promotion accepts only an inactive planned task whose dependencies are done. It rejects stale
revisions, invalid task or generated state, and a dirty state checkout. The transition and the
regeneration of CURRENT.md and STATUS.md share one lock, validation, signed DCO commit and
replication transaction.

A signed local commit is a durable effect even when replication fails: reconcile and retry the
push rather than repeating the transition.

## Finishing work, and who may call it done

A worker may not release its own task: release is the coordinator's transition after an
independent review. Release was also the only way out of `in_progress`, so a worker that had
finished correctly could only hold a live claim and stop. From outside that is
indistinguishable from a worker that died, and once the lease lapsed `doctor` reported an
expired claim against work that was complete.

`in_review` closes that gap. A worker hands finished work over with:

    tools/handoffctl submit AR-NNNN --owner OWNER --note "..."

The transition clears `owner` and `claim_expires`, so no lease runs under finished work and no
expiry can ever be reported for it, and it records `submitted_by`. It names no destination
status, so a worker cannot record its own verdict; and clearing the claim is also what
withdraws the worker's authority, because every owner-authenticated transition — release,
update, heartbeat, and submit itself — is refused once `owner` is empty. That holds because
the empty string is a sentinel for an unheld task and never a credential: a caller passing
`--owner ""` is refused for having named no identity, before any comparison against the
cleared field, so the absence of an owner cannot be presented as one. The note is the
worker's last word on the task, so the reviewed head belongs in it.

`release` cannot name `done` at any time, not only after a submission. Release is owner
authenticated, so whoever calls it holds the claim, and a release naming `done` would be a
worker recording its own verdict. Its remaining targets are `open`, `blocked`, `planned`,
`future`, `cancelled` and `superseded`, and it cannot name `in_review` either — that is what
`submit` is for.

Work awaiting review is not claimable: `claim` accepts only an `open` task. Returning it to the
queue, or accepting it, is a coordinator decision:

    tools/handoffctl review AR-NNNN --reviewer REVIEWER --expected-revision REVISION \
        --status STATUS --note "..."

`review` is the only transition out of `in_review`, and from there the only route to `done`. It
is not owner authenticated, because the claim is already gone, and it refuses a `--reviewer`
equal to the recorded `submitted_by`, so certifying work takes a second, named identity. It
also refuses a stale revision, a task that is not awaiting review, an empty reviewer or note,
and `in_progress` or `in_review` as a decision — neither active status is reachable by naming
it, since one is reached only by claiming and the other only by submitting.

`--reviewer`, like `--owner`, is self-asserted. The rule makes self-certification impossible to
do by accident and impossible to do without writing a false name into an append-only log; it is
not authentication, and does not pretend to be.

`doctor` holds the record consistent from both sides: a `submitted_by` on a task that is not
awaiting review is reported, and so is a task awaiting review with no recorded submitter.
`in_review` is not a finished status, so `complete-milestone` still refuses to close a series
that holds one. Both generated views list the queue under its own heading, and the owner column
of a task awaiting review names the worker that submitted it.
