# Coordination history and restart record

This file records how coordination of Game Experiment restarted when the project was published,
and what the coordination process had already done before that. It is the place where every
reference to `AR-0001` … `AR-0018` in the product repository resolves.

## What this is

Until publication, the project's agentic development workflow — the coordinator
`tools/handoffctl`, its schemas and fault tests, the milestone and task documents, and their
evidence logs — lived in a **separate private coordination repository**. When the product was
prepared for open source, two things happened at once:

- The coordinator was copied into this repository under `coordination/`, with its rules and
  tests unchanged in substance. [SETUP.md](SETUP.md#where-state-lives) records the three things
  that changed because state now shares a repository with the product: only the canonical
  checkout on `main` writes state, a state commit contains only state, and product pull requests
  move `main` underneath the replica.
- The product repository itself was published as a fresh repository with no history.

Neither repository's history was published. **The task set restarts from empty**: `tasks/` and
`plans/` hold nothing carried over, and `STATUS.md` reports zero ARs. Only milestone M0 survives,
as [milestones/M0.md](milestones/M0.md) with status `complete`, because the M1 series needs a
predecessor whose outcome it can build on and because the M0 document is the most compact
statement of why this process exists at all.

The reason for not carrying the tasks over is mechanical, not editorial. Every evidence log
cites commits, pull requests and workflow runs by identifier, and every one of those identifiers
belongs to private history that does not exist in this repository. Imported verbatim, eighteen
task files would have been eighteen documents whose every reference dangles — the precise
failure this process was built to prevent. Rewritten, they would no longer be append-only
evidence. So the evidence stays where it was written, and this file is the summary of record.

The old coordinator covered **2026-09-07 to 2026-09-10**, four days from its first commit to the
recording of M0 as complete. In that time it accumulated 1,452 commits, of which 1,363 were
`chore(state):` transactions written by the tool itself and 89 were content commits, and it ran
one milestone and eighteen ARs. Everything below is reconstructed from its README, status views,
deferred list, provenance record, milestone document, and the eighteen task files and plans.

## Milestone M0 — Process foundation

**Outcome sought:** a working coordination machine, and a product repository whose quality
claims are verifiable by someone who did not write the code. No game feature work.

**Why:** the process before M0 (`docs/superpowers/`) had produced about 2.03 MB of Markdown
against 1.44 MB of Rust, roughly a quarter of it still describing the current tree. Its failure
modes were missing mechanisms, not lapses of diligence: no gate forced a document to change with
the code, artifacts had no identity outside the session that made them, the status vocabulary
had no terminal state, and a mandated checkbox convention ran at 0 of 1531 compliance for three
months.

**How each exit criterion was met:**

1. *Coordinator passes `doctor --live` and its own quality sequence locally and on GitHub
   Actions.* The coordinator was bootstrapped by AR-0001 and put under a hosted workflow by
   AR-0003, made live by AR-0013. At close the suite stood at 140 fault tests and 98% branch
   coverage against a 95% floor, with ruff, strict mypy, the schema check and
   `render-status --check` all green.
2. *`tools/quality/run-gates.sh <SHA>` runs to completion on a clean checkout of `main`, every
   gate passing.* AR-0006 built the runner; it merged honestly red, with three gates not armed.
   AR-0008 armed gate 8, AR-0009 armed gate 7 and AR-0007 implemented gate 13. The post-merge run
   on `main` after AR-0007 and again after AR-0017 reported 15 of 15 gates passing.
3. *Every gate is proven able to fail.* AR-0007's negative-fixture suite plants a defect per
   gate and asserts the specific diagnostic; at close it held 28 fixtures across 15 gates, and a
   gate added without a fixture fails the suite.
4. *`cargo fmt --check`, `cargo clippy … -D warnings` and `cargo test --workspace` green with no
   suppressions carried forward.* AR-0005 cleared 181 clippy warnings and one formatting hunk,
   pinned the toolchain, and added a workspace lint policy; the reviewer confirmed the test
   inventory byte-identical to baseline.
5. *`docs/superpowers/` gone, no dangling link.* AR-0010 deleted the abandoned corpus,
   migrated the six documents something still cited, and built the link and index checker that
   became gate 15.
6. *`CLAUDE.md` and the architecture index describe the crates that exist.* AR-0004 rewrote the
   entry point; AR-0010 brought the architecture index into agreement with
   `cargo metadata --no-deps`; gate 14 (`tools/quality/check_crate_table.py`) checks it.
7. *Every AR in series 00 is `done`, with evidence recorded.* All eighteen were accepted through
   independent review, and `complete-milestone M0` recorded the milestone complete on
   2026-09-10.

Criterion 7 had a structural problem worth recording: it moved every time a reviewer did its job
well. AR-0016, AR-0017 and AR-0018 did not exist when M0 opened; each was individually
justified, and the aggregate was a milestone that would not end. The series was therefore
**frozen at AR-0018**, and from that point a newly found defect was written to a deferred list
rather than scheduled into M0, unless it falsified one of criteria 1–6. That list is reproduced
under [Open items carried forward](#open-items-carried-forward).

## Per-AR record

Status vocabulary, transitions and gate numbers below are the ones in force today; see
[README.md](README.md), [../docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md) and
[../docs/QUALITY_GATES.md](../docs/QUALITY_GATES.md). Every AR ended `done`. A file named here
exists in this repository unless the text says it lived in the old coordinator only.

### AR-0001 — Bootstrap the coordination repository

P0, done. The project had no coordination tool, so the task that built one could not be claimed
through it; it is the single recorded exception to the claim-and-wrapper process. The coordinator
was adapted from the MIT-licensed
[agent-systems-benchmark-state](https://github.com/martin-beck/agent-systems-benchmark-state) at
a pinned revision, as recorded in [PROVENANCE.md](PROVENANCE.md): `tools/handoffctl` and
`tools/handoffctl.py`, `tools/status_renderer.py`, the task schema, the fault-test suite, and a
locked `uv` quality environment (ruff, strict mypy, coverage, jsonschema). Adaptations: the
product renamed, the status graph's series relabelled as milestones, GPG instead of SSH commit
signing, and a privacy scanner that drops the reference's workstation alias, adds two rules for
agent session references, and applies the same rules to new commit messages (`check-commits`).
It shipped with 40 fault tests at 97% branch coverage, and authored M0 with its first eleven
tasks and plans.

The first independent review rejected the head on three counts: no evidence was recorded; the
scanner never inspected commit messages while the repository's own early history broke the rule;
and a rule meant to catch the project's account name could never match it, because a regex word
boundary cannot precede a digit. That rule was removed rather than repaired — the inherited
home-path rule already covers the real leak — and a formal retraction was logged for two claims
that had been false. A re-review mutation-tested every adapted rule and diffed the code against
upstream before accepting. Left open: the extensionless `tools/handoffctl` entry point is outside
lint, typing and coverage, as it is upstream.

### AR-0002 — Add the milestone layer to the coordinator

P1, done. The reference coordinator had no notion of a milestone. This task made the two digits
after `AR-` the milestone series and gave milestones a document contract — JSON front matter
(`schema_version`, `id`, `label`, `status`) and mandatory `Outcome`, `Exit criteria` and
`Out of scope` sections — published as `schema/milestone-schema.json`. Series labels in the
status graph come from those documents, passed into a renderer that stays stdlib-only and does no
file I/O. `validate()` gained the contract, one document per observed series, and a refusal to
hold a milestone complete while one of its ARs is unfinished; the `complete-milestone`
transaction names every AR short of `done` on standard output, in the commit message and in the
milestone document itself. It also added the rule that a description may not still say work has
not started once evidence exists, prompted by drift an AR-0001 reviewer had found.

The first review rejected that rule: it scanned the append-only evidence log, so the task's own
evidence describing the rule tripped it, and the gate could become permanently unsatisfiable. It
now reads only the description above the first evidence entry and keys on the presence of
evidence rather than on status. The second review killed 20 of 25 mutants; all five survivors
loosened the evidence-entry boundary towards fail-open, and each was pinned with a near-miss
test. Tests went from 40 to 67, the final mutation battery 24 of 24 killed. Left open and
recorded in [README.md](README.md): an entry-shaped line inside a description silently ends the
scan there, and milestone status still appears in no generated view. The worker also surfaced
the protocol gap that became AR-0012.

### AR-0003 — Enforce coordination state consistency in CI

P1, done. Put the coordinator's validation sequence on GitHub Actions for every push and pull
request, with rejection demonstrated rather than a green run asserted. The workflow — then
`verify.yml` in the old coordinator, now
[`.github/workflows/coordination.yml`](../.github/workflows/coordination.yml) — runs with read-only
permissions and full-commit-id action pins, and gates the introduced commit
range on an exact `Signed-off-by` match, message privacy, and a signature check that reads the
issuer from GitHub's attestation and requires it to be in a static allow-list. Rejections were
demonstrated on real runners for a stale `STATUS.md`, a missing sign-off, an agent identifier in a
message, an unsigned commit, a key removed from the allow-list, and a first push with no
verifiable base.

Three review rounds reshaped it. Round one found that cancellation could permanently drop a push
range, that the signer was unverified, and that 89–91% of `main`'s commits were state
transactions — so runs whose head is a `chore(state):` commit are now skipped. Round two
rejected the concurrency argument, because GitHub's queue holds one pending run and evicts the
rest; the redesign takes the base from the head of the last *successful* run, which heals across
cancellation, eviction and the skip. Round three found a trailing `|| true` that made an API
error look like a first run. Known limits: the check blocked nothing, since branch protection was
not available on the plan in use; signature checking trusts GitHub's attestation; a rebase merge
would discard signatures. A coordinator error is also recorded here: the task was marked `done`
before its post-merge run had finished, which is how AR-0013 was discovered.

### AR-0004 — Publish the product development process and quality contract

P0, done. The product's entry point was materially wrong — it omitted three crates, called the
live Vulkan layer a scaffold, used pre-cutover algebra vocabulary, named a renderer backend that
appears nowhere and never mentioned the accelerator — and there was no process document or gate
contract. Delivered: [../docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md),
[../docs/QUALITY.md](../docs/QUALITY.md), [../docs/QUALITY_GATES.md](../docs/QUALITY_GATES.md),
[../docs/README.md](../docs/README.md), the pinned tool manifest
[../config/quality-tools.json](../config/quality-tools.json) with verified download digests, a
new top-level `README.md` and a corrected `CLAUDE.md`. The quality contract drew the line still
in force: *specified* means named with an owning AR; *enforced* means a command plus a negative
fixture, and nothing met it yet.

The first review rejected gate 12 as specified: it named the coordinator's `check-commits`,
which does no signature or sign-off checking and, run from the product, scanned an empty range
and printed OK — an always-green no-op. The worker withdrew its own evidence as vacuous, and gate
12 was respecified as a product-side `tools/quality/check_commits.py` with GPG verification
against a committed [`config/allowed-keys.asc`](../config/allowed-keys.asc). The re-review found
the fixture table's prose and its rows disagreeing on the count, which is why the table now
carries a count column. The second reviewer built gate 12 and the crate-table check from the
prose alone, and both worked first time.

### AR-0005 — Bring the workspace to a warnings-clean baseline

P0, done. At baseline `cargo fmt` rejected a hunk, clippy over all targets emitted 181 warnings,
there was no toolchain pin and no lint policy, and the rustdoc build was red too. Delivered: an
exact toolchain pin in `rust-toolchain.toml`, every clippy finding fixed, a workspace `[lints]`
table inherited by every crate, `deny.toml` and `.cargo/audit.toml` with each advisory exclusion
carrying the condition that would invalidate it, and SPDX licence headers on first-party sources.
All seven cargo-side checks exited 0 from a clean target directory; tests stood at 378 passed and
9 ignored, and the reviewer confirmed the test inventory unchanged.

Notable findings: **a lint fix changed behaviour** — rewriting an accumulation in `clifford` to
`+=` changed a traced operation count from 128 to 175, because the tracing scalar implements
`Add` and `AddAssign` differently; the line was restored under a scoped `#[expect]` and the other
sites re-checked. Six negated float comparisons were NaN guards and were rewritten through
`partial_cmp` rather than as clippy suggested. The test documented as failing,
`energy_drift_repro`, in fact passes. The task was reopened once: the workspace lint group had
silently downgraded `clippy::correctness` from deny to warn, which was restored and proven both
ways with a planted defect. The worker was killed mid-task by a provider rate limit, the first
sighting of the deadlock AR-0014 later fixed.

### AR-0006 — Implement the hermetic local gate runner

P0, done. No hosted runner can build the workspace, because `crates/newton/build.rs` needs a
Vulkan device, so a local runner has to supply by construction what CI normally gives.
[`tools/quality/run-gates.sh`](../tools/quality/run-gates.sh) refuses a dirty tree or unknown
revision, checks every pinned tool version, refuses without a device, builds a throwaway worktree
with its own target directory, runs the gates in a fixed order and writes `gate-report.json` with
a stable digest; `--list-gates` emits the gate list as JSON. With it came the checkers under
`tools/quality/`: `check_commits.py` and `privacy.py` (gate 12), `check_crate_table.py` (gate 14),
`check_docs.py` (gate 15, moved from `docs/`), `repository_policy.py` (gate 11),
`check_coverage.py` (gate 7), `install-external-tools.sh` (gate 10) and `build_report.py`.

It merged **honestly red**: 12 gates passing, gate 7 not armed, gate 8 not applicable, gate 13
not implemented, and none of those three statuses able to count as a pass. Determinism was shown
by running from a different checkout at a child commit with a failing test, bad formatting and
stale artefacts, and getting the same digest; the reviewer repeated it with a poisoned target
directory and a nightly override. Four review rounds followed. A revision could gut its own
checkers and be recorded as passing, so the report now publishes tooling and per-checker digests
on both sides and blocks on a difference; gate 9 was scanning every ref through the shared object
store and now scans an isolated repository fetched from the one commit; the report had leaked
hardware strings. Round three rejected the head because a later fix had broken `--list-gates` as
JSON — the second unverified interface claim on the task. One run was voided because the worker
edited `run-gates.sh` while it was executing, and bash reads a running script by byte offset.

### AR-0007 — Prove every gate rejects a planted defect

P0, done. An all-green run could not be told apart from a run in which every gate does nothing.
[`tools/quality/test_failure_paths.py`](../tools/quality/test_failure_paths.py) reads the gate list
from `run-gates.sh --list-gates`, runs the real production gate command against a planted defect,
and asserts the specific diagnostic, not merely a non-zero exit; fixtures carry the revision's
own lint configuration, so loosening that configuration breaks them. Gate 13 bootstraps through
`NEGATIVE_FIXTURE_DROP_GATE`, which drops one gate's fixtures and checks that enumeration
notices. Neutering was demonstrated four ways: gate 11's checker stubbed to exit 0, gate 1's
`--check` removed, gate 14 dropped from the list, and a gate 16 added with no fixture.

The first review disproved the suite's own docstring: with four gates short-circuited in the
runner's dispatch the report was green and every fixture passed, gate 7 even upgraded from not
armed to pass. The claim was retracted rather than patched around, and the problem routed to
AR-0017. A later blocker found the armed-floor probe and the checker using different predicates,
so floors written as strings or booleans passed silently. The task also produced a process rule:
the coordinator twice asked for more work on a task already `in_review`, which strands it by
construction since claim, resume and update all refuse; requests for more work now return the
task to `open` in the same action (see [SETUP.md](SETUP.md#asking-a-submitted-task-for-more-work)).
The post-merge run passed 15 of 15 gates, meeting M0 criteria 2 and 3.

### AR-0008 — Prepare dormant hosted workflows for the product

P2, done. Hosted runners cannot build this workspace, so workflows were wanted in tree, pinned and
validated, with enabling them a configuration change rather than a project.
[`.github/workflows/quality-gates.yml`](../.github/workflows/quality-gates.yml) and
[`.github/workflows/repository-verification.yml`](../.github/workflows/repository-verification.yml)
have no push, pull-request or schedule trigger, empty permissions and full-commit-id pins, and
each records why it is dormant. Enabling one was demonstrated on a scratch branch as three added
lines. Gate 8 ran against real workflows for the first time and passed; gate 11 was shown
rejecting a mutable pin and a live trigger.

The notable result is a **negative result that overturned an accepted blocker**. Two earlier
reviews had accepted "zizmor `--pedantic` exits non-zero even on a clean workflow" as a reason
to weaken gate 8's threshold. The worker reproduced it and traced it to two real findings —
checkout without `persist-credentials: false`, and a job without a `name:` — and fixing both gave
exit 0, so no threshold was weakened and no ignore exists anywhere in the tree. Left open: with
shellcheck absent, actionlint had silently skipped shell linting of `run:` blocks (fixed by
AR-0016), and gate 11's forbidden-trigger list misses `create`, `watch`, `issue_comment` and
others.

### AR-0009 — Establish measured coverage floors

P1, done. Gate 7 had no floors. They were set from measurement with
`cargo llvm-cov --locked --workspace --summary-only --json` on the pinned toolchain — workspace
85.06% of lines, reproduced byte-identically four times — and recorded with their baselines in
[../config/quality-tools.json](../config/quality-tools.json), enforced by
`tools/quality/check_coverage.py` (exit 1 below floor, 2 contract fault, 3 not armed). The critical
set is `peano`, `clifford`, `newton` and `aristotle`; `functions` is deliberately excluded because
its bodies are traced into GLSL and never execute under instrumentation. The branches block reads
0/0 on stable, the checkable form of "there is no branch coverage without nightly". Coverage is
framed throughout as a regression constraint, not evidence of correctness, and an AR-0007 control
keeps [QUALITY_GATES.md](../docs/QUALITY_GATES.md#7-coverage-floors) saying so.

The reviewer found `measured_lines` inert — every floor could be set to 0.0 and the gate passed,
printing the baseline one line above the pass — and that a half-point slack is not
scale-invariant: `peano` would go red on losing three covered lines, an obvious invitation to
lower the floor. The fixes: a floor must lie between baseline minus slack and the baseline, or
the checker exits 2; slack became the larger of half a point and ten lines; and `aristotle`, which
holds the world storage's `unsafe`, was added. The worker corrected the premise along the way:
there are 17 executable `unsafe` blocks across `aristotle` and `newton`, the rest of the
workspace's `unsafe` being `unsafe impl` markers with no line to cover. Two of the worker's
hand-off claims to AR-0007 were false and were retracted as such; the recorded lesson is not to
let inference stand in for a one-second grep. The coordinator likewise retracted a scoping
rationale that assumed the product had a unit-test suite, which it does not.

### AR-0010 — Consolidate the documentation onto one governed system

P1, done. Three documentation systems coexisted with nothing choosing between them. The case for
deleting the abandoned spec corpus was measured: 78 files and about 2.03 MB, 39 of 40 specs never
edited after creation, zero deletions in its history, between 14% and 20% of cited Rust paths
still resolving under every reasonable extraction rule, and the checkbox convention at 0 of 1531.
75 documents were deleted in one isolated commit; six were migrated because something surviving
still cited them — three findings into
[../docs/architecture/findings/](../docs/architecture/findings/) and three specs into a new,
closed [../docs/architecture/records/](../docs/architecture/records/) bucket. The architecture
index and the entry point were brought into agreement with `cargo metadata --no-deps`, the
accelerator document stayed at the root, and a link and index checker was written — now
`tools/quality/check_docs.py`, gate 15.

The reviewer caught a count written into three documents that measured differently (13 in-source
references, not 14), and a gate count falsified inside the paragraph that names that very failure
mode. Scope overreach — two link lines in the accelerator document and a few sentences in the
entry point — was declared at the time and granted. The resume bookmark at
[../docs/process/resume-bookmark.md](../docs/process/resume-bookmark.md) was kept, with its stale
session log archived and its falsified claims flagged. Left open: `records/` is closed by
promise, not by a check.

### AR-0011 — Resolve the stranded feature branches

P2, done. Four stale remote branches carried undecided work, some of it 382 commits behind the
trunk. The owner chose to abandon all four and salvage nothing; the task turned that into a record
rather than a disappearance. Two (`a5-lift-effective-zero`, `gpu-remaining-stages`) were proven
ancestors of `main`. The other two carried eleven unique commits whose content was written down
before deletion:

- `c32-backend` evaluated the 24-DOF Jacobian in double-single `f32` pairs. A front-end `Df32`
  scalar was abandoned after trace-time pair expansion produced a graph of about 250k nodes that
  ran out of memory; a backend-only emitter worked, agreeing with the `f64` oracle to about
  3e-14 across 300 outputs at roughly 17 times the single-`f32` instruction cost.
- `viete-fixed` was a deterministic fixed-point `Scalar` on `u32` limbs. Its roughly 4000-fold
  cost blow-up turned out not to be inherent to fixed point but to refusing a multiply-high
  instruction; with `OpUMulExtended` multiply fell to 33–68 times the `f64` cost. Adding limbs
  with one fractional limb buys range only, and the SLP vectorizer cannot help because carries
  are serial.

The branches were deleted with leases naming their exact recorded heads; no bundle or archive was
kept. The record is therefore a specification to reimplement from, not a restore point — and
since that private history is not published, the prose above is now all that remains of it.

### AR-0012 — Close the worker terminal-state gap

P1, done. Only the coordinator may release a task, and release was the only way out of
`in_progress`, so a finished worker could only hold a live claim and stop — indistinguishable from
a hung one — and once the lease lapsed `doctor` went red on complete work. The fix adds the
`in_review` status, the worker's `submit` transition, which clears `owner` and `claim_expires` and
records `submitted_by`, and the coordinator's `review` transition, which is the only way out of
`in_review`, is not owner-authenticated, and refuses a reviewer equal to the submitter. `release`
can no longer name `done`, so the only route to `done` runs through review of submitted work. The
process side is [DEVELOPMENT.md §8](../docs/DEVELOPMENT.md#8-submit-and-stop) and
[§9](../docs/DEVELOPMENT.md#9-review-the-coordinators-decision).

The second review found a real hole: `submit` leaves `owner` empty, and the shared ownership gate
compared the caller's name as a credential, so `--owner ''` authenticated on exactly the tasks
nobody held. Anyone knowing the task id could rewrite the fields a reviewer reads and bump the
revision. Fixed at the shared gate: the empty string is the sentinel for an unheld task and never
a credential. Tests went from 67 to 96. The worker also corrected its own earlier evidence:
`in_review` does not fix the expired-claim deadlock, it only lowers the rate at which claims get
stranded. Left open: a whitespace-only owner still works as a credential.

### AR-0013 — Restore liveness to the coordination workflow

P0, done, with its owed proof handed on. Right after AR-0003 merged, `main` had no successful run:
state commits landed every few seconds and each cancelled the content run before it finished.
Safety held — the base comes from the last successful run, so nothing went unverified — but the
base never advanced, so the gate certified nothing. AR-0003's round three had removed a ref split
"as simpler" without seeing that it was also the liveness guarantee. The fix: a run its job
condition will skip goes into its own concurrency group, the group expression being the exact
complement of the job condition; and cancellation is disabled on `main`.
`tests/test_workflow_gate.py` parses both expressions and asserts that one is the literal
negation of the other.

Fixture branches on real runners showed a content run completing under a burst of state pushes and
the next run covering all seven commits, and showed queue eviction happening without loss of
coverage. The worker corrected its own record against the run API twice. The successful post-merge
run on `main` was never obtained under this task: a provider rate limit killed three workers at
once, their three expired claims blocked every transition, and the run failed at `doctor` on those
claims. That outcome was handed to AR-0014. A `[skip ci]` marker on state commits was recommended
and not adopted.

### AR-0014 — Break the expired-claim deadlock

P0, done. `validate()` runs over every task inside every transaction, so one expired claim refused
every mutation — including `release` and `heartbeat`, the only transitions that could clear it. The
coordinator could not repair itself; recovery meant editing task files by hand, and the hosted
check stayed red on state an unrelated merge had never touched. This happened for real when a rate
limit took down several workers at once. The fix is `expire`: a coordinator transition that
returns an abandoned task to `open`, requires `--cleared-by` and a note, records the abandoned
owner and the lapsed lease in the evidence log, and refuses a lease that is still live. It is the
only transition whose pre-commit validation tolerates a standing error, and only messages that are
exactly an expired claim and were present before the write. `doctor` still reports the claim as an
error and now prints the command that clears it. See
[SETUP.md](SETUP.md#recovering-an-abandoned-claim).

Severity separation and exempting `release` were rejected: the first is the weakening of a check
the process forbids, and the second authenticates an owner who is gone. The reviewer's own mutation
battery found two survivors the worker's had missed — a substring rather than whole-message match,
and computing the tolerated set after the write, which would commit a tree `doctor` rejects — and
the worker's "every neutered build was caught" claim was withdrawn. Three documents had also
claimed that a stale generated view refuses `expire`; it cannot, because `mutate` regenerates the
views before validating. While working, the worker found AR-0006's 908-character `next_action`,
which seeded AR-0015.

### AR-0015 — Reconcile the transaction gate with the schema gate

P1, done. `validate()` enforced no length or format bound while `schema/task-schema.json` stated
many, so a transition could commit state the schema gate then rejected — observed when `update`
wrote a 908-character `next_action` against a 300 limit. Every push carrying it was a state commit,
which AR-0003's skip ignores, so the hosted check never saw it. The fix makes `validate()` read the
published task and milestone schemas and enforce them with a stdlib reader (`schema_gate_errors`),
with `schema_support_errors` refusing any construct the reader does not implement; the hand-written
duplicates were deleted. `jsonschema` was not imported because it is absent from the interpreter
the entry point runs under. `tests/fuzz_schema_parity.py` (a differential fuzz against a real
draft 2020-12 validator) and `tests/attack_schema_support.py` (mutation of the schemas themselves)
are the evidence; the exact bound of the claim is in [PROVENANCE.md](PROVENANCE.md).

Three reviews. The fuzz found two faults the fixtures had missed: `const` treated `true` as `1`,
and `pattern` must be an unanchored search. Round one was rejected because the added tests pushed
a compiled `__pycache__` module past the privacy walk's 200 KiB cap, so CI's `doctor` would fail on
a build artefact; and because the "total safety net" claim was false for three constructs, one of
which raised `TypeError` out of every transaction. Round two was rejected for the same two fault
shapes through boolean subschemas, which the reviewer showed would have crashed `expire` itself.
Round three fixed the class rather than the instances and **bounded the claim**: total over the
vocabulary and shape of the published schemas, not over semantics. The attack suite reported 1,676
mutants with 0 faults, against 451 at the previous head. Before merge, two other tasks' long
`next_action` fields had to be shortened, because both were `in_review` with no owner and nothing
could have fixed them after the stricter gate landed — the deadlock in a new shape.

### AR-0016 — Pin shellcheck so gate 8 lints shell

P1, done. actionlint lints shell in `run:` blocks by delegating to shellcheck, and when shellcheck
is missing it skips that work and exits 0; gate 8 had passed over every `run:` block AR-0008 added.
Installing shellcheck on the host was rejected because the verdict would then depend on undeclared
host state. Instead shellcheck is pinned by per-platform digest in
[../config/quality-tools.json](../config/quality-tools.json), and
`tools/quality/install-external-tools.sh` verifies it before extraction and generates an
`actionlint-with-shellcheck` launcher that first lints a throwaway workflow containing a defect
only shellcheck catches, printing `SHELL LINTING DID NOT RUN` and exiting 3 if it is not caught.
Gate 8 runs under `env -u SHELLCHECK_OPTS`. The coverage statement is in
[QUALITY_GATES.md §8](../docs/QUALITY_GATES.md#8-workflow-linting).

The reviewer attacked the probe — deleted, unreadable, stubbed, `SHELLCHECK_OPTS`, broken
`TMPDIR` — and every case failed closed. One correction ran the other way: the coordinator had
instructed a `trap on_failure EXIT INT TERM`, the worker deliberately did not use it because a
returning signal handler leaves the shell running, and the reviewer confirmed the worker right.
The worker also found that the runner's argv digest was built with NUL separators in a bash
variable, which cannot hold NUL, so `a b` and `ab` hashed the same; that went to AR-0017.

### AR-0017 — Assert that a gate executed its command

P0, done after five reviews. A reviewer short-circuited four gates in `run-gates.sh`'s dispatch:
each recorded `pass` in a millisecond with a plausible digest, every fixture still passed, and the
stable digest was identical. The fix is an execution ledger: `run_step` appends the argv digest and
the exit status the command actually returned, a gate's recorded digest, duration and `steps`
table are derived from it, and `build_report.py` publishes any gate whose status the ledger does
not support as `unsupported`. `--argv-digest` exposes the corrected digest function so it can be
recomputed outside the runner. The fixture suite grew to 28 fixtures across 15 gates plus a new
category of runner assertions, including a fifth gate 7 fixture for a half-armed coverage manifest.

The review sequence is M0's sharpest lesson about overclaiming text. Round one disproved the claim
that disabling a gate now needed edits in two files: one line per dispatch arm forged records
identical to honest ones. The false claim then survived in comments, a banner, a docstring and a
table — nine sites, then a tenth in the recorder itself — because each sweep was a line-based grep
and the phrase wrapped. The reviewer's verdict was that nine occurrences meant a missing mechanism,
not a diligence failure, and the result is the `no-unqualified-execution-claim` control in
`test_failure_paths.py`, which flattens text and requires a qualifier in the same sentence. Its
first version was itself green on the flagship banner, because the worker had tested it against the
reviewer's paraphrase rather than the sentence in the file. Stated non-coverage: the ledger is
evidence the runner writes about itself, so anyone editing `run-gates.sh` can still forge an entry,
and the stable digest proves two runs saw the same evidence, not that anything ran.

### AR-0018 — Scope the privacy walk to the repository root

P0, done in one round. `privacy_errors()` matched its exclusions against the parts of the absolute
path, so an excluded name anywhere above the root — such as the `workspaces/` directory holding
every worker's worktree — suppressed everything below it. `doctor` run from a worktree examined
zero files and reported clean. AR-0015's worker found it and correctly declined to fix it inside an
unrelated review. Exclusions are now decided relative to the root, named in `PRIVACY_EXCLUSIONS`,
and pruned during the walk, which also made it about a hundred times cheaper; an empty walk is now
an error, documented as a floor against total vacuity and nothing more. The reviewer ran twenty
ancestor-scoping attacks, including symlinks and `..` traversal, and none suppressed anything.
Tests stood at 140 and branch coverage at 98% when M0 closed.

## Open items carried forward

None of these was lost; none is scheduled. They were found during M0, mostly after the freeze at
AR-0018, and are offered as a list a future milestone can pick from. The old coordinator's deferred
list lived there only; its content is here.

### From M0's own out-of-scope list

- Physics-invariant property tests — energy, momentum and angular-momentum drift, Kepler and
  Lagrange-point checks — and the determinism gate.
- A CPU-versus-GPU differential oracle as a required gate (new work, not a promotion; see
  [../docs/QUALITY.md](../docs/QUALITY.md)).
- Criterion performance budgets for the accelerator.
- **Structurally enforcing the "at most one writer per slot" invariant** under the world
  storage's `unsafe`. The highest-consequence open risk in the repository and the leading
  candidate to open M1.
- The two `FIXME: deadlock?` sites in the implicit solver and their ignored reproducers.
- Any renderer, input layer, scene layer or game content.

### Product quality documents and gates

- At the restart, `docs/QUALITY.md`'s status table and the opening paragraph of
  `docs/QUALITY_GATES.md` still described the state before the runner existed — "nothing is
  enforced automatically yet", two gates not armed, the runner "honestly red" — when it had since
  reported 15 of 15. AR-0006's review flagged the same staleness in `docs/README.md` and
  `docs/DEVELOPMENT.md`. The same class of defect as a comment claiming more than is tested,
  pointed the other way.
- `docs/QUALITY_GATES.md` §13 keeps a bolded "there are no exemptions" and narrows it three
  sentences later without recording the earlier claim as having been false.
- Four stale `file:line` references in `README.md` and `docs/QUALITY_GATES.md`.
- `tools/quality/install-external-tools.sh` prints a truncated `--help`: a hardcoded `sed` line
  range over a header that has outgrown it. The same shape was fixed in `run-gates.sh`; any
  interface that depends on an unchecked line number recurs.
- Gate 8 lints no first-party shell, including `run-gates.sh` and `install-external-tools.sh`;
  two of them cannot be shellchecked until a comment is reworded. `run:` blocks in a non-shell
  dialect are not covered at all.
- The shellcheck probe does not lint through the invocation gate 8 performs: an actionlint config
  `paths: … ignore:` entry would silence the real run while the probe stays green. Fix: pass the
  same `-config-file` to the probe.
- A shellcheck stub keyed on the probe's own text defeats the probe (needs write access that
  already permits replacing actionlint outright).
- SIGHUP during a manual install can leave a bare `actionlint`; add `trap 'exit 129' HUP`.
- `run-gates.sh` runs without `-e`, so a failed copy of the gate 13 table would leave gate 13
  reporting `pass` with the table absent. Close together with `run_step`'s step-log copy.
- The report does not name which shellcheck ran, and the launcher is outside the provenance
  digests. aarch64 binaries are pinned and digest-checked but have never been executed.
- Gate 7: the `llvm-cov` measurement step has no fixture; a null `coverage` section crashes probe
  and checker alike; a `nan` floor passes; the workspace-null direction of a half-armed manifest
  raises `TypeError`; the manifest-contract rejections have no automated test.
- Gate 11's dormant-trigger deny-list is narrow (`create`, `watch`, `issue_comment` and others).
- The dormant workflows' base-resolution fallback can narrow the introduced range to one commit on
  a new branch or after a force-push; untestable while dormant.
- Ten pre-existing `allow` attributes carry no rationale and need deciding one by one.
- The `orbit_radius` escape detector reads zero at every sample, so the test's second assertion
  rests on the energy check alone — physics work for M1.
- The runner cannot vouch for itself: invoked from the head under review, its tooling comparison
  is trivially true, and a build script can rewrite the run's record (`execution_limit`).

### Coordinator

- Container `const`/`enum` equality: `_json_equal` separates `true` from `1` only at the top
  level, so `{"const": [1]}` accepts `[true]` where a real validator refuses. A real latent parity
  hole, unreachable by both harnesses today.
- `tests/attack_schema_support.py`: its pass condition is weaker than documented and is silent if
  any repository document is already schema-invalid; it has no mutant-count guard; `positions()`
  does not descend into a branch's own `properties`; and it reimplements rather than calls the
  ordering the fix rests on.
- A catastrophic-backtracking `pattern` wedges the transaction (a liveness hazard, not parity).
- `then`/`else` without `if` is applied by the transaction and ignored by a validator.
- A field-level `description` in a published schema is refused as an unsupported constraint and
  turns the tree red.
- The deliberate `1.0`-is-not-an-integer strictness would flip a branch inside an `if`
  (unreachable in the shipped schemas).
- The privacy test's "binary" fixture writes four ASCII characters, so the `UnicodeDecodeError`
  branch has never run; and the empty-walk floor counts walked entries rather than files read.
- `tests/test_handoffctl.py` is approaching the privacy walk's 200 KiB file cap.
- A whitespace-only `--owner` still works as a credential; `require_active_owner` keeps a bare
  name comparison.
- `--owner`, `--reviewer` and `--cleared-by` are self-asserted records, not authentication.
- Milestone status is not surfaced in `STATUS.md` or `CURRENT.md` (see [README.md](README.md)).
- An `in_progress` task with an absent or unreadable `claim_expires` is refused by `expire`, and
  while it stands no other claim can be cleared through the tool.
- The hosted check skips `chore(state):` commits, and that blind spot lines up exactly with the
  defect class state commits produce. Closing the write gate (AR-0015) removed the known source;
  the detection gap remains. A `[skip ci]` marker was recommended and not adopted.
- A pull request opened during heavy state churn once received no run at all, because GitHub did
  not compute a test merge in time; not fully diagnosed.
- The hosted coordinator check blocks nothing without branch protection.

## Numbering

The identifiers `AR-0001` … `AR-0018` and the milestone `M0` are **retired**. They are cited from
[../docs/DEVELOPMENT.md](../docs/DEVELOPMENT.md), [../docs/QUALITY.md](../docs/QUALITY.md),
[../docs/QUALITY_GATES.md](../docs/QUALITY_GATES.md),
[../config/quality-tools.json](../config/quality-tools.json), the scripts and tests under
`tools/quality/`, and the coordinator's own tests and documents, which name the AR that introduced
a rule. Those citations remain correct and resolve here, in [Per-AR record](#per-ar-record).

The two digits after `AR-` are the milestone series. New work starts at **milestone M1 and
AR-0101**, so no identifier is ever reused and an old citation can never be mistaken for new work.
Series `00` is not reopened: a defect in something an old AR delivered is new work in
the current series that cites the old identifier.

## Lessons that shaped the process

- **Self-certification.** A worker recording its own verdict is the first thing to go. Release
  cannot name `done`; `review` refuses the submitter as reviewer; and gates are run by a reviewer,
  not the implementer ([DEVELOPMENT.md §6](../docs/DEVELOPMENT.md#6-independent-review)). Nearly
  every AR's evidence log carries a catch the worker could not have made about its own work.
- **A finished worker needs somewhere to go.** Without `in_review`, finished work looked exactly
  like a dead worker.
- **Liveness is a property, not an accident.** A gate that is safe but never completes certifies
  nothing (AR-0013); a simplification that removes a liveness guarantee needs to know it is doing
  so.
- **The tool must be able to repair the state it reports.** One expired claim deadlocked every
  transition until `expire` existed, and the same shape recurred with over-long `next_action`
  fields on unowned tasks.
- **One constraint set, not two.** A permissive write gate in front of a strict check produces red
  trees after the fact; the transaction now enforces the published schema itself.
- **Vacuous gates pass over nothing.** A missing shellcheck, a commit check scanning an empty
  range, a privacy walk examining zero files, a fuzz generating zero documents, and a dispatch
  that recorded `pass` without running anything were all green. Each now fails on emptiness or
  proves it ran.
- **Claims outlive their evidence.** False counts, retracted interface claims and an overclaiming
  sentence repeated across ten sites recurred until a mechanism checked them. Bound a claim to what
  is shown, and retract a false one explicitly rather than quietly narrowing it.
- **Test the real artefact.** Fixtures checked against a paraphrase, an excerpt, a live mid-edit
  worktree or a script edited while running all produced wrong verdicts.
- **`next_action` is the next action, not a summary.** Long hand-off notes in that field broke the
  schema bound twice; the rule is in [SETUP.md](SETUP.md).
- **A milestone needs a freeze.** "Every AR done" is unbounded while reviewers keep finding real
  defects; after a freeze, new findings go to a deferred list unless they falsify an exit
  criterion.
