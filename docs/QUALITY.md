# Quality contract

This document says what quality means in this repository, and — more importantly — draws a hard
line between **what is enforced** and **what is merely intended**. The concrete commands are in
[QUALITY_GATES.md](QUALITY_GATES.md); the pinned versions and digests are in
[../config/quality-tools.json](../config/quality-tools.json).

## The rule this document exists to obey

**Do not claim a gate that is not implemented.** A quality document that describes aspirations
in the present tense is worse than no document: it makes a reader stop checking. Every claim
below is either in the *enforced* section, where a command and an exit code back it, or in the
*deferred* section, where it carries the milestone that owns it and nothing else.

The previous documentation system failed exactly here. It mandated a convention that ran at
0 of 1531 compliance for three months, because nothing read it. The trap is not only in the
gate list: a paragraph that says a document is correct *because* a tool checks it, written
before that tool exists, is the same false green in miniature. The general form of that
failure is stated in [README.md](README.md): a document that no gate reads and no commit is
required to update will rot. A gate list is a document; it rots the same way, and faster,
because its rot is invisible until an incident.

## Status as of this commit

**Nothing is enforced automatically yet.** There is no gate runner, no hosted CI for the
product, and the workspace is not yet known to be warnings-clean. That is the honest state, and
it is the reason M0 exists.

| | Owner | State |
| --- | --- | --- |
| Warnings-clean workspace baseline | AR-0005 | not started |
| Hermetic gate runner `tools/quality/run-gates.sh` | AR-0006 | not started, depends on this document |
| Proof that each gate can fail | AR-0007 | not started |
| Dormant hosted workflows | AR-0008 | not started |
| Measured coverage floors | AR-0009 | not started; floors are `null` in the manifest |
| Documentation link and index checks | AR-0010 | **implemented, not yet a gate** — see below |

One check is an exception, and it is worth being exact about what kind. [`check_docs.py`](../tools/quality/check_docs.py)
runs today, offline, from one command, and is proven able to fail: it verifies that every relative
Markdown link and `#anchor` in the tracked tree resolves, and that every governed document is
reachable from the architecture index. It is **not enforced** by the definition below — nothing
runs it hermetically at an exact commit, and no reviewer is required to. It is a working
implementation waiting for a runner, specified in
[QUALITY_GATES.md](QUALITY_GATES.md#15-architecture-index-completeness) for AR-0006 to adopt into
`tools/quality/`. Until it is called from `run-gates.sh`, a green run of it is a developer's
pre-check and not evidence.

Until AR-0005 and AR-0006 land, the gate list in [QUALITY_GATES.md](QUALITY_GATES.md) is a
*specification for AR-0006*, not a description of a running system. Its obligations are written
as obligations — "the runner must", "the gate must reject" — for exactly that reason. A few
sentences describing what an off-the-shelf tool does are in the plain present ("`cargo fmt`
establishes that the tree matches `.rustfmt.toml`"); those describe the tool, not this
repository's use of it, and the file's opening paragraph states outright that none of it runs
here yet.

## What "enforced" is going to mean

A gate counts as enforced only when all four of these hold. Three of four is not enforcement.

1. **It runs from one command**, on an exact commit, without a human choosing which checks to
   include.
2. **It is hermetic**: a throwaway worktree at that commit, its own target directory, the
   pinned toolchain. Nothing in the developer's working tree, and no stale build artefact, can
   change the answer.
3. **It is proven able to fail** — a planted defect of the kind it is meant to catch is
   rejected, with the expected diagnostic and not merely a non-zero exit (AR-0007). An
   always-green check is not a gate; it is decoration.
4. **A reviewer runs it, not the implementer.** See
   [DEVELOPMENT.md](DEVELOPMENT.md#6-independent-review).

A gate that is skipped because a tool is missing must fail, not pass. Fail-closed is not a
preference here; a silently skipped gate is the single most expensive kind of false green,
because it looks identical to a real one for as long as it takes to matter.

## What is deferred, and to which milestone

Everything in this section is **not enforced** and is **not claimed**. It is listed so that a
reader knows it was considered and consciously postponed, rather than overlooked.

Deferred to **M1 and later**:

- **Physics-invariant property tests.** Energy, momentum and angular-momentum drift bounds;
  Kepler and Lagrange-point analytic checks. There is currently no test that would catch an
  integrator that conserves nothing.
- **The determinism gate.** The determinism contract is scoped to integrated state only, and
  its implementation is in flux pending the unresolved fixed-point-versus-`f64` question. No
  gate checks bit-reproducibility today.
- **A CPU-versus-GPU differential oracle.** No such comparison exists today, as a gate or as
  a test. The oracle that does exist,
  `crates/viete/tests/kernel_differential_oracle.rs`, compares the traced kernel run through a
  Lua carrier against a CPU `Differential` — it is trace-versus-CPU, and it does not check that
  the GPU agrees with anything. Building a GPU-versus-CPU check is new work rather than the
  promotion of an existing test, and running it as a gate needs a graphics device on the gate
  machine, which interacts with the build-time device requirement described in
  [README.md](../README.md).
- **Criterion performance budgets for the accelerator.** `[profile.bench]` is configured for
  comparable numbers, but no budget is asserted and no regression fails anything.
- **Structurally enforcing the world-storage writer invariant.** The world storage rests on
  `unsafe` blocks that are sound only under an "at most one writer per slot" invariant which is
  today maintained by convention and reasoning, not by a type or a runtime check. The milestone
  records this as the highest-consequence open risk in the repository and the leading candidate
  to open M1. It is named here, unenforced, on purpose: an unenforced invariant that everyone
  knows about is survivable, and one that is quietly assumed is not.

Also deferred, and tracked outside this document: the two `FIXME: deadlock?` sites in the
implicit solver together with their ignored reproducers.

Explicitly out of scope for M0 in general: any renderer, input layer, scene layer or game
content.

## Licensing and authorship

The repository is MIT ([LICENSE](../LICENSE)). First-party sources do not yet carry
`SPDX-License-Identifier: MIT` headers; AR-0005 adds them, and gate 11 will then enforce that
they stay.

Commits are expected to be signed and to carry a `Signed-off-by:` trailer matching the author
exactly. **Nothing checks this today.** Gate 12 in [QUALITY_GATES.md](QUALITY_GATES.md) specifies
the check and AR-0006 built it as `tools/quality/check_commits.py`; the coordinator's
`handoffctl check-commits` is not that
check and does not examine product commits at all. Contributors certify their own authorship —
do not write another person's sign-off.

## Reuse

The structure of this document, of [QUALITY_GATES.md](QUALITY_GATES.md) and of
[../config/quality-tools.json](../config/quality-tools.json) is adapted from a differently
scoped MIT-licensed project; see [README.md](README.md#reuse) for the attribution and for what
was deliberately not carried over.
