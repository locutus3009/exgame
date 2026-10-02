# Documentation map

This repository has, historically, had more than one documentation system, and a reader could
not tell which one to believe. This file settles that. If two documents disagree, the one higher
in this list wins.

## The systems

### 1. `docs/architecture/` — the governed design system (authoritative)

The design record, and the only documentation system in this repository that survives. It is
authoritative because it is the only one with **governance rules**, stated in its own
[overview](architecture/overview.md#governance): what may be written where, who may move a
document between buckets, and what must happen first. Those rules were carried across the move
from `.claude/architecture/` unaltered: the `## Governance` section extracted before and after
the move is byte-identical, sha256 `42389a8a74e37710f106c81d306f97ced1e28b07cfc3872754e434bf08d46605`.
(`git diff -M` reported the index as a pure rename at the move commit itself; a later commit in
the same branch rewrote the crate table, so a diff across the whole branch shows a rewrite. The
byte-identical section is the claim that holds either way.)

Seven buckets, and the bucket is the claim:

- `decided/` — invariants. Changing one requires an explicit design-review discussion first;
  only a mechanical edit (cross-reference refresh, typo) is exempt.
- `open/` — live questions. A new one is added whenever a question arises that nothing covers,
  and it is cross-linked from what it touches.
- `lessons/` — decisions that were once `decided/` and then failed. They are *moved*, not
  deleted, with the reason. This is the bucket that makes the system trustworthy: a design
  record that only ever accumulates successes is a record that has been edited.
- `planned/` — intent that is not yet a commitment.
- `crates/` — per-crate design documents, added when a crate ships a non-trivial public API.
  Five crates now qualify and have none; the index names them rather than leaving `(no doc)`
  to imply there is nothing to say.
- `findings/` — measurement notes a decision cites: what was measured, on what code, and what
  the numbers ruled out. Evidence, not decisions. One of the three exists because the
  measurement refuted the design it was checked against.
- `records/` — three design specs migrated out of the deleted corpus because live source or
  `ACCELERATOR.md` cites them. Their claim is deliberately the weakest here: a record is the
  spec a shipped piece of code was written to, not an obligation on it, and where it disagrees
  with the code it loses. The bucket is closed; nothing may be added to it.

A document in this system carries an obligation. That is the whole difference. Since AR-0010,
the obligation has a mechanism behind it: [check_docs.py](../tools/quality/check_docs.py) fails when a governed
document leaves the index or a link stops resolving.

`docs/process/` belongs to the same system and moved with it. Two files:
[design-review-method.md](process/design-review-method.md), the Socratic review protocol the
governance rules require before a `decided/` document changes — the owner values it explicitly
and it is kept unchanged — and [resume-bookmark.md](process/resume-bookmark.md), which the
governance rules point a returning reader at. The bookmark went ten weeks without an edit while
the code moved under it, so its session log is now an explicitly-dated archive with its falsified
claims named, and its live half points at the coordination repository, where a task has an owner,
a lease and recorded evidence — an identity outside the session that wrote it, which is exactly
what the bookmark never had.

### 2. `ACCELERATOR.md` — a living subsystem document (authoritative for its subsystem)

The GPU accelerator: build-time GLSL codegen, async dispatch, write-back and locking, and the
open questions. It is *living* — it is edited as the subsystem changes rather than written once
— and it keeps two separate axes that most documents collapse into one: **design maturity**
(settled / weakly worked out / not yet touched) marked inline, and **implementation state**
(landed / placeholder / not wired) in its status section. Keeping them separate is why it is
still accurate; a single "status" field would have hidden a settled design that nothing
implements, and an implemented stage whose design was never worked out.

It is authoritative for the accelerator, and it is the reason the accelerator survived being
invisible from the crate table. It is indexed from
[architecture/overview.md](architecture/overview.md#subsystem-documents) as a subsystem document
of system 1, with both axes named there so a later editor does not collapse them. **It stays at
the repository root**: thirteen in-source comments point at it by that path, and those
back-references are the mechanism that kept it maintained while the abandoned corpus rotted.

### 3. `docs/` — process and quality (authoritative for process)

This directory, excluding `architecture/` and `process/`, which belong to system 1.
[DEVELOPMENT.md](DEVELOPMENT.md), [QUALITY.md](QUALITY.md) and
[QUALITY_GATES.md](QUALITY_GATES.md) plus [../config/quality-tools.json](../config/quality-tools.json)
are the process and gate contract, and [check_docs.py](../tools/quality/check_docs.py) is the one check in it
that runs today. The two long-form design documents that used to sit here,
`IMPLICIT_SOLVER_DESIGN.md` and `setting_patera.md`, moved into system 1 under
[architecture/planned/](architecture/planned/) — neither describes what is implemented, which
is what `planned/` means.

### 4. `CLAUDE.md` / `AGENTS.md` — the entry point (authoritative for orientation)

One file, two names, at the repository root. It says what exists, what the vocabulary is, and
where to go next. It is not a design document and must not grow into one: its job is to stop a
reader from starting out wrong.

### 5. `docs/superpowers/` — deleted

A specification and plan corpus produced by the previous process. **It no longer exists.** It
was removed by AR-0010 and remains recoverable from git history.

It is worth knowing why it was deleted rather than tidied, because the reason is the point of
this file. Measured before removal: 78 files, 2,027,281 bytes of Markdown against 1,442,836
bytes of Rust under `crates/` — 1.4 times more documentation than the code it documented. For
39 of the 40 specs the date the file was added equalled the date it was last modified; the
fortieth moved by one day. Across 1085 commits, 116 touched it: 81 additions, 40 modifications,
**zero deletions**. Of the distinct `*.rs` paths it cited, **40 still resolved**. The
denominator depends on how a citation is extracted: backtick-quoted tokens matching
`[A-Za-z0-9_][A-Za-z0-9_./-]*\.rs`, deduplicated, gives 264 and 15.2%; other reasonable
extraction rules give denominators from 203 to 278, and every variant lands between 14% and 20%.
The numerator is exact and the conclusion does not turn on the rule chosen. Its own
mandated step-tracking convention stood at 1531 unchecked and 0 checked.

It was not a diligence failure. It failed for missing *mechanisms*: no gate forced a document to
change when the code changed, artifacts had no identity outside the session that made them, the
status vocabulary had no terminal state, so a corpus could only grow, and a mandated convention
ran at 0 of 1531 compliance for three months without anyone noticing, because nothing read it.

Tidying it would have produced a second corpus with the same missing mechanism. Six documents
were migrated instead — the three `findings/` notes the governed documents link, and the three
specs in `records/` that live source or `ACCELERATOR.md` cites — and the other 75 were deleted
in one commit whose message carries the numbers above.

## The rule that prevents recurrence

> **A document that no gate reads, and that no commit is required to update, will rot.**

Not *may* rot. Will. The rate varies; the direction does not. It follows that adding a document
without also adding the thing that forces it to stay true is not a neutral act — it creates a
future false statement and puts a reader's trust behind it.

So, for anything written here:

- **Prefer a check to a paragraph.** Two mechanisms are *specified* for exactly this reason:
  gate 14 will compare the `CLAUDE.md` crate table against `cargo metadata` and fail on any
  difference ([QUALITY_GATES.md](QUALITY_GATES.md#14-entry-point-crate-table-consistency)), and
  gate 11 will resolve every relative Markdown link and anchor
  ([QUALITY_GATES.md](QUALITY_GATES.md#11-repository-policy)).

  **Both now run as gates.** `tools/quality/` is in this repository; AR-0006 built
  both. What does exist is [check_docs.py](../tools/quality/check_docs.py), which implements gate 11's link
  property and the new gate 15 (every governed document reachable from the index, every index
  entry resolving), passes on this commit, and is proven able to fail against four fixtures. It
  is *not* enforced: nothing runs it hermetically at an exact commit and no reviewer is required
  to. The crate table is still correct only because it was checked by hand against
  `cargo metadata --no-deps` at the commit that wrote it — which is precisely the condition this
  rule says will not hold.
  Saying otherwise here would be the same failure, committed inside the paragraph that names it.
- **Where no check is possible, name the obligation and the owner.** The governed system does
  this with its buckets and its review rule. "Someone should keep this up to date" is not an
  obligation.
- **Where neither is possible, say so in the document.** An explicitly unverified claim is
  survivable. An unmarked one is not.
- **Say what is not true.** [QUALITY.md](QUALITY.md) lists what is deferred, by milestone, in
  the same breath as what is enforced. A document that only lists strengths teaches readers to
  stop reading it.

## Reuse

The structure of [DEVELOPMENT.md](DEVELOPMENT.md), [QUALITY.md](QUALITY.md),
[QUALITY_GATES.md](QUALITY_GATES.md) and
[../config/quality-tools.json](../config/quality-tools.json) is adapted from the MIT-licensed
[agent-systems-benchmark](https://github.com/martin-beck/agent-systems-benchmark) project, whose
equivalent documents were used as a model for the shape of a gate contract and a pinned tool
manifest.

The content is this project's own: that project benchmarks agent systems, has no GPU or graphics
constraint, targets a large distribution matrix, and signs with SSH. What was carried over is
the *form* — separating the gate list from the quality contract, pinning tool versions and
download digests in a machine-readable manifest, and requiring a negative fixture per gate.

Deliberately not carried over: its platform support matrix, its assurance roadmap table, its
Kani/Loom/Miri/fuzz/mutation program, and its coverage floors. Claiming any of those here would
break the rule this repository is trying to install.

The three external analyzer digests in the manifest are the same values that project records.
They are not taken on trust: each asset was downloaded and hashed for this repository, all six
matched, and a corrupted control was rejected. See the manifest's `external._note`.
