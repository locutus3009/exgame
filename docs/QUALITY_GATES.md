# Quality gates

This is the specification for `tools/quality/run-gates.sh`, which **AR-0006 will build**. It
exists at `tools/quality/run-gates.sh`. The gates below are implemented and run against an exact
revision from a hermetic worktree. Two of them are not yet *armed*: workflow linting until AR-0008
adds workflows, and the negative-fixture suite until AR-0007 builds it. The runner reports those as
`not_applicable` and `not_implemented`, and is green only when every gate passes -- so it is
honestly red until they land. Coverage was the third; AR-0009 armed it from a measured baseline,
and the runner still reports a null floor as `not_armed` rather than as a pass.

It is written as a specification — "the runner must", "the gate must reject" — deliberately, so
that no sentence here can be quoted as a description of a working system. AR-0007 must plant one
defect per gate against it. See [DEVELOPMENT.md](DEVELOPMENT.md#the-unit-of-work) for what an AR
is, and [QUALITY.md](QUALITY.md) for what is and is not enforced today.

Versions, digests and coverage floors must come from
[../config/quality-tools.json](../config/quality-tools.json). The runner must read that file
rather than hard-coding a version, and so must this document.

## How the runner must run

**Hermetic.** It must refuse a dirty tree and refuse an unknown revision. It must create a
throwaway worktree at the exact commit, give it its own target directory, and use the pinned
toolchain from the manifest, so that nothing in the developer's working tree and no stale build
artefact can change the answer. It must remove the worktree on success and on failure alike.

**Invoked by a reviewer, not by the implementer.** The runner exists so that someone who did not
write the change can confirm the claims made about it, at an exact immutable commit. An
implementer running it on their own branch is a useful pre-check and is not evidence.

**Every gate, every run.** It must capture each gate's exit code and duration rather than
stopping at the first failure, so that one run reports everything that is wrong.

**Fail closed.** A missing tool, an unexpected version, a stale command syntax or a digest
mismatch must fail the run. It must never skip a gate silently. Coverage floors that are still
`null` in the manifest must be reported as *not yet armed*, named as such in the report, and
must not be counted as a pass.

**A pass must have a ledger entry behind it.** On the honest path every command a gate runs goes
through the runner's one executor, which records it in an *execution ledger* -- the argv digest, and
the exit status the command returned rather than one the dispatch chose -- after it returns. Each
gate's published `argv_sha256`, `duration_seconds` and `steps` are derived from that ledger; the
dispatch does not pass them. `build_report.py` then refuses to publish a status the ledger does not
support, and records `unsupported` instead, which is not a pass and is blocking.

The rule below is over **ledger entries**, not over executions. The difference is the whole of what
this mechanism buys and is easy to lose in a sentence — the paragraph after the table says why, and
this document and the tooling beside it stated it the wrong way nine times across four review rounds
before a control was added to hold the distinction:

| Status | What the ledger must show |
| --- | --- |
| `pass` | at least one entry for the gate, and every entry at 0 |
| `not_armed` | an entry returning the code that means unarmed (gate 7's floor check exits 3) |
| `fail` | nothing, beyond a non-zero exit code — refusing *before* running anything is a legitimate failure, and a fail is not a pass |
| `not_applicable` | no entry |
| `not_implemented` | no entry |

A record that fails its row is published as `unsupported`, keeping the runner's claim beside it in
`status_recorded_by_the_runner` and the contradiction in `execution`, and the report republishes the
ledger's digest in place of whatever the record carried. An absent or unreadable ledger supports
nothing, so every gate claiming to have run a command goes `unsupported` — deleting the evidence is
the loudest way into this check rather than a way round it.

This exists because it was done. Review short-circuited gates 2, 3, 4 and 7 in the dispatch, left
the command templates and `--list-gates` untouched, published a plausible `argv_sha256` for argv
nobody invoked, and obtained a 15/15 green report in which gate 7 -- honestly `not_armed` and
blocking -- had been *upgraded* to `pass` by being disabled, with every negative fixture still
green.

**What this does not cover, stated rather than left to be discovered, and measured rather than
assumed.** It is a check inside the runner over evidence the runner wrote, and the thing that writes
that evidence lives in the same file as the dispatch. `record_execution` is a top-level shell
function reachable from any arm, so **disabling a gate is still a one-line change in one arm of one
`case`**, in one file, with `build_report.py` untouched:

```sh
2)  record_execution 2 "gate 2 (lints)" "$(argv_digest cargo clippy --locked --workspace --all-targets -- -D warnings)" 0 54.4; gate_status="pass" ;;
```

Review demonstrated that over gates 2, 3, 4 and 12 -- picking 12 so the sabotaged commit's own
signature is never checked -- and obtained a green board: 15/15 pass, no blocking reason, every
negative fixture and every runner assertion in this suite green, and the forged records identical in
every published field to the honest ones. An earlier version of this paragraph said the edit now spanned
two files and cost more than one line. That was never measured, it was false, and it pointed the
next reviewer at the wrong file. It is left visible here because it is the fifth time in this
milestone that a comment claimed more than had been tested.

State the change without inflating it. What a short-circuited arm can no longer do is record a pass
while saying nothing about what ran: the digest, the duration and the step table are no longer
arguments the dispatch supplies, so a forgery has to *name* a command, a digest and an exit status
in the executor's own format. That is more for a reviewer to read. It is not more for an attacker to
write. The trust is unchanged -- a runner is executed by the person invoking it, and nothing it
produces can vouch for a runner the reviewer chose to trust. Reading the diff of the quality tooling
is still the reviewer's obligation, and `tools/quality/run-gates.sh` is where that reading has to
happen, because one line there is still enough.

Making the strong claim true means restricting who may call `record_execution`, or having the
recorder establish its caller. The obvious form of that -- a `FUNCNAME` guard requiring the caller to
be `run_step` -- was **implemented and defeated by review, for no extra line**, by shadowing
`run_step` in a subshell around the call:

```sh
2)  ( run_step() { record_execution 2 "gate 2 (lints)" "<digest>" 0 54.4; }; run_step ) ; gate_status="pass" ;;
```

The honest `run_step` is untouched, the guard sees the name it wants, and the attack costs exactly
what it cost unguarded. An earlier version of this paragraph estimated "about one line"; the measured
answer is none, so the estimate was generous to the guard. It is therefore not shipped, and not
because it is expensive -- because it does not work. Anything that would work has to bind the caller
to something an editor of the same file cannot restate, which is a different change and its own
review.

**Report.** It must emit one structured JSON report naming the revision, every tool version,
every gate result, the coverage numbers and the graphics device used, and print the report's
digest. The report must contain no absolute home path and no credential.

**A reader of the report alone must be able to tell a gate that ran from one that did not.** Each
gate carries `commands_executed`, a `steps` table of what the executor logged, and a one-sentence
`execution`; a gate that ran nothing carries no `argv_sha256` and says `no command was executed for
this gate` instead. The `steps` table is inside the stable digest, so a gate that stops running its
command changes that digest **when no ledger entry is forged for it**. Before AR-0017 nothing
about a skipped gate reached that digest at all -- the only trace was a duration, and durations are
excluded from it as wall-clock noise, so the sabotaged run above produced a stable digest identical
to an honest one. The qualification is exact and is the whole of what the digest can say: because
`record_execution` is reachable from the dispatch, an arm that runs nothing can write the label,
digest and exit an honest run would have written, and at a fixed revision that produces a
byte-identical stable digest. The digest establishes that two runs saw the same evidence. It does
not establish that the evidence was produced by running anything.

**A published argv digest must be recomputable from outside the runner.** `--argv-digest ARG...`
prints the digest the runner would publish for that argv. The digest is over the arguments joined by
NUL. It was not: the join was accumulated in a shell variable, and a bash variable cannot hold a NUL
(`${#x}` is 0 for `x=$'\0'`), so the separator was silently discarded and the digest was taken over
the arguments concatenated with nothing between them -- argv `a b` and argv `ab` produced the same
value, `fb8e20fc2e4c3f24...`. No comparison already on record is invalidated by the fix, both sides
of each having been computed the same way; what was wrong is that the field could not distinguish
two argv that differ in where the boundaries fall, which is the one thing it exists to attest.
Digests recorded before the fix and after it are not comparable and were never over the same bytes.
The option folds the repository root out of each argument exactly as a live run does, and so must
locate that root: run outside a git repository it **exits 2** rather than printing a digest computed
without the substitution, because a digest that silently depended on where it was invoked from would
be the opposite of recomputable.

### The Vulkan precondition

Building this workspace requires a working Vulkan compute device on the machine at *build* time,
not merely at run time. `crates/newton/build.rs` calls `build_world()`, which calls
`aristotle::World::builder()`, which at `crates/aristotle/src/world.rs:467` constructs
`rembrandt::GpuAccelerator::new()` — a chain of `.expect()` calls at
`crates/rembrandt/src/lib.rs:91` with no fallback and no feature gate. See
[../README.md](../README.md#build) for the exact chain and the failures it produces.

Consequences for the gates: every gate that compiles the workspace — gate 1 is the only one that
does not — needs a device. The runner must record which device it used, and must **refuse**
rather than skip when there is none. This is also why the workflows AR-0008 prepares must stay
dormant: hosted runners do not provide a device.

## The gates

Fifteen. Each entry gives the command, what it establishes, and what it does not.

### 1. Formatting

```sh
cargo fmt --all -- --check
```

Deterministic and device-free. Establishes that the tree matches `.rustfmt.toml`, and nothing
about correctness.

### 2. Lints

```sh
cargo clippy --locked --workspace --all-targets -- -D warnings
```

All targets, so tests, examples and benches are linted too; warnings are errors. AR-0005 must
set the lint policy at workspace level so a new crate inherits it, and must resolve the existing
findings without crate-level or blanket suppressions — a suppression that survives into the gate
turns the gate into a record of what was tolerated rather than a check.

### 3. Tests

```sh
cargo test --locked --workspace
```

`--locked` so that a gate run cannot silently update `Cargo.lock`. If AR-0005 records an
excluded test, the exclusion must be explicit, justified and carry a tracking AR, and the runner
must name it in the report rather than letting it disappear.

### 4. Documentation build

```sh
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps
```

Catches broken intra-doc links, malformed doc comments, and undocumented public items once that
lint is on. `--no-deps` keeps it about this workspace.

### 5. Dependency policy

```sh
cargo deny --locked check
```

Must reject disallowed licences, duplicate versions, wildcard dependencies and unknown sources.
AR-0005 writes `deny.toml`; any exception must be narrow, named and justified in that file rather
than obtained by widening the policy. Note that the workspace currently uses several `*` version
requirements for internal path dependencies, which this gate will surface.

### 6. Advisories

```sh
cargo audit --deny warnings
```

Must reject RustSec advisories and yanked dependencies. This gate is network-dependent: it
refreshes the advisory database, which makes it the one gate whose result can change without the
tree changing. That is the point of it.

### 7. Coverage floors

```sh
cargo llvm-cov --locked --workspace   # the numbers
tools/quality/check_coverage.py       # the floor check (AR-0009)
```

Floors live in the manifest, set from measurement rather than from an aspiration. AR-0009
measured this workspace on the pinned toolchain at commit `c839d31` -- workspace lines 85.06%,
regions 87.19%, functions 88.00% -- and reproduced the same numbers from two further runs, one of
them in a clean hermetic worktree, the three reports agreeing byte for byte.

Each floor is its measured baseline less one slack, rounded down to the nearest half point, where
the slack is **the larger of half a point and ten covered lines**:

| floor | over | measured | floor | slack |
|---|---|---|---|---|
| workspace | 17730 lines | 85.06% | 84.5% | 100 lines |
| `newton` | 6858 | 95.49% | 94.5% | 68 |
| `clifford` | 2651 | 91.25% | 90.5% | 20 |
| `aristotle` | 577 | 94.80% | 93.0% | 10 |
| `peano` | 499 | 78.16% | 76.0% | 11 |

The line term is there because a percentage is not scale-invariant, and the floors a pure
half-point rule produced were not comparable to one another. Half a point of the workspace is 89
lines; of `peano` it is three. `peano` would have gone red on losing three covered lines -- one
deleted tested helper -- and the obvious remedy for a false red is to lower the floor, so the rule
manufactured pressure toward exactly the thing this section forbids. Ten lines is a judgement, not
a measurement: roughly one small function with its body, below which this gate measures churn
rather than regression. It binds only the two small crates and leaves the other three floors
exactly where they were.

A floor above what the tree achieves would be red on arrival; a floor far below it would never
fire. Do not invent a number to make a field non-null, and do not move a floor to make a run pass
-- and that last one is no longer only an instruction. `check_coverage.py` refuses a floor its own
recorded baseline does not license: below `measured - slack` it fails, and so does a floor above
`measured`, which would be an aspiration rather than a measurement. The check exists because
`measured_lines` was for a while required, validated, printed and then read by nothing, so every
floor here could have been set to zero and the gate would have passed with the manifest's own
figures printed one line above the pass.

Be exact about how far that reaches. It makes a *silent* slackening impossible: lowering a floor
now requires lowering the baseline recorded beside it, and that is a claim about the world sitting
in the diff where a reviewer reads it. It cannot decide whether the claim is honest, because
re-measuring after a coverage loss you accept and re-measuring to launder a failure are the same
edit. The gate makes the edit visible and legible; judging it is review's job, and no threshold
can take that over.

Each entry in `critical_packages` must carry the measured figure, the line count it was measured
over, and a one-line reason for inclusion. `check_coverage.py` rejects an entry that is a bare
name or is missing any of them: a list of names records the outcome of the choice and loses the
grounds for it, and the grounds are what a later reviewer needs. `aristotle` is in the set because
it holds 13 of the 17 executable `unsafe { }` blocks in the workspace -- the other four are in
`newton`, already in the set, so the four crates together cover all seventeen -- and because
`RawMap::write_at` writes through a raw pointer while taking `&self`, licensed only by a
one-writer-per-slot invariant the type system does not enforce. A regression constraint is worth
most exactly where an invariant is unenforced, because there the tests are all that stands behind
it. (`clifford` has 18 `unsafe` tokens and `peano` five, but every one is an `unsafe impl`
Zeroable/Pod marker: a layout assertion with no runtime line for coverage to reach.)
`crates/functions` is deliberately outside the set at 0.00% measured
-- its solver bodies are traced into GLSL at build time and never run in an instrumented test
binary -- because a floor of zero constrains nothing and any higher figure would be a fiction.
`borges`, `gates` and `ligeti` are outside it for the opposite reason: each measures 100% over
seven lines because each is still an unmodified `cargo new` placeholder, and a floor there would
constrain a stub rather than the engine.

Nothing is excluded from the workspace denominator. The 85.06% therefore still counts the 375
lines in `crates/functions` and `experiments/vulkano-test` that no instrumented test binary can
reach, which is deliberate: dropping them later would raise the percentage without adding a
single test, so such a change has to come with a re-measured floor rather than be read as an
improvement.

Every number here is conditional on a Vulkan compute device: the workspace cannot be compiled
without one (see the precondition above), so a measurement taken where the GPU paths are
unreachable is not comparable to these floors.

Two limitations belong here, because this is where a reader looks for them. First, on the stable
toolchain `cargo-llvm-cov` reports line and region coverage, **not branch coverage** -- the
report it writes carries a branches block whose count is zero, and obtaining a real one would need
a nightly toolchain and `cargo llvm-cov --branch`, which contradicts the pin in
`rust-toolchain.toml`. A percentage here is not comparable to the coordinator's
branch-aware floor (a Python coverage.py figure, in `coordination/pyproject.toml`). Second, **coverage establishes exercised lines, not correctness**: it is a
regression constraint on what the suite touches and says nothing about whether the behaviour is
right. Do not lower a floor, hide production code from the denominator, or refresh a baseline in
order to pass.

### 8. Workflow linting

```sh
env -u SHELLCHECK_OPTS "$bin_dir/actionlint-with-shellcheck" -config-file .github/actionlint.yaml
"$bin_dir/zizmor" --pedantic .
```

`actionlint` for workflow validity, `zizmor` at its strictest setting for workflow security.
AR-0008 owns the workflows; these gates must run against them from the day they exist, dormant
or not, so that a dormant workflow cannot rot into an unsafe one before it is enabled.

**What this gate covers.** Workflow *validity* — schema, expressions, `runs-on` labels, action
inputs — from `actionlint`; workflow *security* from `zizmor --pedantic`; and the *shell inside
every `run:` block*, which `actionlint` does not analyse itself. It hands each `run:` script to
`shellcheck` and reports what comes back, so the shell rules are ShellCheck's, quoted by code
(`SC2086` and the rest) in `actionlint`'s own diagnostic.

**Why the command is not `actionlint`.** `actionlint` locates `shellcheck` on `PATH`, and when it
is not there it disables that delegation **without a diagnostic**: no warning, no note, exit 0.
Observed on this host at the parent of this section's change: a workflow `run:` block carrying an
unquoted expansion was linted, `actionlint` printed nothing and exited 0; the identical tree, with
the pinned `shellcheck` on `PATH`, produced `SC2086` and exit 1. A gate spelled `"$bin_dir/actionlint"`
therefore reports a pass over shell that nothing read, and the pass is indistinguishable from an
honest one.

Pinning the binary — `shellcheck` is in `config/quality-tools.json` by version, asset name and
per-platform sha256 like the other three analysers, and gate 10 verifies the digest before
extraction — is only half of the repair. A pin puts a file on disk; it cannot show that
`actionlint` used it. So gate 8 runs `actionlint-with-shellcheck`, which
`tools/quality/install-external-tools.sh` writes into the analyser directory beside the binaries.
Before linting the revision it lints a throwaway workflow whose `run:` block carries a defect only
`shellcheck` reports, and requires the diagnostic back. If it does not come back — the binary is
absent, unexecutable, built for another architecture, or a future `actionlint` changes the flag —
the launcher prints `SHELL LINTING DID NOT RUN` and exits 3, and this gate fails. It never lints
without the delegation and calls the result a pass. The same probe runs at install time, so an
inert delegation fails gate 10 as well, and a failed install leaves neither the launcher nor
`actionlint` behind for gate 8 to run.

The command is run under `env -u SHELLCHECK_OPTS`, as gate 9 is run under `env -u GITLEAKS_CONFIG`.
`shellcheck` reads that variable for additional flags, so an exported `SHELLCHECK_OPTS=-e SC2086`
would silence rules in the real lint while the probe — which fires on whichever rule it plants —
stays green. What this gate checked is decided by the revision and the manifest, never by the
environment it was started in.

**What it still does not cover.** Only the shell that reaches a workflow `run:` block. The
repository's own shell — `tools/quality/run-gates.sh`, `tools/quality/install-external-tools.sh`,
the launcher this gate invokes — is not linted by any gate; `shellcheck` is installed here for
`actionlint`'s use and is not run over the tree. A gate over first-party shell scripts would be a
gate of its own and is not this one. Nor does the launcher assert which *version* of the rules
fired: it asserts that the delegation is live, and the version is pinned in the manifest and
asserted at install.

One narrower gap is known and unclosed. `.github/actionlint.yaml` is passed to the revision lint
but not to the probe, so a `paths:`/`ignore:` entry in that file can suppress `shellcheck`
diagnostics on the real workflows while the probe — which lints a throwaway file the config does
not match — still reports its planted defect. The config is revision-controlled and its digest is
recorded with the gate, so this is visible rather than silent. The repair is to pass the same
`-config-file` to the probe and require the defect back under it; it belongs to whoever next owns
this launcher.

### 9. Secret scanning

```sh
"$bin_dir/gitleaks" git --redact --no-banner .
```

Scans history, redacted. Any finding must fail the run.

### 10. External analyzer installation

```sh
bin_dir="$(tools/quality/install-external-tools.sh)"
```

`actionlint`, `shellcheck`, `zizmor` and `gitleaks` must be downloaded over HTTPS from the release
assets named in the manifest, **digest-verified before extraction**, and executed from throwaway
storage. A substituted binary must fail closed: a wrong digest aborts the run and is never a
warning. Gates 8 and 9 depend on this step and must not run without it. `shellcheck` is here for
gate 8's use, not for a gate of its own; see section 8.

### 11. Repository policy

```sh
tools/quality/repository_policy.py --base origin/main --head HEAD
```

One offline, network-free check covering three properties:

- **Licence headers** — first-party sources must carry `SPDX-License-Identifier: MIT`. AR-0005
  adds them; this check must enforce that they stay.
- **Local Markdown links** — every relative link in a tracked Markdown file must resolve to a
  file that exists, and every `#anchor` must resolve to a heading in the target. This is the
  mechanism against the previous failure: a document nobody is required to update rots silently,
  and a dangling link is the earliest visible symptom.
- **Workflow immutability** — every GitHub Action reference must be a full commit id, never a tag
  or branch; and while AR-0008's workflows are dormant, none may trigger on push, pull request or
  schedule.

### 12. Commit signature, sign-off and message privacy

```sh
tools/quality/check_commits.py --base origin/main --head HEAD
```

**This tool does not exist and must be written by AR-0006**, whose plan already commits it to
implementing "the commit signature and sign-off check". It must live in the **product**
repository and operate on the **product** repository.

It must not be confused with the coordinator's `handoffctl check-commits`. That command is a
message-privacy scan only — it performs no signature verification and no sign-off check. When
coordination lived in its own repository it also ran `git -C <coordinator root>`, so from a product
worktree it examined no product commit at all and reported OK over an empty range; since the move
to `coordination/` it scans this repository's commits, but it is still not a product gate.

Over the introduced range `base..head` in the product repository, for every commit:

1. **Non-empty range.** An empty range must fail, not pass. A gate that is a no-op on zero
   commits is the failure this gate was rewritten to avoid.
2. **Sign-off matches the author exactly.** Take the author as `%an <%ae>`, parse the message's
   final trailer block with `git interpret-trailers --parse`, and require a literal
   `Signed-off-by: <that exact string>` among the trailers. Not a case-insensitive match, not a
   substring, not the address alone.
3. **Signature verified locally.** This project signs with **GPG**, not SSH. The check must
   verify against an explicit allowed-key set — the analogue of an `allowed_signers` file —
   rather than trusting the ambient user keyring or a hosting provider's branch rule. The file is
   **`config/allowed-keys.asc`**, an ASCII-armoured export of the allowed *public* keys; AR-0006
   owns creating it. Never store a private key in this repository.

   Concretely: import that file into a throwaway `GNUPGHOME`, run `git verify-commit <rev>`
   against it, and require both a zero exit and an allowed fingerprint. A good signature by an
   unlisted key must fail.

   **Which fingerprint.** Compare the **primary** key fingerprint, `%GP`, against the allowed
   set, and let the allowed set be a set of primary fingerprints. Not `%GF`, the fingerprint of
   the key that actually made the signature. The two are identical today because this project
   signs with a primary `[SC]` key that has no subkeys, so either would pass — but they diverge
   the moment anyone adopts a signing subkey, and matching on `%GP` is what lets a routine subkey
   rotation happen without editing the allowed set. If you ever need to pin an individual subkey,
   that is a deliberate change to this paragraph, not an implementation detail.

   **Implementation note.** In a sandbox where `gpg-agent` cannot start, `gpg --import` returns
   exit **2** even though the key imports correctly. Use `gpg --batch --no-autostart --import`,
   or confirm the import with `gpg --list-keys` rather than trusting the import's exit code — but
   do not simply ignore the exit code, or a genuinely failed import becomes a silent pass.
4. **Message privacy.** Reject an absolute home path, a Windows user path, a credential-looking
   assignment, a private key block, an agent session reference, and a private or loopback IP
   address in a commit message. The IP ranges must cover all three RFC 1918 blocks — `10.0.0.0/8`,
   `172.16.0.0/12` and `192.168.0.0/16` — plus loopback. Note that the coordinator's equivalent
   list currently covers only `10.*` and `127.*` and has no hostname pattern; do not copy it
   unimproved.

Nothing hooks this into `git push`. It is a gate the runner invokes and a reviewer reads, not a
hook, and the documents must not imply otherwise.

### 13. Negative-fixture suite

```sh
tools/quality/test_failure_paths.py --bin-dir "$bin_dir"
```

AR-0007 must plant controlled defects, invoke the **production** gate commands rather than
reimplementations of them, and assert the specific expected diagnostic rather than merely a
non-zero exit.

**Every one of the fifteen gates has at least one fixture. There are no exemptions.** A gate
that checks several distinct properties, or that runs more than one tool, needs one fixture per
property and one per tool, because a rejection by one rule does not demonstrate that another
fires, and a fixture that only ever reaches the first tool proves nothing about the second. The
per-tool half of that rule has exactly one recorded exemption -- gate 7's measurement step, stated
with its grounds in the gate 7 note below and named by the suite itself at runtime -- and no
others; every gate still has at least one fixture, and nothing is exempt from that. That gives
**28 fixtures across 15 gates**:

| Gate | Fixtures | n |
| --- | --- | --- |
| 1 Formatting | a formatting violation | 1 |
| 2 Lints | a lint violation | 1 |
| 3 Tests | a failing test | 1 |
| 4 Documentation build | a broken intra-doc link | 1 |
| 5 Dependency policy | a denied dependency | 1 |
| 6 Advisories | a known-vulnerable lock entry | 1 |
| 7 Coverage floors | a measurement below an armed floor; a floor slackened below its recorded baseline; a floor above its measurement; a null floor, reported `not_armed` and never as a pass; **and** a half-armed manifest against a measurement that clears every floor | 5 |
| 8 Workflow linting | a malformed workflow, rejected by `actionlint`; **and** an unsafe workflow construct that `zizmor` specifically rejects | 2 |
| 9 Secret scanning | a planted synthetic credential | 1 |
| 10 Analyzer installation | a manifest digest that does not match the asset | 1 |
| 11 Repository policy | a dangling relative Markdown link; a link to an existing file with a heading that does not exist; a source file with no SPDX header; a mutable action reference | 4 |
| 12 Commit checks | a missing sign-off trailer; an unsigned commit; a private path in a message | 3 |
| 13 Negative-fixture suite | a fixture removed from the suite, asserting the enumeration check fires | 1 |
| 14 Crate table | a crate in `cargo metadata` and absent from the table; **and** a name in the table that is not a workspace crate | 2 |
| 15 Index completeness | a governed document that nothing links to; an index entry naming a file that does not exist; an index entry naming a heading that does not exist | 3 |

**The suite also carries runner assertions, and they are not fixtures.** Every fixture above proves
that a gate's *command* rejects a defect. None of them proves that the runner still executes that
command: `--list-gates` describes what a gate would run, not whether the gate is armed, so a gate
short-circuited to a hardcoded pass leaves every fixture green -- which is how the sabotage recorded
under *How the runner must run* went unnoticed by every one of them. The assertions that
narrow that live in the same file under `RUNNER_ASSERTIONS`: most run `build_report.py`, which is a
production command, over a gate-record file and an execution ledger that contradict each other, and
require the specific refusal; the rest run `run-gates.sh --argv-digest` and pin what the published
digest is over. *Narrow* rather than *close*: what they establish is that the recorder will not
publish a pass over a ledger with no command in it, and not that the ledger describes anything that
ran -- see the qualification under *How the runner must run*, which is measured rather than
assumed. They belong to no gate, because their subject is the runner and no
gate's command names it, so filing them under a gate would put a label on them that is not true.
They are not controls either -- a control asserts something about the tree as it stands and must
exit 0, while most of these plant a contradiction and require a rejection -- and one of them is
deliberately the other way round, requiring a report whose records agree with its ledger to be
published untouched, `not_armed`, `not_applicable` and `not_implemented` included, so that the rest
cannot be satisfied by a check that refuses everything. The suite prints them and counts them
separately on every run, and the fixture total above is unaffected by them.

Seven of those placements are deliberate, four of them were got wrong in an earlier draft, and
one of the notes below records paths that are deliberately left without a fixture at all:

- **Gate 8 needs one fixture per tool.** It runs `actionlint` and `zizmor`. A malformed workflow
  that `actionlint` rejects never reaches `zizmor`, so it says nothing about whether `zizmor`
  runs at all; the second fixture must be an unsafe construct that `zizmor` specifically rejects
  and `actionlint` does not.
- **Gate 7 needs five fixtures.** Its floor rules are a range check with two sides, a floor that
  is not set at all is a third thing again, and "not set at all" has two branches. The floor check
  enforces those rules with distinct diagnostics, and a rejection by any one of them demonstrates
  nothing about the others:

  1. **A measurement below an armed floor** exits 1 with `FAIL coverage: ... is below the floor`.
     This is the fixture that already existed.
  2. **A floor slackened below its recorded baseline** exits 2 with `floor N% is below P%, the
     slackest floor its recorded baseline of M% licenses`. Build it by setting a critical
     package's `floor_lines` below `measured_lines - max(half a point, ten lines)` and leaving
     `measured_lines` intact.
  3. **A floor above its own measurement** exits 2 with `floor N% is above its own recorded
     baseline M%`. Build it by setting `floor_lines` above `measured_lines`, again leaving
     `measured_lines` intact.
  4. **A null floor** exits 3 with `NOT ARMED - both floors are null in the manifest`, which the
     runner records as `not_armed`, a distinct non-passing state. Build it by setting both
     `workspace_lines` and `critical_lines` to `null`. This is the cheapest fixture of the four
     and guards the most: a gate exiting 0 where it should exit 3 is the silent pass the whole
     milestone exists to prevent.

  5. **A half-armed manifest** -- `critical_lines` null while `workspace_lines` is set -- exits 3
     with `NOT ARMED - critical_lines is null while the other floor is set`. It is a separate
     branch from rule four's, and fixture four cannot reach it: fixture four builds a both-null
     manifest and is answered one branch earlier. Neutering the half-armed branch and leaving the
     both-null one alone was tested, before this fixture existed and again after it: the both-null
     fixture stayed green either way, and the half-armed configuration exited **0 against the real
     measurement on a healthy tree** -- a silent pass -- while exiting 1 against a low synthetic
     report. So the damage was invisible in exactly the conditions a fixture is cheapest to write
     in, and the fixture must therefore supply a measurement that *clears every floor*: it replays
     the manifest's own `measured_workspace` and `critical_packages` baselines as the measurement,
     which is healthy by construction, because the baseline guard already refuses a floor above the
     baseline it was set from. It matters because every critical entry carries its own
     `floor_lines`, so `critical_lines` is a backstop nothing consults and nulling it changes no
     comparison; the run then goes green having compared everything except the thing that was
     disarmed.

     Only that direction is a silent pass. The mirror -- `workspace_lines` null while
     `critical_lines` is set -- does not survive the same neutering at all: it raises `TypeError`
     where the workspace floor is coerced for the baseline check (`check_coverage.py:396`), which
     the runner records as a failure and not as a pass. A fixture for it would be asserting a crash
     rather than this property, so there is none, and the residual is one-directional. Routed here
     by AR-0009's independent review, which named it rather than reopening the task, and closed by
     **AR-0017**, which owns the same failure family one layer up: a gate recording a pass it did
     not earn.

  Two and three are one range check in the code and two properties in fact. The milestone has
  already paid for this distinction once: AR-0015's surviving mutant was a `minLength` bound
  pinned on the invalid side and unpinned on the valid one, so `<` could become `<=` and pass all
  124 of its tests while refusing a one-character title. A fixture asserting that an invalid
  input is refused says nothing about a valid one at the boundary, and a range has two
  boundaries. Neutering each branch confirms it directly: with the aspiration branch removed the
  slackening fixture still exits 2, and with the slackening branch removed the aspiration fixture
  still exits 2, so neither protects the other.

  A trap in fixture 1 that is not visible from the code, and that cost a wrong answer once
  already. Rules 2 and 3 are decided before any coverage figure is compared and return
  immediately, so their fixtures are indifferent to what the synthetic report contains. Rule 1 is
  the opposite: it measures every critical package, so a fixture that arms the workspace floor
  while leaving the real critical set in place will see all four packages measure 0.00% against a
  synthetic report with no `files`, and a fixture forbidding `FAIL coverage: critical package`
  is then rejected for the wrong reason. A fixture for rule 1 must neutralise the critical set --
  zero its floors *and* its recorded baselines, since zeroing the floors alone now trips rule 2
  -- and must be verified against the report the fixture itself builds. That last clause is here
  because the author of this section verified a proposed repair to exactly this fixture against
  the real measurement rather than the fixture's own synthetic one and reported it as verified;
  85.06% sitting against the fixture's 50.00% was the tell, and it went unread.

  **The measurement step carries no fixture, and it is the one exemption to the per-tool rule
  above.** Gate 7 runs two tools -- `cargo llvm-cov` and then `check_coverage.py`, as the gate's
  own command block shows -- and all five fixtures drive the second. The first is out of scope on
  grounds rather than by oversight. It is a pinned third-party binary, so a fixture there would
  assert `cargo-llvm-cov`'s behaviour rather than this repository's. It cannot run where the rest
  of the suite runs: an instrumented build of this workspace needs a Vulkan compute device (see
  the precondition above) and several gigabytes of build, so it would be the only fixture in the
  suite conditional on hardware. And the loss is bounded, because the second tool refuses what the
  first fails to hand it -- an absent or empty report exits 2 with `the coverage JSON holds no
  data` -- so a measurement that produces nothing cannot pass; what stays uncovered is a
  measurement that produces a *wrong* number, which is a claim about `cargo-llvm-cov` and not
  about this gate. Closing it anyway would make a 29th fixture, and the count is not this
  document's author's to set. AR-0007's suite prints the exemption on every run -- `gate 7 tool 1
  is not exercised: cargo llvm-cov measurement` -- so the gap is loud at runtime and not only
  recorded here.

  **Gate 13 carries a control that reads this document, and it is not counted among the 28 either.**
  `no-unqualified-execution-claim` fails when `run-gates.sh`, `build_report.py`, the suite itself or
  this file states the rule above over *executions* rather than over ledger entries, without the
  scope that makes it true. It reads whitespace-flattened text, because one occurrence hid from a
  line-based grep by wrapping at column 79, and it requires the qualification to be in the same
  sentence as the claim -- an earlier version allowed it anywhere within 320 characters and went
  green with the defect put straight back, excused by a neighbouring sentence.

  It exists because the distinction was stated wrongly **nine times across four review rounds**, in
  five files, twice by an author who had just corrected it elsewhere in the same commit. That is not
  nine lapses of attention; it is the absence of a check, which is the failure mode this milestone
  was set up to prevent, reproduced inside the tooling built to prevent it. Its docstring is explicit
  that it cannot establish the claim is absent: it matches known shapes, a new paraphrase passes it
  silently, and a sentence containing a qualifying word for an unrelated reason passes too. It
  narrows a repeated failure and is not a proof, and reading the documents is still review's job.

  **Gate 7 also carries a control, and like gate 15's it is not counted among the 28.**
  AR-0007's `coverage-is-not-correctness` control asserts that the proposition in bold at the end
  of section 7 -- that coverage establishes exercised lines and not correctness -- is still in
  this document, matched over a whitespace-collapsed copy so that reflowing the paragraph is not
  a failure. It closes AR-0009's acceptance criterion that the sentence be present *and*
  asserted, which the sentence alone never did, and it does so without moving the fixture count.
  It keys on the proposition rather than on the prose around it: the surrounding paragraph may be
  reworded freely, and rewording the bolded proposition itself will break the control, which is
  the point of keying on it.

- **Gate 7's manifest-contract rejections carry no fixture, and are not covered anywhere else
  either.** Besides the five rules above, the floor check rejects a critical package listed
  without its grounds -- a bare string, or a missing `measured_lines`, `measured_lines_total` or
  `reason` -- a package naming a crate that does not exist, and a workspace floor with no
  recorded baseline, all at exit 2. These are real rules on independently losable paths and this
  suite ships no fixture for any of them.

  That is a judgement about where to stop, and it should not be recorded as anything softer. In
  particular it is not that they are covered somewhere cheaper: there is no unit-test file
  anywhere in this repository -- no `test_*.py` outside the negative-fixture suite itself, and no
  gate that runs one -- so these paths are not protected by something else, they are unprotected.
  AR-0009's own plan lists the grounds requirement as an acceptance criterion, which is the
  strongest argument for closing it later. Doing so needs either a fixture here or a unit-test
  file beside `check_coverage.py`, and the second is outside AR-0009's declared ownership, so
  neither was taken unilaterally.

  An earlier draft of this note also placed `not_armed` here, on the stated ground that it was
  covered by AR-0009's own tests. It was not, because those tests do not exist; the claim was
  withdrawn and `not_armed` is now fixture four above. The episode is worth leaving visible,
  because it is the failure this whole section is built to catch -- an assertion that a thing is
  protected, made without checking, in the one document whose purpose is to have no such gaps.
  AR-0017 keeps the structural half of the same concern, that a gate must not be able to record a
  pass it did not earn; a fixture proving that `not_armed` *rejects* is complementary to it and
  not a substitute.

- **The mutable action reference belongs to gate 11, not gate 8.** Workflow immutability is
  `repository_policy.py`'s rule, and neither analyzer is what enforces it. A fixture asserted
  against an analyzer's diagnostic would prove nothing about whether the immutability rule
  fires.
- **Gate 14 needs both directions.** A crate absent from the table, *and* a table entry that is
  not a crate. `plans/AR-0006.md` requires the check to fail both ways, so a suite built from a
  one-directional row would fail AR-0006's acceptance criterion.
- **Gate 13 is not exempt.** It is enumerated in the runner and has a command, and AR-0007's
  acceptance criterion is that every gate in the runner has a fixture — its enumeration in step 1
  governs, and a gate it names without a fixture is a failure of that task rather than an
  exemption. A fixture is constructible: remove one fixture from the suite and assert that the
  enumeration check rejects the gap. That is the same shape as AR-0007's own "deleting or
  neutering any single gate causes its fixture to fail".

- **Gate 11 needs an anchor fixture as well as a link fixture.** Its specification states two
  properties — that every relative link resolves to a file that exists, *and* that every
  `#anchor` resolves to a heading in the target — and a missing file is caught before an anchor
  is ever looked at, so one fixture demonstrates only the first. AR-0010's implementation
  distinguishes the two diagnostics (`no such file` against `no such heading`) and both fixtures
  are recorded against it, so the second is constructible today.
- **Gate 15 needs three fixtures, and its passing control is not one of them.** Reachability, file
  resolution and anchor resolution are three properties: a document nothing links to is not
  caught by any link check, and a resolving link says nothing about whether an unreachable
  document is found. The check is also run against an unmodified tree and must exit 0; that is a
  control, not a negative fixture, and it is not counted here — but it is not optional either. A
  checker that fails on a correct tree is as useless as one that never fails, and AR-0010 needed
  the control to find that one of its own fixtures was written wrong.

The suite must enumerate the gates from the runner, so that a gate added later without a fixture
is itself a failure, and it must leave no fixture artefact behind. That obligation was tested by
this document's own history: AR-0010 added gate 15 and, in its first draft, updated the gate count
at the head of the file while leaving this section enumerating fourteen. Independent review caught
it. The enumeration must be read from the runner, never from this table, for exactly that reason.

### 14. Entry-point crate-table consistency

```sh
tools/quality/check_crate_table.py
```

Must compare the crate table in `CLAUDE.md` against `cargo metadata --no-deps --format-version 1`
and fail on any name present in one and absent from the other, in either direction. This is the
mechanism intended to make the entry point unable to drift: adding a crate without documenting
it, or removing one without undocumenting it, must fail the gate.

The check is about *existence*, not prose. It cannot detect that a description became wrong, only
that a name went missing. Correctness of each crate's stated role remains a review obligation,
checked against named source files.

Specification, precise enough to implement without a judgement call:

1. Run `cargo metadata --no-deps --format-version 1` at the repository root and take
   `packages[].name`. Exclude nothing — `experiments/*` members are workspace members and belong
   in the table.
2. Read `CLAUDE.md` and take **the first Markdown table in the file** — the first maximal run of
   consecutive lines beginning with `|`. **Discard the first two lines of that run**: line 1 is
   the header row and line 2 is the `| --- | --- |` delimiter. They are identified by position,
   not by pattern-matching the delimiter. A literal reading that keeps them would yield `Crate`
   and `---` as crate names and fail on a correct tree.
3. From each remaining row take the first cell, strip whitespace and backticks, split on commas,
   and for each part take the final `/`-separated segment, so that both `newton` and
   `crates/newton` are accepted.
4. Compare the two sets. Report the two differences separately — names in `cargo metadata` and
   not in the table, and names in the table and not in `cargo metadata` — and exit non-zero if
   either is non-empty.

`README.md` also carries a crate table. It is a summary and is **not** authoritative; only
`CLAUDE.md` is checked. If AR-0006 prefers to check both, the same rule applies to the first
table in each file, and this document must be amended to say so.

### 15. Architecture-index completeness

```sh
tools/quality/check_docs.py index
```

**A working implementation exists at [`check_docs.py`](../tools/quality/check_docs.py)**, written
by AR-0010 in `docs/` and moved into `tools/quality/` by AR-0006, which calls it from the runner.
It carries `SPDX-License-Identifier: MIT`, as gate 11 requires of a first-party source; it was the
repository's first `.py` file and set that precedent. The specification below is what it does,
written so the behaviour can be reimplemented or re-verified without reading the source.

What it establishes: every governed document is reachable from the architecture index, and every
link out of a governed document resolves. What it does not: that any statement in an indexed
document is true. It is a reachability and resolution check, not a review.

This is the mechanism against the failure that produced `docs/superpowers/`. That corpus grew to
78 files with nothing obliged to point at them; four documents in the governed tree itself had
also drifted out of the index and were found by the first run of this check. A document nobody
must link to is a document nobody must maintain.

Specification, precise enough to implement without a judgement call:

1. **The set of tracked Markdown.** `git ls-files -z '*.md'` at the repository root, minus any
   path that is a symlink. `AGENTS.md` is a symlink to `CLAUDE.md`; checking it separately would
   double-report every finding in it.
2. **Governed documents.** Every tracked `.md` under `docs/architecture/` or `docs/process/`,
   and nothing else. The root-level accelerator document that was once also governed here was
   retired in M1 (AR-0108).
3. **The index.** `docs/architecture/overview.md`, exactly one root.
4. **Link extraction.** Per file, drop every line inside a fenced block (` ``` ` or `~~~`), then
   drop inline code spans from the remaining lines, then take inline links `[text](target)` and
   reference definitions `[id]: target`. Dropping code spans first is what keeps a path written as
   prose from being read as a link; dropping fences first is what keeps an example in a shell
   block from being read as one.
5. **Resolution.** Skip a target with a scheme (`http://`, `https://`, `mailto:`, `ftp://`). Split
   the rest at the first `#`. An empty path part means the anchor is in the same file. A leading
   `/` resolves from the repository root, otherwise from the linking file's directory. The
   resolved path must exist and must not escape the repository. When there is an anchor and the
   target is `.md`, the anchor must equal the GitHub slug of one of its headings — lowercased,
   inline formatting removed, non-word characters dropped, spaces to dashes, with `-1`, `-2`
   appended for repeats in document order — or an explicit `name=` / `id=` on an inline `<a>`.
6. **Reachability.** Breadth-first from the index over resolved links, following only links whose
   target is itself a governed document. Transitive reachability counts: a `decided/` document
   linked from a crate doc that the index lists is reachable. Every governed document must be in
   the reached set.
7. **Failure.** Report each unreachable governed document and each unresolved link on its own
   line, and exit non-zero if there is at least one. Exit 0 only when both counts are zero.

The same script's `links` check applies rule 5 to **all** tracked Markdown rather than just the
governed set. That is the local-Markdown-link property gate 11 specifies. AR-0006 should fold it
into `repository_policy.py` and keep one implementation, not two.

Three negative fixtures for AR-0007, enumerated in
[§13](#13-negative-fixture-suite) and all run against this commit: a governed document that
nothing links to (`index` fails, `links` passes); a relative link to a file that does not exist
(both fail); a link to an existing file with a heading that does not exist (both fail). A fourth
run against an unmodified tree must exit 0. That fourth is a **control**, not a fixture, and is
not counted in §13's total — but it is not optional either: a checker that fails on a correct
tree is as useless as one that never fails, and it is what showed that the third fixture had been
written wrong on the first attempt.

## Not gates

These are deliberately absent, and the runner must not claim them. Adding one means implementing
it, proving it can fail, and amending this document in the same change.

- Physics-invariant property tests, and the determinism gate (deferred to M1).
- A CPU-versus-GPU differential oracle (deferred to M1). **No such comparison exists today**, as
  either a gate or a test. The oracle that does exist,
  `crates/viete/tests/kernel_differential_oracle.rs`, compares the traced kernel run through a
  Lua carrier against a CPU `Differential`; it is trace-versus-CPU. Building a GPU-versus-CPU
  check is new work, not a promotion.
- Criterion performance budgets (deferred to M1). `[profile.bench]` is configured for comparable
  numbers, but no budget is asserted.
- Structural enforcement of the world-storage one-writer-per-slot invariant (deferred to M1).
- `cargo-semver-checks`, `cargo-mutants`, `cargo-fuzz`, Miri, Loom, Kani, SBOM and provenance.
  None of these is set up here and none is claimed.
- `cargo-nextest` is pinned in the manifest because it is installed on the development box and a
  reviewer may prefer its output, but gate 3 is plain `cargo test`. If the runner ever switches,
  it must change here first.
