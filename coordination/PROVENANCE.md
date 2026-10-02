# Coordination framework provenance

The coordinator in `tools/`, its fault tests in `tests/`, the task schema and the locked
Python quality configuration are adapted from
[agent-systems-benchmark-state](https://github.com/martin-beck/agent-systems-benchmark-state),
at revision `c109ebb82a38ca2a7dc4707f88fa15aac46bedd6`, which is MIT licensed. That project
in turn documents its own adaptation from `agent-relay-state`.

Only generic coordinator code is reused: the locked transaction, revision and lease model,
the deterministic status renderer, the dependency-graph validation, the privacy scanner and
their tests. **No task history, plan content, runtime configuration or Git history is imported.**

Adaptations made here:

- Renamed the product to Game Experiment; the worktree key pattern now matches
  `game-experiment(-suffix)`.
- The reference's hardcoded `SERIES` map in `tools/status_renderer.py` labelled thematic
  work series. Here the series digits identify a **milestone**, and the labels are derived
  from the documents in `milestones/` instead of being hardcoded. `status_renderer.py`
  still imports only the standard library and does no file I/O: `handoffctl` reads the
  documents and passes a series-to-label mapping into `render_status`, which escapes every
  label through the same inert-text helper as the rest of the untrusted front matter. The
  reference has no milestone layer, no milestone schema and no `complete-milestone`
  transaction.
- `validate()` gains milestone rules the reference has no equivalent for: the document
  contract, one document per observed series, and the refusal to hold a milestone complete
  while an AR in its series is unfinished. It also gains a prose rule, because an
  independent review of AR-0001 found a task body that still told the reader implementation
  had not started after the status said otherwise, and no gate could see it. The rule scans
  only the description above the first appended evidence entry, and fires on the presence of
  evidence rather than on status: the evidence log is append-only, so a rule its entries
  could trip would become permanently unsatisfiable for any AR whose evidence describes the
  rule itself.
- The worker terminal state is added here. The reference offers `in_progress` and release
  only, so a worker that had finished could not leave `in_progress` without either certifying
  its own work or holding a lease that would eventually report an expired claim against work
  that was complete. This project adds the `in_review` status, the `submit` transition that
  clears the claim and records the submitter, the `review` transition that is not owner
  authenticated and refuses the submitter as its own reviewer, and the `validate()` rule that
  binds a recorded submitter to that status and no other. Release, which is owner
  authenticated, no longer accepts `done`, so the only route to `done` runs through a review of
  submitted work. Two choices inside this are recorded rather than left to be rediscovered.
  `review` is deliberately not given the preflight that `promote` and `resume` carry: it is
  fenced by `--expected-revision` and by the full `validate()` that `mutate` runs before it
  commits, and the transitions that close a claim — release, submit and update — carry no
  preflight either. And the presentation colour for `in_review`, `#ad1457`, was chosen for a
  contrast ratio of roughly 7:1 against its white text, calculated by hand: neither this
  repository nor its reference checks contrast anywhere, so that figure is an author's claim
  and not a gate's.
- The expiry recovery transition is added here; the reference has no equivalent, because it
  was built for human workers who release their own tasks. `validate()` runs over every task
  inside every transaction and refuses on any error, so in the reference a single expired
  claim refuses every mutation -- including `release` and `heartbeat`, the only two
  transitions that could clear one. Observed on 2026-09-08, when a provider rate limit
  terminated three workers at once: recovery required editing task files outside the tool, and
  the hosted check, which runs `doctor`, stayed red on state an unrelated merge had never
  touched. This project adds `expire`, a coordinator transition that returns an abandoned task
  to `open`, records the clearing identity, the abandoned owner, the lapsed lease and a
  mandatory reason in the evidence log, and refuses a claim that has not lapsed. It is not
  owner authenticated, for the same reason `review` is not: the identity it would authenticate
  is precisely the one that has gone away. Two choices inside this are recorded rather than
  left to be rediscovered. An expired claim is deliberately **not** downgraded to a warning,
  which would have cleared the deadlock by destroying the lease; instead `mutate` tolerates a
  standing validation error for `expire` alone, and only whole messages that are exactly an
  expired claim and were already present before the write -- so every other transition stays
  fenced, `expire` is still refused by any other fault including one whose message merely
  contains the tolerated text, and a lease that lapses part-way through the transaction fails
  closed rather than being excused by the transaction that raced it. The one fault the
  tolerance cannot be narrowed against is a stale generated view, because no transition can be
  refused by one: `mutate` regenerates both views from the tasks immediately before it
  validates. And `expire` names no destination but `open`, because reopening the queue
  is the whole repair and every other status already has a transition that governs it.
- `apply_owned_change` refuses an empty `--owner` before comparing it to the recorded owner.
  The reference compares alone, which reads as an ownership check but is not one: an unheld
  task records the empty string, so an empty argument matched on exactly the tasks with no
  owner left to authorise anything. That was harmless while every unowned status was inert,
  and stops being harmless with `in_review`, whose safety argument is that submitting withdrew
  the worker's authority. The refusal sits at the shared gate rather than in any one
  transition, because the empty string is a sentinel for an unheld task and never a
  credential. `apply_update` additionally states the precondition on the current status that
  the other owned transitions already state, so its safety does not rest on the coupling
  between an owner and an active task that only `validate()` enforces.
- The transaction gate reads the published schemas instead of restating them. In the
  reference, and here until AR-0015, `validate()` enforced presence, references, the
  dependency graph, claims and privacy, and **no length or format bound at all**, while
  `schema/task-schema.json` enforced several. The permissive gate was the one guarding the
  write: `update --next-action` wrote whatever it was given, `validate()` passed, the
  transaction committed, and `tests/validate_schema.py` then rejected the result. Observed,
  not hypothesised: AR-0006 carried a `next_action` of 908 characters, `doctor` reported the
  tree consistent, and the schema gate exited 1. The hand-maintained `REQ` tuple named nine
  of the schema's sixteen required fields, so seven were unenforced at the write as well.
  `schema_gate_errors` now applies both published schemas to the documents they govern,
  inside the transaction, and the hand-written presence and unknown-field rules that
  duplicated them exactly were deleted rather than left to drift.

  Three choices inside this are recorded rather than left to be rediscovered. First, the
  enforcement reads the schema files but does **not** call `jsonschema`: `validate()` runs
  inside every transaction and every wrapped command triggers one, and `jsonschema` is a
  `quality` dependency group member, absent from the plain interpreter the `tools/handoffctl`
  entry point runs under. Importing it there would have made the coordinator unusable outside
  the quality environment to close a divergence a stdlib reader closes just as completely.
  Second, because a reader of a schema subset can only enforce the constructs it implements,
  `schema_support_errors` refuses the ones it does not — and what that refusal covers is stated
  here as a bound, not as an absolute. It is total over the **vocabulary and shape** of the
  published schemas. The reader walks four kinds of position: the document object, a
  `properties` entry, an `items` entry, and an `allOf` branch with its `if`/`then`/`else`. At
  each one it accepts a closed set of keyword names and reports every other name as an error;
  it refuses any value in a subschema position that is not a JSON object, because draft 2020-12
  permits a boolean there and this reader enforces neither form; it refuses any bound whose
  *value* is not the shape its checker reads, since a keyword whose bound cannot be read is
  unenforced or raises exactly as an unimplemented one would; and `schema_gate_errors` enforces
  a schema **only** when the net returned nothing, so the enforcement pass never walks a
  construct the net has not accepted. It is **not** total over semantics: that each implemented
  keyword agrees with draft 2020-12 on every value is a claim about the checkers rather than
  about coverage, and it rests on the differential fuzz below, with the one declared `1.0`
  exception. Two rounds of review each found a construct the earlier absolute claim did not
  cover, which is why the claim is now the property the code is built to make true and
  `tests/attack_schema_support.py` is the harness that tries to falsify it. Third, `doctor`
  is deliberately **not** given a call to `tests/validate_schema.py`. That would re-introduce
  the dependency, and it would leave two gates to agree; instead the constraint set is one set,
  and a test proves per constraint that `validate()` and a real `Draft202012Validator` refuse
  the same documents, and that no tree `doctor` accepts is one the schema gate rejects.

  The parity was checked by differential fuzzing as well as by the fixtures: 1.44 million
  generated documents were put to both `schema_object_errors` and a real
  `Draft202012Validator`, across both published schemas. No document is refused by the schema
  and accepted by the transaction. Two faults were found this way that the fixture list had
  missed, and both are now fixed and pinned. `const` and `enum` compared with Python
  equality, under which `true == 1`, so a `schema_version` of `true` passed a `const` of `1`;
  `_json_equal` now separates the two the way JSON does. And `pattern` must stay an
  unanchored `re.search`: every pattern the schemas state today is anchored with `^` except
  the milestone `label`'s `\S`, so a `match` or `fullmatch` here would pass every fixture
  while refusing labels the schema gate accepts.

  Exactly one difference remains, and it is deliberate. Draft 2020-12 counts a number with
  zero fractional part as an integer, so `1.0` satisfies `"type": "integer"`; the transaction
  refuses it for `task_revision` and `observed_dirty`. A revision counter that arrived as a
  float would be written back as `2.0` and fence every later transition on a float, so the
  stricter reading is the safe one. It cannot break the property that matters -- a tree
  `doctor` accepts is still one the schema gate accepts -- and a test pins it so it stays a
  decision rather than becoming a drift.

  Two limits are worth stating plainly. The equality proved is *the schema's constraints are
  enforced by `validate()`*, not the converse: `validate()` stays deliberately stricter, with
  cross-document rules — duplicate identifiers, a plan file that must exist, the dependency
  graph, milestone coverage, claim uniqueness, an expired lease, prose and privacy — that a
  per-document schema cannot state. And a document already broken can now draw a message from
  both the hand-written rule that names the field in this tool's vocabulary and the generic
  rule that names the bound in the schema's; the alternative was rewriting the expectations of
  a hundred existing tests, and the pair is complementary rather than contradictory.

  Two smaller repairs came with it, both found by the fixtures rather than by inspection.
  `value_errors` and `reference_errors` reached a regex and a path join with unconverted front
  matter, so a `checkpoint_commit`, `plan` or `id` of the wrong JSON type raised `TypeError`
  out of `validate()` instead of being reported; they go through `str` now. And
  `generated_view_errors` guarded the renderers on status alone, so a task missing `summary` or
  carrying a non-string `priority` crashed `render_current` from inside `validate()`; the guard
  is now `renderable`, which checks everything the two generated views index.

  Review of the first submission added four repairs and two recorded hazards. The size cap in
  `privacy_errors` now excludes `__pycache__`. CI runs the fault tests and `doctor` in the
  same job, so the suite writes `tests/__pycache__` and `doctor` then walks it; 442 added test
  lines took the compiled test module past the 200 KiB cap and failed the tree for a build
  artefact rather than for state. The related fact is worth stating because it is not fixed:
  `tests/test_handoffctl.py` is itself 160,456 bytes against the same cap at this head, and no
  exclusion would be appropriate for a source file. The fact stands and the number moves: the
  figure carried here through two rounds was the pre-fix 148,578 and was stale by the time it
  was read, so it is restated with the head it belongs to and re-measured whenever the file is
  touched. Headroom is 39,544 bytes. That one is a real future problem.

  `schema_support_errors` was not the total safety net it claimed. Three constructs now
  refuse that previously passed in silence. A **changed top-level `type`** is the dangerous
  one: `schema_object_errors` deliberately does not re-check the document type, because
  `read_task` already refuses front matter that is not a JSON object, but that reasoning holds
  only while the schema says `object`. A schema stating `array` refuses every document while
  the transaction accepted them all -- the original divergence restored by a one-word edit.
  The **schema-object form of `additionalProperties`** states a per-property bound this reader
  does not implement, and was ignored rather than refused. And a **`type` stated as a list**,
  which draft 2020-12 permits, is unhashable, so the membership test raised `TypeError` out of
  `validate()` and out of every transaction with it -- a crash where the design promises a
  loud refusal. The declared limit in the previous paragraph was under-bounded, and is now
  stated as what it is: the document type is unchecked *because it is pinned to `object` by
  the support net*, not because it is unchecked.

  The second review found the same two fault shapes again, through **boolean subschemas**,
  which draft 2020-12 permits wherever a subschema may appear. `"items": false` on `depends_on`
  was silently unenforced: a real validator refuses every task that declares a dependency while
  `doctor` exited 0 — the original divergence restored by a one-word edit, and this reader's own
  argument for why the top-level `type` check had to exist. A `properties` entry stated as a
  boolean raised `TypeError: 'bool' object is not iterable` out of `validate()` and out of every
  transaction with it, `release` and `expire` included. Both are fixed, and the fix is not a
  patch of the two counterexamples: every subschema position must now hold a JSON object, every
  bound must have the shape its checker reads, `properties`, `required` and `allOf` must hold
  the container the reader walks, and an unsupported schema is no longer enforced at all — which
  is what closes the crash surface rather than moving it from the net to the enforcer.

  What is still **not** proved, said plainly. That the net is total over vocabulary and shape is
  argued from the code and attacked by a harness; it is not a proof, and a third construct of a
  shape neither of us thought of would falsify it the way the first two did. Nothing here says
  the published schemas will only ever need constructs this reader implements: refusing `oneOf`
  is a different and weaker thing than implementing it, so the day a schema legitimately needs
  one, the tree goes red until someone writes the checker. That is an obligation recorded here
  and enforced nowhere but by that red tree. And the equality between the two gates is checked
  on the constructs the schemas use today; a keyword added tomorrow is caught by the net, not by
  a test that knows what it means.

  `tests/attack_schema_support.py` is committed for the same reason the fuzz harness is: the
  claim above is a completeness claim, and a fixture list cannot supply evidence for one. It
  mutates each published schema at every position the reader walks — 1,676 mutants — with 23
  unimplemented draft 2020-12 keywords, six non-object values in every subschema position, 19
  bounds whose value the checker cannot read, and wrong-shaped `properties`, `required` and
  `allOf` containers, and it fails a mutant unless the transaction either refuses it or agrees
  with a real `Draft202012Validator` about every document in the repository. At this head: 1,676
  mutants, 0 faults. Against the previous head it reports 451 — 308 crashes and 143 silently
  unenforced — including both constructs the review named, so the harness is not vacuous.

  The `minLength` boundary is now pinned as well as `maxLength`. An exclusive comparison
  passed the whole suite while refusing a one-character `title`, `summary` or milestone
  `label`, a 20-character `claim_expires` and a one-character `submitted_by`, all of which the
  schema accepts. Its direction is the transaction refusing what the schema allows, the same
  direction as the unanchored-pattern fault, which is the direction the per-constraint
  fixtures are structurally blind to: a fixture asserts that an invalid document is refused
  and says nothing about a valid one at the boundary. The milestone `label` was pinned only at
  the helper level while the task fields were pinned through `validate()` against a real
  validator; both boundaries now go through `validate()`, so the two documents are checked the
  same way rather than one of them being checked one layer short.

  `tests/fuzz_schema_parity.py` is committed rather than left in a scratch directory, because
  it is the only evidence for the completeness claim and a fixture list cannot supply that.
  It runs two generators: one perturbs a valid document, and one builds documents from random
  key subsets and never starts from a valid one, which is what exercises `required` and
  `additionalProperties` rather than per-field bounds. It is not in the gate sequence -- it is
  slow and randomised -- and it exits non-zero only on a parity hole, never on the deliberate
  over-strictness. Neutering `required` and `maxLength` makes it report 7,955 holes in 80,000
  documents, so it is not vacuous. It also asserts a minimum document count, because an exit
  code alone conflated "no holes" with "nothing generated": `fuzz_schema_parity.py 0` used to
  print `documents fuzzed: 0` and exit 0, which is the same fault class as a bound that is
  silently unenforced — a check that passes by doing nothing.

  A latent hazard is recorded rather than fixed. The deliberate `1.0` strictness is safe
  where it is applied to a document, but the same predicate is used inside an `if` condition,
  where being stricter *flips a branch* rather than adding an error: with
  `if: {task_revision: {type: integer}}` and a document carrying `1.0`, this reader takes
  `else` where the validator takes `then`, and the two gates could then disagree in either
  direction. It is unreachable in the shipped schemas, no `if` in them tests an integer type,
  and `schema_support_errors` does not catch it because every construct involved is supported.

  A second observation, found while reproducing the size blocker and **not** fixed: the
  exclusions in `privacy_errors` test `path.parts` of an absolute path, so a checkout whose
  own directory is named `.git`, `workspaces` or any other excluded component skips every
  file it holds. Running `doctor` inside `workspaces/AR-0015` therefore reports a clean
  privacy walk over an empty set. That is why the blocker did not reproduce in the worktree
  and had to be reproduced at a path outside it. It weakens `doctor` wherever the coordinator
  is checked out under such a path, and it belongs to the privacy gate rather than to this
  change.

  One finding is recorded here and deliberately **not** fixed by this change, because it
  belongs to the hosted check rather than to the coordinator. Every push that carried the
  908-character field was a `chore(state):` commit, which the cost condition added by AR-0013
  skips, so no run ever evaluated it. The skip's blind spot is aligned exactly with the defect
  class state commits produce, and the failure sat latent waiting for an unrelated content push
  to sweep it up. Closing the write gate removes the source; the detection gap remains.

- `read_task` rejects front matter that parses as JSON but is not an object. The reference
  casts the parsed value to a mapping unchecked, so `---\nnull\n---` reached the validators
  as `None` and surfaced as an `AttributeError` rather than a validation error.
- The privacy scanner drops the reference's private-workstation alias, which named a host
  that has no bearing here. No account-name rule replaces it: this project's GitHub handle
  appears legitimately in repository URLs, and the leak vector that matters — the local
  account inside an absolute path — is already covered by the inherited home-path rule.
- Two rules are added for agent session references, one for the URL form and one for a bare
  session identifier, because the inherited UUID rule does not match either shape.
- `privacy_errors()` inspects the working tree only, so `commit_privacy_errors()` and the
  `check-commits` subcommand apply the same rules to the messages of newly introduced
  commits.
- Commits are GPG signed rather than SSH signed, matching this project's existing key.

The `AR-NNNN` identifier format is retained deliberately, for compatibility with the reused
coordinator, its schema and its tests. All new product code is Rust.

## Moved in-tree

Until the project was published, this coordinator ran from a repository of its own, beside the
product. It now lives under `coordination/` in the product repository; neither repository's
history was carried over, and the task set restarted from empty, as recorded in
[HISTORY.md](HISTORY.md). The move made three changes to the tool, each forced by sharing a
repository with the product:

- `require_state_branch()` refuses to commit or replicate state from a linked worktree or from
  any branch other than `main`. A worker's worktree now holds its own copy of the tool, and a
  transaction run from it would take a private lock and commit task state to a feature branch.
- `commit()` commits only the paths its transaction wrote (`git commit -- <paths>`), because the
  canonical checkout's index may hold staged product changes.
- `product_checkout()` defaults to the repository holding `coordination/`, so the runtime
  configuration no longer needs an absolute project root.

The coordination workflow moved with it, to `.github/workflows/coordination.yml`, and became
dormant like every other workflow here, because product gate 11 forbids live triggers.

Source is MIT licensed; see [`../LICENSE`](../LICENSE).
