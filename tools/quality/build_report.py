#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Assemble the gate runner's structured report.

Called by `run-gates.sh` with everything it collected in the environment. Kept
separate from the shell so that the report's shape is one readable thing rather
than a series of `printf` calls, and so that the digest is computed over a
canonical serialisation.

Two digests are published, and the difference matters:

`stable_digest`
    over the report with every field removed that describes the *occasion* of
    the run rather than its *subject*: the generation time, each gate's
    duration, the invoking checkout's commit, and the literal ref string typed
    after `--base`. Two runs of the same revision on the same host must produce
    the same value, which is what makes "run it again and check" a meaningful
    instruction. It is computed over a canonical serialisation (sorted keys, no
    insignificant whitespace) so that key order cannot change it.

    `revision.base_ref` is excluded for a reason worth stating. It is whatever
    the caller typed -- `origin/main`, a branch name, a raw sha -- and three
    spellings of one commit gave three different digests, so the digest changed
    when nothing that was verified had changed. `revision.base_commit` pins the
    thing that matters and stays inside the digest; `base_ref` stays in the
    report because it tells a human what was meant.

the file's own sha256
    printed by the runner, not stored here; it changes with every run, because
    the timings do.

The report carries no absolute path, no raw command output and no credential:
the runner scans it with `privacy.py` before accepting it, and commands appear
as the sha256 of their argv with the run's throwaway paths folded out.

Exit codes: 0 the report was written and every gate record is supported by the
execution ledger; 3 the report was written and at least one is not; anything
else, the report could not be built. 3 rather than 1, so that an unhandled
exception -- which Python exits 1 for -- can never be read as the specific
finding below.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

NOT_A_PASS = {
    "fail": "failed",
    "not_armed": "is not armed",
    "not_applicable": "had nothing to run against",
    "not_implemented": "is not implemented yet",
    "unsupported": "recorded a status the execution ledger does not support",
}

# Which statuses require a ledger entry, which require none, and which require
# neither. The rule is over entries, because that is all this file can see: the
# ledger's writer is a top-level function in `run-gates.sh` that the dispatch can
# call in one line, so an entry is a claim that a command ran and not the fact.
#
# This table said `at least one command ran`, character for character the row in
# `docs/QUALITY_GATES.md` that was rewritten to speak of entries -- and it
# survived three rounds of correcting that document while sitting twenty lines
# above the paragraph explaining why the document was wrong. The document's table
# was corrected and the code's identical table was left standing, which is the
# whole failure this task keeps reproducing.
#
#   pass            at least one entry for the gate, and every entry at 0.
#   not_armed       an entry returning the code that means "unarmed"; gate 7's
#                   floor check exits 3. Nothing else produces it.
#   fail            unconstrained here. Refusing *before* running anything is a
#                   legitimate failure -- gate 9 refuses when its isolated scan
#                   repository did not come out isolated, having logged nothing --
#                   and a fail is not a pass, so there is nothing to protect.
#                   What is still required is a non-zero exit code: a failure
#                   that exited 0 is a contradiction whichever way it arose.
#   not_applicable  no entry, because the gate had no input.
#   not_implemented no entry, because the checker is absent from the revision.
NEEDS_AN_ENTRY = ("pass", "not_armed")
NEEDS_NO_ENTRY = ("not_applicable", "not_implemented")

GATE_NAMES = {
    1: "formatting",
    2: "lints",
    3: "tests",
    4: "documentation build",
    5: "dependency policy",
    6: "advisories",
    7: "coverage floors",
    8: "workflow linting",
    9: "secret scanning",
    10: "external analyzer installation",
    11: "repository policy",
    12: "commit signature, sign-off and message privacy",
    13: "negative-fixture suite",
    14: "entry-point crate-table consistency",
    15: "architecture-index completeness",
}


def environment(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def coverage_section(path: str, manifest: dict[str, object]) -> dict[str, object]:
    floors = dict(manifest.get("coverage", {}) or {})
    workspace_floor = floors.get("workspace_lines")
    critical_floor = floors.get("critical_lines")
    section: dict[str, object] = {
        "measured": False,
        "armed": workspace_floor is not None and critical_floor is not None,
        "floors": {
            "workspace_lines": workspace_floor,
            "critical_lines": critical_floor,
            "critical_packages": floors.get("critical_packages") or [],
        },
        "branch_coverage_available": bool(floors.get("branch_coverage_available")),
        "note": (
            "Line and region coverage only: cargo-llvm-cov on the pinned stable toolchain "
            "does not report branch coverage, so no branch figure is claimed. Coverage "
            "establishes exercised lines, not correctness."
        ),
    }
    if not section["armed"]:
        section["armed_note"] = (
            "The manifest's coverage floors are null, so nothing was compared. AR-0009 sets "
            "them from measurement. A null floor is reported as not armed and is not a pass."
        )
    if not path or not Path(path).is_file():
        return section
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        totals = (report.get("data") or [{}])[0].get("totals", {})
    except (OSError, json.JSONDecodeError, IndexError, AttributeError):
        return section

    def block(kind: str) -> dict[str, float] | None:
        raw = totals.get(kind)
        if not isinstance(raw, dict):
            return None
        count = float(raw.get("count", 0) or 0)
        covered = float(raw.get("covered", 0) or 0)
        return {
            "count": count,
            "covered": covered,
            "percent": round(100.0 * covered / count, 4) if count else 0.0,
        }

    section["measured"] = True
    section["workspace_lines"] = block("lines")
    section["workspace_regions"] = block("regions")
    section["workspace_functions"] = block("functions")
    return section


def kernel_series(release: str) -> str:
    """`7.1.9-arch1-2` -> `7.1`.

    The exact kernel release names the distribution and the build, which narrows
    the host much further than the answer needs: what can change a gate's result
    is the kernel series, not its patch level or its packaging. The full string
    is still inside `host.fingerprint`, so two runs on the same host can still be
    recognised as such.
    """
    parts = release.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else release


# Every revision-controlled input that can change a gate's verdict, not only the
# checkers. A gate is decided as much by the policy it reads as by the script
# that reads it, and several of these can redirect a gate without touching a
# checker at all:
#
#   deny.toml            gate 5's entire policy
#   .cargo/audit.toml    gate 6's ignore list
#   .rustfmt.toml        gate 1's definition of formatted -- and `rustfmt.toml`
#                        without the dot, which rustfmt reads identically.
#                        `tab_spaces = 8` in either spelling produces the same
#                        diff, so comparing only the dotted one compares half a
#                        policy.
#   clippy.toml          gate 2's configuration, in both spellings, read by
#                        clippy itself.
#   Cargo.toml           `[workspace.lints]` sets gate 2's severities, and every
#                        *member* manifest can override them: a `[lints.clippy]`
#                        table in one crate took `cargo clippy -- -D warnings`
#                        from 101 to 0, so the member manifests are compared
#                        too and the earlier claim that the workspace manifest
#                        covered this was wrong. Each is hashed whole rather
#                        than by table, because a digest of a hand-parsed
#                        fragment is a second thing to get wrong; the price is
#                        that an unrelated manifest edit also shows up as a
#                        difference, named.
#   .cargo/config.toml   the sharpest of them. An `[alias]` entry shadows the
#                        external subcommand of the same name -- a revision
#                        supplying `clippy = ["check", "--quiet"]` makes gate 2
#                        run `cargo check` -- and `[build] rustc-wrapper` or
#                        `[build] rustc` redirects the compiler itself for gates
#                        1 to 4 and 7.
#   .github/actionlint.yaml  gate 8's configuration
#
# Cargo.lock and the crate sources are deliberately absent: they are the subject
# of the gates, not the instruments.
COMPARED_PATHS = (
    "tools/quality",
    "config/quality-tools.json",
    "config/allowed-keys.asc",
    "Cargo.toml",
    "*/Cargo.toml",
    ".rustfmt.toml",
    "rustfmt.toml",
    "clippy.toml",
    ".clippy.toml",
    "rust-toolchain.toml",
    "rust-toolchain",
    "deny.toml",
    ".cargo/audit.toml",
    ".cargo/config.toml",
    ".cargo/config",
    ".github/actionlint.yaml",
    ".github/actionlint.yml",
)


def git_lines(repo: str, *args: str) -> list[str]:
    """NUL-separated `git` output as a list, or an empty list if git failed."""
    try:
        out = subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [entry for entry in out.split("\0") if entry]


def content_manifest(repo: str) -> dict[str, str | None]:
    """sha256 of the *worktree bytes* of every compared path tracked in `repo`.

    The bytes on disk, not the blob ids in the index: `git ls-files -s` reports
    what was staged, so a file modified after checkout hashes to its old value
    there. The revision's own build scripts run during gates 2, 3, 4 and 7,
    before the report is built, and they run as the invoking user -- so the
    index is exactly the wrong thing to trust here.
    """
    manifest: dict[str, str | None] = {}
    for name in git_lines(repo, "ls-files", "-z", "--", *COMPARED_PATHS):
        try:
            data = (Path(repo) / name).read_bytes()
        except OSError:
            manifest[name] = None  # tracked but not readable on disk
            continue
        manifest[name] = "sha256:" + hashlib.sha256(data).hexdigest()
    return manifest


def uncommitted(repo: str) -> list[str]:
    """Compared paths that differ from what the revision committed.

    Reconciles the worktree against the index and the index against HEAD, and
    reports additions too: an untracked `.cargo/config.toml` steers cargo just
    as well as a tracked one.
    """
    entries = git_lines(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all",
                        "--", *COMPARED_PATHS)
    names: list[str] = []
    skip = False
    for entry in entries:
        if skip:  # the second half of a rename record
            skip = False
            continue
        if len(entry) < 4:
            continue
        code, name = entry[:2], entry[3:]
        if code[0] in "RC":
            skip = True
        names.append(name)
    return sorted(names)


def digest_of(manifest: dict[str, str | None]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def provenance() -> tuple[dict[str, object], list[str]]:
    """Whose checkers ran, on what inputs, and whether that is self-certifying.

    Returns the section and every blocking reason the boundary itself produces.
    """
    root = environment("GATE_ROOT")
    worktree = environment("GATE_WORKTREE")
    invoking_files = content_manifest(root)
    revision_files = content_manifest(worktree)
    invoking = digest_of(invoking_files)
    revision = digest_of(revision_files)
    matches = invoking == revision
    differing = sorted(
        name
        for name in set(invoking_files) | set(revision_files)
        if invoking_files.get(name) != revision_files.get(name)
    )
    dirty = {"invoking_checkout": uncommitted(root), "revision_worktree": uncommitted(worktree)}

    section: dict[str, object] = {
        "invoking_checkout_commit": environment("GATE_INVOKING_COMMIT"),
        "checkers_executed_from": "the revision under test",
        "report_built_from": "the invoking checkout",
        "compared_paths": list(COMPARED_PATHS),
        "compared_paths_note": (
            "Not only the checkers. A gate is decided as much by the policy it reads as "
            "by the script that reads it, so the comparison covers every "
            "revision-controlled input that can change a verdict: the dependency policy, "
            "the advisory ignore list, the formatting and toolchain files, "
"`[lints]` in the workspace manifest and in every member manifest, "
            "clippy's own configuration, the workflow-analyzer configuration, and "
            "cargo's own configuration -- which can shadow a gate's subcommand with an "
            "`[alias]` or redirect the compiler with `[build] rustc-wrapper`. Where a "
            "tool reads two spellings of one file, both are compared: `.rustfmt.toml` "
            "and `rustfmt.toml`, `clippy.toml` and `.clippy.toml`, `.cargo/config.toml` "
            "and `.cargo/config`, `.github/actionlint.yaml` and `.yml`. Cargo.lock and "
            "the crate sources are not compared: they are what the gates judge, not what "
            "judges them. Each gate also lists the subset that decides it, as `inputs`."
        ),
        "compared_by": (
            "the sha256 of each file's bytes in the worktree, not its blob id in the "
            "index, so that a file modified after checkout -- by a build script of the "
            "revision, for instance -- cannot hash to the value it was checked out with."
        ),
        "quality_tooling_invoking_checkout": invoking,
        "quality_tooling_revision": revision,
        "quality_tooling_matches": matches,
        "differing_paths": differing,
        "invoking_checkout_files": invoking_files,
        "invoking_files_note": (
            "Per file, so that a reader who sees the two digests disagree can tell which "
            "input disagrees without re-deriving it. The revision's side is not repeated "
            "wholesale: each gate's `scripts` entry already carries the sha256 of every "
            "checker it executed, and `differing_paths` names the rest."
        ),
        "uncommitted_compared_paths": dirty,
        "note": (
            "Gates 7 and 10 to 15 execute a checker out of the revision under test; this "
            "report and its privacy scan are built by the invoking checkout. That is "
            "deliberate: it is the only arrangement in which the report is a function of "
            "the revision alone, so that two reviewers with differently versioned tooling "
            "cannot get different answers for the same commit, and so that a change to a "
            "checker can be exercised by the run that reviews it. The cost is that a "
            "revision is graded by its own instruments. That cost is paid in the open "
            "here: every gate lists the sha256 of each checker file it executed, and the "
            "digests above let a reader see at a glance whether those files, and the "
            "policies beside them, are the ones in the checkout that invoked the run."
        ),
        "prescribed_workflow_limit": (
            "Said plainly, because it is the weakest point of this section: the workflow "
            "this runner prescribes -- invoke it from a checkout of the head under review "
            "-- makes `quality_tooling_matches` true for an attacking revision as easily "
            "as for an honest one, because both sides are then the same tree. The flag "
            "catches the reviewer who runs from somewhere else; it cannot catch the "
            "prescribed case, and it is not evidence that the instruments are honest. In "
            "that case the whole obligation is on the reader: read the diff of these "
            "compared paths before quoting any `pass` in this report."
        ),
        "execution_limit": (
            "The revision under test executes arbitrary code as the invoking user during "
            "this run: crates/newton/build.rs and every other build script run during "
            "gates 2, 3, 4 and 7, and the revision's own installer runs as gate 10, "
            "before this report is assembled. The run's temporary directory holds the "
            "gate records the report is built from, and the invoking checkout is "
            "writable by that code. This was demonstrated, not assumed -- a build script "
            "given only CARGO_TARGET_DIR located the record file and rewrote a recorded "
            "gate from not_applicable to pass. It is inherent to compiling untrusted code "
            "on a local machine and this runner does not fix it. The run directory is "
            "mode 700, which `mktemp -d` already gives it whatever the umask, so that is "
            "a stated property and not a change: it keeps out another user on the box "
            "and nobody else. The radius is wider than the record file. That same "
            "directory holds the analyzer binaries gate 10 installs by running the "
            "revision's own installer, and gate 9's isolated scan repository, and gate 10 "
            "runs first of all -- so gate 9's isolation, and the analyzer versions this "
            "report quotes, sit inside the blast radius described here rather than "
            "outside it. Every guarantee in this report is bounded by that."
        ),
        "limit": (
            "The runner cannot vouch for itself. `run-gates.sh` is executed from the "
            "invoking checkout, so a reader must obtain the runner from a source they "
            "trust; nothing inside a report can establish that."
        ),
    }

    blocking: list[str] = []
    if dirty["invoking_checkout"] or dirty["revision_worktree"]:
        section["inputs_intact"] = False
        blocking.append(
            "provenance: a compared input differs from what its commit records, so the "
            "digests below describe bytes that were not committed anywhere. See "
            "runner.provenance.uncommitted_compared_paths. The working tree was clean "
            "when the run started, so anything listed there was written while the gates "
            "ran."
        )
    if not matches:
        section["self_certifying"] = False
        blocking.append(
            "provenance: the gate-deciding inputs in this revision differ from those in "
            "the checkout that invoked the run, and the gate checkers ran from the "
            "revision. No `pass` above is self-certifying until that diff has been read; "
            "the paths are runner.provenance.differing_paths ("
            + (", ".join(differing) if differing else "none")
            + ")."
        )
    return section, blocking


def second_opinion(path: str) -> dict[str, object]:
    """The base commit's checkers, re-run against this revision, as advice only.

    Cheap -- these four checkers are offline and finish in tenths of a second --
    and it answers a question the provenance digests can only pose. When a
    revision changes a checker, the digests say the instruments moved; this says
    whether the verdict moved with them. Advisory, never blocking: a checker
    that legitimately grows stricter, or that legitimately loses a false
    positive, disagrees with its predecessor for good reasons, and only a reader
    can tell which happened. Silence here is a fact about the run, not a pass.
    """
    entries: list[dict[str, object]] = []
    if path and Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                entry["name"] = GATE_NAMES.get(int(entry["id"]), "unknown")
                entries.append(entry)
    entries.sort(key=lambda entry: int(entry["id"]))
    disagree = [entry for entry in entries if entry.get("agrees") is False]
    # A comparison against a byte-identical file is not a comparison. Saying how
    # many of these actually exercised anything keeps the section from reading
    # as more assurance than it is.
    exercised = [
        entry["id"]
        for entry in entries
        if entry.get("agrees") is not None and not entry.get("identical_to_revision")
    ]
    return {
        "gates": entries,
        "disagreements": [
            f"gate {entry['id']} ({entry['name']}): the revision's checker says "
            f"{entry['revision_status']}, the base commit's says {entry['base_status']}"
            for entry in disagree
        ],
        "not_compared": [
            f"gate {entry['id']} ({entry['name']}): {entry['reason']}"
            for entry in entries
            if entry.get("agrees") is None and entry.get("reason")
        ],
        "checkers_compared": len(exercised),
        "checkers_that_differ_from_the_revision": exercised,
        "advisory": True,
        "note": (
            "The base commit's copy of each offline checker, run against this revision "
            "from the same worktree. It is a second opinion on the instruments, not a "
            "gate: a disagreement is reported and never blocks, because a checker can "
            "differ from its predecessor by getting stricter, by shedding a false "
            "positive, or by being weakened, and the report cannot tell those apart. "
            "Only exit 0 and exit 1 are verdicts. `unavailable` means the base commit "
            "has no such checker, which is the honest answer for a branch that "
            "introduces one, and `error` means this harness could not drive the base's "
            "checker to a verdict -- an option it does not accept, or its own exit 2 for "
            "an environment error. Neither is counted as a disagreement, because an "
            "advisory signal that cries wolf is one nobody reads. "
            "`checkers_that_differ_from_the_revision` is the honest measure of what this "
            "section examined: an entry whose base checker is byte-identical to the "
            "revision's cannot disagree with it and establishes nothing."
        ),
    }


def execution_ledger(path: str) -> tuple[dict[int, list[dict[str, object]]], str]:
    """What the runner's execution ledger records, by gate, and why it may be empty.

    One line per command: on the honest path `run_step` writes it after the
    command returned, carrying the exit status *it* returned. Read it as the
    runner's claim about what ran rather than as proof that anything did -- the
    writer is a top-level shell function in `run-gates.sh` and the dispatch can
    call it directly, which review demonstrated in one line per gate. This
    function is the only thing that puts a *number* on that claim, so it is read
    fail-closed: a ledger that is missing, unreadable or malformed yields no
    entries and a reason -- never a shrug -- and every gate then claiming an entry
    it does not have is reported as unsupported below. Deleting the ledger is
    therefore not a way round this check; it is the loudest way into it. Forging
    an entry is the way round it, and nothing here closes that.
    """
    if not path:
        return {}, "the runner passed no execution ledger, so nothing recorded what ran"
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        return {}, f"the execution ledger could not be read ({error.strerror})"
    by_gate: dict[int, list[dict[str, object]]] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            gate_id = int(entry["gate"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return {}, "the execution ledger is not readable as one JSON object per line"
        by_gate.setdefault(gate_id, []).append(entry)
    return by_gate, ""


def unsupported_records(
    gates: list[dict[str, object]], by_gate: dict[int, list[dict[str, object]]]
) -> dict[int, str]:
    """Gate records the execution ledger does not support, and how.

    This is the check AR-0017 exists for, and it is worth being exact about
    what it does and does not establish.

    What it establishes. `run-gates.sh`'s dispatch decides each gate's status
    in an arm of a `case`, and an arm short-circuited to `gate_status="pass"`
    used to be indistinguishable from an honest pass: review demonstrated it on
    gates 2, 3, 4 and 7, leaving the command templates and `--list-gates`
    untouched, publishing a plausible `argv_sha256` for argv nobody invoked,
    and producing a 15/15 green report in which gate 7 -- honestly `not_armed`
    and blocking -- had been *upgraded* to `pass` by being disabled. The
    dispatch can no longer publish a digest or a duration *through `record`*:
    both are derived from the ledger there rather than passed to it. Scope that
    exactly -- an earlier version of this sentence said `at all`, and that is
    false. `record_execution` is a top-level function in `run-gates.sh` and an
    arm can call it in one line with a digest and a duration of its choosing;
    the paragraph below says what that means. What reaches this function either
    way is a claimed status, and this is where that claim meets the ledger. A
    claim the ledger does not support does not become a pass; it becomes
    `unsupported`, which is not a pass, is blocking, and carries the claimed
    status beside it so the contradiction is legible rather than silently
    corrected.

    What it does not establish, stated here rather than left for a reader to
    discover. This is a check inside the runner on evidence the runner wrote,
    and the writer of that evidence is a shell function in the same file as the
    dispatch. `record_execution` is top level and reachable from any arm, so
    **disabling a gate is still a one-line change in one arm of one `case`**,
    in one file, with this file untouched:

        2)  record_execution 2 "gate 2 (lints)" "$(argv_digest <the honest argv>)" 0 54.4
            gate_status="pass" ;;

    Review demonstrated exactly that over gates 2, 3, 4 and 12 and got a green
    board: every gate passing, no blocking reason, every negative fixture and
    every runner assertion below still green, and the forged records identical
    in every published field to honest ones. An earlier version of this
    paragraph claimed the edit now spanned two files and cost more than one
    line. That was wrong, it was never measured, and it pointed the next
    reviewer at the wrong file.

    So state the change without inflating it. What a short-circuited arm can no
    longer do is record a pass while saying nothing about what ran: the digest,
    the duration and the step table are not arguments to `record` any more, so a
    forgery has to *name* a command, a digest and an exit status in the
    executor's own format. That is more for a reviewer to read, and it is not
    more for an attacker to write. The trust is unchanged: a runner is executed
    by the person invoking it, and no in-band mechanism can vouch for a runner
    the reviewer chose to trust. Reading the diff of the quality tooling is
    still the reviewer's obligation and is not discharged by anything here --
    and `run-gates.sh` is where that reading has to happen, because one line
    there is still enough.
    """
    findings: dict[int, str] = {}
    for gate in gates:
        gate_id = int(gate["id"])
        status = str(gate["status"])
        steps = by_gate.get(gate_id, [])
        exits = [int(step.get("exit_code", -1)) for step in steps]
        try:
            code = int(gate["exit_code"])
        except (TypeError, ValueError):
            findings[gate_id] = "the record carries no readable exit code"
            continue

        if status not in NOT_A_PASS and status != "pass":
            findings[gate_id] = (
                f"the record names the status {status!r}, which this report has no "
                "meaning for; a status nobody defined cannot be read as a pass"
            )
        elif status in NEEDS_AN_ENTRY and not steps:
            findings[gate_id] = (
                f"the record claims {status!r}, but the ledger holds no entry for this "
                "gate, so there is nothing behind the status"
            )
        elif status == "pass" and (code != 0 or any(exit_code != 0 for exit_code in exits)):
            findings[gate_id] = (
                f"the record claims 'pass' with exit {code} over ledger entries that "
                f"returned {', '.join(str(exit_code) for exit_code in exits)}; a pass needs "
                "every entry for the gate to be 0"
            )
        elif status == "not_armed" and (code == 0 or code not in exits):
            findings[gate_id] = (
                f"the record claims 'not_armed' with exit {code}, which no ledger entry "
                f"for this gate carries (they carry {', '.join(str(e) for e in exits)}); "
                "an unarmed gate is an entry saying so, not a status chosen for it"
            )
        elif status == "fail" and code == 0:
            findings[gate_id] = "the record claims 'fail' with exit 0, which is a contradiction"
        elif status in NEEDS_NO_ENTRY and steps:
            findings[gate_id] = (
                f"the record claims {status!r}, which requires no ledger entry, while the "
                f"ledger holds {len(steps)} for this gate"
            )
        elif status in NEEDS_NO_ENTRY and code != 0:
            findings[gate_id] = (
                f"the record claims {status!r} with exit {code}; the ledger holds no entry "
                "for this gate, so there is nothing that could have returned it"
            )
        else:
            # The published digest, step table and duration are derived from the
            # ledger by `record`, so on an honest run they agree by construction.
            # Checking anyway is what makes this function meaningful against a
            # records file that was written by something other than `record`.
            expected = "+".join(str(step.get("argv_sha256", "")) for step in steps)
            if str(gate.get("argv_sha256", "")) != expected:
                findings[gate_id] = (
                    "the record's argv digest is not the digest of the commands the "
                    "executor logged for this gate"
                )
            elif int(gate.get("commands_executed", -1)) != len(steps):
                findings[gate_id] = (
                    f"the record says {gate.get('commands_executed')} command(s) where the "
                    f"ledger holds {len(steps)} entr(y/ies)"
                )
    return findings


def alias_shadowing() -> tuple[list[dict[str, str]], list[str]]:
    """Cargo's own report that a revision-supplied alias replaced a gate's tool.

    The provenance comparison can tell a reader that `.cargo/config.toml`
    differs. Cargo says something better and says it during the run -- "user-
    defined alias `clippy` is shadowing an external subcommand" -- and that line
    was being discarded with the rest of the raw output. It is kept only when it
    names a subcommand a gate actually invokes, so an unrelated alias is not a
    finding, and it blocks: a gate that ran something other than the tool this
    report names did not run.
    """
    found: list[dict[str, str]] = []
    for line in environment("GATE_SHADOWING").splitlines():
        if "\x1f" not in line:
            continue
        step, _, alias = line.partition("\x1f")
        found.append({"step": step, "alias": alias})
    reasons = [
        f"a revision-supplied cargo alias shadows the external `{entry['alias']}` "
        f"subcommand, which {entry['step']} invokes: cargo said so while the gate ran, "
        f"so that gate did not execute the tool this report names for it"
        for entry in found
    ]
    return found, reasons


def ignored_tests(path: str) -> list[str]:
    """Every #[ignore]d test gate 3 listed, or an empty list if none were."""
    if not path or not Path(path).is_file():
        return []
    return sorted(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def main() -> int:
    records_path = environment("GATE_RECORDS")
    manifest = json.loads(Path(environment("GATE_MANIFEST")).read_text(encoding="utf-8"))

    gates: list[dict[str, object]] = []
    with open(records_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                record["name"] = GATE_NAMES.get(int(record["id"]), "unknown")
                if not record.get("note"):
                    record.pop("note", None)
                # How the gate was configured is not why it did or did not pass,
                # and never reaches the blocking list.
                if not record.get("configuration"):
                    record.pop("configuration", None)
                # A gate that executed no command publishes no argv digest. The
                # digest of a command nobody ran reads as evidence that one did.
                if not record.get("argv_sha256"):
                    record.pop("argv_sha256", None)
                    record["command_note"] = "no command was executed for this gate"
                gates.append(record)
    gates.sort(key=lambda gate: int(gate["id"]))

    # What ran, according to the thing that ran it. Read before any status in
    # the records above is believed.
    by_gate, ledger_problem = execution_ledger(environment("GATE_EXECUTED"))
    unsupported = unsupported_records(gates, by_gate)
    execution_blocking: list[str] = []
    if ledger_problem:
        execution_blocking.append(
            f"execution: {ledger_problem}. Every gate below claiming a ledger entry it "
            "cannot show is therefore unsupported, which is the fail-closed reading and "
            "not a reason to accept the claims."
        )
    for gate in gates:
        gate_id = int(gate["id"])
        finding = unsupported.get(gate_id)
        if finding is None:
            # Says what the ledger holds, not what happened. The previous
            # wording -- "N command(s) executed, each recorded by the executor
            # with the exit status it returned" -- asserted the execution as
            # fact, which this report cannot establish and which
            # `stable_digest_note` in the same file says it cannot: an arm of
            # the runner's dispatch can write a ledger entry for a command it
            # never ran. Two sentences in one artefact disagreeing is worse than
            # either being wrong alone, because a reader has no way to tell
            # which one the report meant.
            count = int(gate.get("commands_executed", 0))
            gate["execution"] = (
                f"the execution ledger records {count} command(s) for this gate, with the "
                "exit status each returned, and the status above is consistent with them. "
                "The ledger is written by the runner; that it describes commands which ran "
                "is not established here. See stable_digest_note."
            )
            continue
        # The claim is kept beside the contradiction rather than replaced by it.
        # A reader needs to see what the runner said as well as why the report
        # will not repeat it.
        gate["status_recorded_by_the_runner"] = gate["status"]
        gate["status"] = "unsupported"
        gate["execution"] = finding
        # Whatever digest the record carried, the report publishes the ledger's.
        # A digest is this report's only evidence of what ran, so repeating an
        # unfounded one beside the finding that it is unfounded would put the
        # forgery back into the artefact the finding exists to clean.
        derived = "+".join(str(step.get("argv_sha256", "")) for step in by_gate.get(gate_id, []))
        gate.pop("command_note", None)
        if derived:
            gate["argv_sha256"] = derived
        else:
            gate.pop("argv_sha256", None)
            gate["command_note"] = "no command was executed for this gate"
        execution_blocking.append(
            f"gate {gate_id} ({gate['name']}) {NOT_A_PASS['unsupported']}: {finding}"
        )

    # Said out loud as well as written into the report. Every other finding here
    # reaches a reader through the runner's summary; this one says a gate may
    # not have run, which is the finding least safe to leave only in a file
    # somebody has to open.
    for reason in execution_blocking:
        print(f"FAIL {reason}", file=sys.stderr)

    expected = [int(value) for value in environment("GATE_ORDER_LIST").split()]
    missing = sorted(set(expected) - {int(gate["id"]) for gate in gates})

    counts: dict[str, int] = {}
    for gate in gates:
        counts[str(gate["status"])] = counts.get(str(gate["status"]), 0) + 1

    blocking = [
        f"gate {gate['id']} ({gate['name']}) {NOT_A_PASS.get(str(gate['status']), 'did not pass')}"
        + (f": {gate['note']}" if gate.get("note") else "")
        for gate in gates
        # `unsupported` is not a pass either, but it is named once already, in
        # `execution_blocking`, with the specific contradiction attached. Two
        # entries for one gate would read as two findings.
        if gate["status"] not in ("pass", "unsupported")
    ]
    for gate_id in missing:
        blocking.append(f"gate {gate_id} ({GATE_NAMES.get(gate_id, 'unknown')}) produced no record")

    provenance_section, provenance_blocking = provenance()
    shadowing, shadowing_blocking = alias_shadowing()
    # The provenance reasons go first: until they are answered, nothing below
    # them in the list means what it says. A shadowed subcommand goes before
    # those, because it says a gate did not run the tool it is named for -- and
    # an unsupported record goes before everything, because it says a gate may
    # not have run anything at all, which is the same objection one step
    # further back.
    blocking[:0] = provenance_blocking
    blocking[:0] = shadowing_blocking
    blocking[:0] = execution_blocking

    analyzers = {}
    for line in environment("GATE_ANALYZERS").splitlines():
        if "=" in line:
            name, _, version = line.partition("=")
            analyzers[name] = version

    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "game-experiment-gate-report",
        "generated_at": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "runner": {
            "command": "tools/quality/run-gates.sh",
            "execution_order": expected,
            "hermetic": True,
            "note": (
                "Run from a throwaway worktree at the named commit with a private "
                "CARGO_TARGET_DIR, both removed afterwards, so no stale build artefact and "
                "nothing in a developer's working tree can influence the result. That "
                "worktree shares the invoking repository's object store, which made gate 9 "
                "an exception until it was fixed: `gitleaks git` scanned every ref the "
                "invoking repository had, so an unpushed local branch could fail an "
                "innocent revision and two reviewers could get different reports, and "
                "different stable digests, for the same commit. Gate 9 now scans a "
                "repository holding this revision's history and no other ref. The run is "
                "green only when every gate is `pass`."
            ),
            "environment_note": (
                "The gate builds inherit the invoking environment's PATH and any "
                "RUSTFLAGS-family variable; those are not scrubbed. RUSTUP_TOOLCHAIN is "
                "overridden to the pinned toolchain and CARGO_TARGET_DIR to this run's "
                "private directory."
            ),
            "provenance": provenance_section,
            "enumeration_note": (
                "`--list-gates` prints this execution order as JSON, and AR-0007's "
                "negative-fixture suite reads it to discover the gates, so that a gate "
                "added here without a fixture is itself a failure. Every run parses that "
                "output and compares its ids to the order above before any gate starts, "
                "and refuses if it is not valid JSON or does not match. It is checked "
                "because it was once wrong: a command template grew a pair of quotation "
                "marks, the enumeration stopped parsing, and no gate noticed -- an "
                "interface claim with nothing behind it, which is the defect this "
                "project exists to catch."
            ),
            "subcommand_shadowing": shadowing,
            "gate_record_note": (
                "Each gate carries two lists of revision-controlled files, and the "
                "distinction is the point. `scripts` is what the gate executed out of "
                "the revision. `inputs` is the policy it read: deny.toml for gate 5, the "
                "advisory ignore list for gate 6, the formatting and lint configuration "
                "for gates 1 and 2, the manifest whose floors gate 7 compares against. "
                "They were one list, which left `\"scripts\": []` on the four gates that "
                "execute nothing from the revision but are decided outright by a file in "
                "it -- true, and read at face value it says the opposite of the truth. "
                "Both lists give the sha256 of each file as it was in the revision, and "
                "an absent file is recorded as absent rather than omitted."
            ),
        },
        "revision": {
            "commit": environment("GATE_COMMIT"),
            "base_ref": environment("GATE_BASE_REF"),
            "base_commit": environment("GATE_BASE_COMMIT"),
            "base_note": (
                "`base_ref` is the ref as it was typed on the command line and is outside "
                "the stable digest, because how a base is spelled is not part of what was "
                "verified. `base_commit` is what the gates actually ranged over and is "
                "inside it."
            ),
        },
        "toolchain": {
            "pinned": environment("GATE_TOOLCHAIN"),
            "rustc": environment("GATE_RUSTC"),
            "cargo": environment("GATE_CARGO"),
        },
        "tool_versions": {
            "cargo-deny": environment("GATE_DENY"),
            "cargo-audit": environment("GATE_AUDIT"),
            "cargo-llvm-cov": environment("GATE_LLVMCOV"),
            "python3": platform.python_version(),
            **analyzers,
        },
        "graphics": json.loads(environment("GATE_DEVICE") or "{}"),
        "host": {
            "fingerprint": f"sha256:{environment('GATE_HOST')}",
            "system": platform.system(),
            "release_series": kernel_series(platform.release()),
            "machine": platform.machine(),
            "note": (
                "A fingerprint, not a name: the host's identity is local configuration and "
                "does not belong in a public artefact. The same rule applies to everything "
                "beside the fingerprint -- the kernel is reported as a series rather than "
                "an exact release, and the graphics section carries device classes and "
                "fingerprints rather than retail model names. The full kernel release is "
                "one of the inputs to the fingerprint above. What holds that property is "
                "collection, not scanning: the hardware and kernel strings are hashed at "
                "the point they are read and the plaintext never reaches a field. The "
                "runner's privacy scan is no evidence for it either way, because "
                "privacy.py carries no hardware or kernel pattern -- so a regression that "
                "started publishing a model name again would pass that scan. Not "
                "collecting is the stronger guarantee; it is also the unwatched one."
            ),
        },
        "coverage": coverage_section(environment("GATE_COVERAGE"), manifest),
        "tests": {
            "ignored": ignored_tests(environment("GATE_IGNORED")),
            "note": (
                "Tests carrying #[ignore] are named here rather than disappearing into a green "
                "gate 3. Whether an exclusion is justified and carries a tracking AR is a review "
                "obligation, not something this runner can decide."
            ),
        },
        "gates": gates,
        "second_opinion": second_opinion(environment("GATE_SECOND_OPINION")),
        "blocking": blocking,
        "summary": {
            "total": len(gates),
            "counts": dict(sorted(counts.items())),
            "result": "pass" if not blocking else "fail",
        },
    }

    # What the digest is over: the subject of the run, never its occasion.
    stable = json.loads(json.dumps(report))
    stable.pop("generated_at", None)
    stable["revision"].pop("base_ref", None)
    stable["runner"]["provenance"].pop("invoking_checkout_commit", None)
    for gate in stable["gates"]:
        gate.pop("duration_seconds", None)
        # The step table stays inside the digest -- its labels, argv digests and
        # exit codes are what changes when a gate stops running *and no ledger
        # entry is forged for it*, which is the property AR-0017 needed the
        # digest to have and it had none of before. The qualification is not
        # decoration: `record_execution` is callable from the runner's dispatch,
        # so an arm can write the label, digest and exit an honest run would
        # have written, and at a fixed revision that produces a byte-identical
        # digest. `stable_digest_note` says so in the report. Only the
        # wall-clock figure inside each step comes out here, for the same reason
        # the gate's own duration does.
        for step in gate.get("steps", []):
            step.pop("duration_seconds", None)
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    report["stable_digest"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report["stable_digest_note"] = (
        "Computed over this report with generated_at, every gate and step duration, "
        "revision.base_ref and runner.provenance.invoking_checkout_commit removed, "
        "canonically serialised. Two runs of the same revision on the same host produce "
        "the same value however the base was spelled and whichever checkout invoked "
        "them, and whatever local branches that checkout happens to carry, provided its "
        "gate-deciding inputs match the revision's -- and when they do not, the digest "
        "differs, because the two runs are then not the same evidence. Each gate's "
        "`steps` are inside it, so a gate that stops running its command changes this "
        "value WHEN NO LEDGER ENTRY IS FORGED FOR IT; before AR-0017 nothing about a "
        "skipped gate reached this value at all, its only trace being a duration, and "
        "durations are excluded here as wall-clock noise. The qualification is not "
        "hypothetical and is the whole of what this digest can say: `record_execution` "
        "is reachable from the runner's dispatch, so an arm that runs nothing can write "
        "a ledger entry carrying the label, digest and exit an honest run would have "
        "written, and at a fixed revision that produces a byte-identical value here. "
        "This digest establishes that two runs saw the same evidence. It does not "
        "establish that the evidence was produced by running anything."
    )

    Path(environment("GATE_REPORT")).write_text(
        json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    # The report is written either way: a reader needs to see the contradiction,
    # and a run that produced no artefact is harder to review than one that
    # produced a red one. 3, not 1, so an unhandled exception cannot be read as
    # this finding.
    return 3 if (unsupported or ledger_problem) else 0


if __name__ == "__main__":
    sys.exit(main())
