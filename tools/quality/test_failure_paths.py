#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 13: prove every gate rejects a planted defect.

An all-green gate run is worth exactly as much as the evidence that a red run
was possible. This suite supplies that evidence: it plants one controlled
defect per fixture, invokes the *production* gate command against it, and
asserts the specific rejection that gate is supposed to emit. A fixture that
merely observes a non-zero exit proves nothing, because a gate can fail for the
wrong reason; every fixture therefore names the diagnostic it expects, and most
also name the diagnostics that must *not* appear, so that a rejection by one
rule is never mistaken for a rejection by another.

Three couplings make this more than a pile of assertions:

  * The gate list is read from `tools/quality/run-gates.sh --list-gates`, never
    from a list kept here and never from docs/QUALITY_GATES.md. A gate added to
    the runner without a fixture fails this suite, and a fixture naming a gate
    the runner no longer enumerates fails it too. Adding a gate and adding its
    fixture is therefore one piece of work, not two.

  * Each fixture executes the argv the runner's own enumeration describes for
    its gate, so neutering a gate's command in the runner -- replacing it with
    `true`, dropping a flag that arms it -- takes its fixture down with it.

  * Where a gate runs several tools, each tool is a separate fixture against a
    separate defect in a separate run, because a defect that stops the first
    tool says nothing about whether the second one runs at all.

The checkers themselves are executed where they live in the revision under
test, pointed at a throwaway fixture tree with `--repo` (or with the process
working directory, for the one checker that has no such flag). Nothing is
copied and re-run, so a fixture is rejected by the very file the runner
executes: neuter `check_crate_table.py` and gate 14's two fixtures fail.

What this suite cannot do is certify itself. Gate 13's own fixture asserts one
property and only one -- that the coupling above is enforced -- by dropping
every fixture for one gate and requiring the resulting gap to be rejected. It
does not and cannot show that the other twenty-seven fixtures are honest; that
rests on each of them naming a specific diagnostic, and on an outside
demonstration that disabling a gate turns its fixture red.

There is a second boundary, and it is the one a reviewer looking for a
secretly disabled gate will reach first. Every fixture here proves that a
gate's *command* rejects a defect. None of them proves that the *runner* still
executes that command against the real tree: `--list-gates` describes what a
gate would run, not whether the gate is armed, so a gate short-circuited to a
hardcoded pass, or skipped by a condition, leaves every fixture green. That was
demonstrated rather than argued -- review falsified an earlier draft of this
paragraph by doing the thing it claimed was covered, short-circuiting gates 2,
3, 4 and 7 in the runner's *dispatch*, leaving the command templates and
`--list-gates` untouched, and getting a 15/15 green report whose sabotaged
records carried `status: pass`, a plausible `argv_sha256` and real input hashes,
with gate 7 *upgraded* from a blocking `not_armed` to `pass` by being disabled
and all twenty-seven fixtures still green.

AR-0017 closes that, and the assertions that close it are in this file under
`RUNNER_ASSERTIONS`, deliberately not among the fixtures. Every command a gate
runs is now logged by the runner's executor with the exit status the command
returned, a gate's published digest, duration and step table are derived from
that ledger rather than stated by the dispatch, and `build_report.py` refuses to
publish a status the ledger does not support. The rule is over ledger entries
and not over executions, which is the distinction this file has now got wrong
nine times: a `pass` needs at least one entry for the gate with every entry at
0, and `not_applicable` and `not_implemented` need no entry at all. Those
assertions run `build_report.py`, which is a production command with a planted
contradiction, exactly as the fixtures run gate commands with planted defects.

Be exact about how far that reaches. Two versions of this paragraph have now
been wrong in the same direction, and the second was wrong after the first was
corrected, so the correction is written with the measurement beside it.

`record_execution` is a top-level shell function in `run-gates.sh`, reachable
from any arm of the dispatch. **Disabling a gate is still a one-line change in
one arm of one `case`**, in one file, with `build_report.py` untouched: the arm
writes a ledger entry naming the honest label, the digest of the honest argv and
an exit of 0, then sets the status. Review demonstrated it over gates 2, 3, 4
and 12 and got a green board -- every gate passing, no blocking reason, every
fixture and every assertion in this file green, and the forged records identical
in every published field to honest ones. This file previously said the edit now
spanned two files and cost more than one line. It was never measured and it was
false, and saying so here is cheaper than the next reviewer discovering it
again.

What a short-circuited arm can no longer do is record a pass while saying
nothing about what ran. The digest, the duration and the step table are not
arguments the dispatch supplies any more, so a forgery has to name a command, a
digest and an exit status in the executor's own format. That is more for a
reviewer to read; it is not more for an attacker to write. The trust is
unchanged: a runner is executed by the person invoking it, and no artefact it
produces can vouch for a runner the reviewer chose to trust. Reading the diff of
`tools/quality/run-gates.sh` is still the reviewer's obligation, and it is the
file to read, because one line there is still enough.

So the claim of the fixtures is exactly one sentence, with nothing appended to
it: a gate's command rejects the defect it is supposed to reject. The claim of
the runner assertions is one more, and it is about the *recorder* and not about
the runner as a whole: the report will not call a gate a pass over a ledger that
records no command behind it. Whether the ledger describes anything that ran is
outside every assertion here.

One more thing to know before reading a green run: a fixture can be *pending*.
Two of them are, on any revision before AR-0009 arms the coverage floors, since
the property they plant a defect against does not exist there. A pending
fixture is never a pass, is never counted as one, and is printed on every run
and in the JSON. It is also the most dangerous shape in this file, because it
is the one way a check here can report nothing wrong while checking nothing --
review has already caught it doing exactly that twice -- so the condition that
produces it is a three-part conjunction, is set in one place, is written so
that no leg is stricter than the check it stands in for, and has its reach
written down and bounded in `baseline_guard_state`.

Where the results go, and where they do not. `--json PATH` writes the whole
outcome table: every fixture, its gate and tool, the argv, the expected and
forbidden diagnostics, the exit code, and each of the gaps declared below.
**The runner does not pass `--json`.** It runs this suite as a plain command,
so gate 13's record in the gate report is a status, an exit code, a duration
and the checker's digest, and nothing else: grep that report for the fixture tally,
`not exercised` or `shellcheck` and there are no hits. Every gap this suite
declares -- the unexercised tool, the four steps invisible to the enumeration,
the inert capability, the flags no fixture depends on -- is written to stdout
and discarded with the rest of the gate's output. Loud beats silent only up to
that hop, and at that hop it is silent. Until the runner asks for the table,
the only ways to read it are to run this suite directly with `--json`, or to
read the gate's stdout in a `--log-dir` run. Passing `--json` is a
one-argument change in run-gates.sh, which this task does not own.

Everything is built under a throwaway directory and removed afterwards; the
suite asserts at the end that it changed nothing in the repository it read.

    tools/quality/test_failure_paths.py --bin-dir DIR [--json PATH] [--list]

Exit codes: 0 every fixture rejected as expected; 1 a fixture, the enumeration
check or the no-residue check failed; 2 the suite could not run at all.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import string
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Dropping every fixture for one gate. The only supported way to plant a defect
# in this suite, and the mechanism gate 13's own fixture uses. It is safe to
# leave in production code because it can only ever make a run red: removing a
# gate's fixtures leaves that gate unmatched, which is a failure, and the
# invariant is asserted again before exit. There is deliberately no hook that
# removes a single fixture from a gate that has several -- that one could leave
# a hole and still exit 0.
DROP_GATE = "NEGATIVE_FIXTURE_DROP_GATE"

TIMEOUT = 900

# A tool a gate runs that no fixture exercises is a failure, unless it is named
# here with a reason that is printed on every run. Silence is what this suite
# exists to prevent, so the exemption is loud rather than absent.
KNOWN_UNCOVERED = {
    (7, 0): (
        "cargo llvm-cov measurement: docs/QUALITY_GATES.md section 13 specifies five "
        "fixtures for gate 7, every one of them against the floor check and none "
        "against the measurement that feeds it -- section 13 names llvm-cov as the "
        "gate's first tool and makes it the one exemption to its own per-tool rule. "
        "A fixture for the measurement step is constructible (llvm-cov over a "
        "workspace whose tests fail) and is left out on the three grounds section 13 "
        "gives: it would assert a pinned third-party binary's behaviour rather than "
        "this repository's; it could not run where the rest of this suite runs, "
        "needing a Vulkan compute device and a multi-gigabyte instrumented build; "
        "and the loss is bounded, because check_coverage.py exits 2 on a report the "
        "measurement failed to hand it, so what stays uncovered is a wrong number "
        "and not a missing one. Closing it anyway would also make a 29th fixture "
        "against a table that names 28. Reported so the gap is visible rather than "
        "assumed."
    ),
}


# Work a gate does that its own enumeration entry does not describe, and that
# this suite therefore cannot discover by reading `--list-gates`. Listed by hand
# because the alternative is silence, and printed on every run for the same
# reason. Anything here is a limit of the enumeration interface, not of the
# gate: closing it means the runner describing the step, after which the check
# above finds it without help.
UNSEEN_STEPS = (
    (3, "the excluded-test inventory (`cargo test -- --list --ignored`) runs as part "
        "of gate 3 and is absent from the gate's command, so no fixture can be "
        "attached to it by enumeration"),
    (10, "the installed analyzers' versions are cross-checked against the manifest "
         "after the installer returns; that comparison is in the runner, not in the "
         "gate's command, and is not exercised here"),
    (8, "the runner passes -config-file only when .github/actionlint.yaml exists, "
        "while the command template always shows it; the fixture supplies that file "
        "so the described command can be run as described"),
    (7, "the runner passes --coverage-json and --output-path that the command "
        "template does not show; the fixture supplies them"),
)


# Rejection paths a gate has that this suite deliberately does not cover, with
# who does cover them. A decision, recorded so that the next reader sees a
# choice rather than an oversight -- an unrecorded omission being the thing
# this milestone has spent eighteen ARs learning to distrust.
OUT_OF_SCOPE = (
    (7, "manifest-contract violations (a bare string where a record belongs, a "
        "missing `reason`, `measured_lines` or `measured_lines_total`, a "
        "critical package naming a crate that does not exist, a workspace floor "
        "with no recorded baseline), all exit 2",
     "NOT COVERED ANYWHERE. Out of scope for this milestone by coordinator "
     "decision, which is a scope decision and not a coverage claim: there is no "
     "unit-test suite in this repository -- no test file outside this one, and "
     "no gate that runs one -- so nothing cheaper covers these paths. Closing "
     "the gap needs fixtures here or a test file beside check_coverage.py."),
)


# Flags in a gate's command that its fixture does not depend on: remove the
# flag from the runner and the fixture stays green. Recorded because the
# opposite was claimed. Coupling a fixture to the runner's argv catches a
# command weakened in a way the planted defect depends on -- dropping
# `--check` from gate 1, `-D warnings` from gate 2, `RUSTDOCFLAGS` from gate 4
# -- and catches nothing else. Each entry below is a flag whose removal a
# reviewer demonstrated, or this suite demonstrated, to be invisible here.
NON_LOAD_BEARING = (
    (6, "--deny warnings",
     "the planted advisory (RUSTSEC-2020-0071) is a vulnerability, and cargo "
     "audit exits non-zero on a vulnerability with or without the flag. Only a "
     "warning-class advisory -- unmaintained, or yanked -- would depend on it, "
     "and section 13 specifies this fixture as a known-vulnerable lock entry."),
    (9, "--redact",
     "gitleaks 8.30.1 with --no-banner prints no secret material with or "
     "without it, verified both ways, so the fixture's check that the token "
     "never appears in the output is a guard against a future version rather "
     "than evidence about this one."),
    (9, "env -u GITLEAKS_CONFIG",
     "the prefix keeps a developer's exported configuration out of the scan. "
     "No fixture depends on it: attempts to suppress detection through "
     "GITLEAKS_CONFIG, and through an explicit --config naming a rule set with "
     "useDefault = false, both still reported the planted token, so no "
     "reachable configuration was found that removing the prefix would let in."),
)


def inert_capabilities(bin_dir=None) -> list:
    """Checking a gate's tools advertise but silently do not perform here.

    A tool that skips a check without saying so is the exact failure this
    suite exists to make visible, and it is invisible to a fixture: the
    capability is missing, so a defect aimed at it is simply not reported and
    the gate passes. Detected at run time rather than written down, so that the
    note disappears by itself once the gap is closed.

    Look where the tool being probed would look, not where this process
    happens to look. The gate 8 probe below asks whether actionlint can reach
    shellcheck, and actionlint is run out of the analyzer directory the runner
    installs and hands us as `--bin-dir` -- not off this process's PATH. Asking
    `shutil.which` alone got that wrong for a whole milestone: shellcheck sat
    in the analyzer directory from AR-0016 onward and this printed, on every
    real gate 13 run, that the binary was absent, that gate 8 had read no
    shell, and that the fix was a task which had already landed. Three false
    sentences on stdout, from the same defect class as the one two commits
    below this: a probe testing a neighbouring condition rather than the one
    the thing it stands in for tests.
    """
    inert = []
    reachable = shutil.which("shellcheck") is not None or (
        bin_dir is not None and os.access(Path(bin_dir) / "shellcheck", os.X_OK)
    )
    if not reachable:
        inert.append((
            8,
            "actionlint lints the shell in every `run:` block by invoking "
            "shellcheck, and disables that integration without a diagnostic when "
            "it cannot reach the binary. There is no executable shellcheck in the "
            "analyzer directory this suite was given, nor on its PATH, so gate 8 "
            "reports pass over shell it has not read. config/quality-tools.json "
            "has pinned shellcheck since AR-0016, so this is an analyzer "
            "directory that does not hold what the manifest pins rather than a "
            "gap in the manifest. The two fixtures here do not depend on the "
            "delegation (a schema violation and a template injection), so they "
            "remain honest; a fixture aimed at shell content could not be "
            "exercised in this state."
        ))
    return inert


class Environment(Exception):
    """The suite cannot run. Not a finding about the revision."""


@dataclass
class Result:
    exit_code: int
    output: str


def run(argv, cwd=None, env=None, stdin=None) -> Result:
    """Run a command and return its exit code with stdout and stderr merged."""
    merged = dict(os.environ)
    for key, value in (env or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            env=merged,
            input=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=TIMEOUT,
        )
    except FileNotFoundError as error:
        raise Environment(f"cannot execute {argv[0]}: {error.strerror}") from error
    except subprocess.TimeoutExpired as error:
        raise Environment(f"{argv[0]} did not finish within {TIMEOUT}s") from error
    return Result(completed.returncode, completed.stdout or "")


def git(root, *args, env=None, check=True) -> str:
    result = run(["git", "-C", str(root), *args], env=env)
    if check and result.exit_code != 0:
        raise Environment(
            f"git {' '.join(args[:2])} failed in the fixture tree "
            f"(exit {result.exit_code})"
        )
    return result.output


def repo_root() -> Path:
    """The revision under test: the repository holding this script."""
    here = Path(__file__).resolve().parent
    result = run(["git", "-C", str(here), "rev-parse", "--show-toplevel"])
    if result.exit_code != 0:
        raise Environment("not inside a git repository")
    return Path(result.output.strip())


# ------------------------------------------------------------- enumeration --


def enumerated_gates(root: Path) -> list[dict]:
    """The gates, as the runner describes itself.

    Read from the runner and from nowhere else. docs/QUALITY_GATES.md carries
    the same table in prose and has been wrong before -- gate 15 was added with
    the count at the head of the file updated and the section below it still
    enumerating fourteen. Prose cannot be the source here.
    """
    runner = root / "tools" / "quality" / "run-gates.sh"
    if not runner.is_file():
        raise Environment("no tools/quality/run-gates.sh in this revision")
    result = run([str(runner), "--list-gates"], cwd=root)
    if result.exit_code != 0:
        raise Environment(
            f"run-gates.sh --list-gates exited {result.exit_code}; the gate list "
            "cannot be read and this suite has nothing to check itself against"
        )
    try:
        gates = json.loads(result.output)
    except json.JSONDecodeError as error:
        raise Environment(f"--list-gates is not valid JSON: {error}") from error
    if not isinstance(gates, list) or not gates:
        raise Environment("--list-gates did not produce a non-empty JSON array")
    seen = set()
    for gate in gates:
        if not isinstance(gate, dict):
            raise Environment("--list-gates entry is not a JSON object")
        if not isinstance(gate.get("id"), int):
            raise Environment(f"--list-gates entry has no integer id: {gate!r}")
        if not gate.get("name") or not gate.get("command"):
            raise Environment(
                f"--list-gates entry {gate['id']} carries no name or no command"
            )
        if gate["id"] in seen:
            raise Environment(f"--list-gates enumerates gate {gate['id']} twice")
        seen.add(gate["id"])
    return gates


PLACEHOLDER = re.compile(r"<[^<>]+>")
ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


def command_parts(gate: dict) -> list[str]:
    """The tools a gate runs, in order. `;` separates them in the template."""
    return [part.strip() for part in gate["command"].split(";") if part.strip()]


def resolve_command(root: Path, gate: dict, slot: int, subs: dict) -> tuple[dict, list]:
    """Turn the runner's description of one tool into an environment and argv.

    This is what couples a fixture to the runner rather than to a copy of the
    runner's intentions: the fixture runs what `--list-gates` says the gate
    runs. The template is prose-shaped in two places and both are repaired
    here, deterministically, rather than worked around per fixture:

      * a leading `NAME=VALUE` whose value is unquoted and contains a space
        (`RUSTDOCFLAGS=-D warnings`), which shlex splits into two tokens; the
        following tokens are folded back into the value until a token names an
        executable that exists.
      * a repository-relative program path (`tools/quality/...`), resolved
        against the revision under test, so the checker that rejects a fixture
        is the file the runner executes and not a copy of it.

    An unsubstituted placeholder, or a program that cannot be found, raises
    rather than being guessed at.
    """
    parts = command_parts(gate)
    if slot >= len(parts):
        raise Environment(
            f"gate {gate['id']} describes {len(parts)} tool(s); this fixture needs "
            f"tool {slot + 1}. The gate's command changed shape and the fixture did not."
        )
    text = parts[slot]
    for key, value in subs.items():
        text = text.replace(key, str(value))
    unresolved = PLACEHOLDER.findall(text)
    if unresolved:
        raise Environment(
            f"gate {gate['id']} tool {slot + 1} still holds {unresolved} after "
            "substitution; the runner's command template grew a placeholder this "
            "fixture does not know how to fill"
        )
    argv = shlex.split(text)
    env: dict[str, str] = {}
    last_key = None
    while argv and ASSIGNMENT.match(argv[0]):
        key, _, value = argv.pop(0).partition("=")
        env[key] = value
        last_key = key
    while argv and last_key is not None and not _is_program(root, argv[0]):
        env[last_key] = f"{env[last_key]} {argv.pop(0)}"
    if not argv:
        raise Environment(f"gate {gate['id']} tool {slot + 1} names no program")
    argv[0] = _program_path(root, argv[0])
    return env, argv


def _program_path(root: Path, name: str) -> str:
    if "/" in name and not name.startswith("/"):
        return str(root / name)
    return name


def _is_program(root: Path, name: str) -> bool:
    candidate = _program_path(root, name)
    if "/" in candidate:
        return os.path.isfile(candidate) and os.access(candidate, os.X_OK)
    return shutil.which(candidate) is not None


# ----------------------------------------------------------- fixture plumbing --


@dataclass
class Plan:
    """One invocation of a production gate command against a planted defect."""

    argv: list
    cwd: Path
    env: dict = field(default_factory=dict)
    # An identical invocation with the defect removed, which must exit 0. Used
    # where the gate's own diagnostic is too coarse to prove which input it
    # rejected -- gitleaks reports `leaks found: 1` and not the rule that fired.
    control: "Plan | None" = None
    # Text that must not appear in the output of this particular invocation,
    # over and above the fixture's own list. Used where the value is generated
    # and cannot be written into the fixture declaration.
    forbid_extra: list = field(default_factory=list)
    note: str = ""
    # Set when the property this fixture is aimed at does not exist in the
    # revision under test yet. The command is not run and the fixture is
    # reported as pending: never a pass, never counted as one, and loud on
    # every run. Pending is the dangerous state in this suite, because it is
    # the one shape a check can take while reporting nothing wrong: a fixture
    # that pends about a property somebody has deleted is a silent pass wearing
    # a warning label, and review has caught this file doing exactly that
    # twice. What is done about it is narrowness rather than a guarantee --
    # each leg of the condition is written to be no stricter than the check it
    # stands in for, so that a revision the gate will act on is not reported as
    # one where the property is missing. Only `baseline_guard_state` sets this,
    # and its docstring is where the reach of that condition is written down,
    # bounded, and its residual named.
    pending: str = ""


@dataclass
class Fixture:
    gate: int
    slot: int
    name: str
    defect: str
    expect: tuple
    forbid: tuple
    build: object
    # The exact exit code required, where the number is itself the contract.
    # Section 13 gives gate 7's five floor rules their exit codes -- 1, 2, 2, 3
    # and 3 -- and all five are asserted, because the runner reads the number: it
    # maps 3 to `not_armed`, a distinct non-passing state, so a checker that
    # started returning 2 there would still be rejecting and would still be
    # wrong, and a rule-3 mutant was seen exiting 1 where the contract says 2
    # with its diagnostics the only thing that caught it. None means "any
    # non-zero", which is right wherever the gate documents no specific code.
    expect_exit: int | None = None

    @property
    def label(self) -> str:
        return f"{self.name} (gate {self.gate}, tool {self.slot + 1})"


FIXTURES: list[Fixture] = []


def fixture(gate, slot, name, defect, expect, forbid=(), expect_exit=None):
    def register(function):
        FIXTURES.append(
            Fixture(
                gate=gate,
                slot=slot,
                name=name,
                defect=defect,
                expect=tuple(expect),
                forbid=tuple(forbid),
                build=function,
                expect_exit=expect_exit,
            )
        )
        return function

    return register


class Context:
    """Throwaway storage, the analyzers, and one pristine copy of the revision."""

    def __init__(self, root: Path, bin_dir: Path, work: Path, gates: dict):
        self.root = root
        self.bin = bin_dir
        self.work = work
        self.gates = gates
        self._pristine: Path | None = None
        self._pristine_base: str | None = None
        self._counter = 0

    def scratch(self, name: str) -> Path:
        self._counter += 1
        path = self.work / f"{self._counter:02d}-{name}"
        path.mkdir(parents=True)
        return path

    # -- the revision under test, as a throwaway repository -------------------

    def _build_pristine(self) -> None:
        source = self.work / "pristine"
        source.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(self.root), "archive", "--format=tar", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if archive.returncode != 0 or not archive.stdout:
            raise Environment("cannot archive HEAD of the revision under test")
        extract = subprocess.run(
            ["tar", "-x", "-C", str(source)], input=archive.stdout, check=False
        )
        if extract.returncode != 0:
            raise Environment("cannot unpack the revision under test")
        identity = [
            "-c", "user.name=Negative Fixture",
            "-c", "user.email=fixture@example.invalid",
            "-c", "commit.gpgsign=false",
        ]
        git(source, "init", "-q", "-b", "fixture")
        git(source, "add", "-A")
        git(source, *identity, "commit", "-q", "--no-gpg-sign", "-m",
            "the revision under test, as a fixture base")
        self._pristine = source
        self._pristine_base = git(source, "rev-parse", "HEAD").strip()

    def tree(self, name: str) -> Path:
        """A private, writable, committed copy of the revision under test."""
        if self._pristine is None:
            self._build_pristine()
        destination = self.scratch(name) / "tree"
        shutil.copytree(self._pristine, destination, symlinks=True)
        return destination

    @property
    def pristine_base(self) -> str:
        if self._pristine is None:
            self._build_pristine()
        return self._pristine_base or ""

    # -- a minimal cargo package, for the gates that compile ------------------

    def crate(
        self,
        name: str,
        source: str,
        package: str = "gate-fixture",
        policy: tuple = (),
        lints: bool = False,
    ) -> Path:
        """A one-package workspace with no dependencies.

        The gates that compile are exercised against a package of their own
        rather than against a copy of the product workspace, because the
        product workspace opens a Vulkan device in a build script and takes
        minutes to build; the *command* is the production one either way, which
        is what the fixture is about. No dependencies means no registry access
        and a build measured in tenths of a second.

        The price is that such a package does not inherit the product's
        configuration, and a gate can be disarmed by weakening its
        configuration as surely as by deleting it. `policy` copies named
        configuration files out of the revision under test, and `lints` copies
        the workspace lint tables in as the package's own, so that relaxing
        them takes the fixture down. A member manifest's `[lints]` table can
        still override the workspace policy for that member alone, and no
        single fixture package can represent fourteen of those; the runner
        records the sha256 of every member manifest as a gate input for exactly
        that reason.
        """
        directory = self.scratch(name) / "crate"
        (directory / "src").mkdir(parents=True)
        manifest = (
            "[package]\n"
            f'name = "{package}"\n'
            'version = "0.1.0"\n'
            'edition = "2024"\n'
            'license = "MIT"\n'
            "publish = false\n"
        )
        if lints:
            manifest += "\n" + self.workspace_lints()
        (directory / "Cargo.toml").write_text(manifest, encoding="utf-8")
        for filename in policy:
            source_file = self.root / filename
            if source_file.is_file():
                shutil.copy2(source_file, directory / filename)
        (directory / "src" / "lib.rs").write_text(source, encoding="utf-8")
        lock = run(
            ["cargo", "generate-lockfile", "--offline"],
            cwd=directory,
            env={"CARGO_TARGET_DIR": str(directory / "target")},
        )
        if lock.exit_code != 0:
            raise Environment(
                "cannot write a lock file for a fixture package with no dependencies"
            )
        return directory

    def workspace_lints(self) -> str:
        """The product's `[workspace.lints.*]` tables, as a package's own.

        Textual, because the tables are the only part of the manifest wanted
        and a fixture package cannot inherit them without being a member of the
        workspace it is trying not to build.
        """
        wanted, keeping = [], False
        for line in (self.root / "Cargo.toml").read_text(encoding="utf-8").splitlines():
            if line.startswith("[workspace.lints"):
                keeping = True
                wanted.append("[" + line[len("[workspace.") :])
                continue
            if line.startswith("["):
                keeping = False
            if keeping:
                wanted.append(line)
        if not wanted:
            raise Environment(
                "the workspace manifest declares no [workspace.lints] tables, so "
                "this fixture cannot be held to the product's lint policy"
            )
        return "\n".join(wanted).rstrip() + "\n"

    def cargo_env(self, directory: Path) -> dict:
        """Keep fixture builds out of the run's own target directory."""
        return {"CARGO_TARGET_DIR": str(directory / "target")}

    def command(self, gate: int, slot: int, subs: dict | None = None):
        if gate not in self.gates:
            raise Environment(
                f"the runner no longer enumerates gate {gate}, so there is no "
                "production command to run this fixture against"
            )
        return resolve_command(self.root, self.gates[gate], slot, subs or {})


def identity() -> list:
    return [
        "-c", "user.name=Negative Fixture",
        "-c", "user.email=fixture@example.invalid",
    ]


# ------------------------------------------------------ gates 1 to 4: cargo --


@fixture(
    gate=1,
    slot=0,
    name="formatting-violation",
    defect="a function whose spacing and braces rustfmt would rewrite",
    expect=("Diff in", "badly_formatted"),
)
def formatting_violation(ctx: Context) -> Plan:
    crate = ctx.crate(
        "gate01-formatting",
        "//! A fixture package.\n"
        "\n"
        "/// Correctly formatted.\n"
        "pub fn formatted() -> i32 {\n"
        "    1\n"
        "}\n"
        "\n"
        "/// Deliberately misformatted: rustfmt rewrites the spacing and the body.\n"
        "pub fn  badly_formatted ()->i32{\n"
        "42\n"
        "}\n",
        policy=(".rustfmt.toml", "rustfmt.toml"),
    )
    env, argv = ctx.command(1, 0)
    return Plan(argv, crate, {**env, **ctx.cargo_env(crate)})


@fixture(
    gate=2,
    slot=0,
    name="lint-violation",
    defect="a needless `return`, which the workspace lint policy denies through -D warnings",
    expect=("unneeded `return` statement", "needless_return"),
)
def lint_violation(ctx: Context) -> Plan:
    crate = ctx.crate(
        "gate02-lints",
        "//! A fixture package.\n"
        "\n"
        "/// Trips clippy::needless_return.\n"
        "pub fn increment(value: i32) -> i32 {\n"
        "    return value + 1;\n"
        "}\n",
        policy=("clippy.toml", ".clippy.toml"),
        lints=True,
    )
    env, argv = ctx.command(2, 0)
    return Plan(argv, crate, {**env, **ctx.cargo_env(crate)})


@fixture(
    gate=3,
    slot=0,
    name="failing-test",
    defect="a unit test whose assertion does not hold",
    expect=("test result: FAILED", "planted negative fixture"),
)
def failing_test(ctx: Context) -> Plan:
    crate = ctx.crate(
        "gate03-tests",
        "//! A fixture package.\n"
        "\n"
        "/// Adds two numbers.\n"
        "pub fn add(a: i32, b: i32) -> i32 {\n"
        "    a + b\n"
        "}\n"
        "\n"
        "#[cfg(test)]\n"
        "mod tests {\n"
        "    #[test]\n"
        "    fn arithmetic_holds() {\n"
        '        assert_eq!(super::add(2, 2), 5, "planted negative fixture");\n'
        "    }\n"
        "}\n",
    )
    env, argv = ctx.command(3, 0)
    return Plan(argv, crate, {**env, **ctx.cargo_env(crate)})


@fixture(
    gate=4,
    slot=0,
    name="broken-intra-doc-link",
    defect="a doc link to an item that does not exist, on a public function",
    expect=("unresolved link to `NoSuchItemAnywhere`", "broken-intra-doc-links"),
)
def broken_intra_doc_link(ctx: Context) -> Plan:
    # The link has to sit on a *public* item. rustdoc does not document private
    # items by default, so a broken link inside a private module is never
    # resolved, never warns, and would make this fixture pass against a gate
    # that had been switched off. The private function below carries the same
    # defect and is deliberately not what the assertion is about.
    crate = ctx.crate(
        "gate04-documentation",
        "//! A fixture package.\n"
        "\n"
        "/// Refers to [`NoSuchItemAnywhere`], which does not exist.\n"
        "pub fn documented() {}\n"
        "\n"
        "/// A private item whose broken link to [`NeverReached`] rustdoc never sees.\n"
        "fn undocumented() {}\n",
    )
    env, argv = ctx.command(4, 0)
    return Plan(argv, crate, {**env, **ctx.cargo_env(crate)})


# -------------------------------------------- gates 5 and 6: the dependency --


@fixture(
    gate=5,
    slot=0,
    name="denied-dependency",
    defect="a crate in the graph that the dependency policy bans by name",
    expect=("error[banned]", "explicitly banned", "bans FAILED"),
)
def denied_dependency(ctx: Context) -> Plan:
    crate = ctx.crate(
        "gate05-dependency-policy",
        "//! A fixture package.\n",
        package="gate-fixture-denied",
    )
    (crate / "deny.toml").write_text(
        "# The fixture policy: everything the production deny.toml checks, with one\n"
        "# crate in the graph banned by name.\n"
        "[advisories]\n"
        "version = 2\n"
        "\n"
        "[licenses]\n"
        "version = 2\n"
        'allow = ["MIT"]\n'
        "\n"
        "[bans]\n"
        'multiple-versions = "deny"\n'
        'wildcards = "deny"\n'
        'deny = [{ name = "gate-fixture-denied" }]\n'
        "\n"
        "[sources]\n"
        'unknown-registry = "deny"\n'
        'unknown-git = "deny"\n',
        encoding="utf-8",
    )
    env, argv = ctx.command(5, 0)
    return Plan(argv, crate, {**env, **ctx.cargo_env(crate)})


@fixture(
    gate=6,
    slot=0,
    name="known-vulnerable-lock-entry",
    defect="a lock file pinning time 0.1.44, the subject of RUSTSEC-2020-0071",
    expect=("RUSTSEC-2020-0071", "vulnerability found"),
)
def known_vulnerable_lock_entry(ctx: Context) -> Plan:
    directory = ctx.scratch("gate06-advisories")
    (directory / "Cargo.lock").write_text(
        "version = 3\n"
        "\n"
        "[[package]]\n"
        'name = "gate-fixture"\n'
        'version = "0.1.0"\n'
        "dependencies = [\n"
        ' "time",\n'
        "]\n"
        "\n"
        "[[package]]\n"
        'name = "time"\n'
        'version = "0.1.44"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        'checksum = "6db9e6914ab8b1ae1c260a4ae7a49b6c5611b40328a735b21862567685e73255"\n',
        encoding="utf-8",
    )
    env, argv = ctx.command(6, 0)
    return Plan(argv, directory, env)


# ------------------------------------------------------- gate 7: the floors --


@fixture(
    gate=7,
    slot=1,
    name="unreachable-coverage-floor",
    defect="a workspace line floor of 100% against a measurement of 50%",
    expect=("FAIL coverage:", "is below the floor 100.00%"),
    # The checker names every critical package it measured whether or not the
    # package failed, so the forbidden text is the failure line and not the
    # measurement line: the workspace floor must be the only thing rejected.
    forbid=("NOT ARMED", "FAIL coverage: critical package"),
    # Section 13 names an exit code for each of gate 7's five rules -- 1, 2, 2,
    # 3 and 3 -- and this is rule 1. Asserting the number rather than its non-zeroness
    # is not pedantry here: a mutant aimed at rule 3 was observed exiting 1
    # where the contract says 2, and only the diagnostics caught it. The codes
    # are the gate's interface to the runner, which maps them to different
    # verdicts, so a rejection with the wrong one is a rejection that will be
    # recorded as the wrong thing.
    expect_exit=1,
)
def unreachable_coverage_floor(ctx: Context) -> Plan:
    directory = ctx.scratch("gate07-coverage")
    tree = ctx.tree("gate07-coverage-tree")
    manifest_path = ctx.root / "config" / "quality-tools.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    coverage = manifest.setdefault("coverage", {})
    # Both floors are armed, because exactly one null is `not armed` and not a
    # rejection, and every critical package is driven to a floor of zero so
    # that the workspace floor is the only thing that can fail and the
    # diagnostic is unambiguous. Replacing `critical_packages` wholesale is
    # what an earlier version did and it is wrong once AR-0009 arms the gate:
    # the manifest then carries real critical packages, and this fixture's
    # synthetic measurement has an empty `files` list, so every one of them
    # measures 0.00% and trips the fixture's own `forbid`. Each entry is
    # therefore edited in place, whatever shape it has.
    coverage["workspace_lines"] = 100.0
    coverage["critical_lines"] = 0.0
    # AR-0009 also records the measurement each floor was derived from and
    # rejects a floor it cannot substantiate. That is a second property of the
    # same gate, and this fixture is aimed at the first, so the baseline is
    # raised with the floor to keep the "unsubstantiated" path from firing
    # instead and making the rejection ambiguous.
    measured = coverage.get("measured_workspace")
    if isinstance(measured, dict):
        measured["lines"] = 100.0
    # Two manifest shapes have to work, because this fixture has to be green
    # both before and after AR-0009 arms the gate. Before, the list is empty
    # and a critical floor with nothing to apply to is itself an error, so a
    # package has to be named. After, the list holds a record per package and
    # each one is driven to a floor of zero in place.
    packages = coverage.get("critical_packages")
    if isinstance(packages, list) and any(
        isinstance(entry, dict) for entry in packages
    ):
        for entry in packages:
            if isinstance(entry, dict):
                entry["floor_lines"] = 0.0
                entry["measured_lines"] = 0.0
    else:
        coverage["critical_packages"] = ["peano"]
    fixture_manifest = directory / "quality-tools.json"
    fixture_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    measurement = directory / "coverage.json"
    measurement.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "files": [],
                        "totals": {
                            "lines": {"count": 2, "covered": 1, "percent": 50.0},
                            "regions": {"count": 2, "covered": 1, "percent": 50.0},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    env, argv = ctx.command(7, 1)
    argv += [
        "--coverage-json", str(measurement),
        "--manifest", str(fixture_manifest),
        "--repo", str(tree),
    ]
    return Plan(argv, directory, env, note="the floor check, not the measurement")


def _package_directories(tree: Path) -> dict:
    """Each workspace member's directory, the way check_coverage.py finds it.

    Asked of cargo rather than assumed to be `crates/<name>`, because that is
    what the checker does: it matches a coverage report's filenames against the
    directories `cargo metadata` reports, and a fixture that guessed the layout
    would keep passing after a crate moved while the property it is aimed at
    had stopped being exercised.
    """
    completed = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=str(tree),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=TIMEOUT,
    )
    if completed.returncode != 0:
        raise Environment(
            f"cargo metadata exited {completed.returncode} in the fixture tree, so a "
            "measurement cannot be attributed to the critical packages"
        )
    data = json.loads(completed.stdout)
    directories = {}
    for package in data["packages"]:
        directory = Path(package["manifest_path"]).parent
        try:
            directories[package["name"]] = str(directory.relative_to(tree))
        except ValueError:
            directories[package["name"]] = str(directory)
    return directories


def _healthy_measurement(tree: Path, coverage: dict) -> dict:
    """The manifest's own recorded baselines, replayed as a measurement.

    A healthy tree, by the manifest's definition of healthy and by no number
    this file chose: the workspace totals and every critical package's figure
    are read straight out of `measured_workspace` and `critical_packages`.
    Every floor is at or below the baseline it was set from -- the baseline
    guard exits 2 on a floor above its own measurement -- so a measurement
    equal to the baselines clears every floor by construction, and stays
    clearing them when the floors are re-measured.

    This exists because the synthetic 50% measurement the other gate 7 fixtures
    share cannot see the half-armed branch. Against 50% a checker with that
    branch removed exits 1 on the workspace floor, so the fixture stays green
    while the property is gone; the silent pass is on a *healthy* tree, which
    is the one condition none of those four fixtures set up.
    """
    baseline = coverage["measured_workspace"]
    total = float(baseline["lines_total"])
    covered = round(total * float(baseline["lines"]) / 100.0)
    directories = _package_directories(tree)
    files = []
    for entry in coverage.get("critical_packages", []):
        name = entry["name"]
        if name not in directories:
            raise Environment(
                f"the manifest names a critical package the fixture tree does not "
                f"have: {name}"
            )
        count = float(entry["measured_lines_total"])
        files.append({
            "filename": f"{directories[name]}/src/lib.rs",
            "summary": {"lines": {
                "count": count,
                "covered": round(count * float(entry["measured_lines"]) / 100.0),
            }},
        })
    return {"data": [{"files": files, "totals": {
        "lines": {"count": total, "covered": covered},
        "regions": {"count": total, "covered": covered},
    }}]}


def _coverage_plan(ctx: Context, name: str, edit, healthy: bool = False) -> tuple:
    """A gate 7 floor-check invocation over a manifest `edit` has rewritten.

    The measurement is synthetic. By default it is the one four of the five
    gate 7 fixtures share -- a workspace at 50%, with no per-file records --
    and what differs between them is the manifest, which is what those fixtures
    are about. `healthy=True` replaces it with the manifest's own baselines
    replayed as a measurement, for the one fixture whose defect is invisible
    against a failing tree.
    """
    directory = ctx.scratch(name)
    tree = ctx.tree(name + "-tree")
    manifest = json.loads(
        (ctx.root / "config" / "quality-tools.json").read_text(encoding="utf-8")
    )
    coverage = manifest.setdefault("coverage", {})
    # Read before `edit` nulls anything: the measurement is built from what the
    # manifest recorded, and the defect is planted in the floors beside it.
    report = _healthy_measurement(tree, coverage) if healthy else {"data": [{"files": [], "totals": {
        "lines": {"count": 2, "covered": 1, "percent": 50.0},
        "regions": {"count": 2, "covered": 1, "percent": 50.0},
    }}]}
    edit(coverage)
    fixture_manifest = directory / "quality-tools.json"
    fixture_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    measurement = directory / "coverage.json"
    measurement.write_text(json.dumps(report), encoding="utf-8")
    env, argv = ctx.command(7, 1)
    argv += [
        "--coverage-json", str(measurement),
        "--manifest", str(fixture_manifest),
        "--repo", str(tree),
    ]
    return directory, argv, env


def baseline_guard_state(ctx: Context) -> str:
    """Whether this revision checks a floor against its recorded baseline.

    AR-0009 adds that check, the baselines it reads and the floors it guards,
    all three at once. Before it lands there is nothing to plant a defect
    against: the manifest records no baseline, sets no floor, and the checker
    has no opinion about either, so the two fixtures below are reported pending
    rather than run.

    A skip that can come back is worse than no fixture at all, so pending is a
    conjunction and needs every part of it: the manifest carries no
    `measured_workspace`, *and* the checker does not name the guard, *and*
    neither coverage floor is set. Any one of the three failing puts the
    fixture back to running, which is the only direction that is safe -- a
    fixture that runs can go red, and a fixture that pends cannot.

    The third probe is here because the first two are not enough, and that was
    demonstrated rather than argued. An independent review deleted the guard,
    the `measured_workspace` baseline it reads and the manifest-contract block
    that requires one, all in one edit -- which is how a feature actually comes
    out, as a unit rather than a function at a time. Both of the first two
    probes went true again, while the floors stayed armed at 84.5 and 76.0 and
    gate 7 went on passing a healthy tree. The two fixtures said the property
    had not arrived yet, about a property that had been removed: a pending
    state decayed into a silent pass, in the suite whose whole claim is that
    its comments do not overclaim. So the third probe reads the floors.

    It reads them with `coverage.get(key) is not None`, which is the test
    `check_coverage.py` itself applies to these two keys at the point where it
    decides `NOT_ARMED`. That the two predicates are the same test is the
    property this leg is aiming at, and it is not a matter of taste. A stricter
    test here calls a manifest unarmed that the checker will go on to compare
    against, and the gap between the two is a silent pass and not a cosmetic
    difference: this leg previously asked for a non-bool `int` or `float`, and
    with both floors written as the JSON strings "84.5" and "76.0" on the
    deletion tree above, both fixtures below pended while `check_coverage.py`
    exited 0 against a 85.06% measurement with nothing on stderr. `float()`
    coerces the string, so the comparison happened and passed. Booleans behave
    the same way. That was the same decay, surviving the fix written for it.

    That is a claim about this leg and not about all three, and the difference
    is worth being exact about, because a reader who generalises it will
    generalise it wrongly. Leg 1 asks `isinstance(..., dict)` where the checker
    applies a four-part conjunction to the same value: `measured_workspace =
    {}` satisfies this probe and exits the checker 2. Leg 2 is a substring
    grep, which the checker applies nowhere at all. Both are *looser* than the
    checks they stand in for, which is the safe direction -- a looser leg
    reports the property present and lets the fixture run, and a fixture that
    runs can go red. The rule the whole conjunction is written to obey is that
    no leg may be stricter than the check it stands in for; only leg 3 is the
    same test, and only leg 3 needed to be.

    What this deliberately does not claim, two earlier versions of this
    docstring having claimed things in this neighbourhood and been wrong by
    review. Not that pending is unreachable once AR-0009 has landed: an edit
    that nulls both floors reaches it again. That residual is loudly red by
    another route -- `check_coverage.py` returns `NOT_ARMED`, exit 3, on the
    repository's own manifest, which blocks. (`floor-that-is-not-armed` is not
    what makes it safe. It builds its own synthetic manifest with both floors
    nulled, so it asserts the checker's behaviour on an unarmed manifest and
    says nothing about whether this repository's manifest is armed; it reported
    `ok` throughout the silent-pass state described above.) And not that the
    two predicates can no longer drift apart: nothing here reads
    `check_coverage.py`'s test or fails if it changes. Keeping them the same
    test is an obligation on whoever edits either, recorded here and enforced
    nowhere.
    """
    coverage = json.loads(
        (ctx.root / "config" / "quality-tools.json").read_text(encoding="utf-8")
    ).get("coverage", {})
    has_baseline = isinstance(coverage.get("measured_workspace"), dict)
    checker = (ctx.root / "tools" / "quality" / "check_coverage.py")
    names_guard = (
        "check_floor_against_baseline" in checker.read_text(encoding="utf-8")
        if checker.is_file() else False
    )
    # `is not None`, per key: the same test `check_coverage.py` applies to
    # these two keys at the point where it decides `NOT_ARMED`. See the
    # docstring for why using the checker's own predicate is the property this
    # leg is trying to have, and for what is and is not guaranteed by it.
    #
    # Any floor at all, not both: a manifest that sets one and nulls the other
    # is making a claim about coverage, and gate 7 already treats that state as
    # a rejection rather than a pass. Reading it as "armed" is what keeps the
    # narrower condition -- fewer revisions pend, more of them run.
    floors_armed = any(
        coverage.get(key) is not None
        for key in ("workspace_lines", "critical_lines")
    )
    if not has_baseline and not names_guard and not floors_armed:
        return (
            "this revision records no measured_workspace baseline, its "
            "check_coverage.py does not name check_floor_against_baseline, and "
            "neither coverage floor is set, so the property does not exist here "
            "yet. AR-0009 adds all three; this fixture runs and is expected to "
            "reject from the moment it lands, and runs again the moment any one "
            "of the three comes back without the other two."
        )
    return ""


def _armed_critical_floors(coverage: dict) -> None:
    """Put every critical package exactly on its own baseline.

    A floor equal to its baseline is licensed by the rule, so the critical
    entries contribute no contradiction of their own and the workspace floor is
    the only thing either fixture below is testing.
    """
    packages = coverage.get("critical_packages")
    if isinstance(packages, list):
        for entry in packages:
            if isinstance(entry, dict) and isinstance(
                entry.get("measured_lines"), (int, float)
            ):
                entry["floor_lines"] = float(entry["measured_lines"])


@fixture(
    gate=7,
    slot=1,
    name="floor-below-its-baseline",
    defect="a workspace floor slackened far below what its recorded baseline licenses",
    expect=(
        "workspace floor 50.00% is below",
        "the slackest floor its recorded baseline of 90.00% licenses",
    ),
    # The other three gate 7 fixtures' diagnostics, so a rejection by one is
    # never read as a rejection by another: this must not be the measurement
    # failing, nor an unarmed gate, nor the aspiration branch on the far side of
    # the same range check.
    forbid=(
        "NOT ARMED",
        "is below the floor",
        "is above its own recorded baseline",
    ),
    # Rule 2 of the four. A manifest that contradicts itself is exit 2, decided
    # before any coverage figure is compared; exit 1 here would be the gate
    # reporting a measurement failure it never performed.
    expect_exit=2,
)
def floor_below_its_baseline(ctx: Context) -> Plan:
    # One of the two sides of AR-0009's range check. Both sides get a fixture
    # because a range check has two boundaries and pinning one is how a mutant
    # survives: this milestone's own AR-0015 lost a `minLength` mutant that was
    # pinned on the invalid side and unpinned on the valid one.
    pending = baseline_guard_state(ctx)
    if pending:
        return Plan([], ctx.root, {}, pending=pending)

    def edit(coverage: dict) -> None:
        coverage["workspace_lines"] = 50.0
        coverage["critical_lines"] = 0.0
        baseline = coverage.setdefault("measured_workspace", {})
        baseline["lines"] = 90.0
        baseline["lines_total"] = 10000
        _armed_critical_floors(coverage)

    directory, argv, env = _coverage_plan(ctx, "gate07-slackened-floor", edit)
    return Plan(argv, directory, env,
                note="the floor is loosened without the baseline beside it moving")


@fixture(
    gate=7,
    slot=1,
    name="floor-above-its-measurement",
    defect="a workspace floor set above the measurement it claims to come from",
    expect=(
        "workspace floor 95.00% is above its own recorded baseline 80.00%",
        "a floor is set from what was measured, not from what was hoped for",
    ),
    forbid=(
        "NOT ARMED",
        "is below the floor",
        "the slackest floor its recorded baseline",
    ),
    # Rule 3, and the one that demonstrated why the number is worth asserting:
    # a review's mutant of this path exited 1 where section 13 says 2, and the
    # fixture was saved by its diagnostics alone.
    expect_exit=2,
)
def floor_above_its_measurement(ctx: Context) -> Plan:
    # The other side. An aspiration rather than a measurement: red on arrival,
    # and it tells a reader the opposite of what the number claims.
    pending = baseline_guard_state(ctx)
    if pending:
        return Plan([], ctx.root, {}, pending=pending)

    def edit(coverage: dict) -> None:
        coverage["workspace_lines"] = 95.0
        coverage["critical_lines"] = 0.0
        baseline = coverage.setdefault("measured_workspace", {})
        baseline["lines"] = 80.0
        baseline["lines_total"] = 10000
        _armed_critical_floors(coverage)

    directory, argv, env = _coverage_plan(ctx, "gate07-aspirational-floor", edit)
    return Plan(argv, directory, env,
                note="the floor claims a baseline the manifest does not record")


@fixture(
    gate=7,
    slot=1,
    name="floor-that-is-not-armed",
    defect="both floors null, which must be rejected as unarmed and never as a pass",
    expect=(
        "NOT ARMED",
        "both floors are null in the manifest, so nothing was compared",
    ),
    # The three floor fixtures' diagnostics. An unarmed gate must be rejected
    # for being unarmed, not because some floor happened to fail on the way.
    forbid=(
        "is below the floor",
        "is above its own recorded baseline",
        "the slackest floor its recorded baseline",
    ),
    # The number is the contract, not merely its non-zeroness. The runner maps
    # 3 to `not_armed`, a distinct non-passing state that names AR-0009 and
    # blocks; a checker returning 2 here would still be rejecting and would
    # still be wrong, and one returning 0 would be the silent pass this whole
    # suite exists to prevent.
    expect_exit=3,
)
def floor_that_is_not_armed(ctx: Context) -> Plan:
    # The one gate 7 fixture that needs no baseline and no measurement to
    # compare against, so it runs on every revision: the not-armed path and its
    # message are identical before and after AR-0009 arms the gate, which is
    # the point -- arming the gate must not be able to remove the check that
    # notices it was disarmed again.
    #
    # It was briefly out of scope on the grounds that AR-0009's own tests
    # covered it. They do not; AR-0009 ships no tests and this repository has
    # no unit-test suite. With that reason gone the exclusion had nothing
    # holding it up, and a gate exiting 0 where it owes 3 is exactly the
    # failure this milestone exists to prevent.
    def edit(coverage: dict) -> None:
        coverage["workspace_lines"] = None
        coverage["critical_lines"] = None

    directory, argv, env = _coverage_plan(ctx, "gate07-not-armed", edit)
    return Plan(argv, directory, env,
                note="a disarmed gate is a non-passing result, not a pass")


@fixture(
    gate=7,
    slot=1,
    name="half-armed-floors-on-a-healthy-tree",
    defect=(
        "critical_lines null while workspace_lines is set, against a measurement "
        "that clears every floor"
    ),
    expect=(
        "NOT ARMED",
        "critical_lines is null while the other floor is set",
        "a half-armed gate is not a pass",
    ),
    # Everything the other four gate 7 fixtures are aimed at, plus the both-null
    # message. A half-armed manifest must be rejected for being half-armed, and
    # in particular not by the branch that fixture four already covers: the
    # whole point of this one is that the two branches are separate and only one
    # of them has a fixture.
    forbid=(
        "both floors are null in the manifest",
        "is below the floor",
        "is above its own recorded baseline",
        "the slackest floor its recorded baseline",
    ),
    # Rule 4's second configuration, and the same exit code as its first: the
    # runner maps 3 to `not_armed`, which blocks. 0 here is the silent pass this
    # fixture exists to make impossible.
    expect_exit=3,
)
def half_armed_floors_on_a_healthy_tree(ctx: Context) -> Plan:
    # Routed here by AR-0009's independent review and recorded in
    # docs/QUALITY_GATES.md section 13 as a known residual of fixture four.
    # Reproduced rather than argued: with the half-armed branch removed from
    # check_coverage.py, fixture four stays green -- it builds a both-null
    # manifest and takes the branch above this one -- and this configuration
    # exits 0 against the real measurement on a healthy tree. It is reachable
    # because all four critical entries carry their own `floor_lines`, leaving
    # `critical_lines` a backstop nothing consults, so nulling it changes no
    # comparison and the run goes green having compared everything except the
    # thing that was disarmed.
    #
    # Only this direction. The mirror -- `workspace_lines` null while
    # `critical_lines` is set -- raises TypeError where the workspace floor is
    # coerced for the baseline check, which the runner records as a failure and
    # not as a pass, so a fixture for it would be asserting a crash rather than
    # this property. Saying so here keeps the claim the size of the evidence.
    def edit(coverage: dict) -> None:
        coverage["critical_lines"] = None

    directory, argv, env = _coverage_plan(
        ctx, "gate07-half-armed", edit, healthy=True
    )
    return Plan(argv, directory, env,
                note="the measurement is the manifest's own baselines, so every "
                     "floor that is still armed passes and only the disarmed one "
                     "can be the reason for a rejection")


# ------------------------------------------------ gate 8: the two analyzers --


def _workflow_repository(ctx: Context, name: str, workflows: dict) -> Path:
    directory = ctx.scratch(name)
    (directory / ".github" / "workflows").mkdir(parents=True)
    (directory / ".github" / "actionlint.yaml").write_text(
        "self-hosted-runner:\n  labels: []\n", encoding="utf-8"
    )
    for filename, body in workflows.items():
        (directory / ".github" / "workflows" / filename).write_text(body, encoding="utf-8")
    # actionlint refuses to look for workflows outside a repository.
    git(directory, "init", "-q", "-b", "fixture")
    git(directory, "add", "-A")
    return directory


@fixture(
    gate=8,
    slot=0,
    name="malformed-workflow",
    defect="a step key actionlint's schema does not allow",
    expect=('unexpected key "iff"', "syntax-check"),
)
def malformed_workflow(ctx: Context) -> Plan:
    # Separate file *and* separate run from the zizmor fixture, deliberately.
    # The original reason was that a schema-invalid workflow makes zizmor
    # collect no inputs and report `no inputs collected` instead of auditing.
    # That no longer reproduces on zizmor 1.30.0, which audits the valid
    # workflow beside the invalid one and reports six findings; the rationale
    # is kept here only because it explains the shape, not because it still
    # holds. The separation is still right for the reason section 13 gives: a
    # defect that stops the first tool proves nothing about the second, so each
    # tool is exercised in a run where it is the only thing that can fail.
    directory = _workflow_repository(
        ctx,
        "gate08-actionlint",
        {
            "malformed.yml": "name: malformed\n"
            "on:\n"
            "  workflow_dispatch:\n"
            "jobs:\n"
            "  build:\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - run: echo hello\n"
            "        iff: always()\n"
        },
    )
    env, argv = ctx.command(8, 0, {"<bin>": ctx.bin})
    return Plan(argv, directory, env)


@fixture(
    gate=8,
    slot=1,
    name="unsafe-workflow-construct",
    defect="attacker-controllable expression interpolated into a run block",
    # `concurrency-limits` is a pedantic-only audit: zizmor reports it on this
    # workflow with --pedantic and not without, verified both ways. Asserting
    # it alongside the injection is what makes the runner's --pedantic load
    # bearing, so dropping that flag turns this fixture red instead of leaving
    # it green. The injection itself is the fixture's subject and is reported
    # at either setting.
    expect=(
        "template-injection",
        "may expand into attacker-controllable code",
        "concurrency-limits",
    ),
    forbid=("no inputs collected",),
)
def unsafe_workflow_construct(ctx: Context) -> Plan:
    # A literal deviation from section 13 worth naming: it asks for a construct
    # that zizmor rejects "and `actionlint` does not". actionlint 1.7.12 also
    # objects to this workflow. That does not weaken the fixture -- the two
    # tools run in separate runs, and the assertions above are zizmor's own
    # diagnostics, which actionlint never emits -- but the clause is not
    # literally satisfied and should not be read as though it were.
    directory = _workflow_repository(
        ctx,
        "gate08-zizmor",
        {
            "injection.yml": "name: injection\n"
            "on:\n"
            "  workflow_dispatch:\n"
            "permissions: {}\n"
            "jobs:\n"
            "  greet:\n"
            "    name: greet\n"
            "    runs-on: ubuntu-latest\n"
            "    steps:\n"
            "      - name: greet\n"
            '        run: echo "hello ${{ github.event.issue.title }}"\n'
        },
    )
    env, argv = ctx.command(8, 1, {"<bin>": ctx.bin})
    return Plan(argv, directory, env)


# ---------------------------------------------------- gate 9: the scan path --


@fixture(
    gate=9,
    slot=0,
    name="planted-credential",
    defect="a synthetic GitHub personal access token committed to the history",
    expect=("leaks found: 1",),
    forbid=("no leaks found",),
)
def planted_credential(ctx: Context) -> Plan:
    # gitleaks' rule for this prefix carries an entropy threshold: a token of
    # the right shape built out of a predictable string is silently missed, and
    # a fixture using one would pass against a gate that never ran. The token
    # is therefore generated, never written down, and only its shape is fixed.
    alphabet = string.ascii_letters + string.digits
    token = "ghp_" + "".join(secrets.choice(alphabet) for _ in range(36))
    leaky = ctx.scratch("gate09-secret-scanning")
    clean = ctx.scratch("gate09-secret-scanning-control")
    plans = []
    for directory, body in ((leaky, token), (clean, "not-a-credential")):
        git(directory, "init", "-q", "-b", "fixture")
        (directory / "settings.py").write_text(
            f'ACCESS = "{body}"\n', encoding="utf-8"
        )
        git(directory, "add", "-A")
        git(directory, *identity(), "commit", "-q", "--no-gpg-sign", "-m",
            "fixture: application settings")
        commit = git(directory, "rev-parse", "HEAD").strip()
        env, argv = ctx.command(
            9,
            0,
            {
                "<bin>": ctx.bin,
                "<commit>": commit,
                "<isolated scan repository>": directory,
            },
        )
        plans.append(Plan(argv, directory, env))
    # The control is the same command over the same repository with a value of
    # the same length that is not a credential: it must find nothing. Without
    # it the assertion would be `leaks found: 1` with no evidence about which
    # input produced the one.
    plans[0].control = plans[1]
    # Kept, but not evidence about this version: gitleaks 8.30.1 with
    # --no-banner prints no secret material whether or not --redact is passed,
    # verified both ways, so this cannot fail here and does not demonstrate
    # that the flag does anything. It stays as a guard against a future version
    # that prints more, and is listed in NON_LOAD_BEARING so that nobody reads
    # it as proof that the gate redacts.
    plans[0].forbid_extra = [token]
    return plans[0]


# ----------------------------------------------- gate 10: the analyzer feed --


@fixture(
    gate=10,
    slot=0,
    name="analyzer-digest-mismatch",
    defect="a manifest whose recorded sha256 does not match the release asset",
    expect=("sha256 mismatch for", "refusing to extract"),
)
def analyzer_digest_mismatch(ctx: Context) -> Plan:
    directory = ctx.scratch("gate10-analyzer-install")
    cache = directory / "cache"
    cache.mkdir()
    installed = directory / "bin"
    installed.mkdir()
    manifest = json.loads((ctx.root / "config" / "quality-tools.json").read_text("utf-8"))
    machine = os.uname().machine
    platform = "linux_aarch64" if machine in ("aarch64", "arm64") else "linux_x86_64"
    tool = "actionlint"  # first in the installer's loop, so it fails before the rest
    entry = manifest["external"][tool]
    asset = entry[platform]["asset"]
    version = entry["version"]
    # Prefer the real asset out of the shared archive cache, so that the defect
    # is exactly the one the row describes: a manifest digest that does not
    # match the asset it names. When the cache holds nothing -- a first run on a
    # cold machine -- a stand-in of the same name is written instead, which
    # exercises the same comparison and is recorded as such in the note.
    shared = Path(
        os.environ.get("GATE_TOOL_CACHE")
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "game-experiment-gate-tools"
    )
    archive = cache / f"{tool}-{version}-{platform}-{asset}"
    origin = shared / archive.name
    if origin.is_file():
        shutil.copy2(origin, archive)
        note = "the real release asset against a tampered manifest digest"
    else:
        archive.write_bytes(b"not the asset this manifest names\n")
        note = "a stand-in asset: the shared archive cache held no copy to tamper against"
    entry[platform]["sha256"] = "0" * 63 + "1"
    tampered = directory / "quality-tools.json"
    tampered.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env, argv = ctx.command(10, 0, {"<bin>": installed})
    argv += ["--cache-dir", str(cache), "--manifest", str(tampered)]
    return Plan(argv, ctx.root, env, note=note)


# ------------------------------------------------------ gate 11: the policy --


def _policy_plan(ctx: Context, tree: Path) -> Plan:
    git(tree, "add", "-A")
    env, argv = ctx.command(
        11, 0, {"<base>": ctx.pristine_base, "<head>": "HEAD"}
    )
    argv += ["--repo", str(tree)]
    return Plan(argv, tree, env)


@fixture(
    gate=11,
    slot=0,
    name="dangling-markdown-link",
    defect="a relative link to a file that is not in the repository",
    expect=("FAIL markdown link:", "no such file: ./no-such-document.md"),
    forbid=("licence header:", "workflow:", "no such heading"),
)
def dangling_markdown_link(ctx: Context) -> Plan:
    tree = ctx.tree("gate11-dangling-link")
    (tree / "fixture-dangling-link.md").write_text(
        "# Fixture\n\nA link to [nothing](./no-such-document.md).\n", encoding="utf-8"
    )
    return _policy_plan(ctx, tree)


@fixture(
    gate=11,
    slot=0,
    name="dangling-markdown-anchor",
    defect="a link to a file that exists, naming a heading that does not",
    expect=(
        "FAIL markdown link:",
        "no such heading: ./fixture-dangling-anchor.md#no-such-heading",
    ),
    forbid=("licence header:", "workflow:", "no such file"),
)
def dangling_markdown_anchor(ctx: Context) -> Plan:
    # Separate from the fixture above on purpose: a missing file is rejected
    # before an anchor is ever looked up, so one fixture demonstrates only the
    # first of the two properties the gate claims.
    tree = ctx.tree("gate11-dangling-anchor")
    (tree / "fixture-dangling-anchor.md").write_text(
        "# Fixture\n\nA link to [a heading that is not here]"
        "(./fixture-dangling-anchor.md#no-such-heading).\n",
        encoding="utf-8",
    )
    return _policy_plan(ctx, tree)


@fixture(
    gate=11,
    slot=0,
    name="missing-licence-header",
    defect="a tracked source file with no SPDX identifier in its first lines",
    expect=(
        "FAIL licence header:",
        "no SPDX-License-Identifier: MIT in the first 5 lines",
    ),
    forbid=("markdown link:", "workflow:"),
)
def missing_licence_header(ctx: Context) -> Plan:
    tree = ctx.tree("gate11-licence-header")
    (tree / "tools" / "quality" / "fixture_unlicensed.py").write_text(
        "#!/usr/bin/env python3\n"
        '"""A tracked source file carrying no licence identifier."""\n',
        encoding="utf-8",
    )
    return _policy_plan(ctx, tree)


@fixture(
    gate=11,
    slot=0,
    name="mutable-action-reference",
    defect="a workflow step pinning an action to a tag rather than a commit id",
    expect=(
        "FAIL workflow:",
        "action reference is mutable, a full 40-character commit id is required: "
        "actions/checkout@v4",
    ),
    forbid=("markdown link:", "licence header:", "dormant workflows must not trigger"),
)
def mutable_action_reference(ctx: Context) -> Plan:
    # This belongs to gate 11 and not to gate 8: immutability is
    # repository_policy.py's rule, and neither analyzer enforces it. A fixture
    # asserted against actionlint or zizmor would say nothing about whether
    # this rule fires. The trigger is `workflow_dispatch` so that the dormancy
    # rule cannot be what rejects it.
    tree = ctx.tree("gate11-mutable-action")
    workflows = tree / ".github" / "workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    (workflows / "fixture-mutable.yml").write_text(
        "name: fixture\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "permissions: {}\n"
        "jobs:\n"
        "  check:\n"
        "    name: check\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n",
        encoding="utf-8",
    )
    return _policy_plan(ctx, tree)


# ----------------------------------------------------- gate 12: the commits --


def _commit_plan(ctx: Context, tree: Path, message: str, sign: bool) -> Plan:
    (tree / "fixture-commit.txt").write_text("a change to commit\n", encoding="utf-8")
    git(tree, "add", "-A")
    argv = ["commit", "-q", "-m", message]
    argv.append("-S" if sign else "--no-gpg-sign")
    result = run(["git", "-C", str(tree), *identity(), *argv])
    if result.exit_code != 0:
        raise Environment(
            "cannot create the fixture commit "
            f"({'signed' if sign else 'unsigned'}, exit {result.exit_code})"
        )
    head = git(tree, "rev-parse", "HEAD").strip()
    env, argv = ctx.command(12, 0, {"<base>": ctx.pristine_base, "<head>": head})
    argv += ["--repo", str(tree)]
    return Plan(argv, tree, env)


AUTHOR = "Negative Fixture <fixture@example.invalid>"


@fixture(
    gate=12,
    slot=0,
    name="missing-sign-off",
    defect="a signed commit whose message carries no Signed-off-by trailer",
    expect=("no Signed-off-by trailer for author " + AUTHOR,),
    forbid=("signature does not verify", "message privacy"),
)
def missing_sign_off(ctx: Context) -> Plan:
    tree = ctx.tree("gate12-sign-off")
    return _commit_plan(ctx, tree, "fixture: a commit with no sign-off", sign=True)


@fixture(
    gate=12,
    slot=0,
    name="unsigned-commit",
    defect="a correctly signed-off commit that carries no signature",
    expect=("signature does not verify against the allowed keys",),
    forbid=("no Signed-off-by trailer", "message privacy"),
)
def unsigned_commit(ctx: Context) -> Plan:
    tree = ctx.tree("gate12-signature")
    return _commit_plan(
        ctx,
        tree,
        f"fixture: a commit with no signature\n\nSigned-off-by: {AUTHOR}",
        sign=False,
    )


@fixture(
    gate=12,
    slot=0,
    name="private-path-in-message",
    defect="a commit message quoting an absolute path from a developer's account",
    expect=("message privacy: absolute Linux home path",),
    forbid=("no Signed-off-by trailer", "signature does not verify"),
)
def private_path_in_message(ctx: Context) -> Plan:
    # Assembled at run time rather than written out, so that this file does not
    # itself contain the shape of path the gate exists to keep out of the
    # repository. privacy.py builds its own pattern the same way.
    private = "/" + "home/" + "someone/checkout/notes.txt"
    tree = ctx.tree("gate12-privacy")
    return _commit_plan(
        ctx,
        tree,
        f"fixture: a commit naming {private}\n\nSigned-off-by: {AUTHOR}",
        sign=True,
    )


# ------------------------------------------------------ gate 13: this suite --


@fixture(
    gate=13,
    slot=0,
    name="fixture-removed-from-the-suite",
    defect="every fixture for one gate dropped, leaving that gate unmatched",
    expect=(
        "FAIL enumeration:",
        "is enumerated by the runner but has no fixture",
        "gate 14",
    ),
)
def fixture_removed_from_the_suite(ctx: Context) -> Plan:
    # The bootstrap, stated rather than assumed. Gate 13 runs this suite and
    # this suite enumerates gate 13, so the question is what a fixture for gate
    # 13 can honestly assert about itself. It cannot assert that the suite is
    # right -- nothing can bootstrap its own credibility, and a suite that ran
    # itself in full would only be slower, not more convincing. It asserts the
    # one property that is genuinely self-referential and genuinely load
    # bearing: that the coupling between the runner's enumeration and this
    # fixture table is enforced, so a gate cannot be added later without a
    # fixture.
    #
    # The defect is planted through NEGATIVE_FIXTURE_DROP_GATE, which removes
    # every fixture for one gate. The child therefore has a real gap, reports
    # it and exits non-zero. Dropping by gate rather than by fixture is what
    # makes the hook safe to ship: it cannot leave a gate half-covered and
    # still green, and the drop mode never runs fixtures, so there is no
    # recursion to bound.
    env, argv = ctx.command(13, 0, {"<bin>": ctx.bin})
    return Plan(
        argv,
        ctx.root,
        {**env, DROP_GATE: "14"},
        note="the enumeration check only; the child runs no fixtures",
    )


# -------------------------------------------------- gate 14: the crate table --


def _crate_table_plan(ctx: Context, tree: Path) -> Plan:
    env, argv = ctx.command(14, 0)
    argv += ["--repo", str(tree)]
    return Plan(argv, tree, env)


def _entry_point_lines(tree: Path) -> list:
    return (tree / "CLAUDE.md").read_text(encoding="utf-8").splitlines()


def _first_table_bounds(lines: list) -> tuple:
    start = None
    for number, line in enumerate(lines):
        if line.lstrip().startswith("|"):
            if start is None:
                start = number
        elif start is not None:
            return start, number
    if start is None:
        raise Environment("the entry point holds no Markdown table to plant a defect in")
    return start, len(lines)


@fixture(
    gate=14,
    slot=0,
    name="crate-absent-from-the-table",
    defect="a workspace crate whose row is missing from the entry-point table",
    expect=("FAIL peano: a workspace crate that CLAUDE.md does not name",),
    forbid=("named in CLAUDE.md but not a workspace crate",),
)
def crate_absent_from_the_table(ctx: Context) -> Plan:
    tree = ctx.tree("gate14-undocumented-crate")
    lines = _entry_point_lines(tree)
    start, end = _first_table_bounds(lines)
    kept = [
        line
        for number, line in enumerate(lines)
        if not (start + 2 <= number < end and line.split("|")[1].strip().strip("`") == "peano")
    ]
    if len(kept) == len(lines):
        raise Environment("the entry-point table names no crate this fixture can remove")
    (tree / "CLAUDE.md").write_text("\n".join(kept) + "\n", encoding="utf-8")
    return _crate_table_plan(ctx, tree)


@fixture(
    gate=14,
    slot=0,
    name="table-entry-that-is-not-a-crate",
    defect="a row in the entry-point table naming a crate the workspace does not have",
    expect=("FAIL fixture-phantom-crate: named in CLAUDE.md but not a workspace crate",),
    forbid=("a workspace crate that CLAUDE.md does not name",),
)
def table_entry_that_is_not_a_crate(ctx: Context) -> Plan:
    # The other direction, and not a duplicate of it: AR-0006's checker has to
    # fail both ways, and a suite built from one of them would leave the other
    # unproven.
    tree = ctx.tree("gate14-phantom-crate")
    lines = _entry_point_lines(tree)
    _, end = _first_table_bounds(lines)
    lines.insert(end, "| `fixture-phantom-crate` | A crate that does not exist. |")
    (tree / "CLAUDE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _crate_table_plan(ctx, tree)


# ------------------------------------------------------- gate 15: the index --


def _index_plan(ctx: Context, tree: Path) -> Plan:
    git(tree, "add", "-A")
    # check_docs.py resolves the repository from the working directory and has
    # no --repo, so the fixture tree is supplied as the working directory.
    env, argv = ctx.command(15, 0)
    return Plan(argv, tree, env)


INDEX = "docs/architecture/overview.md"


@fixture(
    gate=15,
    slot=0,
    name="unreachable-governed-document",
    defect="a governed document that nothing links to",
    expect=(
        "FAIL docs/process/fixture-orphan.md: governed document is not reachable "
        f"from {INDEX}",
    ),
    forbid=("index entry does not resolve",),
)
def unreachable_governed_document(ctx: Context) -> Plan:
    tree = ctx.tree("gate15-unreachable")
    (tree / "docs" / "process" / "fixture-orphan.md").write_text(
        "# Fixture orphan\n\nA governed document the index does not reach.\n",
        encoding="utf-8",
    )
    return _index_plan(ctx, tree)


@fixture(
    gate=15,
    slot=0,
    name="index-entry-without-a-file",
    defect="an index entry naming a document that does not exist",
    expect=("index entry does not resolve: no such file: ./fixture-absent.md",),
    forbid=("governed document is not reachable",),
)
def index_entry_without_a_file(ctx: Context) -> Plan:
    tree = ctx.tree("gate15-missing-file")
    with (tree / INDEX).open("a", encoding="utf-8") as handle:
        handle.write("\n[a document that is not here](./fixture-absent.md)\n")
    return _index_plan(ctx, tree)


@fixture(
    gate=15,
    slot=0,
    name="index-entry-without-a-heading",
    defect="an index entry naming a heading the target document does not have",
    expect=(
        "index entry does not resolve: no such heading: "
        "../process/design-review-method.md#no-such-heading",
    ),
    forbid=("governed document is not reachable", "no such file"),
)
def index_entry_without_a_heading(ctx: Context) -> Plan:
    tree = ctx.tree("gate15-missing-heading")
    with (tree / INDEX).open("a", encoding="utf-8") as handle:
        handle.write(
            "\n[a heading that is not there]"
            "(../process/design-review-method.md#no-such-heading)\n"
        )
    return _index_plan(ctx, tree)


# ------------------------------------------------------- the runner itself --
#
# Everything above asserts that a gate's *command* rejects a defect. These
# assert that the *recorder* will not publish a gate as a pass over an
# execution ledger with no command in it. Not that the gate ran: the ledger's
# writer is a top-level function in `run-gates.sh` and the dispatch can call it
# in one line, which review demonstrated. This is the second boundary the
# docstring above used to call unverified anywhere -- narrowed, not closed, and
# the reason AR-0017 exists.
#
# They are not fixtures and are not counted among them. A fixture belongs to a
# gate, executes the argv the runner's own enumeration describes for that gate,
# and is coupled to that enumeration so a gate cannot be added without one.
# These belong to no gate: their subject is `run-gates.sh` and
# `build_report.py`, which no gate's command names. Filing them under some gate
# to reuse the fixture table would put a label on them that is not true, in a
# file whose entire claim is that its labels are not larger than its evidence.
# They are also not controls, which assert something about the tree as it
# stands with nothing planted in it and must exit 0; most of these plant a
# contradiction and require a rejection.
#
# The production command they run is `build_report.py`, invoked exactly as the
# runner invokes it, over gate records and an execution ledger this file writes.
# That is the same shape as every fixture above -- a production command, a
# planted defect, a named diagnostic -- with the artefact under test being the
# report rather than a gate.


@dataclass
class RunnerAssertion:
    """One property of the runner's own reporting, with a planted defect."""

    name: str
    claim: str
    build: object
    expect: tuple = ()
    forbid: tuple = ()
    # False for the assertions whose planted state is legitimate and must be
    # *accepted*. Without at least one of those, everything here would be
    # satisfied by a checker that rejects unconditionally, and the statuses
    # that legitimately require no ledger entry would be collateral.
    must_reject: bool = True
    expect_exit: int | None = None


RUNNER_ASSERTIONS: list[RunnerAssertion] = []


def runner_assertion(name, claim, expect=(), forbid=(), must_reject=True, expect_exit=None):
    def register(function):
        RUNNER_ASSERTIONS.append(
            RunnerAssertion(
                name=name,
                claim=claim,
                build=function,
                expect=tuple(expect),
                forbid=tuple(forbid),
                must_reject=must_reject,
                expect_exit=expect_exit,
            )
        )
        return function

    return register


# build_report.py's own exit code for the finding these assertions are about:
# the report was written and at least one gate record is not supported by the
# execution ledger. Asserted by number, for the reason section 13 gives about
# gate 7's four codes -- the number is the runner's interface to the checker,
# and a rejection with the wrong one is a rejection recorded as the wrong thing.
UNSUPPORTED_RECORD = 3


def _ledger_line(gate: int, label: str, digest: str, code: int, duration: float = 0.5) -> dict:
    return {
        "gate": gate,
        "label": label,
        "argv_sha256": digest,
        "exit_code": code,
        "duration_seconds": duration,
    }


def _gate_record(gate: int, status: str, code: int, steps: list, **overrides) -> dict:
    """A gate record shaped exactly as `record` in run-gates.sh writes one."""
    record = {
        "id": gate,
        "status": status,
        "exit_code": code,
        "duration_seconds": round(sum(step["duration_seconds"] for step in steps), 3),
        "argv_sha256": "+".join(step["argv_sha256"] for step in steps),
        "commands_executed": len(steps),
        "steps": [
            {
                "label": step["label"],
                "argv_sha256": step["argv_sha256"],
                "exit_code": step["exit_code"],
                "duration_seconds": step["duration_seconds"],
            }
            for step in steps
        ],
        "note": "",
        "configuration": "",
        "scripts": [],
        "inputs": [],
    }
    record.update(overrides)
    return record


def _report_plan(ctx: Context, name: str, records: list, ledger: list,
                 ledger_path: str | None = None) -> Plan:
    """Invoke `build_report.py` over records and a ledger written here.

    The command, its environment variable names and the file formats are the
    runner's, not this file's invention: `run-gates.sh` builds every report
    through exactly this call. What is planted is the pair of files, which is
    the only place a claim about what ran can disagree with what ran.
    """
    directory = ctx.scratch(name)
    tree = ctx.tree(name + "-tree")
    records_file = directory / "gates.jsonl"
    records_file.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    executed_file = directory / "executed.jsonl"
    executed_file.write_text(
        "".join(json.dumps(entry) + "\n" for entry in ledger), encoding="utf-8"
    )
    report = directory / "gate-report.json"
    env = {
        "GATE_RECORDS": str(records_file),
        "GATE_EXECUTED": str(executed_file) if ledger_path is None else ledger_path,
        "GATE_REPORT": str(report),
        "GATE_ORDER_LIST": " ".join(str(record["id"]) for record in records),
        "GATE_MANIFEST": str(ctx.root / "config" / "quality-tools.json"),
        "GATE_ROOT": str(tree),
        "GATE_WORKTREE": str(tree),
        "GATE_COMMIT": ctx.pristine_base,
        "GATE_BASE_COMMIT": ctx.pristine_base,
    }
    plan = Plan(
        ["python3", str(ctx.root / "tools" / "quality" / "build_report.py")],
        directory,
        env,
    )
    # The report is part of what this command produced, and some of these
    # assertions are about what must *not* be in it. Searched alongside the
    # command's output.
    plan.artefact = report
    return plan


# The honest shape every assertion below deviates from by exactly one thing.
# Five gates, covering every status the report defines: a pass over one command
# that returned 0, a fail over one that returned 101, a `not_armed` over two
# commands the second of which returned 3, and the two statuses that require no
# entry at all.
def _honest_records() -> tuple:
    gate1 = [_ledger_line(1, "gate 1 (formatting)", "d1", 0)]
    gate2 = [_ledger_line(2, "gate 2 (lints)", "d2", 101)]
    gate7 = [
        _ledger_line(7, "gate 7 (coverage measurement)", "d7a", 0),
        _ledger_line(7, "gate 7 (floor check)", "d7b", 3),
    ]
    records = [
        _gate_record(1, "pass", 0, gate1),
        _gate_record(2, "fail", 101, gate2),
        _gate_record(7, "not_armed", 3, gate7),
        _gate_record(8, "not_applicable", 0, []),
        _gate_record(13, "not_implemented", 0, []),
    ]
    return records, gate1 + gate2 + gate7


@runner_assertion(
    name="honest-records-are-accepted",
    claim=(
        "a report whose records agree with the execution ledger is published "
        "unchanged, `not_armed`, `not_applicable` and `not_implemented` included"
    ),
    expect=('"status": "not_armed"',
            '"status": "not_applicable"',
            '"status": "not_implemented"',
            '"status": "pass"'),
    forbid=("unsupported",),
    must_reject=False,
    expect_exit=0,
)
def honest_records_are_accepted(ctx: Context) -> Plan:
    # The assertion that keeps the six below honest. Every one of them requires
    # a rejection, and a check that rejected everything would satisfy all six
    # while destroying the three statuses that legitimately need no passing
    # entry -- which AR-0017's acceptance criteria protect by name, and whose
    # value gate 7 spent this milestone demonstrating.
    records, ledger = _honest_records()
    return _report_plan(ctx, "runner-honest", records, ledger)


@runner_assertion(
    name="pass-with-no-command-executed",
    claim="a gate recorded `pass` with no command in the ledger is not published as a pass",
    expect=(
        "gate 2 (lints) recorded a status the execution ledger does not support",
        "the ledger holds no entry for this gate",
        '"status": "unsupported"',
        '"status_recorded_by_the_runner": "pass"',
        "no command was executed for this gate",
    ),
    expect_exit=UNSUPPORTED_RECORD,
)
def pass_with_no_command_executed(ctx: Context) -> Plan:
    # The naive form of the demonstrated attack: an arm of the dispatch
    # short-circuited to `gate_status="pass"` with no `run_step` call. Before
    # AR-0017 this produced a record indistinguishable from an honest pass.
    return _report_plan(ctx, "runner-no-command", [_gate_record(2, "pass", 0, [])], [])


@runner_assertion(
    name="pass-with-a-forged-argv-digest",
    claim=(
        "a gate recorded `pass` with a plausible argv digest and no command in "
        "the ledger is not published as a pass, and the forged digest is not "
        "republished"
    ),
    expect=(
        "gate 2 (lints) recorded a status the execution ledger does not support",
        "the ledger holds no entry for this gate",
        "no command was executed for this gate",
    ),
    # The digest itself. This is the demonstrated attack rather than the naive
    # one: the reviewer's sabotage computed `argv_digest` over the honest argv
    # without running it, so `argv_sha256` was present, well-formed and
    # meaningless. A report that repeated it beside the finding that it is
    # unfounded would have put the forgery back into the artefact.
    forbid=("f" * 64,),
    expect_exit=UNSUPPORTED_RECORD,
)
def pass_with_a_forged_argv_digest(ctx: Context) -> Plan:
    forged = _gate_record(2, "pass", 0, [])
    forged["argv_sha256"] = "f" * 64
    forged["commands_executed"] = 1
    return _report_plan(ctx, "runner-forged-digest", [forged], [])


@runner_assertion(
    name="pass-over-a-command-that-failed",
    claim=(
        "a gate recorded `pass` over a command the executor saw exit non-zero is "
        "not published as a pass"
    ),
    expect=(
        "gate 2 (lints) recorded a status the execution ledger does not support",
        "a pass needs every entry for the gate to be 0",
    ),
    expect_exit=UNSUPPORTED_RECORD,
)
def pass_over_a_command_that_failed(ctx: Context) -> Plan:
    # The other half of the same property, and the reason the ledger carries an
    # exit status at all. A dispatch that runs the command and then ignores what
    # it returned is a skipped gate wearing a duration.
    steps = [_ledger_line(2, "gate 2 (lints)", "d2", 101)]
    return _report_plan(ctx, "runner-ignored-exit",
                        [_gate_record(2, "pass", 0, steps)], steps)


@runner_assertion(
    name="not-armed-with-exit-zero",
    claim="a gate recorded `not_armed` at exit 0 is not published at all",
    expect=(
        "gate 7 (coverage floors) recorded a status the execution ledger does not support",
        "an unarmed gate is an entry saying so, not a status chosen for it",
    ),
    expect_exit=UNSUPPORTED_RECORD,
)
def not_armed_with_exit_zero(ctx: Context) -> Plan:
    # AR-0009 established that nothing automated would notice a `not_armed` gate
    # that had started exiting 0, and gate 7 was then observed being *upgraded*
    # from a blocking `not_armed` to `pass` by being disabled. A half-armed gate
    # reporting a pass is the same silent pass as a skipped one, so `not_armed`
    # is held to the same rule: an entry exists, and the code it carries is the
    # one recorded.
    steps = [
        _ledger_line(7, "gate 7 (coverage measurement)", "d7a", 0),
        _ledger_line(7, "gate 7 (floor check)", "d7b", 0),
    ]
    return _report_plan(ctx, "runner-not-armed-zero",
                        [_gate_record(7, "not_armed", 0, steps)], steps)


@runner_assertion(
    name="not-applicable-over-an-executed-command",
    claim="a status requiring no ledger entry is not published over a ledger that holds one",
    expect=(
        "gate 8 (workflow linting) recorded a status the execution ledger does not support",
        "which requires no ledger entry, while the ledger holds 1 for this gate",
    ),
    expect_exit=UNSUPPORTED_RECORD,
)
def not_applicable_over_an_executed_command(ctx: Context) -> Plan:
    # The mirror of the two above, and not a hypothetical: `not_applicable`,
    # `not_implemented` and `not_armed` were once believed to *be* the check
    # that a gate had not run. They are claims like any other, and this is the
    # direction in which one of them can hide a command that ran and failed.
    steps = [_ledger_line(8, "gate 8 (actionlint)", "d8", 1)]
    return _report_plan(ctx, "runner-na-over-command",
                        [_gate_record(8, "not_applicable", 0, [])], steps)


@runner_assertion(
    name="an-absent-ledger-supports-nothing",
    claim="a report built without an execution ledger publishes no gate as a pass",
    expect=(
        "the execution ledger could not be read",
        "which is the fail-closed reading and not a reason to accept the claims",
        "gate 1 (formatting) recorded a status the execution ledger does not support",
    ),
    expect_exit=UNSUPPORTED_RECORD,
)
def an_absent_ledger_supports_nothing(ctx: Context) -> Plan:
    # Deleting the evidence must not be easier than forging it. Every gate
    # claiming to have run something goes unsupported, which is the loudest
    # state this report has.
    records, ledger = _honest_records()
    directory = ctx.scratch("runner-absent-ledger")
    return _report_plan(ctx, "runner-absent-ledger-inner", records, ledger,
                        ledger_path=str(directory / "no-such-ledger.jsonl"))


def _digest_assertion(ctx: Context, name: str, argv: list) -> Plan:
    directory = ctx.scratch(name)
    return Plan(
        [str(ctx.root / "tools" / "quality" / "run-gates.sh"), "--argv-digest", *argv],
        directory,
        {},
    )


# Computed here with hashlib rather than copied from a run of the thing under
# test: the assertion is that the runner joins its arguments with a NUL, and an
# expected value taken from the runner would assert only that it is consistent
# with itself.
_NUL_JOINED_A_B = hashlib.sha256(b"a\0b\0").hexdigest()
_CONCATENATED_AB = hashlib.sha256(b"ab\0").hexdigest()


@runner_assertion(
    name="argv-digest-separates-two-arguments",
    claim="the digest of argv `a b` is the digest of the arguments joined by NUL",
    expect=(_NUL_JOINED_A_B,),
    forbid=(_CONCATENATED_AB,),
    must_reject=False,
    expect_exit=0,
)
def argv_digest_separates_two_arguments(ctx: Context) -> Plan:
    # `argv_digest` built its input in a shell variable -- `out+="$part"$'\0'`
    # -- and a bash variable cannot hold a NUL, so the separator was dropped and
    # the digest was taken over the arguments concatenated with nothing between
    # them: argv `a b` and argv `ab` hashed identically. Two argv that differ
    # only in where the boundaries fall were indistinguishable, in the one field
    # this report carries as evidence of what was run, in a milestone that has
    # used that field throughout to establish that two people ran the same
    # thing.
    return _digest_assertion(ctx, "argv-digest-two", ["a", "b"])


@runner_assertion(
    name="argv-digest-separates-one-argument",
    claim="the digest of argv `ab` is not the digest of argv `a b`",
    expect=(_CONCATENATED_AB,),
    forbid=(_NUL_JOINED_A_B,),
    must_reject=False,
    expect_exit=0,
)
def argv_digest_separates_one_argument(ctx: Context) -> Plan:
    # The other side of the collision, so that the pair proves two *different*
    # argv give two different digests rather than one of them giving a
    # particular digest.
    return _digest_assertion(ctx, "argv-digest-one", ["ab"])


# The digest of an argv holding the repository root, with the root folded out.
# Computed here, so that the assertion is about the substitution having happened
# and not about the runner agreeing with itself.
_REPO_FOLDED = hashlib.sha256(b"<repo>/x\0").hexdigest()


@runner_assertion(
    name="argv-digest-folds-out-the-repository-root",
    claim=(
        "`--argv-digest` folds out the repository root exactly as a live run "
        "does, so a published digest is recomputable from outside the runner"
    ),
    expect=(_REPO_FOLDED,),
    must_reject=False,
    expect_exit=0,
)
def argv_digest_folds_out_the_repository_root(ctx: Context) -> Plan:
    # `--argv-digest` is answered inside the argument loop, and `root` used to be
    # assigned below it, so this substitution was skipped there while a live run
    # applied it: the same argv could get two digests depending on which side of
    # the loop asked. That made the digest not recomputable from outside, which
    # is the only thing the option exists for. It was harmless in fact -- no
    # gate's argv holds the repository root -- and the fact that made it
    # harmless was not one anything checked, which is the shape this suite
    # exists to refuse. Asserted against a value computed here rather than taken
    # from the runner, so the runner agreeing with itself cannot satisfy it.
    directory = ctx.scratch("argv-digest-repo-root")
    return Plan(
        [str(ctx.root / "tools" / "quality" / "run-gates.sh"),
         "--argv-digest", f"{ctx.root}/x"],
        directory,
        {},
    )


def evaluate_runner(assertion: RunnerAssertion, plan: Plan, paths: dict) -> dict:
    """Run one runner assertion and say precisely why it did or did not hold."""
    outcome = {
        "fixture": assertion.name,
        "kind": "runner",
        "defect": assertion.claim,
        "argv": [scrub(part, paths) for part in plan.argv],
        "expected": list(assertion.expect),
        "expected_exit": assertion.expect_exit,
        "forbidden": list(assertion.forbid),
        "must_reject": assertion.must_reject,
        "status": "pass",
        "reasons": [],
    }
    result = run(plan.argv, cwd=plan.cwd, env=plan.env)
    outcome["exit_code"] = result.exit_code
    # The report is part of what the command produced, and several of these
    # assertions are about what must and must not be in it.
    text = result.output
    artefact = getattr(plan, "artefact", None)
    if artefact is not None and Path(artefact).is_file():
        text += "\n" + Path(artefact).read_text(encoding="utf-8")
    if assertion.must_reject and result.exit_code == 0:
        outcome["reasons"].append(
            "the production command accepted the planted contradiction (exit 0)"
        )
    if not assertion.must_reject and result.exit_code != 0:
        outcome["reasons"].append(
            f"the production command rejected a legitimate report (exit {result.exit_code})"
        )
    if assertion.expect_exit is not None and result.exit_code != assertion.expect_exit:
        outcome["reasons"].append(
            f"exited {result.exit_code}, not the exit {assertion.expect_exit} this "
            "path is required to return"
        )
    for wanted in assertion.expect:
        if wanted not in text:
            outcome["reasons"].append(f"expected diagnostic not found: {wanted!r}")
    for unwanted in assertion.forbid:
        if unwanted in text:
            outcome["reasons"].append(
                f"the wrong outcome: {unwanted!r} appeared in the output or the report"
            )
    if outcome["reasons"]:
        outcome["status"] = "fail"
        outcome["output"] = excerpt(text, paths)
    return outcome


# ----------------------------------------------------------------- controls --
#
# A control asserts something about the tree as it stands, with nothing planted
# in it, and must exit 0. Section 13 requires the gate 15 one by name and says
# in the same breath that a control is not a negative fixture and is not counted
# among them -- so a control can be added without moving the fixture total,
# which is why the second one below is a control and not a twenty-eighth
# fixture. Not optional either: a checker that fails on a correct tree is as
# useless as one that never fails, and the gate 15 control is what found one of
# AR-0010's own fixtures written wrong.


def index_control(ctx: Context) -> Plan:
    tree = ctx.tree("gate15-control")
    git(tree, "add", "-A")
    env, argv = ctx.command(15, 0)
    return Plan(argv, tree, env)


# docs/QUALITY_GATES.md's statement of what a coverage figure is and is not.
# AR-0009 asked for it to be asserted, and it is worth asserting for the reason
# this suite exists: the sentence is prose, no gate reads it, and deleting the
# one line that stops a percentage being read as evidence of correctness would
# turn nothing red anywhere. This is the assertion.
#
# What is asserted is the bolded proposition and not the paragraph around it.
# That paragraph is worded one way before AR-0009 and another after -- AR-0009
# extends the sentence with `it is a regression constraint on what the suite
# touches` -- and a control that goes red on a rewording is a control the next
# author deletes rather than reads. The proposition itself is unchanged across
# both, which is the part that carries the claim.
COVERAGE_IS_NOT_CORRECTNESS = "**coverage establishes exercised lines, not correctness**"


def coverage_meaning_control(ctx: Context) -> Plan:
    """Assert the document still says a coverage number is not correctness."""
    directory = ctx.scratch("gate07-coverage-meaning")
    document = (ctx.root / "docs" / "QUALITY_GATES.md")
    if not document.is_file():
        raise Environment("no docs/QUALITY_GATES.md in this revision")
    # Collapsed into the throwaway directory first, so that the assertion is
    # about the text and not about where the paragraph happens to wrap: the
    # proposition sits on one line today and a reflow could put it across two,
    # which is not a change to the document's meaning and must not read as one.
    collapsed = directory / "QUALITY_GATES.collapsed.md"
    collapsed.write_text(
        " ".join(document.read_text(encoding="utf-8").split()) + "\n",
        encoding="utf-8",
    )
    return Plan(
        ["grep", "-F", "-q", "-e", COVERAGE_IS_NOT_CORRECTNESS, str(collapsed)],
        directory,
    )


# ------------------------------------- the claim this milestone keeps making --
#
# One distinction in this repository has now been stated wrongly nine times
# across four review rounds, in five files, by an author who had just corrected
# it elsewhere in the same commit. The rule the report enforces is over *ledger
# entries*; the wrong form asserts it over *executions* -- that a pass means a
# command ran. It is wrong because the ledger's writer is a top-level function
# in `run-gates.sh` that the dispatch can call in one line, which review
# demonstrated by doing it.
#
# Nine occurrences is not nine lapses of attention. It is a missing check: no
# gate made a document change when the code it described changed, which is the
# failure mode this whole milestone exists to prevent, reproduced inside the
# tooling that exists to prevent it. So the distinction gets a check rather
# than another round of care.
#
# Read over whitespace-flattened text, because the eighth occurrence hid from a
# line-based grep by wrapping at column 79 -- `a command that` ended one line
# and `ran and returned 0` began the next, and the pattern that would have
# caught it did not, for no reason anyone could have predicted from reading it.

EXECUTION_CLAIM_FILES = (
    "tools/quality/run-gates.sh",
    "tools/quality/build_report.py",
    "tools/quality/test_failure_paths.py",
    "docs/QUALITY_GATES.md",
)


# Words whose presence near a match means the sentence is doing one of the three
# legitimate things: stating the rule over entries, scoping the claim to what it
# really covers, or quoting the wrong form in order to name it as wrong.
#
# This list is why the control fails on the *claim* rather than on a string. It
# is also its main weakness, and the weakness runs toward silence: a sentence
# that makes the unqualified claim while happening to contain one of these words
# is passed. `entries` and `arguments` are the loosest, and they are here because
# the corrected text uses them constantly. Tightening them produced false
# positives on honest paragraphs, which is the failure that gets a control
# deleted rather than read.
EXECUTION_CLAIM_QUALIFIERS = (
    "ledger entry", "ledger entries", "no entry", "entries", "an entry",
    "forged", "forgery", "record_execution",
    "honest path", "honest tree", "on an honest", "property of the tree",
    "through `record`", "through this function", "parameters of it",
    "does not pass them",
    "not established", "narrowed", "not closed",
    "earlier version", "previous wording", "was wrong", "was false",
    "never measured", "got wrong", "nine times",
)
# Two words were in that list and are not any more, each having excused the
# occurrence it was standing next to:
#
#   used to     excused the flagship banner, whose sentence ended `...the second
#               boundary this file's docstring used to name as unverified
#               anywhere`. The phrase was about the docstring's history and had
#               nothing to do with scoping the claim in front of it.
#   arguments   excused the comment above `record()`, whose sentence went on to
#               say the digest and duration `are no longer arguments to this
#               function`. That is a qualification, but the word alone is far
#               too common to stand for one.
#
# Both were added to keep a corrected passage from firing, and both turned out
# to be doing that job for a passage that had another qualifier anyway. Removing
# them costs nothing and was checked: zero false positives on the honest tree.
#
# This list is where the check gets switched off by a word that looks harmless,
# and a reading will not find it. Only running the control over the real text of
# a real occurrence does. How that is done, and what it has caught, is in
# `unqualified_execution_claims`.
# `no longer` was in that list for one round and silently disabled the
# `digest-or-duration-impossible` shape, because `can no longer` is part of the
# claim rather than of its qualification: every match carried its own excuse.
# Caught by testing the control against the nine occurrences it exists for --
# two of them walked straight through it. A qualifier list is a place where a
# check can be switched off by a plausible-looking edit, which is the shape this
# suite exists to distrust, so the test below is the thing keeping it honest and
# not this comment.

# A qualifier counts only inside the sentence that makes the claim.
#
# The first version of this control used a 320-character window either side, and
# it passed with the ninth occurrence put straight back: the corrected paragraph
# opens `A pass must have a ledger entry behind it. On the honest path ...`, so
# restoring the bad title left `honest path` sitting one sentence away, and the
# claim was excused by its own neighbour. A control that goes green on the defect
# it was written for is worse than no control, so the scope is the sentence now.
# Every corrected passage in this repository qualifies inside the sentence that
# makes the claim, which is what made the tighter rule available -- and it is the
# better writing rule anyway, because a reader who stops at the full stop must
# not have been misled by what came before it.
#
# Every shape below is period-free by construction, so a match cannot straddle a
# sentence boundary and its containing sentence is always well defined.
SENTENCE_BREAK = re.compile(r"""(?<=[.!?])[)\]"'`*]*\s+""")


# Each entry is a shape, not a sentence: a regular expression over flattened
# text, with the wording it is meant to catch described rather than quoted --
# quoting it here would trip the control on its own definition.
#
# The `why` is printed when a claim fires, because a bare regex name tells the
# next author nothing about what to write instead.
EXECUTION_CLAIMS = (
    (
        "pass-implies-execution",
        r"\bpass(?:ing)?\b[^.]{0,70}?\bwithout\b[^.]{0,45}?\b(?:run|ran|running)\b",
        "asserts that a pass implies execution. What the recorder checks is the "
        "ledger, and an arm of the dispatch can write the ledger. Say what the "
        "recorder checks.",
    ),
    (
        "pass-needs-a-command",
        r"\bpass[`'\"]?\s*(?:needs|requires)\b[^.]{0,60}?\bcommand that ran\b",
        "states the rule over executions. The rule is: at least one entry for the "
        "gate, every entry at 0.",
    ),
    (
        "pass-has-a-command-behind-it",
        r"\bpass\b[^.]{0,25}?\b(?:must have|has|have)\s+a\s+command\s+behind\s+it",
        "the same rule as a slogan. A pass has a ledger entry behind it.",
    ),
    (
        "stable-digest-changes",
        r"\bstops running\b[^.]{0,90}?\bchanges?\b[^.]{0,35}?\bdigest\b",
        "the stable digest changes only when no ledger entry is forged for the "
        "gate. Unqualified, this is the claim a forged entry falsifies.",
    ),
    (
        "stable-digest-changes-reversed",
        r"\bchanges?\b[^.]{0,45}?\bwhen a gate stops running\b",
        "the same claim with the clauses the other way round, which the shape "
        "above requires in one order and therefore missed for a whole round.",
    ),
    (
        "a-command-ran",
        r"\b(?:no|at least one|every|one|a) commands? ran\b|\bnothing ran\b",
        "a bare count of what ran. The recorder can count entries and nothing "
        "else; say entries.",
    ),
    (
        "ledger-is-evidence-of-execution",
        r"\bevidence\b[^.]{0,70}?\b(?:a gate|the gate|command)\b[^.]{0,45}?\b(?:ran|executed)\b",
        "the ledger is the runner's claim about what ran, not evidence that "
        "anything did; its writer is callable from the dispatch.",
    ),
    (
        "ledger-sole-writer",
        r"\b(?:only thing that appends|only writer|from nothing else)\b",
        "true of an honest tree, false as an invariant: the ledger writer is "
        "top-level and callable from the dispatch. Say which of the two you mean.",
    ),
    (
        "digest-or-duration-impossible",
        r"\bcan no longer\b[^.]{0,90}?\b(?:digest|duration)\b",
        "scope it to the recorder's parameters. Publishing either is still "
        "possible through the ledger writer.",
    ),
    (
        "commands-executed-asserted",
        r"\bcommand\(s\) executed\b",
        "the report cannot establish that a command executed, and says so in "
        "stable_digest_note. Report what the ledger records.",
    ),
)


COMMENT_LEAD = re.compile(r"^[ \t]*(?:#+|//+|\*)[ \t]?")


def flatten_for_claims(text: str) -> str:
    """Collapse whitespace and line-leading comment markers into running prose.

    Whitespace first, because a claim that wrapped at column 79 hid from a
    line-based grep for a whole round. The comment markers came second and for
    the same kind of reason: a two-word qualifier split across two comment lines
    reads as `previous # wording` once the lines are joined, so it matched
    nothing, and the historical quotation it was excusing fired as though it were
    a live claim. Both are the same defect -- a check whose answer depends on
    where a line happens to break.
    """
    lines = [COMMENT_LEAD.sub("", line) for line in text.splitlines()]
    return " ".join(" ".join(lines).split())


def sentence_spans(flat: str) -> list:
    """(start, end) of every sentence in flattened text."""
    spans, start = [], 0
    for break_match in SENTENCE_BREAK.finditer(flat):
        spans.append((start, break_match.start()))
        start = break_match.end()
    spans.append((start, len(flat)))
    return spans


# An upper bound on the sentence, in characters either side of the match.
#
# Sentence splitting is reliable in prose and unreliable in code, which has few
# full stops: a "sentence" inside a Python function can run from the end of the
# docstring to the next string literal that happens to contain one, hundreds of
# characters away. Two of the round-five sites were excused exactly that way, by
# `on an honest` sitting in a docstring far above the finding it excused. So the
# qualifier must be in the same sentence AND within this many characters. The
# bound only ever tightens the rule -- it can excuse nothing the sentence rule
# would have refused.
EXECUTION_CLAIM_REACH = 200


def containing_sentence(spans: list, flat: str, match) -> str:
    lower = max(0, match.start() - EXECUTION_CLAIM_REACH)
    upper = min(len(flat), match.end() + EXECUTION_CLAIM_REACH)
    for start, end in spans:
        if start <= match.start() < end:
            return flat[max(start, lower):min(end, upper)]
    return flat[match.start():match.end()]


def unqualified_execution_claims(root: Path) -> list:
    """Every place the execution claim is made without the scope that makes it true.

    Be exact about what this establishes, because a control that reads as proof
    of absence would be this task's own defect one level up.

    What it does. It fails when one of the shapes in `EXECUTION_CLAIMS` appears
    with no qualifying word in the same sentence. Those shapes are the occurrences
    actually found, generalised: each is a pattern with the specifics left open,
    so rewording within the shape -- a different verb, a clause inserted, a line
    wrapped anywhere -- still fires. That is a regression guard on a mistake this
    repository has demonstrably made nine times, which is the case for having it.

    What it does not. It cannot establish that the claim is absent. A new way of
    saying the same wrong thing, in words none of these shapes cover, passes
    silently, and no pattern list can close that -- the space of English
    paraphrase is not enumerable. It also passes any sentence containing one of
    the qualifying words for an unrelated reason, which is a deliberate choice in
    the direction of silence, because a control with false positives on honest
    prose gets deleted rather than read. And it reads only the four files listed;
    the same claim in a commit message, a task file or a plan is invisible to it.

    The qualifier must sit in the same sentence as the claim. That is tighter
    than this started -- a character window let a qualification one sentence away
    excuse a claim, and the control went green with the ninth occurrence put
    straight back, which is the failure it exists to prevent happening to it.

    So: this narrows a known and repeated failure. It is not a proof, and a
    reviewer who wants to know whether the documents are honest still has to
    read them.

    How this is verified, because the control has now been wrong four times and
    every one was found the same way. Take the text of an occurrence *from the
    file it was in*, at the commit before it was corrected, and run the control
    over the whole file. Not over a hand-typed excerpt: sentence boundaries and
    qualifier reach both depend on surrounding text, and an excerpt fires where
    the file does not. Not over a description of the wording either -- the
    control was once reported as catching all nine occurrences when it had been
    tested against a *paraphrase* of the flagship one, which fired, while the
    sentence actually in the file did not. Four defects found this way:

      * `no longer` in the qualifier list, which is part of the claim rather
        than of its qualification, so two occurrences carried their own excuse.
      * a 320-character qualifier window, which let a neighbouring sentence
        excuse a claim; the rule is the sentence now.
      * the sentence rule alone, which is unreliable in code: source has few full
        stops, so a `sentence` ran from a docstring to a string literal hundreds
        of characters away and picked up a qualifier there. Hence
        `EXECUTION_CLAIM_REACH`, which can only tighten.
      * flattening that kept line-leading comment markers, so a two-word
        qualifier split across two comment lines read as `previous # wording`
        and matched nothing, and a historical quotation fired as a live claim.

    The standing check is: every occurrence named in review fires, taken verbatim
    from the commit where it was corrected, and the honest tree reports none.

    One self-reference to declare. This file is one of the four it scans, and the
    shapes above are stored here as text. Each pattern's own source is removed
    from this file's flattened copy before matching, or every pattern would find
    itself. The hole that leaves is exactly one: prose that is character-for-
    character identical to a declared pattern's source, which is a regular
    expression and not a sentence anybody would write by accident.
    """
    findings = []
    for relative in EXECUTION_CLAIM_FILES:
        path = root / relative
        if not path.is_file():
            raise Environment(
                f"the execution-claim control expects {relative} in this revision "
                "and it is not there; a control that cannot read its subject is not "
                "a control that passed"
            )
        flat = flatten_for_claims(path.read_text(encoding="utf-8"))
        if relative == "tools/quality/test_failure_paths.py":
            for _name, pattern, _why in EXECUTION_CLAIMS:
                flat = flat.replace(pattern, " ")
        sentences = sentence_spans(flat)
        for name, pattern, why in EXECUTION_CLAIMS:
            for match in re.finditer(pattern, flat, re.IGNORECASE):
                sentence = containing_sentence(sentences, flat, match).lower()
                if any(word in sentence for word in EXECUTION_CLAIM_QUALIFIERS):
                    continue
                quoted = flat[max(0, match.start() - 60):match.end() + 60]
                findings.append(
                    f"{relative}: [{name}] ...{quoted}...\n"
                    f"    why this is wrong: {why}"
                )
    return findings


def execution_claim_control(ctx: Context) -> Plan:
    """Assert the unqualified execution claim is not in the quality tooling."""
    directory = ctx.scratch("execution-claim")
    findings = unqualified_execution_claims(ctx.root)
    report = directory / "unqualified-execution-claims.txt"
    report.write_text(
        "".join(finding + "\n\n" for finding in findings), encoding="utf-8"
    )
    # The assertion is the command, and the file it reads is the scan's output,
    # so a reader of the JSON table sees both what was asserted and where the
    # evidence for it went. An empty file is the pass.
    return Plan(
        [
            "python3",
            "-c",
            "import sys; text = open(sys.argv[1], encoding='utf-8').read(); "
            "sys.stdout.write(text or 'no unqualified execution claims\\n'); "
            "sys.exit(1 if text.strip() else 0)",
            str(report),
        ],
        directory,
    )


# Each control carries the sentence to print when it fails, because the controls
# no longer all assert the same kind of thing and one shared message would have
# to be vague enough to fit both.
CONTROLS = [
    (
        7,
        "coverage-is-not-correctness",
        "docs/QUALITY_GATES.md must still say that coverage measures exercised "
        "lines and not correctness",
        coverage_meaning_control,
        "the sentence establishing that a coverage floor is a regression "
        "constraint and not evidence of correctness is no longer in "
        "docs/QUALITY_GATES.md",
    ),
    (
        15,
        "unmodified-tree-passes",
        "the architecture index over an unmodified tree must exit 0",
        index_control,
        "the checker rejected a tree with no planted defect",
    ),
    (
        13,
        "no-unqualified-execution-claim",
        "the quality tooling must not state the pass rule over executions",
        execution_claim_control,
        "a document or comment states the execution claim without the scope that "
        "makes it true; the lines are printed above, each with what to say "
        "instead. This is the mistake this repository has made nine times across "
        "four review rounds, which is why it is checked rather than watched for",
    ),
]


# -------------------------------------------------------------- the harness --


def scrub(text: str, paths: dict) -> str:
    for value, name in paths.items():
        if value:
            text = text.replace(str(value), name)
    return text


def excerpt(text: str, paths: dict, limit: int = 24) -> str:
    lines = [line for line in scrub(text, paths).splitlines() if line.strip()]
    kept = lines[:limit]
    if len(lines) > limit:
        kept.append(f"... {len(lines) - limit} further line(s) not shown")
    return "\n".join("    | " + line for line in kept)


def check_enumeration(gates: dict, fixtures: list, dropped: int | None) -> list:
    """The coupling: gates and fixtures must account for each other exactly."""
    failures = []
    covered = {fixture.gate for fixture in fixtures}
    for gate_id in sorted(gates):
        if gate_id not in covered:
            failures.append(
                f"gate {gate_id} ({gates[gate_id]['name']}) is enumerated by the "
                "runner but has no fixture"
            )
    for gate_id in sorted(covered - set(gates)):
        names = ", ".join(f.name for f in fixtures if f.gate == gate_id)
        failures.append(
            f"fixture(s) {names} name gate {gate_id}, which the runner does not "
            "enumerate; a gate was removed or renumbered and its fixtures were left behind"
        )
    for gate_id in sorted(set(gates) & covered):
        parts = command_parts(gates[gate_id])
        slots = {f.slot for f in fixtures if f.gate == gate_id}
        for slot in range(len(parts)):
            if slot in slots:
                continue
            reason = KNOWN_UNCOVERED.get((gate_id, slot))
            if reason is None:
                failures.append(
                    f"gate {gate_id} runs {len(parts)} tools and tool {slot + 1} "
                    "has no fixture; a defect that stops one tool proves nothing "
                    "about the next"
                )
        for slot in sorted(slots):
            if slot >= len(parts):
                failures.append(
                    f"gate {gate_id} fixture claims tool {slot + 1} but the gate "
                    f"now runs {len(parts)}"
                )
    if dropped is not None and not failures:
        failures.append(
            f"{DROP_GATE}={dropped} removed every fixture for that gate and the "
            "enumeration check still found nothing; the coupling this suite rests "
            "on is not working"
        )
    return failures


def evaluate(fixture: Fixture, plan: Plan, paths: dict) -> dict:
    outcome = {
        "fixture": fixture.name,
        "gate": fixture.gate,
        "tool": fixture.slot + 1,
        "defect": fixture.defect,
        "argv": [scrub(part, paths) for part in plan.argv],
        "note": plan.note,
        "expected": list(fixture.expect),
        "expected_exit": fixture.expect_exit,
        "forbidden": list(fixture.forbid),
        "status": "pass",
        "reasons": [],
    }
    if plan.pending:
        # Never a pass. The command is not run, the fixture is reported as
        # pending on every run and in the JSON, and the tally counts it apart
        # from the fixtures that were exercised.
        outcome["status"] = "pending"
        outcome["exit_code"] = None
        outcome["pending"] = plan.pending
        return outcome
    result = run(plan.argv, cwd=plan.cwd, env=plan.env)
    outcome["exit_code"] = result.exit_code
    if result.exit_code == 0:
        outcome["reasons"].append(
            "the production command accepted the planted defect (exit 0)"
        )
    elif fixture.expect_exit is not None and result.exit_code != fixture.expect_exit:
        outcome["reasons"].append(
            f"rejected with exit {result.exit_code}, not the exit "
            f"{fixture.expect_exit} this path is required to return"
        )
    for wanted in fixture.expect:
        if wanted not in result.output:
            outcome["reasons"].append(f"expected diagnostic not found: {wanted!r}")
    for unwanted in list(fixture.forbid) + list(getattr(plan, "forbid_extra", [])):
        if unwanted in result.output:
            outcome["reasons"].append(
                f"rejected for the wrong reason: {unwanted!r} appeared in the output"
            )
    if plan.control is not None:
        control = run(plan.control.argv, cwd=plan.control.cwd, env=plan.control.env)
        outcome["control_exit_code"] = control.exit_code
        if control.exit_code != 0:
            outcome["reasons"].append(
                "the control -- the same command with the defect removed -- did not "
                f"pass (exit {control.exit_code}), so the rejection above is not "
                "evidence about the defect"
            )
    if outcome["reasons"]:
        outcome["status"] = "fail"
        outcome["output"] = excerpt(result.output, paths)
    return outcome


def list_fixtures(gates: dict, bin_dir=None) -> None:
    print(f"{len(FIXTURES)} fixtures across {len({f.gate for f in FIXTURES})} gates")
    for fixture in sorted(FIXTURES, key=lambda f: (f.gate, f.slot, f.name)):
        name = gates.get(fixture.gate, {}).get("name", "not enumerated by the runner")
        print(f"  gate {fixture.gate:>2} tool {fixture.slot + 1} [{name}] "
              f"{fixture.name}: {fixture.defect}")
    for gate_id, name, description, _build, _failure in CONTROLS:
        print(f"  gate {gate_id:>2} CONTROL (not one of the fixtures above) "
              f"{name}: {description}")
    for assertion in RUNNER_ASSERTIONS:
        print(f"     RUNNER (belongs to no gate, not one of the fixtures above) "
              f"{assertion.name}: {assertion.claim}")
    for (gate_id, slot), reason in sorted(KNOWN_UNCOVERED.items()):
        print(f"  gate {gate_id:>2} tool {slot + 1} NOT EXERCISED: {reason}")
    for gate_id, reason in UNSEEN_STEPS:
        print(f"  gate {gate_id:>2} NOT VISIBLE IN THE ENUMERATION: {reason}")
    for gate_id, reason in inert_capabilities(bin_dir):
        print(f"  gate {gate_id:>2} INERT ON THIS HOST: {reason}")
    for gate_id, flag, reason in NON_LOAD_BEARING:
        print(f"  gate {gate_id:>2} FLAG NOT LOAD BEARING [{flag}]: {reason}")
    for gate_id, path, why in OUT_OF_SCOPE:
        print(f"  gate {gate_id:>2} DELIBERATELY OUT OF SCOPE [{path}]: {why}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Plant a defect per gate and require the production command to reject it."
    )
    parser.add_argument("--bin-dir", help="directory holding the external analyzers")
    parser.add_argument("--json", dest="json_path", help="write the outcome table here")
    parser.add_argument("--list", action="store_true", help="list the fixtures and exit")
    parser.add_argument("--only", action="append", default=[],
                        help="run only the named fixture (repeatable; development aid)")
    args = parser.parse_args(argv)

    root = repo_root()
    gates = {gate["id"]: gate for gate in enumerated_gates(root)}
    print(f"fixtures: the runner enumerates {len(gates)} gates: "
          f"{', '.join(str(i) for i in sorted(gates))}")

    if args.list:
        list_fixtures(gates, args.bin_dir)
        return 0

    dropped = None
    if os.environ.get(DROP_GATE):
        try:
            dropped = int(os.environ[DROP_GATE])
        except ValueError:
            raise Environment(f"{DROP_GATE} is not a gate number")

    fixtures = [f for f in FIXTURES if f.gate != dropped]
    if args.only:
        fixtures = [f for f in fixtures if f.name in args.only]

    enumeration = check_enumeration(gates, fixtures, dropped)
    for failure in enumeration:
        print(f"FAIL enumeration: {failure}", file=sys.stderr)
    for (gate_id, slot), reason in sorted(KNOWN_UNCOVERED.items()):
        print(f"fixtures: gate {gate_id} tool {slot + 1} is not exercised: {reason}")
    for gate_id, reason in UNSEEN_STEPS:
        print(f"fixtures: gate {gate_id} not visible in the enumeration: {reason}")
    inert = inert_capabilities(args.bin_dir)
    for gate_id, reason in inert:
        print(f"fixtures: gate {gate_id} runs a tool whose check is inert here: {reason}")
    for gate_id, flag, reason in NON_LOAD_BEARING:
        print(f"fixtures: gate {gate_id} flag {flag!r} is not load bearing: {reason}")
    for gate_id, path, why in OUT_OF_SCOPE:
        print(f"fixtures: gate {gate_id} path deliberately not covered here -- {path}: {why}")

    if dropped is not None:
        # The drop hook exists for gate 13's own fixture and can only ever make
        # a run red. Never running the fixtures in this mode is what keeps the
        # suite from recursing into itself, and exiting non-zero unconditionally
        # is what stops the hook from being a way to skip anything.
        print(f"fixtures: {DROP_GATE}={dropped} dropped "
              f"{len([f for f in FIXTURES if f.gate == dropped])} fixture(s); only the "
              "enumeration check ran")
        return 1

    if not args.bin_dir:
        raise Environment(
            "--bin-dir is required: gates 8, 9 and 10 are exercised with the "
            "analyzers the runner installed, and a fixture that cannot run is not "
            "a fixture that passed"
        )
    bin_dir = Path(args.bin_dir).resolve()
    for tool in ("actionlint", "zizmor", "gitleaks"):
        if not os.access(bin_dir / tool, os.X_OK):
            raise Environment(f"no executable {tool} in the analyzer directory")

    before = run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"])

    work = Path(tempfile.mkdtemp(prefix="gate13-fixtures-"))
    paths = {work: "<fixtures>", bin_dir: "<bin>", root: "<repo>", Path.home(): "<home>"}
    outcomes = []
    residue = []
    try:
        ctx = Context(root, bin_dir, work, gates)
        for gate_id, name, description, build, failure in CONTROLS:
            if gate_id not in gates:
                continue
            plan = build(ctx)
            result = run(plan.argv, cwd=plan.cwd, env=plan.env)
            status = "pass" if result.exit_code == 0 else "fail"
            outcomes.append({
                "fixture": name,
                "gate": gate_id,
                "kind": "control",
                "defect": description,
                "argv": [scrub(part, paths) for part in plan.argv],
                "status": status,
                "exit_code": result.exit_code,
                "reasons": [] if status == "pass" else [failure],
            })
            mark = "ok  " if status == "pass" else "FAIL"
            print(f"{mark} control gate {gate_id:>2} {name}: {description}")
            if status == "fail":
                print(f"     {failure}", file=sys.stderr)
                print(excerpt(result.output, paths), file=sys.stderr)

        # Before the fixtures, deliberately. Each fixture below proves that a
        # gate's command rejects a defect; these prove that the *recorder* will
        # not report a gate as passing over a ledger with no command in it. Not
        # that the gate ran -- an arm of the dispatch can write the ledger entry
        # itself, which review demonstrated in one line. If the recorder half is
        # broken, the fixtures are answering a question nobody asked, which is
        # why they run first.
        for assertion in RUNNER_ASSERTIONS:
            try:
                plan = assertion.build(ctx)
            except Exception as error:  # noqa: BLE001 - same reason as below
                if not isinstance(error, Environment):
                    error = Environment(f"{type(error).__name__}: {error}")
                outcomes.append({
                    "fixture": assertion.name,
                    "kind": "runner",
                    "defect": assertion.claim,
                    "status": "fail",
                    "exit_code": None,
                    "reasons": [f"the assertion could not be built: {error}"],
                })
                print(f"FAIL runner {assertion.name}: cannot plant it: {error}",
                      file=sys.stderr)
                continue
            outcome = evaluate_runner(assertion, plan, paths)
            outcomes.append(outcome)
            mark = "ok  " if outcome["status"] == "pass" else "FAIL"
            print(f"{mark} runner {assertion.name}: {assertion.claim} "
                  f"-> exit {outcome['exit_code']}")
            if outcome["status"] == "fail":
                for reason in outcome["reasons"]:
                    print(f"     {reason}", file=sys.stderr)
                if outcome.get("output"):
                    print(outcome["output"], file=sys.stderr)

        for fixture in sorted(fixtures, key=lambda f: (f.gate, f.slot, f.name)):
            try:
                plan = fixture.build(ctx)
            except Exception as error:  # noqa: BLE001 - a fixture that cannot be
                # planted is a failure of this suite, not a reason to abandon the
                # remaining fixtures and report nothing about them.
                if not isinstance(error, Environment):
                    error = Environment(f"{type(error).__name__}: {error}")
                outcomes.append({
                    "fixture": fixture.name,
                    "gate": fixture.gate,
                    "tool": fixture.slot + 1,
                    "kind": "fixture",
                    "defect": fixture.defect,
                    "status": "fail",
                    "exit_code": None,
                    "reasons": [f"the fixture could not be built: {error}"],
                })
                print(f"FAIL {fixture.label}: cannot plant the defect: {error}",
                      file=sys.stderr)
                continue
            outcome = evaluate(fixture, plan, paths)
            outcome["kind"] = "fixture"
            outcomes.append(outcome)
            if outcome["status"] == "pending":
                print(f"PEND {fixture.label}: {fixture.defect} "
                      f"-> NOT RUN: {outcome['pending']}")
            else:
                mark = "ok  " if outcome["status"] == "pass" else "FAIL"
                print(f"{mark} {fixture.label}: {fixture.defect} "
                      f"-> exit {outcome['exit_code']}")
            if outcome["status"] == "fail":
                for reason in outcome["reasons"]:
                    print(f"     {reason}", file=sys.stderr)
                if outcome.get("output"):
                    print(outcome["output"], file=sys.stderr)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    after = run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"])
    if after.output != before.output:
        residue.append(
            "the suite changed the working tree: a fixture escaped its throwaway "
            "directory"
        )
        for line in sorted(set(after.output.splitlines()) - set(before.output.splitlines())):
            residue.append(f"left behind: {line.strip()}")
    for line in residue:
        print(f"FAIL residue: {line}", file=sys.stderr)

    failed = [o for o in outcomes if o["status"] == "fail"]
    pended = [o for o in outcomes if o["status"] == "pending"]
    print(f"fixtures: {len([o for o in outcomes if o['kind'] == 'fixture'])} fixture(s) "
          f"across {len({o['gate'] for o in outcomes if o['kind'] == 'fixture'})} gates, "
          f"{len(CONTROLS)} control(s), {len(RUNNER_ASSERTIONS)} runner assertion(s), "
          f"{len(failed)} failing, {len(pended)} pending "
          f"(not run, not passed), "
          f"{len(enumeration)} enumeration failure(s), {len(residue)} residue failure(s)")

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "gates_enumerated": sorted(gates),
                    "enumeration_failures": enumeration,
                    "residue_failures": residue,
                    "not_visible_in_the_enumeration": [
                        {"gate": gate_id, "reason": reason}
                        for gate_id, reason in UNSEEN_STEPS
                    ],
                    "inert_capabilities": [
                        {"gate": gate_id, "reason": reason}
                        for gate_id, reason in inert
                    ],
                    "flags_no_fixture_depends_on": [
                        {"gate": gate_id, "flag": flag, "reason": reason}
                        for gate_id, flag, reason in NON_LOAD_BEARING
                    ],
                    "deliberately_out_of_scope": [
                        {"gate": gate_id, "path": path, "covered_by": why}
                        for gate_id, path, why in OUT_OF_SCOPE
                    ],
                    "not_exercised": [
                        {"gate": gate_id, "tool": slot + 1, "reason": reason}
                        for (gate_id, slot), reason in sorted(KNOWN_UNCOVERED.items())
                    ],
                    "runner_assertions": [
                        {"name": a.name, "claim": a.claim, "must_reject": a.must_reject}
                        for a in RUNNER_ASSERTIONS
                    ],
                    "outcomes": outcomes,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    return 1 if (failed or enumeration or residue) else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Environment as error:
        print(f"fixtures: {error}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
