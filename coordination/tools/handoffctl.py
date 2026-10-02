#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Transactional coordination state for Game Experiment."""

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

from status_renderer import (
    StatusRenderError,
    graph_errors,
    render_status,
    submission_label,
)

# The coordination directory inside the product repository. State is versioned in the product's
# own history, so every Git command below runs against the product repository through this path.
ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "tasks"
RUNTIME = ROOT / ".runtime"
LOCK = RUNTIME / "state.lock"
CONFIG = RUNTIME / "config.json"
# The only branch a state transaction may commit to or replicate. State shares the product
# repository, so a transaction run from a worker's worktree would otherwise take that worktree's
# own lock and commit task state onto a feature branch, where nobody else would ever see it.
STATE_BRANCH = "main"
STATUSES = (
    "in_progress",
    # Finished work with no claim. A worker may put its own task here and nowhere else, so
    # no lease runs under work that is done, and no worker records its own verdict.
    "in_review",
    "open",
    "blocked",
    "planned",
    "future",
    "done",
    "cancelled",
    "superseded",
)
# Statuses a transition out of an active state may name. Neither active state is among them:
# in_progress is reached only by claiming, and in_review only by submitting.
DECIDED = tuple(value for value in STATUSES if value not in ("in_progress", "in_review"))
# Statuses a live claim may be closed into by the worker that holds it. Done is not among
# them: a claim is owner authenticated, so an owner-authenticated release naming done is a
# worker certifying its own work. Done is reachable only through review, after submission.
RELEASABLE = tuple(value for value in DECIDED if value != "done")
PRIORITIES = ("P0", "P1", "P2", "P3", "P4")
# Statuses that assert work is finished, for the purposes of closing a milestone.
FINISHED = ("done", "cancelled", "superseded")
# The first appended evidence entry ends a task's description. Entries are written by
# mutate() as "- <ISO timestamp>: ..." wrapped at 100 columns, and the log is append-only, so
# only the description above it may be corrected. A gate that a task's own immutable evidence
# can trip is a gate that becomes permanently unsatisfiable.
EVIDENCE_ENTRY = re.compile(r"^- \d{4}-\d{2}-\d{2}T[0-9:.+\-Z]{5,}: ", re.M)
# Anything a reader would take for a log entry. If a body carries one of these but no entry
# the strict pattern can read, the log is malformed and the prose rule fails closed rather
# than silently exempting the task.
MALFORMED_ENTRY = re.compile(r"^-\s+\S*\d{4}-\d{2}-\d{2}", re.M)
# A boilerplate detector, not a semantic one: it catches the wordings this repository has
# actually used, and cannot judge an arbitrary claim about progress.
UNSTARTED_PROSE = (
    re.compile(
        r"\b(?:implementation|work|development|coding)\b[^.\n]{0,60}?"
        r"\bnot\s+(?:yet\s+)?(?:been\s+)?(?:started|begun)\b",
        re.I,
    ),
    re.compile(
        r"\bno\s+(?:implementation|work|development|coding)\b[^.\n]{0,60}?"
        r"\b(?:been\s+)?(?:started|begun)\b",
        re.I,
    ),
)
# The one validation error a recovery transition may find already standing. It is matched
# exactly, against the message active_expiry_errors() emits, so the tolerance can never widen
# to a different fault that happens to mention a claim.
EXPIRED_CLAIM = re.compile(r"AR-\d{4}: expired claim")
# Printed beneath a doctor run that found one, so a red check names the way out of itself.
EXPIRY_HINT = (
    "HINT: a worker has gone away rather than the state being corrupt. Clear each abandoned "
    "claim with: handoffctl expire AR-NNNN --expected-revision N --cleared-by WHO --note WHY"
)
# A milestone document is MN.md for AR series NN, zero padded: M0 governs AR-00NN.
MILESTONE_ID = re.compile(r"M(?:0|[1-9][0-9]?)")
MILESTONE_STATUSES = ("active", "complete")
# Kept identical to schema/milestone-schema.json, so both gates accept the same documents.
MILESTONE_LABEL_MAX = 80
MILESTONE_SECTIONS = ("## Outcome", "## Exit criteria", "## Out of scope")
# Field presence, unknown fields and every mechanical bound are read from the published
# schemas rather than restated here. The hand-maintained REQ and FIELDS tuples that used to
# sit at this point stated a strict subset of what schema/task-schema.json states -- nine of
# its sixteen required fields, and no length, pattern or format bound at all -- so a
# transition could commit a document the schema gate then rejected. See schema_gate_errors.
#
# The schemas ship beside the tool, so they are located from this source file and not from
# ROOT, which the tests repoint at a fixture repository.
SCHEMA = Path(__file__).resolve().parent.parent / "schema"
# Object-level constructs the enforcer implements, and the annotations it may ignore.
SCHEMA_OBJECT_KEYWORDS = frozenset(
    {"type", "additionalProperties", "required", "properties", "allOf"}
)
SCHEMA_ANNOTATIONS = frozenset({"$schema", "$id", "title", "description"})
SCHEMA_BRANCHES = ("if", "then", "else")
JSON_TYPE_NAMES = frozenset({"string", "integer", "array", "object"})
# RFC 3339, the shape the schema gate's date-time format checker accepts.
RFC3339 = re.compile(r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})")
type Meta = dict[str, Any]
type Task = tuple[Path, Meta, str]
type Milestone = tuple[Path, Meta, str]
type State = dict[str, Any]

PRIVATE = (
    (re.compile("/" + "home/"), "absolute Linux home path"),
    (re.compile(r"[A-Za-z]:\\Users\\", re.I), "absolute Windows user path"),
    (re.compile(r"\b(?:10|127)\.(?:\d{1,3}\.){2}\d{1,3}\b"), "private or loopback IP"),
    (
        re.compile(
            r"\b(?:password|passwd|token|secret|api[_-]?key)\s*[:=]\s*[^\s<]+",
            re.I,
        ),
        "possible credential",
    ),
    (re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----"), "private key"),
    (
        re.compile(r"claude\.ai/" + r"code/session", re.I),
        "private agent session reference",
    ),
    (
        re.compile(r"\bses" + r"sion_[A-Za-z0-9]{16,}\b", re.I),
        "private agent session identifier",
    ),
    (
        re.compile(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            re.I,
        ),
        "session-like UUID",
    ),
)
# Directory names the privacy walk never examines. Every one is matched against the path
# **relative to ROOT** and never against the absolute path: an absolute match lets a directory
# *above* the root suppress the whole tree beneath it. Worker checkouts live at
# workspaces/AR-NNNN, so every absolute path inside one contains `workspaces`, and the walk
# examined nothing at all -- reporting clean over an empty set for every worker that validated
# from its own worktree. The same shape applies to every other name here: a checkout beneath a
# directory called `.venv` or `.git` would be skipped whole.
PRIVACY_EXCLUSIONS = frozenset(
    {
        ".git",
        ".runtime",
        ".venv",
        ".mypy_cache",
        ".ruff_cache",
        "private-archive",
        "workspaces",
        "__pycache__",
    }
)


def now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=capture, check=False)
    if check and proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr}")
    return proc


def config() -> Meta:
    if not CONFIG.exists():
        raise RuntimeError(f"missing private runtime config: {CONFIG}")
    return cast(Meta, json.loads(CONFIG.read_text()))


@contextlib.contextmanager
def locked(*, exclusive: bool = True) -> Iterator[None]:
    RUNTIME.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temp_path = Path(tmp)
        temp_path.chmod(0o600)
        temp_path.replace(path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temp_path = Path(tmp)
        if temp_path.exists():
            temp_path.unlink()


def append_evidence(body: str, timestamp: str, note: str) -> str:
    """Append one evidence entry, wrapped so no continuation line can look like an entry.

    The two-space continuation indent is what keeps ``EVIDENCE_ENTRY`` unambiguous: an
    entry begins at column zero with "- ", and a wrapped line never does.
    """
    entry = textwrap.fill(
        f"{timestamp}: {note}",
        width=100,
        initial_indent="- ",
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )
    return f"{body}\n{entry}\n"


def read_task(path: Path) -> tuple[Meta, str]:
    text = path.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"{path}: no front matter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError(f"{path}: unterminated front matter")
    meta = json.loads(text[4:end])
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: front matter is not a JSON object")
    return cast(Meta, meta), text[end + 5 :]


def write_task(path: Path, meta: Meta, body: str) -> None:
    atomic(path, "---\n" + json.dumps(meta, indent=2, sort_keys=True) + "\n---\n" + body)


def all_tasks() -> list[Task]:
    result: list[Task] = []
    for path in sorted(TASKS.glob("AR-*.md")):
        meta, body = read_task(path)
        result.append((path, meta, body))
    return result


def milestone_documents() -> list[Milestone]:
    """Return every milestone document; a malformed one yields empty metadata to validate."""
    result: list[Milestone] = []
    for path in sorted((ROOT / "milestones").glob("M*.md")):
        try:
            meta, body = read_task(path)
        except ValueError:
            result.append((path, {}, ""))
        else:
            result.append((path, meta, body))
    return result


def milestone_series(meta: Meta) -> str:
    """Return the two-digit AR series a milestone governs, or empty when unidentifiable."""
    identifier = str(meta.get("id", ""))
    if not MILESTONE_ID.fullmatch(identifier):
        return ""
    return f"{int(identifier[1:]):02d}"


def milestone_labels() -> dict[str, str]:
    """Map each governed AR series to its milestone label for the deterministic renderer."""
    labels: dict[str, str] = {}
    for _, meta, _ in milestone_documents():
        series = milestone_series(meta)
        label = meta.get("label")
        if series and isinstance(label, str) and label.strip():
            labels.setdefault(series, label)
    return labels


def render_current(tasks: list[Task]) -> str:
    groups: dict[str, list[Meta]] = {status: [] for status in STATUSES}
    for _, meta, _ in tasks:
        groups[meta["status"]].append(meta)
    labels = {x: x.replace("_", " ").title() for x in STATUSES}
    lines = [
        "# Game Experiment current coordination state",
        "",
        "This file is generated. Read `README.md`, then use `tools/handoffctl snapshot`.",
        "Never edit this file directly.",
        "",
    ]
    by_id = {meta["id"]: path.name for path, meta, _ in tasks}

    def clean(value: object) -> str:
        return str(value or "-").replace("|", "\\|").replace("\n", " ")

    for status in STATUSES:
        rows = sorted(
            groups[status], key=lambda meta: (PRIORITIES.index(meta["priority"]), meta["id"])
        )
        if not rows:
            continue
        lines += [
            f"## {labels[status]}",
            "",
            "| Priority | Task | Summary | Next action | Owner |",
            "| --- | --- | --- | --- | --- |",
        ]
        for meta in rows:
            link = f"[{meta['id']}](tasks/{by_id[meta['id']]})"
            row = (
                f"| {meta['priority']} | {link}: {clean(meta['title'])} | "
                f"{clean(meta['summary'])} | {clean(meta['next_action'])} | "
                f"{clean(submission_label(meta) or meta.get('owner'))} |"
            )
            lines.append(row)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def product_checkout(settings: Meta) -> Path:
    """Return the product checkout whose worktrees are inventoried.

    Coordination lives inside the product repository, so by default that is the checkout holding
    ``ROOT``. ``projects_root`` and ``product_worktree`` remain as an explicit override.
    """
    if "projects_root" in settings and "product_worktree" in settings:
        return Path(str(settings["projects_root"])) / str(settings["product_worktree"])
    top = run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"]).stdout.strip()
    return Path(top)


def project_scan() -> State:
    settings = config()
    repo = product_checkout(settings)
    raw = run(["git", "-C", str(repo), "worktree", "list", "--porcelain"]).stdout
    paths = [Path(line[9:]) for line in raw.splitlines() if line.startswith("worktree ")]
    worktrees = []
    for path in paths:
        head = run(["git", "-C", str(path), "rev-parse", "HEAD"]).stdout.strip()
        branch = (
            run(
                ["git", "-C", str(path), "symbolic-ref", "--short", "-q", "HEAD"], check=False
            ).stdout.strip()
            or "DETACHED"
        )
        changed = run(["git", "-C", str(path), "status", "--porcelain=v1"]).stdout.splitlines()
        counts = run(
            ["git", "-C", str(path), "rev-list", "--left-right", "--count", "origin/main...HEAD"],
            check=False,
        ).stdout.split()
        worktrees.append(
            {
                "key": path.name,
                "branch": branch,
                "head": head,
                "dirty": len(changed),
                "paths": [line[3:] for line in changed[:50]],
                "behind": int(counts[0]) if len(counts) == 2 else None,
                "ahead": int(counts[1]) if len(counts) == 2 else None,
            }
        )
    github = settings["github_repository"]
    prs = json.loads(
        run(
            [
                "gh",
                "pr",
                "list",
                "-R",
                github,
                "--state",
                "open",
                "--limit",
                "100",
                "--json",
                "number,title,headRefName,headRefOid,baseRefName,isDraft,mergeStateStatus,statusCheckRollup",
            ]
        ).stdout
    )
    runs = json.loads(
        run(
            [
                "gh",
                "run",
                "list",
                "-R",
                github,
                "--limit",
                "12",
                "--json",
                "databaseId,headSha,status,conclusion,workflowName,event",
            ]
        ).stdout
    )
    remote_line = run(
        ["git", "-C", str(repo), "ls-remote", "origin", "refs/heads/main"]
    ).stdout.strip()
    if not remote_line:
        raise RuntimeError("remote main is missing")
    return {
        "remote_main": remote_line.split()[0],
        "origin_main": run(["git", "-C", str(repo), "rev-parse", "origin/main"]).stdout.strip(),
        "primary_head": run(["git", "-C", str(repo), "rev-parse", "HEAD"]).stdout.strip(),
        "worktrees": worktrees,
        "prs": prs,
        "runs": runs,
    }


def live_docs(state: State) -> tuple[str, str]:
    project = [
        "# Game Experiment live project state",
        "",
        "Generated from local Git and GitHub. Do not edit.",
        "",
        f"- Product remote main: `{state['remote_main']}`",
        f"- Local origin/main: `{state['origin_main']}`",
        f"- Primary worktree head: `{state['primary_head']}`",
        "",
        "## Open pull requests",
        "",
        "| PR | Head | Base | Merge | Checks | Title |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pr in sorted(state["prs"], key=lambda item: item["number"]):
        checks = [
            (item.get("status") or "") + ":" + (item.get("conclusion") or "")
            for item in pr.get("statusCheckRollup") or []
        ]
        title = pr["title"].replace("|", "/")
        project.append(
            f"| #{pr['number']} | `{pr['headRefName']}@{pr['headRefOid'][:12]}` | "
            f"`{pr['baseRefName']}` | {pr['mergeStateStatus']} | "
            f"{', '.join(checks) or '-'} | {title} |"
        )
    project += [
        "",
        "## Recent workflows",
        "",
        "| Run | SHA | Event | Workflow | State |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in state["runs"]:
        project.append(
            f"| {item['databaseId']} | `{item['headSha'][:12]}` | "
            f"{item['event']} | {item['workflowName']} | "
            f"{item['status']}:{item.get('conclusion') or '-'} |"
        )
    worktrees = [
        "# Game Experiment worktree inventory",
        "",
        "Generated from live Git. Paths are privacy-safe worktree keys.",
        "",
        "| Worktree | Branch | Head | Dirty | vs origin/main |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for item in state["worktrees"]:
        worktrees.append(
            f"| `{item['key']}` | `{item['branch']}` | "
            f"`{item['head'][:12]}` | {item['dirty']} | "
            f"behind {item['behind']}, ahead {item['ahead']} |"
        )
        if item["dirty"]:
            paths = ", ".join("`" + value + "`" for value in item["paths"])
            worktrees.append(f"| changed files | - | - | - | {paths} |")
    return "\n".join(project) + "\n", "\n".join(worktrees) + "\n"


def sync_task_observations(tasks: list[Task], state: State) -> None:
    observed = {item["key"]: item for item in state["worktrees"]}
    for path, meta, body in tasks:
        key = meta.get("worktree_key")
        if not key or key not in observed:
            continue
        item = observed[key]
        values = {
            "observed_branch": item["branch"],
            "observed_head": item["head"],
            "observed_dirty": item["dirty"],
        }
        if any(meta.get(name) != value for name, value in values.items()):
            meta.update(values)
            meta["updated_at"] = now()
            meta["task_revision"] += 1
            write_task(path, meta, body)


def privacy_walk() -> list[Path]:
    """Return every file the privacy walk examines, as paths relative to ``ROOT``.

    Exclusions are decided on the **relative** path, so nothing outside the repository can
    suppress anything inside it, and excluded directories are pruned rather than filtered, so
    an excluded tree is never descended: ``workspaces/`` holds whole product checkouts, and
    enumerating them on every transaction is what the exclusion exists to avoid. Pruning removes
    every excluded *ancestor* below the root, so the remaining test is on the file's own name --
    which is not redundant, because a worktree's ``.git`` is a file and not a directory.
    """
    files: list[Path] = []
    for directory, subdirectories, names in os.walk(ROOT):
        subdirectories[:] = [name for name in subdirectories if name not in PRIVACY_EXCLUSIONS]
        parent = Path(directory).relative_to(ROOT)
        files.extend(parent / name for name in names if name not in PRIVACY_EXCLUSIONS)
    return sorted(files)


def privacy_errors() -> list[str]:
    """Report private references and oversized state files, and refuse to pass over nothing.

    The empty walk is an error in itself. A gate that reports clean because it examined no
    file is indistinguishable, in its output, from one that examined the repository and found
    it clean -- which is exactly how this walk reported clean for every worker validating from
    a worktree under ``workspaces/``. Relative scoping is what fixes that; the floor is the
    check on the checker, and it holds whatever the reason for the emptiness: a misconfigured
    root, an exclusion that swallows the tree, or a future regression in either.

    It is a floor and not a proof of coverage: it catches total vacuity only, and a walk that
    examines one file of seven thousand still passes it. The measured cost of keeping it is
    nil -- no walk over any fixture in the suite examines nothing -- and the alternative is a
    gate whose only report of having looked at nothing is silence.
    """
    errors: list[str] = []
    walked = privacy_walk()
    if not walked:
        errors.append("privacy walk examined no file: the root is empty or excluded entirely")
    for relative in walked:
        path = ROOT / relative
        if not path.is_file():
            continue
        if path.stat().st_size > 200000:
            errors.append(f"{relative}: state file exceeds 200 KiB")
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for regex, label in PRIVATE:
            if regex.search(text):
                errors.append(f"{relative}: {label}")
    return errors


def commit_privacy_errors(base: str, head: str) -> list[str]:
    """Return private references found in the messages of newly introduced commits."""
    span = f"{base}..{head}" if base else head
    revisions = run(["git", "-C", str(ROOT), "rev-list", span]).stdout.split()
    errors: list[str] = []
    for revision in revisions:
        message = run(["git", "-C", str(ROOT), "show", "-s", "--format=%B", revision]).stdout
        errors.extend(
            f"{revision[:12]}: {label}" for regex, label in PRIVATE if regex.search(message)
        )
    return errors


def schema_document(name: str) -> Meta:
    """Return one published schema, parsed."""
    return cast(Meta, json.loads((SCHEMA / name).read_text()))


def _json_type_error(value: object, name: object) -> bool:
    """Report whether a value fails the JSON Schema primitive type named here.

    ``integer`` excludes ``bool`` exactly as JSON Schema does and Python does not, so a
    ``task_revision`` of ``true`` is refused by both gates rather than by one of them.
    """
    if name == "integer":
        return isinstance(value, bool) or not isinstance(value, int)
    if name == "string":
        return not isinstance(value, str)
    if name == "array":
        return not isinstance(value, list)
    return not isinstance(value, dict)


def _format_error(value: object, name: object) -> bool:
    """Report whether a string fails the one string format the published schemas state.

    ``fromisoformat`` alone is not the RFC 3339 the schema gate's format checker applies: it
    also accepts a bare date, a space separator and a missing offset. The pattern states the
    shape and the parse states that the instant is real.
    """
    if not isinstance(value, str):
        return False
    if name != "date-time":
        return True
    if RFC3339.fullmatch(value) is None:
        return True
    try:
        dt.datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
    except ValueError:
        return True
    return False


def _length_error(value: object, bound: object, *, longer: bool) -> bool:
    """Report whether a string breaches a stated length bound."""
    if not isinstance(value, str) or not isinstance(bound, int):
        return False
    return len(value) > bound if longer else len(value) < bound


def _unique_error(value: object, bound: object) -> bool:
    """Report whether an array that must hold distinct items repeats one."""
    if bound is not True or not isinstance(value, list):
        return False
    rendered = [json.dumps(item, sort_keys=True) for item in value]
    return len(rendered) != len(set(rendered))


def _pattern_error(value: object, bound: object) -> bool:
    """Report whether a string fails a stated pattern, which JSON Schema leaves unanchored."""
    return isinstance(value, str) and re.search(str(bound), value) is None


def _json_equal(value: object, bound: object) -> bool:
    """Report JSON equality, where ``true`` and ``1`` are different values and Python is not.

    ``_json_type_error`` already separates the two for ``integer``; a ``const`` or ``enum``
    stated as a number would otherwise accept the boolean the schema gate refuses, which is
    the same divergence in miniature.
    """
    if isinstance(value, bool) != isinstance(bound, bool):
        return False
    return bool(value == bound)


def _minimum_error(value: object, bound: object) -> bool:
    """Report whether a number falls below a stated minimum."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    return isinstance(bound, int | float) and value < bound


SCHEMA_CONSTRAINTS: dict[str, Callable[[object, object], bool]] = {
    "const": lambda value, bound: not _json_equal(value, bound),
    "enum": lambda value, bound: (
        isinstance(bound, list) and not any(_json_equal(value, item) for item in bound)
    ),
    "format": _format_error,
    "maxLength": lambda value, bound: _length_error(value, bound, longer=True),
    "minLength": lambda value, bound: _length_error(value, bound, longer=False),
    "minimum": _minimum_error,
    "pattern": _pattern_error,
    "type": _json_type_error,
    "uniqueItems": _unique_error,
}


def _bound_message(label: str, field: str, keyword: str, bound: object, value: object) -> str:
    """Name the field and the bound it breaches, in the schema's own vocabulary.

    A worker that writes an over-long value learns the field, the limit and what it wrote at
    the transition that refused it, rather than at whatever gate runs next.
    """
    observed = ""
    if keyword in ("maxLength", "minLength") and isinstance(value, str):
        observed = f" (length {len(value)})"
    return f"{label}: {field} violates {keyword} {json.dumps(bound, sort_keys=True)}{observed}"


def schema_value_errors(label: str, field: str, value: object, rules: Meta) -> list[str]:
    """Report every bound the schema states for one field that this value breaches."""
    errors = [
        _bound_message(label, field, keyword, bound, value)
        for keyword, bound in sorted(rules.items())
        if keyword in SCHEMA_CONSTRAINTS and SCHEMA_CONSTRAINTS[keyword](value, bound)
    ]
    items = rules.get("items")
    if isinstance(items, dict) and isinstance(value, list):
        for index, item in enumerate(value):
            errors.extend(schema_value_errors(label, f"{field}[{index}]", item, cast(Meta, items)))
    return errors


def schema_object_errors(label: str, meta: Meta, schema: Meta) -> list[str]:
    """Report every constraint an object schema states that this document breaches.

    ``type: object`` is not re-checked here: ``read_task`` already refuses front matter that
    is not a JSON object, so nothing else can reach this point. That holds only while the
    schema says ``object``, which is why ``_object_support_errors`` refuses any other
    top-level type rather than leaving this function to accept documents the schema refuses.
    """
    properties = cast(Meta, schema.get("properties", {}))
    errors = [
        f"{label}: missing {name}"
        for name in cast(list[str], schema.get("required", []))
        if name not in meta
    ]
    if schema.get("additionalProperties") is False:
        errors.extend(
            f"{label}: unknown field {name}" for name in sorted(set(meta) - set(properties))
        )
    for field, rules in sorted(properties.items()):
        if field in meta:
            errors.extend(schema_value_errors(label, field, meta[field], cast(Meta, rules)))
    errors.extend(_conditional_errors(label, meta, schema))
    return errors


def _conditional_errors(label: str, meta: Meta, schema: Meta) -> list[str]:
    """Apply each if/then/else branch the schema states, the way a validator would."""
    errors: list[str] = []
    for branch in cast(list[Meta], schema.get("allOf", [])):
        condition = branch.get("if")
        matched = not isinstance(condition, dict) or not schema_object_errors(
            label, meta, cast(Meta, condition)
        )
        applied = branch.get("then" if matched else "else")
        if isinstance(applied, dict):
            errors.extend(schema_object_errors(label, meta, cast(Meta, applied)))
    return errors


def _is_length_bound(bound: object) -> bool:
    """Report whether a length bound is the non-negative integer ``_length_error`` needs."""
    return isinstance(bound, int) and not isinstance(bound, bool) and bound >= 0


def _is_pattern_bound(bound: object) -> bool:
    """Report whether a pattern is a string this interpreter can actually compile."""
    if not isinstance(bound, str):
        return False
    try:
        re.compile(bound)
    except re.error:
        return False
    return True


SCHEMA_BOUND_SHAPES: dict[str, Callable[[object], bool]] = {
    "enum": lambda bound: isinstance(bound, list),
    "maxLength": _is_length_bound,
    "minLength": _is_length_bound,
    "minimum": lambda bound: isinstance(bound, int | float) and not isinstance(bound, bool),
    "pattern": _is_pattern_bound,
    "uniqueItems": lambda bound: isinstance(bound, bool),
}


def _unsupported(label: str, field: str, what: str, value: object) -> str:
    """Name one construct the reader refuses, quoting it as the schema states it."""
    where = f"{label}: {field} states" if field else f"{label}: states"
    return f"{where} unsupported {what} {json.dumps(value, sort_keys=True)}"


def _bound_shape_errors(label: str, field: str, rules: Meta) -> list[str]:
    """Refuse a bound whose *value* is not the shape its checker can apply.

    A keyword's name being implemented is not enough: ``SCHEMA_CONSTRAINTS`` reaches its
    bound directly, so ``"minimum": "3"`` and ``"enum": {}`` are silently unenforced while
    ``"maxLength": "300"`` and ``"pattern": "["`` raise out of ``validate()`` and out of
    every transaction with it. Both are the fault class this net exists to prevent, so the
    net checks the shape of each bound and not only the keyword that states it.
    """
    return [
        _unsupported(label, field, f"{keyword} bound", bound)
        for keyword, bound in sorted(rules.items())
        if keyword in SCHEMA_BOUND_SHAPES and not SCHEMA_BOUND_SHAPES[keyword](bound)
    ]


def _field_support_errors(label: str, field: str, rules: object) -> list[str]:
    """Refuse one field constraint the transaction gate cannot enforce itself.

    Draft 2020-12 permits a **boolean** wherever a subschema may appear, and ``false`` there
    is a real constraint: ``"items": false`` refuses every element of an array. This reader
    enforces neither form, so a non-object subschema is refused rather than skipped -- and
    that refusal is also what keeps ``set(rules)`` from raising ``TypeError`` on a boolean.
    """
    if not isinstance(rules, dict):
        return [_unsupported(label, field, "subschema", rules)]
    unsupported = set(rules) - set(SCHEMA_CONSTRAINTS) - {"items"}
    errors = [
        f"{label}: {field} states unsupported constraint {name}" for name in sorted(unsupported)
    ]
    # A type stated as a list -- ["string", "null"] is legal draft 2020-12 -- is unhashable,
    # so testing membership first would raise TypeError out of validate() and out of every
    # transaction with it. The design promises a loud refusal, not a crash.
    if "type" in rules and not (
        isinstance(rules["type"], str) and rules["type"] in JSON_TYPE_NAMES
    ):
        errors.append(f"{label}: {field} states unsupported type {rules['type']}")
    if "format" in rules and rules["format"] != "date-time":
        errors.append(f"{label}: {field} states unsupported format {rules['format']}")
    errors.extend(_bound_shape_errors(label, field, rules))
    if "items" in rules:
        errors.extend(_field_support_errors(label, f"{field}[]", rules["items"]))
    return errors


def _object_support_errors(label: str, schema: Meta) -> list[str]:
    """Refuse object-level constructs the enforcer cannot apply to a whole document.

    A changed top-level ``type`` is the dangerous one: ``schema_object_errors`` deliberately
    does not re-check it, because ``read_task`` already refuses front matter that is not a
    JSON object. That reasoning holds only while the schema says ``object``. A schema stating
    ``array`` refuses every document, so leaving it unenforced would accept everything the
    schema gate rejects -- the original divergence, restored by a one-word edit.

    ``additionalProperties`` is enforced only in its boolean form; the schema-object form
    states a bound on each additional property that this reader does not implement.

    ``properties``, ``required`` and ``allOf`` are refused unless they hold the container the
    enforcer walks, because a walk over anything else is a ``TypeError`` inside the
    transaction rather than a refusal.
    """
    errors = [
        f"{label}: unsupported schema keyword {name}"
        for name in sorted(set(schema) - SCHEMA_OBJECT_KEYWORDS - SCHEMA_ANNOTATIONS)
    ]
    declared = schema.get("type", "object")
    if declared != "object":
        errors.append(f"{label}: states unsupported document type {declared}")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        errors.append(
            f"{label}: states unsupported additionalProperties {schema['additionalProperties']}"
        )
    for name, container in (("properties", dict), ("required", list), ("allOf", list)):
        if name in schema and not isinstance(schema[name], container):
            errors.append(_unsupported(label, "", name, schema[name]))
    required = schema.get("required", [])
    if isinstance(required, list) and not all(isinstance(item, str) for item in required):
        errors.append(_unsupported(label, "", "required", required))
    return errors


def _branch_support_errors(label: str, branch: object) -> list[str]:
    """Refuse one ``allOf`` entry the conditional reader cannot apply."""
    if not isinstance(branch, dict):
        return [f"{label}: states unsupported allOf branch {json.dumps(branch, sort_keys=True)}"]
    errors = [
        f"{label}: unsupported conditional keyword {name}"
        for name in sorted(set(branch) - set(SCHEMA_BRANCHES))
    ]
    for name in SCHEMA_BRANCHES:
        if name in branch:
            errors.extend(schema_support_errors(f"{label} {name}", branch[name]))
    return errors


def schema_support_errors(label: str, schema: object) -> list[str]:
    """Refuse every construct of the published schema this reader cannot enforce itself.

    What this establishes, exactly. The reader walks a schema through four kinds of position:
    the document object, a ``properties`` entry, an ``items`` entry, and an ``allOf`` branch
    with its ``if``/``then``/``else``. At each one it accepts a closed vocabulary of keywords
    -- ``SCHEMA_OBJECT_KEYWORDS`` and ``SCHEMA_ANNOTATIONS`` for an object, ``SCHEMA_CONSTRAINTS``
    plus ``items`` for a field, ``SCHEMA_BRANCHES`` for a branch -- refuses every other name,
    refuses any value in a subschema position that is not a JSON object (draft 2020-12 allows
    a boolean there, and this reader enforces neither form), and refuses any bound whose shape
    its checker cannot apply. ``schema_gate_errors`` then enforces a schema **only** when this
    function returned nothing, so the enforcement path never runs over a construct the reader
    has not accepted. Together those give the property the reader needs: no keyword, subschema
    or bound of the published schemas can be silently unenforced or raise out of a transaction
    without first being reported here.

    What it does not establish: that each *implemented* keyword agrees with draft 2020-12 on
    every value. That is a claim about the checkers in ``SCHEMA_CONSTRAINTS``, not about
    coverage, and it rests on the differential fuzz in ``tests/fuzz_schema_parity.py`` -- with
    one declared exception, ``1.0`` counted as an integer by the validator and not here.
    """
    if not isinstance(schema, dict):
        return [_unsupported(label, "", "subschema", schema)]
    errors = _object_support_errors(label, schema)
    properties = schema.get("properties", {})
    if isinstance(properties, dict):
        for field, rules in sorted(properties.items()):
            errors.extend(_field_support_errors(label, field, rules))
    branches = schema.get("allOf", [])
    if isinstance(branches, list):
        for branch in branches:
            errors.extend(_branch_support_errors(label, branch))
    return errors


def schema_gate_errors(tasks: list[Task]) -> list[str]:
    """Apply both published schemas inside the transaction that writes the documents.

    This is what makes the two gates one gate. ``validate()`` states no length or format
    bound of its own; it reads the bounds the schema publishes and enforces them where the
    write happens, so a transition can no longer commit state ``tests/validate_schema.py``
    would reject.
    """
    errors: list[str] = []
    groups: tuple[tuple[str, list[Task]], ...] = (
        ("task-schema.json", tasks),
        ("milestone-schema.json", milestone_documents()),
    )
    for name, documents in groups:
        try:
            schema = schema_document(name)
        except (OSError, ValueError) as error:
            errors.append(f"{name}: unreadable published schema ({type(error).__name__})")
            continue
        support = schema_support_errors(name, schema)
        errors.extend(support)
        if support:
            # The enforcer walks the schema too, and a construct the net refuses is exactly
            # one it cannot walk. Enforcing an unsupported schema would raise from inside the
            # transaction; the tree is already refused by the support errors themselves.
            continue
        for path, meta, _ in documents:
            errors.extend(schema_object_errors(path.name, meta, schema))
    return errors


def value_errors(path: Path, meta: Meta) -> list[str]:
    """Report the per-field faults whose message names the field in this tool's vocabulary.

    Every value reaches a regex or a path join through ``str``: front matter is untrusted, so
    a field of the wrong JSON type must produce a validation error and never a TypeError.
    """
    errors: list[str] = []
    task_id = str(meta.get("id", ""))
    if not re.fullmatch(r"AR-\d{4}", task_id):
        errors.append(f"{path.name}: invalid id")
    if meta.get("status") not in STATUSES:
        errors.append(f"{task_id}: invalid status")
    if meta.get("priority") not in PRIORITIES:
        errors.append(f"{task_id}: invalid priority")
    revision = meta.get("task_revision")
    if not isinstance(revision, int) or revision < 1:
        errors.append(f"{task_id}: invalid revision")
    errors.extend(
        f"{task_id}: invalid {name}"
        for name in ("title", "summary", "next_action", "updated_at")
        if not isinstance(meta.get(name), str) or not meta.get(name)
    )
    return errors


def reference_errors(path: Path, meta: Meta) -> list[str]:
    errors: list[str] = []
    task_id = str(meta.get("id", ""))
    try:
        dt.datetime.fromisoformat(str(meta.get("updated_at", "")).replace("Z", "+00:00"))
    except ValueError:
        errors.append(f"{task_id}: invalid updated_at")
    checkpoint = meta.get("checkpoint_commit")
    if checkpoint and not re.fullmatch(r"[0-9a-f]{40}", str(checkpoint)):
        errors.append(f"{task_id}: invalid checkpoint commit")
    plan = meta.get("plan")
    if plan and not (path.parent / str(plan)).resolve().is_file():
        errors.append(f"{task_id}: missing plan {plan}")
    return errors


def basic_task_errors(path: Path, meta: Meta) -> list[str]:
    """Return the per-task rules the published schema cannot state.

    Presence and unknown fields are not restated here: ``schema_gate_errors`` derives both
    from the schema's ``required`` and ``additionalProperties`` and emits the same messages,
    so a single fault is still reported once.
    """
    return [*value_errors(path, meta), *reference_errors(path, meta)]


def active_expiry_errors(task_id: str, expiry: object) -> list[str]:
    if not expiry:
        return [f"{task_id}: active without claim"]
    if not isinstance(expiry, str):
        return [f"{task_id}: invalid claim expiry"]
    try:
        parsed_expiry = dt.datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        if parsed_expiry.tzinfo is None:
            raise ValueError
    except ValueError:
        return [f"{task_id}: invalid claim expiry"]
    if parsed_expiry <= dt.datetime.now(dt.UTC):
        return [f"{task_id}: expired claim"]
    return []


def submission_errors(task_id: str, meta: Meta) -> list[str]:
    """A recorded submitter belongs to work awaiting review, and to nothing else.

    The field is what a review decision is checked against, so a stale one left on an active
    or finished task would name a worker the coordinator could then refuse as the reviewer of
    work that worker never submitted, and an absent one would let anybody accept the task.
    """
    submitted = meta.get("submitted_by")
    if meta.get("status") != "in_review":
        return [f"{task_id}: submission recorded outside review"] if submitted else []
    if not isinstance(submitted, str) or not submitted:
        return [f"{task_id}: awaiting review without a recorded submitter"]
    return []


def claim_errors(
    meta: Meta,
    active_owners: dict[str, str],
    active_worktrees: dict[str, str],
    active_branches: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    task_id = str(meta.get("id", ""))
    errors.extend(submission_errors(task_id, meta))
    if meta.get("status") != "in_progress":
        if meta.get("owner") or meta.get("claim_expires"):
            errors.append(f"{task_id}: inactive task retains claim")
        return errors
    if not meta.get("owner"):
        errors.append(f"{task_id}: active without claim")
    errors.extend(active_expiry_errors(task_id, meta.get("claim_expires")))
    for field, seen in (
        ("owner", active_owners),
        ("worktree_key", active_worktrees),
        ("branch", active_branches),
    ):
        value = meta.get(field)
        if value and value in seen:
            errors.append(f"{task_id}: active {field} also used by {seen[value]}")
        elif value:
            seen[value] = task_id
    return errors


def task_description(body: str) -> tuple[str, bool]:
    """Split a task body into its editable description and whether evidence is recorded."""
    entry = EVIDENCE_ENTRY.search(body)
    if entry is None:
        return body, False
    return body[: entry.start()], True


def prose_errors(meta: Meta, body: str) -> list[str]:
    """Reject a description that still claims work is unstarted once evidence exists.

    Recorded evidence, not status, is the trigger: a task nobody ever worked on may say so
    truthfully at any status, including cancelled and superseded, while a task with a
    recorded history must describe itself honestly whatever became of it.
    """
    task_id = meta.get("id", "")
    description, recorded = task_description(body)
    if not recorded:
        if MALFORMED_ENTRY.search(body):
            return [f"{task_id}: evidence log is not in the recorded entry format"]
        return []
    if not any(regex.search(description) for regex in UNSTARTED_PROSE):
        return []
    return [f"{task_id}: description claims work has not started"]


def milestone_field_errors(path: Path, meta: Meta) -> list[str]:
    """Reject milestone front matter that does not declare the contract."""
    identifier = str(meta.get("id", ""))
    label = meta.get("label")
    checks = (
        (
            identifier == path.stem and MILESTONE_ID.fullmatch(identifier) is not None,
            "identifier does not match its filename",
        ),
        (meta.get("schema_version") == 1, "unsupported milestone schema version"),
        (meta.get("status") in MILESTONE_STATUSES, "invalid milestone status"),
        (
            isinstance(label, str) and bool(label.strip()) and len(label) <= MILESTONE_LABEL_MAX,
            "invalid milestone label",
        ),
    )
    # Presence and unknown fields come from milestone-schema.json through schema_gate_errors,
    # which emits the same messages; only the rules the schema cannot state remain here.
    return [f"{path.name}: {message}" for passed, message in checks if not passed]


def milestone_body_errors(path: Path, meta: Meta, body: str) -> list[str]:
    """Reject milestone prose that omits or contradicts the declared contract."""
    errors = [
        f"{path.name}: missing section {section}"
        for section in MILESTONE_SECTIONS
        if f"\n{section}\n" not in body
    ]
    label = str(meta.get("label", ""))
    heading = f"# {meta.get('id', '')} \u2014 {label}\n"
    if label.strip() and heading not in body:
        errors.append(f"{path.name}: heading does not match the recorded label")
    return errors


def milestone_coverage_errors(documents: list[Milestone], tasks: list[Task]) -> list[str]:
    """Every AR series needs exactly one milestone document to govern it."""
    errors: list[str] = []
    governed: dict[str, str] = {}
    for path, meta, _ in documents:
        series = milestone_series(meta)
        if not series:
            continue
        if series in governed:
            errors.append(f"{path.name}: series {series} already governed by {governed[series]}")
        else:
            governed[series] = path.name
    observed = {
        str(meta["id"])[3:5]
        for _, meta, _ in tasks
        if re.fullmatch(r"AR-\d{4}", str(meta.get("id")))
    }
    errors.extend(
        f"series {series}: no milestone document" for series in sorted(observed - set(governed))
    )
    return errors


def milestone_completion_errors(documents: list[Milestone], tasks: list[Task]) -> list[str]:
    """A milestone recorded complete must have no unfinished AR in its series."""
    errors: list[str] = []
    for path, meta, _ in documents:
        if meta.get("status") != "complete":
            continue
        series = milestone_series(meta)
        pending = sorted(
            str(item["id"])
            for _, item, _ in tasks
            if str(item.get("id", ""))[3:5] == series and item.get("status") not in FINISHED
        )
        if pending:
            errors.append(f"{path.name}: unfinished ARs {', '.join(pending)}")
    return errors


def milestone_errors(tasks: list[Task]) -> list[str]:
    """Return every milestone contract, coverage and completion error, deterministically."""
    documents = milestone_documents()
    errors: list[str] = []
    for path, meta, body in documents:
        errors.extend(milestone_field_errors(path, meta))
        errors.extend(milestone_body_errors(path, meta, body))
    errors.extend(milestone_coverage_errors(documents, tasks))
    errors.extend(milestone_completion_errors(documents, tasks))
    return errors


def render_status_view(tasks: list[Task]) -> str:
    """Render the public task dashboard with coordinator presentation constants."""
    return render_status(tasks, STATUSES, PRIORITIES, milestone_labels())


def renderable(tasks: list[Task]) -> bool:
    """Report whether every task carries what the two generated views index directly.

    A document missing one of these fields, or carrying one of the wrong JSON type, cannot be
    rendered at all. The per-task rules already report it by name, and a renderer crash here
    would replace that report with a traceback from inside ``validate()``.
    """
    return all(
        meta.get("status") in STATUSES
        and meta.get("priority") in PRIORITIES
        and all(
            isinstance(meta.get(name), str) for name in ("id", "title", "summary", "next_action")
        )
        for _, meta, _ in tasks
    )


def generated_view_errors(tasks: list[Task]) -> list[str]:
    """Check both task-derived views without allowing renderer errors to escape."""
    errors: list[str] = []
    if not renderable(tasks):
        return errors
    current = ROOT / "CURRENT.md"
    if current.exists() and current.read_text() != render_current(tasks):
        errors.append("CURRENT.md differs from generated tasks")
    try:
        expected_status = render_status_view(tasks)
    except StatusRenderError as error:
        errors.extend(str(error).splitlines())
    else:
        status = ROOT / "STATUS.md"
        if not status.exists() or status.read_text() != expected_status:
            errors.append("STATUS.md differs from generated tasks")
    return errors


def validate(*, live: bool = False) -> list[str]:
    errors: list[str] = []
    tasks = all_tasks()
    ids: dict[str, Path] = {}
    active_owners: dict[str, str] = {}
    active_worktrees: dict[str, str] = {}
    active_branches: dict[str, str] = {}
    for path, meta, body in tasks:
        task_id = str(meta.get("id", ""))
        if task_id in ids:
            errors.append(f"duplicate {task_id}")
        ids[task_id] = path
        errors.extend(basic_task_errors(path, meta))
        errors.extend(prose_errors(meta, body))
        errors.extend(claim_errors(meta, active_owners, active_worktrees, active_branches))
    errors.extend(schema_gate_errors(tasks))
    errors.extend(graph_errors(tasks))
    errors.extend(milestone_errors(tasks))
    errors.extend(generated_view_errors(tasks))
    errors.extend(privacy_errors())
    if live:
        state = project_scan()
        project, worktrees = live_docs(state)
        if (
            not (ROOT / "PROJECT_STATE.md").exists()
            or (ROOT / "PROJECT_STATE.md").read_text() != project
        ):
            errors.append("PROJECT_STATE.md is stale")
        if not (ROOT / "WORKTREES.md").exists() or (ROOT / "WORKTREES.md").read_text() != worktrees:
            errors.append("WORKTREES.md is stale")
    return errors


def require_state_branch() -> None:
    """Refuse to write or replicate state anywhere but the canonical checkout on ``main``.

    A linked worktree is refused even before its branch is read: it carries its own copy of this
    tool and of ``.runtime``, so a transaction there would hold a lock nobody else contends for.
    """
    git_dir = run(["git", "-C", str(ROOT), "rev-parse", "--absolute-git-dir"], check=False)
    common = run(
        ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        check=False,
    )
    if git_dir.stdout.strip() != common.stdout.strip():
        raise RuntimeError(
            "state transactions run only in the canonical product checkout, not in a linked "
            "worktree; invoke coordination/tools/handoffctl from the canonical checkout"
        )
    branch = run(
        ["git", "-C", str(ROOT), "symbolic-ref", "--short", "-q", "HEAD"], check=False
    ).stdout.strip()
    if branch != STATE_BRANCH:
        raise RuntimeError(
            f"state transactions commit only to {STATE_BRANCH}; the canonical checkout is on "
            f"{branch or 'a detached HEAD'}"
        )


def commit(message: str, paths: list[Path]) -> bool:
    """Commit exactly ``paths``, signed and signed off, on the state branch.

    The commit is limited to those paths. The canonical checkout is also a product checkout, so
    its index may hold staged product changes that are no business of a state transaction.
    """
    require_state_branch()
    relative = [str(path.relative_to(ROOT)) for path in paths]
    run(["git", "-C", str(ROOT), "add", "--", *relative])
    try:
        quiet = ["git", "-C", str(ROOT), "diff", "--cached", "--quiet", "--", *relative]
        if run(quiet, check=False).returncode == 0:
            return False
        run(
            ["git", "-C", str(ROOT), "commit", "-S", "-s", "-m", message, "--", *relative],
            capture=False,
        )
        return True
    except Exception:
        run(["git", "-C", str(ROOT), "reset", "--", *relative], check=False)
        raise


def push_replica() -> None:
    if not CONFIG.exists():
        return
    if not config().get("push_enabled", False):
        return
    require_state_branch()
    remote = run(["git", "-C", str(ROOT), "remote", "get-url", "origin"], check=False)
    if remote.returncode:
        raise RuntimeError("state replication is enabled but origin is missing")
    run(["git", "-C", str(ROOT), "fetch", "--no-tags", "origin", "main"])
    local_head = run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip()
    remote_head = run(["git", "-C", str(ROOT), "rev-parse", "FETCH_HEAD"]).stdout.strip()
    if local_head == remote_head:
        return
    ancestry = run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", remote_head, local_head],
        check=False,
    )
    if ancestry.returncode:
        raise RuntimeError(
            "state replica diverged; refusing non-fast-forward push. Product pull requests also "
            "advance origin/main: run `git pull --rebase origin main` in the canonical checkout, "
            "then `handoffctl reconcile --commit --push`"
        )
    run(["git", "-C", str(ROOT), "push", "origin", f"{local_head}:refs/heads/main"])


def reconcile(*, do_commit: bool, push: bool = False) -> bool:
    with locked():
        state = project_scan()
        generated = [
            ROOT / name for name in ("CURRENT.md", "STATUS.md", "PROJECT_STATE.md", "WORKTREES.md")
        ]
        before: dict[Path, str | None] = {path: path.read_text() for path, _, _ in all_tasks()}
        before.update({path: path.read_text() if path.exists() else None for path in generated})
        committed = False
        try:
            sync_task_observations(all_tasks(), state)
            tasks = all_tasks()
            atomic(ROOT / "CURRENT.md", render_current(tasks))
            atomic(ROOT / "STATUS.md", render_status_view(tasks))
            project, worktrees = live_docs(state)
            atomic(ROOT / "PROJECT_STATE.md", project)
            atomic(ROOT / "WORKTREES.md", worktrees)
            errors = validate(live=False)
            if errors:
                raise RuntimeError("validation failed:\n" + "\n".join(errors))
            touched = [
                path for path, old in before.items() if path.exists() and path.read_text() != old
            ]
            committed = (
                commit("chore(state): reconcile Game Experiment", touched) if do_commit else False
            )
            head = run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=False).stdout.strip()
            if push:
                push_replica()
            atomic(
                RUNTIME / "last-reconcile.json",
                json.dumps(
                    {"at": now(), "state_commit": head, "project_main": state["remote_main"]},
                    indent=2,
                )
                + "\n",
            )
            return committed if do_commit else True
        except Exception:
            if not committed:
                for path, old in before.items():
                    if old is None:
                        path.unlink(missing_ok=True)
                    else:
                        atomic(path, old)
            raise


def locate(task_id: str) -> Task:
    for path, meta, body in all_tasks():
        if meta["id"] == task_id:
            return path, meta, body
    raise RuntimeError(f"unknown task {task_id}")


def lease_expiry(minutes: int) -> str:
    """Return the instant a lease of this many minutes expires."""
    return (
        (dt.datetime.now(dt.UTC) + dt.timedelta(minutes=minutes)).replace(microsecond=0).isoformat()
    )


def apply_claim(args: argparse.Namespace, meta: Meta, tasks: list[Task]) -> str:
    if args.lease_minutes <= 0:
        raise RuntimeError("lease must be positive")
    if meta.get("status") != "open":
        raise RuntimeError(f"{args.task} is not open")
    states = {item["id"]: item["status"] for _, item, _ in tasks}
    pending = [item for item in meta.get("depends_on", []) if states.get(item) != "done"]
    if pending:
        raise RuntimeError("unfinished dependencies: " + ", ".join(pending))
    held = [
        item["id"]
        for _, item, _ in tasks
        if item.get("status") == "in_progress"
        and item.get("owner") == args.owner
        and item["id"] != args.task
    ]
    if held:
        raise RuntimeError(f"owner already holds {held[0]}")
    meta["owner"] = args.owner
    meta["status"] = "in_progress"
    meta["claim_expires"] = lease_expiry(args.lease_minutes)
    return f"Claimed by {args.owner}."


def dirty_state_paths() -> list[str]:
    """Return public state-repository changes that would make promotion ambiguous."""
    result = run(
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ]
    )
    return result.stdout.splitlines()


def apply_promote(args: argparse.Namespace, meta: Meta, tasks: list[Task]) -> str:
    """Validate the sole planned-to-open transition before changing task state."""
    if args.expected_revision != meta["task_revision"]:
        raise RuntimeError(
            f"stale revision: expected {args.expected_revision}, current {meta['task_revision']}"
        )
    if meta.get("status") != "planned":
        raise RuntimeError(f"{args.task} is not planned")
    if meta.get("owner") or meta.get("claim_expires"):
        raise RuntimeError(f"{args.task} has active claim metadata")
    states = {item["id"]: item["status"] for _, item, _ in tasks}
    pending = [item for item in meta.get("depends_on", []) if states.get(item) != "done"]
    if pending:
        raise RuntimeError("unfinished dependencies: " + ", ".join(pending))
    if not args.note.strip():
        raise RuntimeError("promotion note must not be empty")
    meta["status"] = "open"
    return str(args.note)


def apply_resume(args: argparse.Namespace, meta: Meta, _tasks: list[Task]) -> str:
    """Reopen a blocked task after an explicit coordinator review."""
    if args.expected_revision != meta["task_revision"]:
        raise RuntimeError(
            f"stale revision: expected {args.expected_revision}, current {meta['task_revision']}"
        )
    if meta.get("status") != "blocked":
        raise RuntimeError(f"{args.task} is not blocked")
    if meta.get("owner") or meta.get("claim_expires"):
        raise RuntimeError(f"{args.task} has active claim metadata")
    if not args.note.strip():
        raise RuntimeError("resume note must not be empty")
    meta["status"] = "open"
    return str(args.note)


def require_promotion_preflight(kind: str) -> None:
    """Reject a promotion before writes when its source checkout is ambiguous."""
    if kind not in ("promote", "resume"):
        return
    errors = validate(live=False)
    if errors:
        raise RuntimeError("promotion preflight failed:\n" + "\n".join(errors))
    if dirty_state_paths():
        raise RuntimeError("promotion requires a clean state repository")


def apply_heartbeat(args: argparse.Namespace, meta: Meta) -> str:
    """Extend a live claim, refusing to assert activity for a task that has none."""
    if args.lease_minutes <= 0 or meta.get("status") != "in_progress":
        raise RuntimeError("heartbeat requires an active task and positive lease")
    meta["claim_expires"] = lease_expiry(args.lease_minutes)
    return f"Heartbeat by {args.owner}."


def apply_submit(args: argparse.Namespace, meta: Meta) -> str:
    """Hand finished work to review and drop the claim that would otherwise expire under it.

    This is the only way a worker may leave ``in_progress``. It names no destination status, so
    a worker cannot record its own verdict, and it clears the lease, so a worker that finished
    is no longer indistinguishable from one that died. Clearing the claim is also what
    withdraws the worker's authority: every owner-authenticated transition — release, update,
    heartbeat, and submit itself — is refused once the owner field is empty, because an empty
    owner never authenticates and no other identity matches the cleared field.
    """
    if meta.get("status") != "in_progress":
        raise RuntimeError(f"{args.task} is not active")
    if not args.note.strip():
        raise RuntimeError("submission note must not be empty")
    meta["status"] = "in_review"
    meta["submitted_by"] = args.owner
    meta["owner"] = ""
    meta["claim_expires"] = ""
    return str(args.note)


def apply_release(args: argparse.Namespace, meta: Meta) -> str:
    """Close a live claim into a status that is not a verdict on the work.

    Release is owner authenticated, so whoever calls it holds the claim: naming ``done`` here
    would be a worker recording its own verdict, which is the invariant this vocabulary exists
    to protect. ``done`` is reachable only through review, and review only after submission.
    """
    if args.status == "in_review":
        raise RuntimeError("use submit to record work as awaiting review")
    if args.status == "done":
        raise RuntimeError("done is a review decision; submit the work for review instead")
    meta["status"] = args.status
    meta["owner"] = ""
    meta["claim_expires"] = ""
    return str(args.note)


def apply_update(args: argparse.Namespace, meta: Meta) -> str:
    """Record progress on a live claim without changing what the task is.

    Every other owned transition states its own precondition on the current status, and this
    one states it too rather than inheriting it from the coupling between an owner and an
    active task. The coupling is a repository-wide invariant checked after the write; a
    precondition checked here is what makes an update on work that is not active a refusal
    with a reason instead of a rollback.
    """
    if args.expected_revision is not None and args.expected_revision != meta["task_revision"]:
        expected = args.expected_revision
        current = meta["task_revision"]
        raise RuntimeError(f"stale revision: expected {expected}, current {current}")
    if meta.get("status") != "in_progress":
        raise RuntimeError(f"{args.task} is not active")
    if args.status is not None and args.status != "in_progress":
        raise RuntimeError("use release or submit for a non-active status")
    for name in ("status", "priority", "summary", "next_action"):
        value = getattr(args, name, None)
        if value is not None:
            meta[name] = value
    return str(args.note)


def apply_owned_change(args: argparse.Namespace, kind: str, meta: Meta) -> str:
    """Dispatch a transition only the recorded owner of a live claim is allowed to make.

    The empty string is how a task records that nobody holds it, so it is a sentinel and never
    a credential. Comparing it as one would authenticate a caller against exactly the tasks
    that have no owner left to authorise anything — submitted work above all, whose entire
    safety argument is that clearing the claim withdrew the worker's authority. It is refused
    before the comparison, so no caller can present the absence of an identity as one.
    """
    if not args.owner:
        raise RuntimeError(f"{args.task} needs an owner to authorise this change")
    if meta.get("owner") != args.owner:
        raise RuntimeError(f"{args.task} is owned by {meta.get('owner') or 'nobody'}")
    if kind == "heartbeat":
        return apply_heartbeat(args, meta)
    if kind == "submit":
        return apply_submit(args, meta)
    if kind == "release":
        return apply_release(args, meta)
    return apply_update(args, meta)


def apply_review(args: argparse.Namespace, meta: Meta, _tasks: list[Task]) -> str:
    """Record the coordinator's decision on submitted work.

    This is the only transition out of ``in_review``, and the only one that can reach ``done``
    from it. It is not owner authenticated, because submitting cleared the claim: no worker can
    reach it with the credential it held. The worker that submitted the work is refused by
    name as well, so certifying one's own work takes a second, named identity.
    """
    if args.expected_revision != meta["task_revision"]:
        raise RuntimeError(
            f"stale revision: expected {args.expected_revision}, current {meta['task_revision']}"
        )
    if meta.get("status") != "in_review":
        raise RuntimeError(f"{args.task} is not awaiting review")
    if args.status in ("in_progress", "in_review"):
        raise RuntimeError("review records a decision, not another active status")
    if not args.reviewer.strip():
        raise RuntimeError("reviewer must not be empty")
    if args.reviewer == meta.get("submitted_by"):
        raise RuntimeError(f"{args.task} cannot be reviewed by the worker that submitted it")
    if not args.note.strip():
        raise RuntimeError("review note must not be empty")
    meta["status"] = args.status
    meta["submitted_by"] = ""
    return str(args.note)


def apply_expire(args: argparse.Namespace, meta: Meta, _tasks: list[Task]) -> str:
    """Clear a lease whose worker has gone away, and record who cleared it and why.

    A lease exists to surface a worker that stopped, and every other transition in this tool
    stays fenced by that report. This one is the exception, and it is deliberately the only
    one: it is the coordinator's repair, so it is not owner authenticated — the owner of an
    abandoned claim is by definition unable to call anything, and a transition that accepted
    the dead worker's name would be an invitation to assert an identity nobody holds.

    A live claim is refused. The condition accepted here is exactly the one ``doctor``
    reports, compared against the message rather than recomputed, so expire can never reach a
    task whose lease is still running and can never be used to take work away from a worker
    that is alive. The task returns to ``open`` and nowhere else: reopening the queue is the
    whole repair, and any other destination would make this a second, unreviewed route to a
    status that its own transition already governs.
    """
    if args.expected_revision != meta["task_revision"]:
        raise RuntimeError(
            f"stale revision: expected {args.expected_revision}, current {meta['task_revision']}"
        )
    if meta.get("status") != "in_progress":
        raise RuntimeError(f"{args.task} is not active")
    task_id = str(meta.get("id", ""))
    expiry = meta.get("claim_expires")
    if active_expiry_errors(task_id, expiry) != [f"{task_id}: expired claim"]:
        raise RuntimeError(f"{args.task} does not hold an expired claim")
    if not args.cleared_by.strip():
        raise RuntimeError("clearing an abandoned claim must name who cleared it")
    if not args.note.strip():
        raise RuntimeError("expiry note must not be empty")
    abandoned = str(meta.get("owner"))
    meta["status"] = "open"
    meta["owner"] = ""
    meta["claim_expires"] = ""
    return (
        f"Abandoned claim of {abandoned}, lease ended {expiry}, "
        f"cleared by {args.cleared_by}: {args.note}"
    )


TRANSITIONS: dict[str, Callable[[argparse.Namespace, Meta, list[Task]], str]] = {
    "claim": apply_claim,
    "expire": apply_expire,
    "promote": apply_promote,
    "resume": apply_resume,
    "review": apply_review,
}


def tolerated_expiry_errors(kind: str) -> frozenset[str]:
    """Return the standing expired-claim errors one recovery transition may proceed despite.

    ``validate()`` runs over every task inside every transaction, so before this existed a
    single abandoned lease refused every mutation — including release and heartbeat, the only
    two transitions that could have cleared one. The coordinator could not repair the
    coordinator, and the only recorded recovery was a human editing task files outside the
    tool.

    The exemption is narrowed on three axes at once, because a wider one would be the
    weakening this repair exists to avoid rather than the repair. It applies to ``expire``
    alone, so every other transition stays fenced by the full check. It matches a *whole*
    message and never a substring, so a different fault that merely contains the tolerated
    text — an unknown field named after it is enough — is not excused, and a corrupt task, a
    broken dependency graph, a milestone fault, a privacy leak or a claim error of any other
    kind still refuses even this transition. And it is the set observed *before* the write, so
    a lease that lapses part-way through the transaction fails closed rather than being
    excused by the transaction that raced it.

    One fault it cannot be narrowed against is a stale generated view, because no transition
    can be refused by one: ``mutate`` regenerates ``CURRENT.md`` and ``STATUS.md`` from the
    tasks immediately before it validates, so ``generated_view_errors`` never fires inside a
    transaction. That is a property of the transaction, identical for ``heartbeat``, and not
    of this tolerance.
    """
    if kind != "expire":
        return frozenset()
    return frozenset(value for value in validate(live=False) if EXPIRED_CLAIM.fullmatch(value))


def apply_transition(args: argparse.Namespace, kind: str, meta: Meta) -> str:
    """Route one transition to the rules that own it, so mutate stays a plain transaction."""
    handler = TRANSITIONS.get(kind)
    if handler is None:
        return apply_owned_change(args, kind, meta)
    return handler(args, meta, all_tasks())


def mutate(args: argparse.Namespace, kind: str) -> None:
    with locked():
        path, meta, body = locate(args.task)
        require_promotion_preflight(kind)
        old_task = path.read_text()
        current_path = ROOT / "CURRENT.md"
        old_current = current_path.read_text() if current_path.exists() else ""
        status_path = ROOT / "STATUS.md"
        old_status = status_path.read_text() if status_path.exists() else None
        committed = False
        note = apply_transition(args, kind, meta)
        tolerated = tolerated_expiry_errors(kind)
        meta["task_revision"] += 1
        meta["updated_at"] = now()
        if note:
            body = append_evidence(body, str(meta["updated_at"]), note)
        try:
            write_task(path, meta, body)
            tasks = all_tasks()
            atomic(current_path, render_current(tasks))
            atomic(status_path, render_status_view(tasks))
            errors = [value for value in validate(live=False) if value not in tolerated]
            if errors:
                raise RuntimeError("\n".join(errors))
            committed = commit(
                f"chore(state): {kind} {args.task}", [path, current_path, status_path]
            )
            push_replica()
        except Exception:
            # A signed local commit is already durable even when replication fails.
            # Keep its worktree representation intact so a later reconcile can safely
            # inspect and retry the push instead of silently rolling state backward.
            if not committed:
                atomic(path, old_task)
                if old_current:
                    atomic(current_path, old_current)
                if old_status is None:
                    status_path.unlink(missing_ok=True)
                else:
                    atomic(status_path, old_status)
            raise


def apply_milestone_completion(identifier: str) -> Milestone:
    """Return the milestone to record complete, or refuse the transition with a reason."""
    tasks = all_tasks()
    for path, meta, body in milestone_documents():
        if meta.get("id") != identifier:
            continue
        contract = milestone_field_errors(path, meta) + milestone_body_errors(path, meta, body)
        if contract:
            raise RuntimeError("milestone contract unmet:\n" + "\n".join(contract))
        if meta.get("status") == "complete":
            raise RuntimeError(f"{identifier} is already complete")
        meta["status"] = "complete"
        unfinished = milestone_completion_errors([(path, meta, body)], tasks)
        if unfinished:
            raise RuntimeError("milestone has unfinished work:\n" + "\n".join(unfinished))
        return path, meta, body
    raise RuntimeError(f"unknown milestone {identifier}")


def milestone_closure_note(series: str, tasks: list[Task]) -> str:
    """Name every AR the milestone is closed over that reached a terminal state short of done.

    ``complete-milestone`` enforces the floor: no AR may still be live. A milestone document
    may demand more of itself, so closing over cancelled or superseded work is recorded
    rather than silent, and the coordinator can see it in the document and in the commit.
    """
    exceptions = sorted(
        f"{item['id']} ({item['status']})"
        for _, item, _ in tasks
        if str(item.get("id", ""))[3:5] == series and item.get("status") != "done"
    )
    if not exceptions:
        return "Recorded complete. Every AR in the series is done."
    return "Recorded complete over ARs that are not done: " + ", ".join(exceptions) + "."


def cmd_complete_milestone(identifier: str) -> None:
    """Record a milestone complete under the coordinator lock, or refuse and change nothing."""
    with locked():
        path, meta, body = apply_milestone_completion(identifier)
        note = milestone_closure_note(milestone_series(meta), all_tasks())
        body = append_evidence(body, now(), note)
        before = path.read_text()
        committed = False
        try:
            write_task(path, meta, body)
            errors = validate(live=False)
            if errors:
                raise RuntimeError("\n".join(errors))
            committed = commit(f"chore(state): complete {identifier}\n\n{note}", [path])
            push_replica()
            print(f"{identifier}: {note}")
        except Exception:
            # A signed local commit is already durable; leave it for a later reconcile.
            if not committed:
                atomic(path, before)
            raise


def cmd_render_status(*, check: bool) -> None:
    """Render STATUS.md or fail if its checked-in form is stale."""
    with locked(exclusive=not check):
        expected = render_status_view(all_tasks())
        path = ROOT / "STATUS.md"
        if check:
            if not path.exists() or path.read_text() != expected:
                raise RuntimeError("STATUS.md differs from generated tasks")
            return
        atomic(path, expected)


def cmd_doctor(*, live: bool) -> int:
    """Validate static state and optionally compare the live generated views."""
    errors = validate(live=live)
    if errors:
        print("\n".join("ERROR: " + value for value in errors))
        if any(EXPIRED_CLAIM.fullmatch(value) for value in errors):
            print(EXPIRY_HINT)
        return 1
    print(
        "OK: structure, references, privacy, generated views"
        + (" and live state" if live else "")
        + " are consistent"
    )
    return 0


def cmd_check_commits(*, base: str, head: str) -> int:
    """Reject private references in the messages of newly introduced commits."""
    errors = commit_privacy_errors(base, head)
    if errors:
        print("\n".join("ERROR: " + value for value in errors))
        return 1
    print("OK: introduced commit messages carry no private reference")
    return 0


def cmd_snapshot() -> None:
    with locked(exclusive=False):
        errors = validate(live=True)
        if errors:
            raise RuntimeError("snapshot refused:\n" + "\n".join(errors))
        print("STATE_COMMIT=" + run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip())
        print((ROOT / "CURRENT.md").read_text(), end="")


def require_active_owner(task_id: str, owner: str) -> None:
    """Fence wrapped commands with a live claim before external effects."""
    with locked(exclusive=False):
        _, meta, _ = locate(task_id)
        if meta.get("owner") != owner:
            raise RuntimeError("task claim does not match owner")
        if meta.get("status") != "in_progress":
            raise RuntimeError("task claim is not active")
        errors = active_expiry_errors(task_id, meta.get("claim_expires"))
        if errors:
            raise RuntimeError(errors[0])


def cmd_run(args: argparse.Namespace) -> int:
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        raise RuntimeError("missing command")
    require_active_owner(args.task, args.owner)
    # Never execute or consume untracked caller input. Scripts and data must be
    # named by argv or a stable file, whose content digest can be recorded too.
    proc = subprocess.run(args.command, check=False, stdin=subprocess.DEVNULL)
    reconcile(do_commit=True, push=True)

    command_hash = hashlib.sha256("\0".join(args.command).encode()).hexdigest()
    update = argparse.Namespace(
        task=args.task,
        owner=args.owner,
        # This internal append is serialized by mutate and must attach to the
        # latest task revision after concurrent observation reconciliation.
        expected_revision=None,
        status=None,
        priority=None,
        summary=None,
        next_action=None,
        note=f"Recorded command exit {proc.returncode}; command argv SHA-256 {command_hash}.",
    )
    mutate(update, "update")
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    """Declare every subcommand and its arguments."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="cmd", required=True)
    item = commands.add_parser("reconcile")
    item.add_argument("--commit", action="store_true")
    item.add_argument("--push", action="store_true")
    commands.add_parser("snapshot")
    item = commands.add_parser("check-commits")
    item.add_argument("--base", default="")
    item.add_argument("--head", default="HEAD")
    item = commands.add_parser("doctor")
    item.add_argument("--live", action="store_true")
    item = commands.add_parser("render-status")
    item.add_argument("--check", action="store_true")
    item = commands.add_parser("complete-milestone")
    item.add_argument("milestone")
    for name in ("claim", "heartbeat"):
        item = commands.add_parser(name)
        item.add_argument("task")
        item.add_argument("--owner", required=True)
        item.add_argument("--lease-minutes", type=int, default=120)
    item = commands.add_parser("release")
    item.add_argument("task")
    item.add_argument("--owner", required=True)
    item.add_argument("--status", required=True, choices=RELEASABLE)
    item.add_argument("--note", required=True)
    item = commands.add_parser("submit")
    item.add_argument("task")
    item.add_argument("--owner", required=True)
    item.add_argument("--note", required=True)
    item = commands.add_parser("review")
    item.add_argument("task")
    item.add_argument("--reviewer", required=True)
    item.add_argument("--expected-revision", type=int, required=True)
    item.add_argument("--status", required=True, choices=DECIDED)
    item.add_argument("--note", required=True)
    item = commands.add_parser("expire")
    item.add_argument("task")
    item.add_argument("--expected-revision", type=int, required=True)
    item.add_argument("--cleared-by", required=True)
    item.add_argument("--note", required=True)
    item = commands.add_parser("promote")
    item.add_argument("task")
    item.add_argument("--expected-revision", type=int, required=True)
    item.add_argument("--note", required=True)
    item = commands.add_parser("resume")
    item.add_argument("task")
    item.add_argument("--expected-revision", type=int, required=True)
    item.add_argument("--note", required=True)
    item = commands.add_parser("update")
    item.add_argument("task")
    item.add_argument("--owner", required=True)
    item.add_argument("--expected-revision", type=int, required=True)
    item.add_argument("--status", choices=STATUSES)
    item.add_argument("--priority", choices=PRIORITIES)
    item.add_argument("--summary")
    item.add_argument("--next-action")
    item.add_argument("--note", required=True)
    item = commands.add_parser("run")
    item.add_argument("task")
    item.add_argument("--owner", required=True)
    item.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    """Parse argv and dispatch one coordination command."""
    args = build_parser().parse_args()
    if args.cmd == "reconcile":
        reconcile(do_commit=args.commit, push=args.push)
    elif args.cmd == "snapshot":
        cmd_snapshot()
    elif args.cmd == "check-commits":
        return cmd_check_commits(base=args.base, head=args.head)
    elif args.cmd == "doctor":
        return cmd_doctor(live=args.live)
    elif args.cmd == "render-status":
        cmd_render_status(check=args.check)
    elif args.cmd == "complete-milestone":
        cmd_complete_milestone(args.milestone)
    elif args.cmd in (
        "claim",
        "expire",
        "heartbeat",
        "release",
        "submit",
        "review",
        "promote",
        "resume",
        "update",
    ):
        mutate(args, args.cmd)
    elif args.cmd == "run":
        return cmd_run(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
