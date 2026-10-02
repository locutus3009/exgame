#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Validate every task and milestone document against its published strict schema."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent


def metadata(path: Path) -> object:
    """Return the JSON-compatible front matter."""
    text = path.read_text()
    end = text.index("\n---\n", 4)
    return json.loads(text[4:end])


def failures(schema_name: str, directory: str, pattern: str) -> list[str]:
    """Validate the named schema itself, then every document it governs."""
    schema = json.loads((ROOT / "schema" / schema_name).read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{path.name}: {error.message}"
        for path in sorted((ROOT / directory).glob(pattern))
        for error in sorted(validator.iter_errors(metadata(path)), key=str)
    ]


def task(**changes: object) -> dict[str, object]:
    """Return a minimal valid task, with the caller's fields applied over it."""
    meta: dict[str, object] = {
        "schema_version": 1,
        "id": "AR-0001",
        "title": "Fixture",
        "status": "open",
        "priority": "P1",
        "summary": "Fixture task.",
        "next_action": "None.",
        "task_revision": 1,
        "updated_at": "2026-09-07T12:00:00+00:00",
        "owner": "",
        "claim_expires": "",
        "worktree_key": "",
        "branch": "",
        "checkpoint_commit": "",
        "plan": "",
        "depends_on": [],
    }
    meta.update(changes)
    return meta


LEASE = "2099-01-01T00:00:00+00:00"
# The claim and submission states the protocol defines, and the ones it forbids. The documents
# in tasks/ only ever exhibit the states the repository happens to be in, so without these the
# conditional half of the schema would be unexercised by any gate.
TASK_FIXTURES: tuple[tuple[str, dict[str, object], bool], ...] = (
    (
        "an active task with an owner and a lease",
        task(status="in_progress", owner="w", claim_expires=LEASE),
        True,
    ),
    ("an active task with no owner", task(status="in_progress", claim_expires=LEASE), False),
    ("an active task with no lease", task(status="in_progress", owner="w"), False),
    (
        "work awaiting review that names its submitter",
        task(status="in_review", submitted_by="w"),
        True,
    ),
    ("work awaiting review with no submitter", task(status="in_review"), False),
    (
        "work awaiting review with an empty submitter",
        task(status="in_review", submitted_by=""),
        False,
    ),
    (
        "work awaiting review that still holds an owner",
        task(status="in_review", submitted_by="w", owner="w"),
        False,
    ),
    (
        "work awaiting review that still holds a lease",
        task(status="in_review", submitted_by="w", claim_expires=LEASE),
        False,
    ),
    ("a submitter recorded on finished work", task(status="done", submitted_by="w"), False),
    (
        "a submitter recorded on an active task",
        task(status="in_progress", owner="w", claim_expires=LEASE, submitted_by="w"),
        False,
    ),
    ("an unknown status", task(status="in_reviewed"), False),
)


def fixture_failures() -> list[str]:
    """Prove the published task schema accepts and rejects the states the protocol defines."""
    schema = json.loads((ROOT / "schema" / "task-schema.json").read_text())
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    problems = []
    for label, meta, expected in TASK_FIXTURES:
        accepted = not list(validator.iter_errors(meta))
        if accepted != expected:
            verdict = "accepted" if accepted else "rejected"
            problems.append(f"task-schema.json: {label} was {verdict}")
    return problems


def main() -> int:
    """Validate the schemas themselves and every document they govern."""
    problems = failures("task-schema.json", "tasks", "AR-*.md")
    problems.extend(failures("milestone-schema.json", "milestones", "M*.md"))
    problems.extend(fixture_failures())
    if problems:
        raise SystemExit("\n".join(problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
