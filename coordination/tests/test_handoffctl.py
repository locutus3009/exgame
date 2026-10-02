# SPDX-License-Identifier: MIT
"""Fault, consistency, claim and generation tests for handoffctl."""

import argparse
import datetime as dt
import fcntl
import importlib.util
import io
import json
import multiprocessing
import os
import re
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest.mock import patch

from jsonschema import Draft202012Validator, FormatChecker

SOURCE = Path(__file__).resolve().parent.parent / "tools/handoffctl.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("handoffctl_core", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load handoffctl")
CORE: Any = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def hold_lock(lock_path: str, ready: Any, release: Any) -> None:
    """Hold an exclusive lock in a separate process."""
    CORE.LOCK = Path(lock_path)
    CORE.RUNTIME = CORE.LOCK.parent
    with CORE.locked():
        ready.set()
        release.wait(5)


def configure_child(root_value: str) -> None:
    """Point the imported coordinator module at a process-shared fixture."""
    root = Path(root_value)
    CORE.ROOT = root
    CORE.TASKS = root / "tasks"
    CORE.RUNTIME = root / ".runtime"
    CORE.LOCK = CORE.RUNTIME / "state.lock"
    CORE.CONFIG = CORE.RUNTIME / "config.json"


def concurrent_claim(root_value: str, start: Any) -> None:
    """Claim through the real locked mutation path in a child process."""
    configure_child(root_value)
    start.wait(5)
    with patch.object(CORE, "commit", return_value=True):
        CORE.mutate(argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim")


def concurrent_reconcile(root_value: str, start: Any) -> None:
    """Reconcile through the real locked generation path in a child process."""
    configure_child(root_value)
    start.wait(5)
    state = {
        "remote_main": "a" * 40,
        "origin_main": "a" * 40,
        "primary_head": "b" * 40,
        "worktrees": [],
        "prs": [],
        "runs": [],
    }
    completed = subprocess.CompletedProcess(["git"], 0, stdout="b" * 40 + "\n", stderr="")
    with (
        patch.object(CORE, "project_scan", return_value=state),
        patch.object(CORE, "run", return_value=completed),
    ):
        CORE.reconcile(do_commit=False)


def concurrent_promote(root_value: str, start: Any) -> None:
    """Promote through the real locked mutation path in a child process."""
    configure_child(root_value)
    start.wait(5)
    args = argparse.Namespace(
        task="AR-0001",
        expected_revision=1,
        note="dependencies verified",
    )
    with (
        patch.object(CORE, "commit", return_value=True),
        patch.object(CORE, "dirty_state_paths", return_value=[]),
    ):
        CORE.mutate(args, "promote")


def concurrent_complete_milestone(root_value: str, start: Any) -> None:
    """Complete a milestone through the real locked transition in a child process."""
    configure_child(root_value)
    start.wait(5)
    with patch.object(CORE, "commit", return_value=True), patch("builtins.print"):
        CORE.cmd_complete_milestone("M0")


def probe_lock(lock_path: str, taken: Any) -> None:
    """Report from another process whether the coordinator lock is currently held."""
    handle = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)
        taken.value = 0
    except BlockingIOError:
        taken.value = 1
    finally:
        os.close(handle)


ABSENT = object()
LEASE = "2099-01-01T00:00:00+00:00"
# One fixture per constraint the published task schema states, keyed by the schema path and
# keyword it breaches. test_every_published_constraint_has_a_fixture derives the same key set
# from the schema itself and refuses any pair without a fixture, so a bound added to the
# schema cannot arrive unproved -- which is the failure this whole change exists to close.
TASK_CONSTRAINT_FIXTURES: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("schema_version", "const", {"schema_version": 2}),
    # true is not 1 to JSON Schema and is to Python, so a const stated as a number must not
    # accept the boolean. Found by fuzzing validate() against the real validator, not by the
    # fixture list, which is why the fuzz is recorded in the evidence.
    ("schema_version", "const", {"schema_version": True}),
    ("id", "type", {"id": 1}),
    ("id", "pattern", {"id": "AR-1"}),
    ("title", "type", {"title": 1}),
    ("title", "minLength", {"title": ""}),
    ("title", "maxLength", {"title": "x" * 121}),
    ("status", "enum", {"status": "in_reviewed"}),
    ("priority", "enum", {"priority": "P9"}),
    ("summary", "type", {"summary": []}),
    ("summary", "minLength", {"summary": ""}),
    ("summary", "maxLength", {"summary": "x" * 301}),
    ("next_action", "type", {"next_action": None}),
    ("next_action", "minLength", {"next_action": ""}),
    ("next_action", "maxLength", {"next_action": "x" * 301}),
    # True is an int to Python and is not an integer to JSON Schema. A gate that used
    # isinstance alone would accept it and the schema gate would not.
    ("task_revision", "type", {"task_revision": True}),
    ("task_revision", "minimum", {"task_revision": 0}),
    ("updated_at", "type", {"updated_at": 20260908}),
    # fromisoformat accepts a bare date; RFC 3339 date-time does not.
    ("updated_at", "format", {"updated_at": "2026-09-08"}),
    ("owner", "type", {"status": "in_progress", "owner": 5, "claim_expires": LEASE}),
    (
        "owner",
        "maxLength",
        {"status": "in_progress", "owner": "w" * 81, "claim_expires": LEASE},
    ),
    ("claim_expires", "type", {"status": "in_progress", "owner": "w", "claim_expires": 5}),
    (
        "claim_expires",
        "maxLength",
        {"status": "in_progress", "owner": "w", "claim_expires": LEASE + "0" * 20},
    ),
    ("submitted_by", "type", {"status": "in_review", "submitted_by": 7}),
    ("submitted_by", "maxLength", {"status": "in_review", "submitted_by": "w" * 81}),
    ("worktree_key", "type", {"worktree_key": 3}),
    ("worktree_key", "pattern", {"worktree_key": "Game-Experiment"}),
    ("branch", "type", {"branch": 1}),
    ("branch", "maxLength", {"branch": "b" * 161}),
    ("checkpoint_commit", "type", {"checkpoint_commit": 1}),
    ("checkpoint_commit", "pattern", {"checkpoint_commit": "z" * 40}),
    ("plan", "type", {"plan": 1}),
    ("plan", "pattern", {"plan": "../plans/notes.md"}),
    ("depends_on", "type", {"depends_on": "AR-0002"}),
    ("depends_on", "uniqueItems", {"depends_on": ["AR-0002", "AR-0002"]}),
    ("depends_on[]", "type", {"depends_on": [1]}),
    ("depends_on[]", "pattern", {"depends_on": ["ar-0002"]}),
    ("observed_branch", "type", {"observed_branch": 1}),
    ("observed_branch", "maxLength", {"observed_branch": "b" * 161}),
    ("observed_head", "type", {"observed_head": 1}),
    ("observed_head", "pattern", {"observed_head": "z" * 40}),
    ("observed_dirty", "type", {"observed_dirty": "1"}),
    ("observed_dirty", "minimum", {"observed_dirty": -1}),
    (
        "allOf[0].then.owner",
        "minLength",
        {"status": "in_progress", "owner": "", "claim_expires": LEASE},
    ),
    (
        "allOf[0].then.claim_expires",
        "minLength",
        {"status": "in_progress", "owner": "w", "claim_expires": "2099"},
    ),
    ("allOf[0].else.owner", "maxLength", {"owner": "w"}),
    ("allOf[0].else.claim_expires", "maxLength", {"claim_expires": LEASE}),
    ("allOf[1].then", "required", {"status": "in_review", "submitted_by": ABSENT}),
    ("allOf[1].then.submitted_by", "minLength", {"status": "in_review", "submitted_by": ""}),
    ("allOf[1].else.submitted_by", "maxLength", {"submitted_by": "w"}),
    ("", "required", {"summary": ABSENT}),
    ("", "additionalProperties", {"extra": "bad"}),
)
MILESTONE_CONSTRAINT_FIXTURES: tuple[tuple[str, str, dict[str, object]], ...] = (
    ("schema_version", "const", {"schema_version": 2}),
    ("id", "type", {"id": 1}),
    ("id", "pattern", {"id": "M007"}),
    ("label", "type", {"label": 1}),
    ("label", "minLength", {"label": ""}),
    ("label", "maxLength", {"label": "x" * 81}),
    ("label", "pattern", {"label": "   "}),
    ("status", "enum", {"status": "retired"}),
    ("", "required", {"label": ABSENT}),
    ("", "additionalProperties", {"stage": "late"}),
)


def stated_field_constraints(prefix: str, field: str, rules: Any) -> dict[tuple[str, str], Any]:
    """Return every bound one property subschema states, flattening array items."""
    stated: dict[tuple[str, str], Any] = {}
    for keyword, bound in rules.items():
        if keyword == "items":
            stated.update(stated_field_constraints(prefix, f"{field}[]", bound))
        else:
            stated[(f"{prefix}{field}", keyword)] = bound
    return stated


def stated_constraints(schema: dict[str, Any], prefix: str = "") -> dict[tuple[str, str], Any]:
    """Return every bound a published schema states, keyed by its path and keyword.

    The ``if`` half of a conditional states a condition rather than a bound, so it is not
    enumerated; ``then`` and ``else`` are, because a document can breach them.
    """
    stated: dict[tuple[str, str], Any] = {}
    if "required" in schema:
        stated[(prefix.rstrip("."), "required")] = schema["required"]
    if schema.get("additionalProperties") is False:
        stated[(prefix.rstrip("."), "additionalProperties")] = False
    for field, rules in schema.get("properties", {}).items():
        stated.update(stated_field_constraints(prefix, field, rules))
    for index, branch in enumerate(schema.get("allOf", [])):
        for name in ("then", "else"):
            if name in branch:
                stated.update(stated_constraints(branch[name], f"{prefix}allOf[{index}].{name}."))
    return stated


def on_main(args: list[str]) -> str:
    """Answer the state-branch guard's branch query as the canonical checkout on main would."""
    return "main\n" if "symbolic-ref" in args else ""


class HandoffTest(unittest.TestCase):
    """Exercise transaction safety without accessing the live project."""

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        CORE.ROOT = root
        CORE.TASKS = root / "tasks"
        CORE.RUNTIME = root / ".runtime"
        CORE.LOCK = CORE.RUNTIME / "state.lock"
        CORE.CONFIG = CORE.RUNTIME / "config.json"
        CORE.TASKS.mkdir()
        (root / "plans").mkdir()
        (root / "milestones").mkdir()
        self.root = root
        self.make_milestone()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_task(self, task_id: str = "AR-0001", **changes: object) -> Path:
        """Create one valid task fixture."""
        meta = {
            "schema_version": 1,
            "id": task_id,
            "title": "Test task",
            "status": "open",
            "priority": "P1",
            "summary": "Ready for testing.",
            "next_action": "Run the test.",
            "task_revision": 1,
            "updated_at": "2026-09-03T20:00:00+00:00",
            "owner": "",
            "claim_expires": "",
            "worktree_key": "",
            "branch": "",
            "checkpoint_commit": "",
            "plan": "",
            "depends_on": [],
        }
        meta.update(changes)
        path = CORE.TASKS / f"{task_id}-test.md"
        CORE.write_task(path, meta, "# Test\n")
        self.refresh_views()
        return cast(Path, path)

    def make_milestone(
        self, identifier: str = "M0", label: str = "Process foundation", **changes: object
    ) -> Path:
        """Create one milestone document that satisfies the declared contract."""
        meta: dict[str, object] = {
            "schema_version": 1,
            "id": identifier,
            "label": label,
            "status": "active",
        }
        meta.update(changes)
        body = (
            f"# {identifier} \u2014 {label}\n\n"
            "## Outcome\n\nA coordination machine whose claims are verifiable.\n\n"
            "## Exit criteria\n\n1. Every AR in the series is finished.\n\n"
            "## Out of scope\n\nProduct feature work.\n"
        )
        path = self.root / "milestones" / f"{identifier}.md"
        CORE.write_task(path, meta, body)
        return path

    def refresh_views(self) -> None:
        """Refresh both deterministic task views in a fixture repository."""
        tasks = CORE.all_tasks()
        CORE.atomic(self.root / "CURRENT.md", CORE.render_current(tasks))
        CORE.atomic(self.root / "STATUS.md", CORE.render_status_view(tasks))

    def test_atomic_task_round_trip_and_render(self) -> None:
        path = self.make_task()
        meta, body = CORE.read_task(path)
        self.assertEqual("AR-0001", meta["id"])
        self.assertEqual("# Test\n", body)
        current = CORE.render_current(CORE.all_tasks())
        self.assertIn("## Open", current)
        self.assertIn("[AR-0001]", current)
        status = CORE.render_status_view(CORE.all_tasks())
        self.assertIn("flowchart LR", status)
        self.assertIn("**1 ARs tracked**", status)

    def test_status_is_deterministic_complete_accessible_and_injection_safe(self) -> None:
        self.make_task(
            "AR-0001",
            title="Hostile ](https://example.invalid) | `code`\n%%{init: bad}%%",
            summary="<script>alert(1)</script>",
        )
        self.make_task("AR-0002", status="done", priority="P0", depends_on=["AR-0001"])
        tasks = CORE.all_tasks()
        status = CORE.render_status_view(tasks)
        self.assertEqual(status, CORE.render_status_view(list(reversed(tasks))))
        self.assertEqual(2, status.count(":::status_"))
        self.assertIn('subgraph series_00["00 - Process foundation"]', status)
        self.assertEqual(1, status.count("AR_0001 --> AR_0002"))
        self.assertIn("Accessible dependency index", status)
        self.assertIn("&#93;(https://example.invalid)", status)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", status)
        self.assertNotIn("%%{init: bad}%%", status)

    def test_future_series_is_never_omitted_from_graph_or_text_fallback(self) -> None:
        self.make_task("AR-1101")
        status = CORE.render_status_view(CORE.all_tasks())
        self.assertEqual(1, status.count('AR_1101["AR-1101 - Open"]'))
        self.assertIn('subgraph series_11["11 - Additional work"]', status)
        self.assertIn("| [AR-1101](tasks/AR-1101-test.md) | None | None |", status)

    def test_status_rejects_malformed_duplicate_self_missing_and_cycles(self) -> None:
        first = self.make_task("AR-0001")
        second = self.make_task("AR-0002")
        meta, body = CORE.read_task(first)
        meta["depends_on"] = ["AR-0001", "AR-9999"]
        CORE.write_task(first, meta, body)
        other, other_body = CORE.read_task(second)
        other["depends_on"] = ["AR-0001", "AR-0001"]
        CORE.write_task(second, other, other_body)
        errors = "\n".join(CORE.graph_errors(CORE.all_tasks()))
        self.assertIn("duplicate dependency AR-0001", errors)
        self.assertIn("self dependency", errors)
        self.assertIn("missing dependency AR-9999", errors)
        self.assertIn("dependency cycle", errors)
        other["depends_on"] = "AR-0001"
        CORE.write_task(second, other, other_body)
        self.assertIn("must be a list", "\n".join(CORE.graph_errors(CORE.all_tasks())))
        duplicate = self.root / "tasks" / "AR-0001-copy.md"
        CORE.write_task(duplicate, meta, body)
        self.assertIn("duplicate graph node", "\n".join(CORE.graph_errors(CORE.all_tasks())))

    def test_status_rejects_unsafe_filename_and_unknown_presentation(self) -> None:
        path = self.make_task()
        task = CORE.all_tasks()[0]
        with self.assertRaisesRegex(CORE.StatusRenderError, "unsafe"):
            CORE.render_status_view([(path.with_name("AR-0001-BAD.md"), task[1], task[2])])
        task[1]["priority"] = "PX"
        with self.assertRaisesRegex(CORE.StatusRenderError, "unknown status or priority"):
            CORE.render_status_view([task])

    def test_rejects_bad_front_matter(self) -> None:
        path = CORE.TASKS / "AR-0001-bad.md"
        path.write_text("bad")
        with self.assertRaisesRegex(ValueError, "no front matter"):
            CORE.read_task(path)
        path.write_text("---\n{}")
        with self.assertRaisesRegex(ValueError, "unterminated"):
            CORE.read_task(path)

    def test_validation_finds_schema_graph_claim_and_privacy_errors(self) -> None:
        first = self.make_task("AR-0001", extra="bad")
        second = self.make_task(
            "AR-0002",
            status="in_progress",
            owner="worker-a",
            claim_expires="",
        )
        first_meta, first_body = CORE.read_task(first)
        second_meta, second_body = CORE.read_task(second)
        first_meta["depends_on"] = ["AR-0002"]
        second_meta["depends_on"] = ["AR-0001"]
        CORE.write_task(first, first_meta, first_body)
        CORE.write_task(second, second_meta, second_body)
        (self.root / "leak.md").write_text("/" + "home/example")
        errors = CORE.validate()
        joined = "\n".join(errors)
        self.assertIn("unknown field extra", joined)
        self.assertIn("active without claim", joined)
        self.assertIn("dependency cycle", joined)
        self.assertIn("absolute Linux home path", joined)

    def test_claim_update_release_and_stale_revision(self) -> None:
        path = self.make_task()
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10),
                "claim",
            )
            meta, _ = CORE.read_task(path)
            self.assertEqual("in_progress", meta["status"])
            revision = meta["task_revision"]
            with self.assertRaisesRegex(RuntimeError, "stale revision"):
                CORE.mutate(
                    argparse.Namespace(
                        task="AR-0001",
                        owner="worker-a",
                        expected_revision=revision - 1,
                        status=None,
                        priority=None,
                        summary=None,
                        next_action=None,
                        note="stale",
                    ),
                    "update",
                )
            CORE.mutate(
                argparse.Namespace(
                    task="AR-0001",
                    owner="worker-a",
                    expected_revision=revision,
                    status=None,
                    priority="P0",
                    summary="Updated.",
                    next_action="Continue.",
                    note="verified",
                ),
                "update",
            )
            CORE.mutate(
                argparse.Namespace(
                    task="AR-0001",
                    owner="worker-a",
                    status="open",
                    note="paused with the next action recorded " * 8,
                ),
                "release",
            )
        meta, _ = CORE.read_task(path)
        self.assertEqual("open", meta["status"])
        self.assertEqual("", meta["owner"])
        self.assertTrue(all(len(line) <= 100 for line in path.read_text().splitlines()))

    def test_claim_enforces_dependencies_owner_and_positive_lease(self) -> None:
        self.make_task("AR-0001")
        self.make_task("AR-0002", depends_on=["AR-0001"])
        with patch.object(CORE, "commit", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "unfinished dependencies"):
                CORE.mutate(
                    argparse.Namespace(task="AR-0002", owner="worker-a", lease_minutes=10),
                    "claim",
                )
            with self.assertRaisesRegex(RuntimeError, "positive"):
                CORE.mutate(
                    argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=0),
                    "claim",
                )
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10),
                "claim",
            )
            with self.assertRaisesRegex(RuntimeError, "not open"):
                CORE.mutate(
                    argparse.Namespace(task="AR-0001", owner="worker-b", lease_minutes=10),
                    "claim",
                )

    def test_promote_is_dependency_revision_and_state_aware(self) -> None:
        dependency = self.make_task("AR-0001", status="done")
        target = self.make_task(
            "AR-0002",
            status="planned",
            depends_on=["AR-0001"],
        )
        before = target.read_text()
        args = argparse.Namespace(
            task="AR-0002",
            expected_revision=0,
            note="dependencies verified",
        )
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "stale revision"),
        ):
            CORE.mutate(args, "promote")
        self.assertEqual(before, target.read_text())

        dependency_meta, dependency_body = CORE.read_task(dependency)
        dependency_meta["status"] = "open"
        CORE.write_task(dependency, dependency_meta, dependency_body)
        self.refresh_views()
        args.expected_revision = 1
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "unfinished dependencies"),
        ):
            CORE.mutate(args, "promote")
        dependency_meta["status"] = "done"
        CORE.write_task(dependency, dependency_meta, dependency_body)
        self.refresh_views()

        args.note = ""
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "must not be empty"),
        ):
            CORE.mutate(args, "promote")
        args.note = "dependencies verified"

        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            patch.object(CORE, "commit", return_value=True),
        ):
            CORE.mutate(args, "promote")
        meta, body = CORE.read_task(target)
        self.assertEqual("open", meta["status"])
        self.assertEqual(2, meta["task_revision"])
        self.assertIn("dependencies verified", body)
        self.assertIn("AR_0001 --> AR_0002", (self.root / "STATUS.md").read_text())
        self.assertIn("## Open", (self.root / "CURRENT.md").read_text())

    def test_promote_rejects_dirty_invalid_claimed_and_nonplanned_state(self) -> None:
        path = self.make_task(status="planned")
        args = argparse.Namespace(task="AR-0001", expected_revision=1, note="ready")
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[" M tasks/other.md"]),
            self.assertRaisesRegex(RuntimeError, "clean state repository"),
        ):
            CORE.mutate(args, "promote")
        self.assertEqual("planned", CORE.read_task(path)[0]["status"])

        (self.root / "STATUS.md").write_text("stale")
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "promotion preflight failed"),
        ):
            CORE.mutate(args, "promote")
        self.refresh_views()

        meta, body = CORE.read_task(path)
        meta["owner"] = "worker-a"
        meta["claim_expires"] = "2099-01-01T00:00:00+00:00"
        CORE.write_task(path, meta, body)
        self.refresh_views()
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "promotion preflight failed"),
        ):
            CORE.mutate(args, "promote")
        meta["owner"] = ""
        meta["claim_expires"] = ""
        meta["status"] = "future"
        CORE.write_task(path, meta, body)
        self.refresh_views()
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "not planned"),
        ):
            CORE.mutate(args, "promote")

    def test_resume_reopens_only_exact_blocked_revision(self) -> None:
        target = self.make_task("AR-0001", status="blocked")
        args = argparse.Namespace(task="AR-0001", expected_revision=0, note="blocker cleared")
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "stale revision"),
        ):
            CORE.mutate(args, "resume")

        args.expected_revision = 1
        args.note = ""
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "must not be empty"),
        ):
            CORE.mutate(args, "resume")

        args.note = "blocker cleared"
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            patch.object(CORE, "commit", return_value=True),
        ):
            CORE.mutate(args, "resume")
        meta, body = CORE.read_task(target)
        self.assertEqual("open", meta["status"])
        self.assertEqual("", meta["owner"])
        self.assertEqual(2, meta["task_revision"])
        self.assertIn("blocker cleared", body)

        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            self.assertRaisesRegex(RuntimeError, "not blocked"),
        ):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", expected_revision=2, note="again"),
                "resume",
            )

    def test_promote_failure_restores_task_and_generated_views(self) -> None:
        path = self.make_task(status="planned")
        before = {
            item: item.read_text()
            for item in (path, self.root / "CURRENT.md", self.root / "STATUS.md")
        }
        args = argparse.Namespace(task="AR-0001", expected_revision=1, note="ready")
        real_atomic = CORE.atomic
        failed = False

        def fail_status_once(target: Path, text: str) -> None:
            nonlocal failed
            if target.name == "STATUS.md" and not failed:
                failed = True
                raise OSError("injected status failure")
            real_atomic(target, text)

        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            patch.object(CORE, "atomic", side_effect=fail_status_once),
            self.assertRaisesRegex(OSError, "injected status failure"),
        ):
            CORE.mutate(args, "promote")
        self.assertTrue(all(item.read_text() == text for item, text in before.items()))

        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            patch.object(CORE, "commit", side_effect=RuntimeError("injected commit failure")),
            self.assertRaisesRegex(RuntimeError, "injected commit failure"),
        ):
            CORE.mutate(args, "promote")
        self.assertTrue(all(item.read_text() == text for item, text in before.items()))

    def test_promote_push_failure_preserves_durable_state(self) -> None:
        path = self.make_task(status="planned")
        args = argparse.Namespace(task="AR-0001", expected_revision=1, note="ready")
        with (
            patch.object(CORE, "dirty_state_paths", return_value=[]),
            patch.object(CORE, "commit", return_value=True),
            patch.object(CORE, "push_replica", side_effect=RuntimeError("push failed")),
            self.assertRaisesRegex(RuntimeError, "push failed"),
        ):
            CORE.mutate(args, "promote")
        self.assertEqual("open", CORE.read_task(path)[0]["status"])
        self.assertEqual(
            CORE.render_status_view(CORE.all_tasks()), (self.root / "STATUS.md").read_text()
        )

    def test_dirty_state_paths_uses_complete_porcelain(self) -> None:
        completed = subprocess.CompletedProcess(
            ["git"],
            0,
            stdout=" M tasks/one.md\n?? tasks/two.md\n",
            stderr="",
        )
        with patch.object(CORE, "run", return_value=completed) as invoked:
            self.assertEqual(
                [" M tasks/one.md", "?? tasks/two.md"],
                CORE.dirty_state_paths(),
            )
        command = invoked.call_args.args[0]
        self.assertIn("--porcelain=v1", command)
        self.assertIn("--untracked-files=all", command)

    def test_concurrent_promote_and_reconcile_keep_source_and_views_atomic(self) -> None:
        self.make_task(status="planned")
        start = multiprocessing.Event()
        promote = multiprocessing.Process(target=concurrent_promote, args=(str(self.root), start))
        reconcile = multiprocessing.Process(
            target=concurrent_reconcile, args=(str(self.root), start)
        )
        promote.start()
        reconcile.start()
        start.set()
        promote.join(10)
        reconcile.join(10)
        self.assertEqual(0, promote.exitcode)
        self.assertEqual(0, reconcile.exitcode)
        meta, _ = CORE.read_task(CORE.locate("AR-0001")[0])
        self.assertEqual("open", meta["status"])
        self.assertEqual(
            CORE.render_current(CORE.all_tasks()), (self.root / "CURRENT.md").read_text()
        )
        self.assertEqual(
            CORE.render_status_view(CORE.all_tasks()), (self.root / "STATUS.md").read_text()
        )

    def test_invalid_update_rolls_back_both_files(self) -> None:
        path = self.make_task()
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10),
                "claim",
            )
            before_task = path.read_text()
            before_current = (self.root / "CURRENT.md").read_text()
            before_status = (self.root / "STATUS.md").read_text()
            revision = CORE.read_task(path)[0]["task_revision"]
            with self.assertRaisesRegex(RuntimeError, "invalid summary"):
                CORE.mutate(
                    argparse.Namespace(
                        task="AR-0001",
                        owner="worker-a",
                        expected_revision=revision,
                        status=None,
                        priority=None,
                        summary="",
                        next_action=None,
                        note="invalid",
                    ),
                    "update",
                )
        self.assertEqual(before_task, path.read_text())
        self.assertEqual(before_current, (self.root / "CURRENT.md").read_text())
        self.assertEqual(before_status, (self.root / "STATUS.md").read_text())

    def test_failed_push_preserves_durable_commit_state(self) -> None:
        path = self.make_task()
        with (
            patch.object(CORE, "commit", return_value=True),
            patch.object(CORE, "push_replica", side_effect=RuntimeError("push failed")),
            self.assertRaisesRegex(RuntimeError, "push failed"),
        ):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10),
                "claim",
            )
        meta, _ = CORE.read_task(path)
        self.assertEqual("in_progress", meta["status"])
        self.assertEqual("worker-a", meta["owner"])
        self.assertEqual(
            CORE.render_current(CORE.all_tasks()), (self.root / "CURRENT.md").read_text()
        )
        self.assertEqual(
            CORE.render_status_view(CORE.all_tasks()), (self.root / "STATUS.md").read_text()
        )

    def test_lock_excludes_a_second_process(self) -> None:
        ready = multiprocessing.Event()
        release = multiprocessing.Event()
        process = multiprocessing.Process(
            target=hold_lock,
            args=(str(CORE.LOCK), ready, release),
        )
        process.start()
        self.assertTrue(ready.wait(5))
        start = time.monotonic()
        release.set()
        with CORE.locked():
            elapsed = time.monotonic() - start
        process.join(5)
        self.assertEqual(0, process.exitcode)
        self.assertGreaterEqual(elapsed, 0)

    def test_concurrent_claim_and_reconcile_keep_status_current(self) -> None:
        self.make_task()
        start = multiprocessing.Event()
        claim = multiprocessing.Process(target=concurrent_claim, args=(str(self.root), start))
        reconcile = multiprocessing.Process(
            target=concurrent_reconcile, args=(str(self.root), start)
        )
        claim.start()
        reconcile.start()
        start.set()
        claim.join(10)
        reconcile.join(10)
        self.assertEqual(0, claim.exitcode)
        self.assertEqual(0, reconcile.exitcode)
        meta, _ = CORE.read_task(CORE.locate("AR-0001")[0])
        self.assertEqual("in_progress", meta["status"])
        self.assertEqual(
            CORE.render_status_view(CORE.all_tasks()), (self.root / "STATUS.md").read_text()
        )

    def test_live_document_generation_and_observation_sync(self) -> None:
        path = self.make_task(worktree_key="game-experiment-test")
        state = {
            "remote_main": "a" * 40,
            "origin_main": "a" * 40,
            "primary_head": "b" * 40,
            "worktrees": [
                {
                    "key": "game-experiment-test",
                    "branch": "feature/test",
                    "head": "c" * 40,
                    "dirty": 2,
                    "paths": ["one", "two"],
                    "behind": 1,
                    "ahead": 2,
                }
            ],
            "prs": [
                {
                    "number": 1,
                    "title": "Test",
                    "headRefName": "feature/test",
                    "headRefOid": "c" * 40,
                    "baseRefName": "main",
                    "mergeStateStatus": "CLEAN",
                    "statusCheckRollup": [{"status": "COMPLETED", "conclusion": "SUCCESS"}],
                }
            ],
            "runs": [
                {
                    "databaseId": 1,
                    "headSha": "c" * 40,
                    "event": "push",
                    "workflowName": "Verify",
                    "status": "completed",
                    "conclusion": "success",
                }
            ],
        }
        project, worktrees = CORE.live_docs(state)
        self.assertIn("Product remote main", project)
        self.assertIn("#1", project)
        self.assertIn("dirty", worktrees.lower())
        self.assertIn("| changed files | - | - | - | `one`, `two` |", worktrees)
        CORE.sync_task_observations(CORE.all_tasks(), state)
        meta, _ = CORE.read_task(path)
        self.assertEqual(2, meta["observed_dirty"])

    def test_run_config_privacy_and_locate_failures(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "command failed"):
            CORE.run(["false"])
        with self.assertRaisesRegex(RuntimeError, "missing private runtime"):
            CORE.config()
        CORE.CONFIG.parent.mkdir()
        CORE.CONFIG.write_text(
            json.dumps(
                {
                    "projects_root": "root",
                    "product_worktree": "repo",
                    "github_repository": "owner/repo",
                }
            )
        )
        self.assertEqual("repo", CORE.config()["product_worktree"])
        with self.assertRaisesRegex(RuntimeError, "unknown task"):
            CORE.locate("AR-9999")
        binary = self.root / "binary"
        binary.write_bytes(b"\\xff")
        large = self.root / "large.md"
        large.write_text("x" * 200001)
        errors = "\n".join(CORE.privacy_errors())
        self.assertIn("exceeds 200 KiB", errors)

    def test_validation_reports_all_basic_reference_and_claim_errors(self) -> None:
        path = self.make_task("AR-0001")
        meta, body = CORE.read_task(path)
        meta.update(
            {
                "id": "bad",
                "status": "wrong",
                "priority": "PX",
                "task_revision": 0,
                "title": "",
                "summary": "",
                "next_action": "",
                "updated_at": "bad",
                "checkpoint_commit": "no",
                "plan": "../plans/AR-0001.md",
                "owner": "orphan",
                "claim_expires": "later",
            }
        )
        CORE.write_task(path, meta, body)
        errors = "\n".join(CORE.validate())
        for phrase in (
            "invalid id",
            "invalid status",
            "invalid priority",
            "invalid revision",
            "invalid title",
            "invalid summary",
            "invalid next_action",
            "invalid updated_at",
            "invalid checkpoint",
            "missing plan",
            "inactive task retains claim",
        ):
            self.assertIn(phrase, errors)

    def test_claim_expiry_is_timezone_aware_and_live(self) -> None:
        path = self.make_task(
            status="in_progress",
            owner="worker-a",
            claim_expires="2000-01-01T00:00:00+00:00",
        )
        self.assertIn("expired claim", "\n".join(CORE.validate()))
        meta, body = CORE.read_task(path)
        meta["claim_expires"] = "2099-01-01T00:00:00"
        CORE.write_task(path, meta, body)
        CORE.atomic(self.root / "CURRENT.md", CORE.render_current(CORE.all_tasks()))
        self.assertIn("invalid claim expiry", "\n".join(CORE.validate()))
        meta["claim_expires"] = 123
        CORE.write_task(path, meta, body)
        CORE.atomic(self.root / "CURRENT.md", CORE.render_current(CORE.all_tasks()))
        self.assertIn("invalid claim expiry", "\n".join(CORE.validate()))

    def test_push_replica_disabled_without_private_config(self) -> None:
        with patch.object(CORE, "run") as run_mock:
            CORE.push_replica()
            run_mock.assert_not_called()

    def test_claim_owner_collision_heartbeat_and_release_failures(self) -> None:
        first = self.make_task("AR-0001")
        self.make_task("AR-0002")
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim"
            )
            with self.assertRaisesRegex(RuntimeError, "already holds"):
                CORE.mutate(
                    argparse.Namespace(task="AR-0002", owner="worker-a", lease_minutes=10), "claim"
                )
            with self.assertRaisesRegex(RuntimeError, "owned by"):
                CORE.mutate(
                    argparse.Namespace(task="AR-0001", owner="worker-b", lease_minutes=10),
                    "heartbeat",
                )
            with self.assertRaisesRegex(RuntimeError, "positive lease"):
                CORE.mutate(
                    argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=0),
                    "heartbeat",
                )
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=20),
                "heartbeat",
            )
            revision = CORE.read_task(first)[0]["task_revision"]
            with self.assertRaisesRegex(RuntimeError, "use release"):
                CORE.mutate(
                    argparse.Namespace(
                        task="AR-0001",
                        owner="worker-a",
                        expected_revision=revision,
                        status="done",
                        priority=None,
                        summary=None,
                        next_action=None,
                        note="bad",
                    ),
                    "update",
                )

    def fake_scan(self) -> dict[str, object]:
        return {
            "remote_main": "a" * 40,
            "origin_main": "a" * 40,
            "primary_head": "b" * 40,
            "worktrees": [],
            "prs": [],
            "runs": [],
        }

    def test_project_scan_covers_dirty_and_detached_worktrees(self) -> None:  # noqa: C901
        product = self.root / "game-experiment"
        second = self.root / "game-experiment-two"
        product.mkdir()
        second.mkdir()
        CORE.CONFIG.parent.mkdir()
        CORE.CONFIG.write_text(
            json.dumps(
                {
                    "projects_root": str(self.root),
                    "product_worktree": product.name,
                    "github_repository": "owner/repo",
                }
            )
        )

        def fake_run(args: list[str], **_: object) -> object:  # noqa: C901
            joined = " ".join(args)
            stdout = ""
            returncode = 0
            if "worktree list" in joined:
                stdout = f"worktree {product}\n\nworktree {second}\n"
            elif "symbolic-ref" in joined and str(second) in joined:
                returncode = 1
            elif "symbolic-ref" in joined:
                stdout = "main\n"
            elif "status --porcelain" in joined and str(product) in joined:
                stdout = " M file\n"
            elif "rev-list -1" in joined:
                # No product-changing commit found: product_head falls back to rev-parse.
                returncode = 128
            elif "rev-list" in joined and str(product) in joined:
                stdout = "1 2\n"
            elif "rev-list" in joined:
                returncode = 1
            elif "ls-remote" in joined:
                stdout = ("a" * 40) + "\trefs/heads/main\n"
            elif "rev-parse origin/main" in joined:
                stdout = ("a" * 40) + "\n"
            elif "rev-parse HEAD" in joined and str(product) in joined:
                stdout = ("b" * 40) + "\n"
            elif "rev-parse HEAD" in joined:
                stdout = ("c" * 40) + "\n"
            elif args[:3] == ["gh", "pr", "list"] or args[:3] == ["gh", "run", "list"]:
                stdout = "[]"
            return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")

        with patch.object(CORE, "run", side_effect=fake_run):
            state = CORE.project_scan()
        self.assertEqual("a" * 40, state["remote_main"])
        self.assertEqual(2, len(state["worktrees"]))
        self.assertEqual(1, state["worktrees"][0]["dirty"])
        self.assertEqual("DETACHED", state["worktrees"][1]["branch"])

    def test_reconcile_and_live_staleness(self) -> None:
        self.make_task()
        with (
            patch.object(CORE, "project_scan", return_value=self.fake_scan()),
            patch.object(CORE, "commit", return_value=True) as commit,
        ):
            self.assertTrue(CORE.reconcile(do_commit=True))
            commit.assert_called_once()
            self.assertEqual([], CORE.validate(live=True))
            (self.root / "PROJECT_STATE.md").write_text("stale")
            errors = CORE.validate(live=True)
            self.assertIn("PROJECT_STATE.md is stale", errors)

    def test_generated_view_validation_reports_stale_and_renderer_errors(self) -> None:
        self.make_task()
        (self.root / "CURRENT.md").write_text("stale")
        (self.root / "STATUS.md").unlink()
        errors = CORE.generated_view_errors(CORE.all_tasks())
        self.assertIn("CURRENT.md differs from generated tasks", errors)
        self.assertIn("STATUS.md differs from generated tasks", errors)
        with patch.object(
            CORE,
            "render_status_view",
            side_effect=CORE.StatusRenderError("hostile graph"),
        ):
            self.assertIn("hostile graph", CORE.generated_view_errors(CORE.all_tasks()))
        path = CORE.locate("AR-0001")[0]
        meta, body = CORE.read_task(path)
        meta["status"] = "invalid"
        CORE.write_task(path, meta, body)
        self.assertEqual([], CORE.generated_view_errors(CORE.all_tasks()))

    def test_reconcile_generation_failure_rolls_back_every_view(self) -> None:
        path = self.make_task(worktree_key="game-experiment-test")
        before_task = path.read_text()
        before_current = (self.root / "CURRENT.md").read_text()
        before_status = (self.root / "STATUS.md").read_text()
        state = self.fake_scan()
        state["worktrees"] = [
            {
                "key": "game-experiment-test",
                "branch": "feature/test",
                "head": "c" * 40,
                "dirty": 1,
                "paths": ["changed"],
                "behind": 0,
                "ahead": 1,
            }
        ]
        real_atomic = CORE.atomic
        failed = False

        def fail_status_once(target: Path, text: str) -> None:
            nonlocal failed
            if target.name == "STATUS.md" and not failed:
                failed = True
                raise OSError("injected status write failure")
            real_atomic(target, text)

        with (
            patch.object(CORE, "project_scan", return_value=state),
            patch.object(CORE, "atomic", side_effect=fail_status_once),
            self.assertRaisesRegex(OSError, "injected status write failure"),
        ):
            CORE.reconcile(do_commit=False)
        self.assertEqual(before_task, path.read_text())
        self.assertEqual(before_current, (self.root / "CURRENT.md").read_text())
        self.assertEqual(before_status, (self.root / "STATUS.md").read_text())
        self.assertFalse((self.root / "PROJECT_STATE.md").exists())
        self.assertFalse((self.root / "WORKTREES.md").exists())

    def test_reconcile_commit_failure_rolls_back_and_push_failure_preserves(self) -> None:
        path = self.make_task(worktree_key="game-experiment-test")
        before = {
            item: item.read_text()
            for item in (path, self.root / "CURRENT.md", self.root / "STATUS.md")
        }
        with (
            patch.object(CORE, "project_scan", return_value=self.fake_scan()),
            patch.object(CORE, "commit", side_effect=RuntimeError("commit failed")),
            self.assertRaisesRegex(RuntimeError, "commit failed"),
        ):
            CORE.reconcile(do_commit=True)
        self.assertTrue(all(item.read_text() == text for item, text in before.items()))
        with (
            patch.object(CORE, "project_scan", return_value=self.fake_scan()),
            patch.object(CORE, "commit", return_value=True),
            patch.object(CORE, "push_replica", side_effect=RuntimeError("push failed")),
            self.assertRaisesRegex(RuntimeError, "push failed"),
        ):
            CORE.reconcile(do_commit=True, push=True)
        self.assertEqual(
            CORE.render_status_view(CORE.all_tasks()), (self.root / "STATUS.md").read_text()
        )

    def test_mutation_render_and_commit_failures_restore_three_files(self) -> None:
        path = self.make_task()
        before = {
            item: item.read_text()
            for item in (path, self.root / "CURRENT.md", self.root / "STATUS.md")
        }
        args = argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10)
        with (
            patch.object(CORE, "render_status_view", side_effect=RuntimeError("render failed")),
            self.assertRaisesRegex(RuntimeError, "render failed"),
        ):
            CORE.mutate(args, "claim")
        self.assertTrue(all(item.read_text() == text for item, text in before.items()))
        (self.root / "STATUS.md").unlink()
        with (
            patch.object(CORE, "render_status_view", side_effect=RuntimeError("render failed")),
            self.assertRaisesRegex(RuntimeError, "render failed"),
        ):
            CORE.mutate(args, "claim")
        self.assertFalse((self.root / "STATUS.md").exists())
        CORE.atomic(self.root / "STATUS.md", before[self.root / "STATUS.md"])
        with (
            patch.object(CORE, "commit", side_effect=RuntimeError("commit failed")),
            self.assertRaisesRegex(RuntimeError, "commit failed"),
        ):
            CORE.mutate(args, "claim")
        self.assertTrue(all(item.read_text() == text for item, text in before.items()))

    def test_commit_no_change_and_change_paths(self) -> None:
        target = self.root / "CURRENT.md"
        target.write_text("x")
        calls = []

        def fake_run(args: list[str], **_: object) -> object:
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=on_main(args), stderr="")

        with patch.object(CORE, "run", side_effect=fake_run):
            self.assertFalse(CORE.commit("test", [target]))
        self.assertFalse(any("commit" in args for args in calls))

        def changed_run(args: list[str], **_: object) -> object:
            calls.append(args)
            code = 1 if "diff" in args else 0
            return subprocess.CompletedProcess(args, code, stdout=on_main(args), stderr="")

        with patch.object(CORE, "run", side_effect=changed_run):
            self.assertTrue(CORE.commit("test", [target]))
        commit_call = next(args for args in calls if "commit" in args)
        self.assertIn("-S", commit_call)
        self.assertIn("-s", commit_call)
        # Limited to the transaction's own paths: the canonical checkout is also a product
        # checkout, and staged product changes must not ride along in a state commit.
        self.assertEqual(["--", "CURRENT.md"], commit_call[-2:])
        diff_call = next(args for args in calls if "diff" in args)
        self.assertEqual(["--", "CURRENT.md"], diff_call[-2:])

        def failing_run(args: list[str], **_: object) -> object:
            calls.append(args)
            if "diff" in args:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
            if "commit" in args:
                raise RuntimeError("signing failed")
            return subprocess.CompletedProcess(args, 0, stdout=on_main(args), stderr="")

        with (
            patch.object(CORE, "run", side_effect=failing_run),
            self.assertRaisesRegex(RuntimeError, "signing failed"),
        ):
            CORE.commit("test", [target])
        self.assertTrue(any("reset" in args for args in calls))

    def test_push_replica_fast_forward_noop_and_divergence(self) -> None:
        CORE.CONFIG.parent.mkdir()
        CORE.CONFIG.write_text('{"push_enabled": true}')
        local = "a" * 40
        remote = "b" * 40
        calls: list[list[str]] = []

        def fake_run(args: list[str], **_: object) -> object:
            calls.append(args)
            joined = " ".join(args)
            stdout = ""
            returncode = 0
            if "symbolic-ref" in joined:
                stdout = "main\n"
            elif "remote get-url" in joined:
                stdout = "git@example.invalid:owner/state.git\n"
            elif "rev-parse HEAD" in joined:
                stdout = local + "\n"
            elif "rev-parse FETCH_HEAD" in joined:
                stdout = remote + "\n"
            return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")

        with patch.object(CORE, "run", side_effect=fake_run):
            CORE.push_replica()
        self.assertTrue(any("push" in args for args in calls))

        calls.clear()
        local = remote
        with patch.object(CORE, "run", side_effect=fake_run):
            CORE.push_replica()
        self.assertFalse(any("push" in args for args in calls))

        local = "c" * 40

        def divergent_run(args: list[str], **kwargs: object) -> object:
            result = fake_run(args, **kwargs)
            if "merge-base" in args:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
            return result

        with (
            patch.object(CORE, "run", side_effect=divergent_run),
            self.assertRaisesRegex(RuntimeError, "diverged"),
        ):
            CORE.push_replica()

    def test_push_replica_requires_origin_when_enabled(self) -> None:
        CORE.CONFIG.parent.mkdir()
        CORE.CONFIG.write_text('{"push_enabled": true}')

        def no_origin(args: list[str], **_: object) -> object:
            if "symbolic-ref" in args:
                return subprocess.CompletedProcess(args, 0, stdout="main\n", stderr="")
            return subprocess.CompletedProcess(args, 2, stdout="", stderr="")

        with (
            patch.object(CORE, "run", side_effect=no_origin),
            self.assertRaisesRegex(RuntimeError, "origin is missing"),
        ):
            CORE.push_replica()

    def test_push_replica_refuses_off_the_state_branch(self) -> None:
        CORE.CONFIG.parent.mkdir()
        CORE.CONFIG.write_text('{"push_enabled": true}')
        calls: list[list[str]] = []

        def feature_branch(args: list[str], **_: object) -> object:
            calls.append(args)
            stdout = "feature/x\n" if "symbolic-ref" in args else ""
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

        with (
            patch.object(CORE, "run", side_effect=feature_branch),
            self.assertRaisesRegex(RuntimeError, "commit only to main.*feature/x"),
        ):
            CORE.push_replica()
        self.assertFalse(any("push" in args or "fetch" in args for args in calls))

    def test_product_checkout_defaults_to_the_repository_holding_root(self) -> None:
        top = subprocess.CompletedProcess(["git"], 0, stdout="/srv/product\n", stderr="")
        with patch.object(CORE, "run", return_value=top) as called:
            self.assertEqual(Path("/srv/product"), CORE.product_checkout({}))
        self.assertIn("--show-toplevel", called.call_args.args[0])
        override = {"projects_root": "/srv", "product_worktree": "other"}
        self.assertEqual(Path("/srv/other"), CORE.product_checkout(override))

    def test_snapshot_and_run_command_paths(self) -> None:
        self.make_task(
            status="in_progress",
            owner="worker-a",
            claim_expires=(
                (dt.datetime.now(dt.UTC) + dt.timedelta(minutes=5))
                .replace(microsecond=0)
                .isoformat()
            ),
        )
        CORE.atomic(self.root / "PROJECT_STATE.md", CORE.live_docs(self.fake_scan())[0])
        CORE.atomic(self.root / "WORKTREES.md", CORE.live_docs(self.fake_scan())[1])
        completed = subprocess.CompletedProcess(["true"], 0, stdout="", stderr="")
        with (
            patch.object(CORE, "project_scan", return_value=self.fake_scan()),
            patch.object(CORE, "run", return_value=completed),
            patch("builtins.print"),
        ):
            CORE.cmd_snapshot()
        args = argparse.Namespace(task="AR-0001", owner="worker-a", command=["true"])
        with (
            patch.object(CORE.subprocess, "run", return_value=completed) as subprocess_run,
            patch.object(CORE, "reconcile"),
            patch.object(CORE, "mutate") as mutate,
        ):
            self.assertEqual(0, CORE.cmd_run(args))
            self.assertIn("command argv SHA-256", mutate.call_args.args[0].note)
            self.assertIsNone(mutate.call_args.args[0].expected_revision)
            subprocess_run.assert_called_once_with(["true"], check=False, stdin=subprocess.DEVNULL)
        with self.assertRaisesRegex(RuntimeError, "claim"):
            CORE.cmd_run(argparse.Namespace(task="AR-0001", owner="wrong", command=["true"]))
        with self.assertRaisesRegex(RuntimeError, "missing command"):
            CORE.cmd_run(argparse.Namespace(task="AR-0001", owner="worker-a", command=[]))
        path = CORE.locate("AR-0001")[0]
        meta, body = CORE.read_task(path)
        meta["claim_expires"] = "2000-01-01T00:00:00+00:00"
        CORE.write_task(path, meta, body)
        with (
            patch.object(CORE.subprocess, "run") as command,
            self.assertRaisesRegex(RuntimeError, "expired claim"),
        ):
            CORE.cmd_run(argparse.Namespace(task="AR-0001", owner="worker-a", command=["true"]))
        command.assert_not_called()

    def test_explicit_status_render_and_stale_check(self) -> None:
        self.make_task()
        CORE.cmd_render_status(check=True)
        (self.root / "STATUS.md").write_text("stale")
        with self.assertRaisesRegex(RuntimeError, "STATUS.md differs"):
            CORE.cmd_render_status(check=True)
        CORE.cmd_render_status(check=False)
        self.assertEqual(
            CORE.render_status_view(CORE.all_tasks()), (self.root / "STATUS.md").read_text()
        )

    def test_commit_messages_are_gated_for_private_agent_references(self) -> None:
        messages = {
            "a" * 40: "Add a feature\n\nSigned-off-by: Someone <someone@example.invalid>\n",
            "b" * 40: "Leak a URL\n\nClaude-Session: https://claude.ai/" + "code/session_abc\n",
            "c" * 40: "Leak a bare id\n\nClaude-Session: ses" + "sion_01HeMgiRgN7PNb8e9xvXz8GX\n",
        }

        def fake_run(args: list[str], **_: object) -> object:
            out = "\n".join(messages) + "\n" if "rev-list" in args else messages[args[-1]]
            return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

        with patch.object(CORE, "run", side_effect=fake_run):
            errors = CORE.commit_privacy_errors("origin/main", "HEAD")
            self.assertEqual(2, len(errors))
            self.assertIn("private agent session reference", errors[0])
            self.assertIn("private agent session identifier", errors[1])
            with patch("builtins.print"):
                self.assertEqual(1, CORE.cmd_check_commits(base="origin/main", head="HEAD"))

        clean = {"d" * 40: "Ordinary message\n"}

        def clean_run(args: list[str], **_: object) -> object:
            out = "\n".join(clean) + "\n" if "rev-list" in args else clean[args[-1]]
            return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

        with patch.object(CORE, "run", side_effect=clean_run) as called:
            with patch("builtins.print"):
                self.assertEqual(0, CORE.cmd_check_commits(base="", head="HEAD"))
            self.assertIn("HEAD", called.call_args_list[0].args[0])

    def test_agent_session_references_are_rejected_in_the_working_tree(self) -> None:
        self.make_task()
        leak = self.root / "note.md"
        leak.write_text("see https://claude.ai/" + "code/session_x for context")
        self.assertIn("private agent session reference", "\n".join(CORE.privacy_errors()))
        leak.write_text("resume ses" + "sion_01HeMgiRgN7PNb8e9xvXz8GX now")
        self.assertIn("private agent session identifier", "\n".join(CORE.privacy_errors()))
        leak.write_text("resume SES" + "SION_01HeMgiRgN7PNb8e9xvXz8GX now")
        self.assertIn("private agent session identifier", "\n".join(CORE.privacy_errors()))
        leak.write_text("owner locutus3009 on github")
        self.assertEqual([], CORE.privacy_errors())

    def test_main_dispatches_every_command(self) -> None:
        cases = [
            (["handoffctl", "reconcile", "--commit"], "reconcile", None),
            (["handoffctl", "snapshot"], "cmd_snapshot", None),
            (["handoffctl", "render-status", "--check"], "cmd_render_status", None),
            (["handoffctl", "complete-milestone", "M0"], "cmd_complete_milestone", None),
            (["handoffctl", "claim", "AR-0001", "--owner", "worker-a"], "mutate", None),
            (["handoffctl", "heartbeat", "AR-0001", "--owner", "worker-a"], "mutate", None),
            (
                [
                    "handoffctl",
                    "promote",
                    "AR-0001",
                    "--expected-revision",
                    "1",
                    "--note",
                    "ready",
                ],
                "mutate",
                None,
            ),
            (
                [
                    "handoffctl",
                    "resume",
                    "AR-0001",
                    "--expected-revision",
                    "1",
                    "--note",
                    "ready",
                ],
                "mutate",
                None,
            ),
            (
                [
                    "handoffctl",
                    "release",
                    "AR-0001",
                    "--owner",
                    "worker-a",
                    "--status",
                    "open",
                    "--note",
                    "pause",
                ],
                "mutate",
                None,
            ),
            (
                [
                    "handoffctl",
                    "update",
                    "AR-0001",
                    "--owner",
                    "worker-a",
                    "--expected-revision",
                    "1",
                    "--note",
                    "update",
                ],
                "mutate",
                None,
            ),
            (
                ["handoffctl", "run", "--owner", "worker-a", "AR-0001", "--", "true"],
                "cmd_run",
                7,
            ),
            (
                ["handoffctl", "check-commits", "--base", "origin/main"],
                "cmd_check_commits",
                3,
            ),
        ]
        for argv, target, result in cases:
            with (
                self.subTest(target=target),
                patch.object(sys, "argv", argv),
                patch.object(CORE, target, return_value=result) as called,
            ):
                self.assertEqual(result or 0, CORE.main())
                called.assert_called_once()
        with (
            patch.object(sys, "argv", ["handoffctl", "doctor"]),
            patch.object(CORE, "validate", return_value=[]),
            patch("builtins.print"),
        ):
            self.assertEqual(0, CORE.main())
        with (
            patch.object(sys, "argv", ["handoffctl", "doctor", "--live"]),
            patch.object(CORE, "validate", return_value=["bad"]),
            patch("builtins.print"),
        ):
            self.assertEqual(1, CORE.main())

    def test_graph_labels_each_series_from_its_milestone_document(self) -> None:
        self.make_task("AR-0001")
        self.assertIn(
            'subgraph series_00["00 - Process foundation"]',
            CORE.render_status_view(CORE.all_tasks()),
        )
        self.make_milestone(label="Renamed foundation")
        status = CORE.render_status_view(CORE.all_tasks())
        self.assertIn('subgraph series_00["00 - Renamed foundation"]', status)
        self.assertNotIn("Process foundation", status)
        self.assertNotIn("00 - Additional work", status)

    def test_graph_label_from_a_milestone_stays_deterministic_and_injection_safe(self) -> None:
        self.make_task("AR-0001")
        self.make_task("AR-0201")
        self.make_milestone(label='Hostile "] %%{init: bad}%% | `code` [x]')
        self.make_milestone("M2", label="Second milestone")
        tasks = CORE.all_tasks()
        status = CORE.render_status_view(tasks)
        self.assertEqual(status, CORE.render_status_view(list(reversed(tasks))))
        self.assertNotIn("%%{init: bad}%%", status)
        self.assertIn("&quot;&#93; &#37;&#37;{init: bad}&#37;&#37;", status)
        self.assertIn('subgraph series_02["02 - Second milestone"]', status)

    def test_milestone_label_is_ignored_when_the_document_cannot_govern_a_series(self) -> None:
        self.make_task("AR-0001")
        (self.root / "milestones" / "M0.md").unlink()
        self.make_milestone("Mx", label="Unidentifiable")
        (self.root / "milestones" / "Mbroken.md").write_text("no front matter here\n")
        self.assertEqual({}, CORE.milestone_labels())
        self.assertIn(
            'subgraph series_00["00 - Additional work"]',
            CORE.render_status_view(CORE.all_tasks()),
        )

    def test_every_ar_series_needs_a_milestone_document(self) -> None:
        self.make_task("AR-1101")
        self.assertIn("series 11: no milestone document", CORE.validate())
        self.make_milestone("M11", label="Later milestone")
        self.assertNotIn("series 11: no milestone document", CORE.validate())

    def test_milestone_document_contract_rejects_every_defect(self) -> None:
        self.make_task("AR-0001")
        document = self.root / "milestones" / "M0.md"
        meta, body = CORE.read_task(document)
        meta.update({"status": "retired", "schema_version": 2, "label": " ", "stage": "late"})
        CORE.write_task(document, meta, body)
        errors = "\n".join(CORE.validate())
        for phrase in (
            "invalid milestone status",
            "unsupported milestone schema version",
            "invalid milestone label",
            "unknown field stage",
        ):
            self.assertIn(phrase, errors)

        CORE.write_task(document, {"id": "M0"}, body)
        errors = "\n".join(CORE.validate())
        self.assertIn("missing schema_version", errors)
        self.assertIn("missing label", errors)
        self.assertIn("missing status", errors)

        self.make_milestone()
        renamed = self.root / "milestones" / "M9.md"
        renamed.write_text(document.read_text())
        errors = "\n".join(CORE.validate())
        self.assertIn("M9.md: identifier does not match its filename", errors)
        self.assertIn("M9.md: series 00 already governed by M0.md", errors)
        renamed.unlink()

    def test_milestone_document_must_carry_its_contract_sections_and_heading(self) -> None:
        self.make_task("AR-0001")
        document = self.root / "milestones" / "M0.md"
        meta, _ = CORE.read_task(document)
        CORE.write_task(document, meta, "# M0 \u2014 Different heading\n\n## Outcome\n\nx\n")
        errors = "\n".join(CORE.validate())
        self.assertIn("heading does not match the recorded label", errors)
        self.assertIn("missing section ## Exit criteria", errors)
        self.assertIn("missing section ## Out of scope", errors)
        self.assertNotIn("missing section ## Outcome", errors)

    def test_unparsable_milestone_front_matter_is_reported_not_raised(self) -> None:
        self.make_task("AR-0001")
        (self.root / "milestones" / "M0.md").write_text("# M0 \u2014 no front matter\n")
        errors = "\n".join(CORE.validate())
        self.assertIn("M0.md: missing id", errors)
        self.assertIn("series 00: no milestone document", errors)

    def test_complete_milestone_is_refused_while_an_ar_is_unfinished(self) -> None:
        self.make_task("AR-0001", status="open")
        self.make_task("AR-0002", status="done")
        document = self.root / "milestones" / "M0.md"
        before = document.read_text()
        with (
            patch.object(CORE, "commit", return_value=True) as commit,
            self.assertRaisesRegex(RuntimeError, "unfinished work"),
        ):
            CORE.cmd_complete_milestone("M0")
        commit.assert_not_called()
        self.assertEqual(before, document.read_text())
        self.assertEqual("active", CORE.read_task(document)[0]["status"])

    def test_a_milestone_recorded_complete_by_hand_with_unfinished_work_is_invalid(self) -> None:
        path = self.make_task("AR-0001", status="open")
        self.make_milestone(status="complete")
        self.assertIn("M0.md: unfinished ARs AR-0001", CORE.validate())
        meta, body = CORE.read_task(path)
        meta["status"] = "cancelled"
        CORE.write_task(path, meta, body)
        self.refresh_views()
        self.assertEqual([], CORE.validate())

    def test_complete_milestone_records_completion_once_every_ar_is_finished(self) -> None:
        self.make_task("AR-0001", status="done")
        document = self.root / "milestones" / "M0.md"
        with self.assertRaisesRegex(RuntimeError, "unknown milestone M7"):
            CORE.cmd_complete_milestone("M7")
        with patch.object(CORE, "commit", return_value=True) as commit, patch("builtins.print"):
            CORE.cmd_complete_milestone("M0")
            commit.assert_called_once()
        self.assertEqual("complete", CORE.read_task(document)[0]["status"])
        self.assertEqual([], CORE.validate())
        with self.assertRaisesRegex(RuntimeError, "already complete"):
            CORE.cmd_complete_milestone("M0")

    def test_complete_milestone_refuses_a_document_that_breaks_the_contract(self) -> None:
        self.make_task("AR-0001", status="done")
        document = self.root / "milestones" / "M0.md"
        meta, _ = CORE.read_task(document)
        CORE.write_task(document, meta, "# M0 \u2014 Process foundation\n")
        with self.assertRaisesRegex(RuntimeError, "milestone contract unmet"):
            CORE.cmd_complete_milestone("M0")

    def test_complete_milestone_rolls_back_when_the_transaction_fails(self) -> None:
        self.make_task("AR-0001", status="done")
        document = self.root / "milestones" / "M0.md"
        before = document.read_text()
        with (
            patch.object(CORE, "commit", side_effect=RuntimeError("commit failed")),
            self.assertRaisesRegex(RuntimeError, "commit failed"),
        ):
            CORE.cmd_complete_milestone("M0")
        self.assertEqual(before, document.read_text())
        with (
            patch.object(CORE, "validate", return_value=["injected inconsistency"]),
            self.assertRaisesRegex(RuntimeError, "injected inconsistency"),
        ):
            CORE.cmd_complete_milestone("M0")
        self.assertEqual(before, document.read_text())
        with (
            patch.object(CORE, "commit", return_value=True),
            patch.object(CORE, "push_replica", side_effect=RuntimeError("push failed")),
            self.assertRaisesRegex(RuntimeError, "push failed"),
        ):
            CORE.cmd_complete_milestone("M0")
        self.assertEqual("complete", CORE.read_task(document)[0]["status"])

    def test_prose_rule_is_scoped_to_the_description_and_keyed_on_recorded_evidence(self) -> None:
        stale = "Implementation has not started. Read the linked plan before claiming.\n"
        entry = "\n- 2026-09-07T15:29:44+00:00: Claimed by worker-a.\n"
        path = self.make_task("AR-0001", status="done")
        meta, _ = CORE.read_task(path)
        CORE.write_task(path, meta, stale)
        self.refresh_views()
        self.assertEqual([], CORE.validate())

        CORE.write_task(path, meta, stale + entry)
        self.refresh_views()
        self.assertIn("AR-0001: description claims work has not started", CORE.validate())

        corrected = (
            "Read the linked plan before claiming.\n"
            + entry
            + "\n- 2026-09-07T16:00:00+00:00: Rejected a description saying implementation"
            " has not started while the status said otherwise.\n"
        )
        CORE.write_task(path, meta, corrected)
        self.refresh_views()
        self.assertEqual([], CORE.validate())
        description, recorded = CORE.task_description(corrected)
        self.assertTrue(recorded)
        self.assertEqual("Read the linked plan before claiming.\n\n", description)

    def test_prose_rule_binds_at_every_status_and_exempts_none(self) -> None:
        entry = "\n- 2026-09-07T15:29:44+00:00: Claimed by worker-a.\n"
        body = "Implementation has not started.\n" + entry
        for status in CORE.STATUSES:
            with self.subTest(status=status):
                self.assertEqual(
                    ["AR-0001: description claims work has not started"],
                    CORE.prose_errors({"id": "AR-0001", "status": status}, body),
                )
                self.assertEqual(
                    [],
                    CORE.prose_errors(
                        {"id": "AR-0001", "status": status},
                        "Implementation has not started.\n",
                    ),
                )

    def test_unstarted_prose_is_rejected_in_every_recorded_wording(self) -> None:
        entry = "\n- 2026-09-07T15:29:44+00:00: Claimed by worker-a.\n"
        meta = {"id": "AR-0001", "status": "done"}
        for description in (
            "Implementation has not started.",
            "No work has begun on this task yet.",
            "Development has not yet been started.",
            "CODING HAS NOT BEGUN.",
        ):
            with self.subTest(description=description):
                self.assertEqual(
                    ["AR-0001: description claims work has not started"],
                    CORE.prose_errors(meta, description + entry),
                )
        self.assertEqual([], CORE.prose_errors(meta, "The gate run started and passed." + entry))

    def test_only_terminal_statuses_let_a_milestone_close(self) -> None:
        self.make_milestone(status="complete")
        documents = CORE.milestone_documents()
        for status in CORE.STATUSES:
            with self.subTest(status=status):
                tasks = [
                    (self.root / "tasks/AR-0001-test.md", {"id": "AR-0001", "status": status}, "")
                ]
                errors = CORE.milestone_completion_errors(documents, tasks)
                if status in ("done", "cancelled", "superseded"):
                    self.assertEqual([], errors)
                else:
                    self.assertEqual(["M0.md: unfinished ARs AR-0001"], errors)

    def test_complete_milestone_is_refused_at_every_unfinished_status(self) -> None:
        path = self.make_task("AR-0001", status="done")
        for status in ("open", "planned", "future", "blocked"):
            with self.subTest(status=status):
                meta, body = CORE.read_task(path)
                meta["status"] = status
                CORE.write_task(path, meta, body)
                self.refresh_views()
                with self.assertRaisesRegex(RuntimeError, "unfinished work"):
                    CORE.cmd_complete_milestone("M0")
        meta, body = CORE.read_task(path)
        meta.update(
            {"status": "in_progress", "owner": "w", "claim_expires": "2099-01-01T00:00:00+00:00"}
        )
        CORE.write_task(path, meta, body)
        self.refresh_views()
        with self.assertRaisesRegex(RuntimeError, "unfinished work"):
            CORE.cmd_complete_milestone("M0")

    def test_complete_milestone_holds_the_coordinator_lock(self) -> None:
        self.make_task("AR-0001", status="done")
        CORE.RUNTIME.mkdir(mode=0o700, exist_ok=True)
        CORE.LOCK.touch()
        taken = multiprocessing.Value("i", -1)

        def probing_commit(_message: str, _paths: list[Path]) -> bool:
            process = multiprocessing.Process(target=probe_lock, args=(str(CORE.LOCK), taken))
            process.start()
            process.join(5)
            self.assertEqual(0, process.exitcode)
            return True

        with patch.object(CORE, "commit", side_effect=probing_commit), patch("builtins.print"):
            CORE.cmd_complete_milestone("M0")
        self.assertEqual(1, taken.value)

    def test_concurrent_completion_and_reconcile_keep_the_document_atomic(self) -> None:
        self.make_task("AR-0001", status="done")
        start = multiprocessing.Event()
        complete = multiprocessing.Process(
            target=concurrent_complete_milestone, args=(str(self.root), start)
        )
        reconcile = multiprocessing.Process(
            target=concurrent_reconcile, args=(str(self.root), start)
        )
        complete.start()
        reconcile.start()
        start.set()
        complete.join(10)
        reconcile.join(10)
        self.assertEqual(0, complete.exitcode)
        self.assertEqual(0, reconcile.exitcode)
        document = self.root / "milestones" / "M0.md"
        self.assertEqual("complete", CORE.read_task(document)[0]["status"])
        self.assertEqual([], CORE.validate())
        self.assertEqual(
            CORE.render_status_view(CORE.all_tasks()), (self.root / "STATUS.md").read_text()
        )

    def test_closure_over_work_that_is_not_done_is_recorded_and_reported(self) -> None:
        self.make_task("AR-0001", status="done")
        self.make_task("AR-0002", status="cancelled")
        self.make_task("AR-0003", status="superseded")
        document = self.root / "milestones" / "M0.md"
        expected = (
            "Recorded complete over ARs that are not done: "
            "AR-0002 (cancelled), AR-0003 (superseded)."
        )
        with (
            patch.object(CORE, "commit", return_value=True) as commit,
            patch("builtins.print") as reported,
        ):
            CORE.cmd_complete_milestone("M0")
        text = document.read_text()
        self.assertIn("Recorded complete over ARs that are not done", text)
        self.assertIn("AR-0002 (cancelled)", text)
        self.assertIn("AR-0003 (superseded)", text)
        self.assertTrue(all(len(line) <= 100 for line in text.splitlines()))
        self.assertIn(expected, commit.call_args.args[0])
        self.assertEqual(f"M0: {expected}", reported.call_args.args[0])
        self.assertEqual([], CORE.validate())

    def test_closure_over_finished_work_records_that_every_ar_is_done(self) -> None:
        self.make_task("AR-0001", status="done")
        with (
            patch.object(CORE, "commit", return_value=True) as commit,
            patch("builtins.print") as reported,
        ):
            CORE.cmd_complete_milestone("M0")
        expected = "Recorded complete. Every AR in the series is done."
        self.assertIn(expected, (self.root / "milestones" / "M0.md").read_text())
        self.assertIn(expected, commit.call_args.args[0])
        self.assertEqual(f"M0: {expected}", reported.call_args.args[0])

    def test_front_matter_that_parses_but_is_not_an_object_is_rejected(self) -> None:
        self.make_task("AR-0001")
        document = self.root / "milestones" / "M0.md"
        for payload in ("[1, 2]", '"scalar"', "null", "3"):
            with self.subTest(payload=payload):
                document.write_text(f"---\n{payload}\n---\n# M0 \u2014 Process foundation\n")
                errors = "\n".join(CORE.validate())
                self.assertIn("M0.md: missing id", errors)
                self.assertIn("series 00: no milestone document", errors)
        broken = CORE.TASKS / "AR-0002-test.md"
        broken.write_text("---\n[1, 2]\n---\nbody\n")
        with self.assertRaisesRegex(ValueError, "front matter is not a JSON object"):
            CORE.read_task(broken)

    def test_milestone_label_bound_agrees_with_the_published_schema(self) -> None:
        schema = json.loads(
            (Path(__file__).resolve().parent.parent / "schema/milestone-schema.json").read_text()
        )
        label = schema["properties"]["label"]
        self.assertEqual(CORE.MILESTONE_LABEL_MAX, label["maxLength"])
        self.assertEqual("\\S", label["pattern"])
        self.make_task("AR-0001")
        self.make_milestone(label="x" * (CORE.MILESTONE_LABEL_MAX + 1))
        self.assertIn("M0.md: invalid milestone label", CORE.validate())

    def test_only_a_well_formed_entry_at_column_zero_ends_the_description(self) -> None:
        """Pin every character class of EVIDENCE_ENTRY that a fail-open mutant would drop."""
        entry = "\n- 2026-09-07T15:29:44+00:00: Claimed by worker-a.\n"
        stale = " Implementation has not started.\n"
        near_misses = (
            # Not at column zero: an entry-shaped span inside a sentence.
            "See - 2026-01-01T00:00:00+00:00: the log format." + stale,
            # A space where the ISO separator must be a T.
            "- 2026-01-01 00:00:00+00:00: spaced separator." + stale,
            # No ": " terminator after the timestamp.
            "- 2026-01-01T00:00:00+00:00 missing terminator." + stale,
            # No leading "- " bullet.
            "2026-01-01T00:00:00+00:00: bare timestamp." + stale,
            # Indented, as a documented sample inside a description must be.
            "  - 2026-01-01T00:00:00+00:00: indented sample." + stale,
        )
        for description in near_misses:
            with self.subTest(description=description.splitlines()[0]):
                self.assertEqual(
                    (description + "\n", True), CORE.task_description(description + entry)
                )
                self.assertEqual(
                    ["AR-0001: description claims work has not started"],
                    CORE.prose_errors({"id": "AR-0001"}, description + entry),
                )
        self.assertEqual(("", True), CORE.task_description(entry.lstrip("\n")))

    def test_appended_entries_can_never_wrap_into_something_entry_shaped(self) -> None:
        """Pin the coupling between the continuation indent and the entry boundary."""
        nested = "2026-02-02T00:00:00+00:00: nested"
        for words in range(40):
            with self.subTest(words=words):
                body = CORE.append_evidence(
                    "", "2026-01-01T00:00:00+00:00", f"{'word ' * words}{nested}"
                )
                self.assertEqual(1, len(CORE.EVIDENCE_ENTRY.findall(body)))
                self.assertTrue(all(len(line) <= 100 for line in body.splitlines()))
                continuations = [line for line in body.splitlines()[2:] if line]
                self.assertTrue(all(line.startswith("  ") for line in continuations))

    def test_evidence_the_boundary_cannot_read_fails_closed(self) -> None:
        stale = "Implementation has not started.\n"
        for malformed in (
            "\n- 2026-01-01 00:00:00: hand written entry.\n",
            "\n-   2026-01-01T00:00:00+00:00 no terminator.\n",
            "\n- [2026-01-01] a bracketed date.\n",
        ):
            with self.subTest(malformed=malformed.strip()):
                self.assertEqual(
                    ["AR-0001: evidence log is not in the recorded entry format"],
                    CORE.prose_errors({"id": "AR-0001"}, stale + malformed),
                )
        self.assertEqual([], CORE.prose_errors({"id": "AR-0001"}, stale))
        self.assertEqual([], CORE.prose_errors({"id": "AR-0001"}, stale + "- a plain bullet\n"))
        self.assertEqual([], CORE.prose_errors({"id": "AR-0001"}, stale + "- 2026 is the target\n"))

    def test_the_real_mutation_path_writes_entries_the_boundary_reads(self) -> None:
        path = self.make_task("AR-0001")
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim"
            )
            CORE.mutate(
                argparse.Namespace(
                    task="AR-0001",
                    owner="worker-a",
                    expected_revision=CORE.read_task(path)[0]["task_revision"],
                    status=None,
                    priority=None,
                    summary=None,
                    next_action=None,
                    note="verified " * 30,
                ),
                "update",
            )
        body = CORE.read_task(path)[1]
        self.assertEqual(2, len(CORE.EVIDENCE_ENTRY.findall(body)))
        self.assertEqual("# Test\n\n", CORE.task_description(body)[0])
        self.assertEqual([], CORE.validate())

    def submit_args(
        self, task_id: str = "AR-0001", owner: str = "worker-a", note: str = "submitted for review"
    ) -> argparse.Namespace:
        """Build the argument namespace the submit transition consumes."""
        return argparse.Namespace(task=task_id, owner=owner, note=note)

    def review_args(
        self,
        revision: int,
        task_id: str = "AR-0001",
        reviewer: str = "coordinator",
        status: str = "done",
        note: str = "reviewed at the exact head",
    ) -> argparse.Namespace:
        """Build the argument namespace the review transition consumes."""
        return argparse.Namespace(
            task=task_id,
            reviewer=reviewer,
            status=status,
            expected_revision=revision,
            note=note,
        )

    def release_args(
        self,
        task_id: str = "AR-0001",
        owner: str = "worker-a",
        status: str = "open",
        note: str = "released",
    ) -> argparse.Namespace:
        """Build the argument namespace the release transition consumes."""
        return argparse.Namespace(task=task_id, owner=owner, status=status, note=note)

    def update_args(
        self,
        revision: int | None,
        task_id: str = "AR-0001",
        owner: str = "worker-a",
        status: str | None = None,
        note: str = "progress",
    ) -> argparse.Namespace:
        """Build the argument namespace the update transition consumes."""
        return argparse.Namespace(
            task=task_id,
            owner=owner,
            expected_revision=revision,
            status=status,
            priority=None,
            summary=None,
            next_action=None,
            note=note,
        )

    def claim_and_submit(self, task_id: str = "AR-0001", owner: str = "worker-a") -> Path:
        """Take a task through the complete worker lifecycle and return its path."""
        path = self.make_task(task_id)
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(argparse.Namespace(task=task_id, owner=owner, lease_minutes=10), "claim")
            CORE.mutate(self.submit_args(task_id=task_id, owner=owner), "submit")
        return path

    def test_submitting_finished_work_clears_the_claim_and_records_the_submitter(self) -> None:
        path = self.claim_and_submit()
        meta, body = CORE.read_task(path)
        self.assertEqual("in_review", meta["status"])
        self.assertEqual("", meta["owner"])
        self.assertEqual("", meta["claim_expires"])
        self.assertEqual("worker-a", meta["submitted_by"])
        self.assertIn("submitted for review", body)
        self.assertEqual([], CORE.validate())
        self.assertEqual([], CORE.claim_errors(meta, {}, {}, {}))

    def test_a_submitted_task_can_never_report_the_expiry_a_held_claim_would(self) -> None:
        """The motivating failure: a finished worker holding a lease goes red once it lapses."""
        past = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5)).replace(microsecond=0)
        path = self.make_task(
            "AR-0001",
            status="in_progress",
            owner="worker-a",
            claim_expires=past.isoformat(),
        )
        self.assertIn("AR-0001: expired claim", CORE.validate())
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(self.submit_args(), "submit")
        meta, _ = CORE.read_task(path)
        self.assertEqual("", meta["claim_expires"])
        self.assertEqual([], CORE.validate())
        with patch.object(
            CORE, "active_expiry_errors", side_effect=AssertionError("lease checked")
        ):
            self.assertEqual([], CORE.claim_errors(meta, {}, {}, {}))

    def test_submit_is_refused_without_a_live_claim_a_matching_owner_or_a_note(self) -> None:
        self.make_task("AR-0001")
        with patch.object(CORE, "commit", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "owned by nobody"):
                CORE.mutate(self.submit_args(), "submit")
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim"
            )
            with self.assertRaisesRegex(RuntimeError, "owned by worker-a"):
                CORE.mutate(self.submit_args(owner="worker-b"), "submit")
            with self.assertRaisesRegex(RuntimeError, "submission note must not be empty"):
                CORE.mutate(self.submit_args(note="   "), "submit")
        meta, _ = CORE.read_task(CORE.TASKS / "AR-0001-test.md")
        self.assertEqual("in_progress", meta["status"])

    def test_submit_refuses_a_task_that_is_not_active_even_when_the_owner_matches(self) -> None:
        self.make_task("AR-0001", status="blocked", owner="worker-a")
        with (
            patch.object(CORE, "commit", return_value=True),
            self.assertRaisesRegex(RuntimeError, "AR-0001 is not active"),
        ):
            CORE.mutate(self.submit_args(), "submit")

    def test_a_worker_cannot_take_its_own_submitted_task_further_by_any_route(self) -> None:
        path = self.claim_and_submit()
        revision = CORE.read_task(path)[0]["task_revision"]
        with patch.object(CORE, "commit", return_value=True):
            # The worker's own name no longer matches the cleared field, and the cleared field
            # itself is not an identity a caller may present, so neither credential gets in.
            for owner, message in (("worker-a", "owned by nobody"), ("", "needs an owner")):
                for kind, args in (
                    ("release", self.release_args(owner=owner, status="done")),
                    ("update", self.update_args(revision, owner=owner)),
                    ("submit", self.submit_args(owner=owner)),
                    (
                        "heartbeat",
                        argparse.Namespace(task="AR-0001", owner=owner, lease_minutes=1),
                    ),
                ):
                    with (
                        self.subTest(kind=kind, owner=owner),
                        self.assertRaisesRegex(RuntimeError, message),
                    ):
                        CORE.mutate(args, kind)
            with self.assertRaisesRegex(RuntimeError, "reviewed by the worker that submitted it"):
                CORE.mutate(self.review_args(revision, reviewer="worker-a"), "review")
            with self.assertRaisesRegex(RuntimeError, "not open"):
                CORE.mutate(
                    argparse.Namespace(task="AR-0001", owner="worker-b", lease_minutes=10), "claim"
                )
        with self.assertRaisesRegex(RuntimeError, "claim does not match owner"):
            CORE.require_active_owner("AR-0001", "worker-a")
        meta, _ = CORE.read_task(path)
        self.assertEqual("in_review", meta["status"])
        self.assertEqual("worker-a", meta["submitted_by"])

    def test_an_empty_owner_cannot_rewrite_or_restage_work_awaiting_review(self) -> None:
        """An unheld task records no owner, so the absence of one must not authenticate.

        The ownership gate compared a caller's name against the recorded field, and submitting
        empties that field, so an empty name matched on exactly the tasks whose safety rests on
        the claim having been withdrawn. Nothing below is refused by the repository check that
        runs after the write: the record a reviewer reads, the append-only evidence log and the
        revision a staged review is pinned to all have to survive the attempt untouched.

        The rule is not conditional on the review state, and the last case is what says so. A
        gate narrowed to work awaiting review would still hand every task nobody holds to a
        caller who names nobody, which is the same defect against a different status.
        """
        path = self.claim_and_submit()
        before = path.read_text()
        revision = CORE.read_task(path)[0]["task_revision"]
        tamper = argparse.Namespace(
            task="AR-0001",
            owner="",
            expected_revision=None,
            status=None,
            priority="P4",
            summary="TAMPERED",
            next_action="Just approve it.",
            note="rewriting what the reviewer reads",
        )
        with patch.object(CORE, "commit", return_value=True):
            with (
                patch.object(CORE, "validate", side_effect=AssertionError("guard bypassed")),
                self.assertRaisesRegex(RuntimeError, "needs an owner to authorise this change"),
            ):
                CORE.mutate(tamper, "update")
            self.assertEqual(before, path.read_text())
            CORE.mutate(self.review_args(revision), "review")
        meta, body = CORE.read_task(path)
        self.assertEqual("done", meta["status"])
        self.assertEqual("P1", meta["priority"])
        self.assertEqual("Ready for testing.", meta["summary"])
        self.assertNotIn("TAMPERED", body)
        self.assertNotIn("Just approve it.", body)
        self.assertEqual([], CORE.validate())
        untouched = self.make_task("AR-0002")
        unheld = untouched.read_text()
        with (
            patch.object(CORE, "commit", return_value=True),
            patch.object(CORE, "validate", side_effect=AssertionError("guard bypassed")),
            self.assertRaisesRegex(RuntimeError, "needs an owner to authorise this change"),
        ):
            CORE.mutate(self.release_args(task_id="AR-0002", owner="", status="planned"), "release")
        self.assertEqual(unheld, untouched.read_text())

    def test_update_refuses_the_review_state_and_inactive_work_by_its_own_guard(self) -> None:
        """Both refusals belong to update, not to the repository check that would roll them back.

        Naming ``in_review`` or updating work that is not active leaves state the whole-record
        validation rejects anyway, so a test that only demanded a failure would pass with the
        guard deleted. Each case here forbids that validation from running at all.
        """
        path = self.make_task("AR-0001")
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim"
            )
            revision = CORE.read_task(path)[0]["task_revision"]
            self.make_task("AR-0002", status="blocked", owner="worker-a")
            cases = (
                (
                    self.update_args(revision, status="in_review"),
                    "use release or submit for a non-active status",
                ),
                (self.update_args(None, task_id="AR-0002"), "AR-0002 is not active"),
            )
            for args, message in cases:
                with (
                    self.subTest(message=message),
                    patch.object(CORE, "validate", side_effect=AssertionError("guard bypassed")),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    CORE.mutate(args, "update")
        self.assertEqual("in_progress", CORE.read_task(path)[0]["status"])

    def test_heartbeat_refuses_submitted_work_even_with_the_owner_still_recorded(self) -> None:
        """Heartbeat asserts a live claim, and work awaiting review has none to assert."""
        self.make_task("AR-0001", status="in_review", owner="worker-a", submitted_by="worker-a")
        with (
            patch.object(CORE, "commit", return_value=True),
            patch.object(CORE, "validate", side_effect=AssertionError("guard bypassed")),
            self.assertRaisesRegex(RuntimeError, "heartbeat requires an active task"),
        ):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "heartbeat"
            )

    def test_review_records_the_decision_and_clears_the_submission(self) -> None:
        path = self.claim_and_submit()
        revision = CORE.read_task(path)[0]["task_revision"]
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(self.review_args(revision), "review")
        meta, body = CORE.read_task(path)
        self.assertEqual("done", meta["status"])
        self.assertEqual("", meta["submitted_by"])
        self.assertEqual("", meta["owner"])
        self.assertEqual("", meta["claim_expires"])
        self.assertIn("reviewed at the exact head", body)
        self.assertEqual([], CORE.validate())

    def test_review_rejects_a_stale_revision_a_wrong_state_and_empty_arguments(self) -> None:
        self.make_task("AR-0002")
        path = self.claim_and_submit()
        revision = CORE.read_task(path)[0]["task_revision"]
        cases = (
            ("stale revision", self.review_args(revision - 1)),
            ("not awaiting review", self.review_args(1, task_id="AR-0002")),
            ("not another active status", self.review_args(revision, status="in_progress")),
            ("not another active status", self.review_args(revision, status="in_review")),
            ("reviewer must not be empty", self.review_args(revision, reviewer="  ")),
            ("review note must not be empty", self.review_args(revision, note=" ")),
        )
        with patch.object(CORE, "commit", return_value=True):
            for message, args in cases:
                with (
                    self.subTest(message=message, status=args.status),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    CORE.mutate(args, "review")
        meta, _ = CORE.read_task(path)
        self.assertEqual("in_review", meta["status"])
        self.assertEqual(revision, meta["task_revision"])

    def test_only_a_review_decision_makes_submitted_work_claimable_again(self) -> None:
        path = self.claim_and_submit()
        revision = CORE.read_task(path)[0]["task_revision"]
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(self.review_args(revision, status="open", note="needs rework"), "review")
            meta, _ = CORE.read_task(path)
            self.assertEqual("open", meta["status"])
            self.assertEqual("", meta["submitted_by"])
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-b", lease_minutes=10), "claim"
            )
        meta, _ = CORE.read_task(path)
        self.assertEqual("in_progress", meta["status"])
        self.assertEqual("worker-b", meta["owner"])
        self.assertEqual([], CORE.validate())

    def test_submitting_frees_its_worker_to_claim_the_next_task(self) -> None:
        self.make_task("AR-0002")
        self.claim_and_submit()
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                argparse.Namespace(task="AR-0002", owner="worker-a", lease_minutes=10), "claim"
            )
        self.assertEqual("in_progress", CORE.read_task(CORE.TASKS / "AR-0002-test.md")[0]["status"])
        self.assertEqual([], CORE.validate())

    def test_a_live_claim_cannot_be_released_into_a_verdict(self) -> None:
        """Release is owner authenticated, so naming done there is a worker certifying itself."""
        path = self.make_task("AR-0001")
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim"
            )
            with self.assertRaisesRegex(RuntimeError, "done is a review decision"):
                CORE.mutate(self.release_args(status="done"), "release")
        meta, _ = CORE.read_task(path)
        self.assertEqual("in_progress", meta["status"])
        self.assertEqual("worker-a", meta["owner"])
        self.assertNotIn("done", CORE.RELEASABLE)
        self.assertIn("done", CORE.DECIDED)
        with (
            patch("sys.stderr"),
            self.assertRaises(SystemExit),
        ):
            CORE.build_parser().parse_args(
                ["release", "AR-0001", "--owner", "w", "--status", "done", "--note", "n"]
            )

    def test_release_cannot_name_the_review_state_from_either_direction(self) -> None:
        self.make_task("AR-0001")
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim"
            )
            with self.assertRaisesRegex(RuntimeError, "use submit"):
                CORE.mutate(self.release_args(status="in_review"), "release")
        for argv in (
            ["release", "AR-0001", "--owner", "w", "--status", "in_review", "--note", "n"],
            ["release", "AR-0001", "--owner", "w", "--status", "in_progress", "--note", "n"],
            [
                "review",
                "AR-0001",
                "--reviewer",
                "c",
                "--expected-revision",
                "1",
                "--status",
                "in_review",
                "--note",
                "n",
            ],
        ):
            with (
                self.subTest(argv=argv[0] + argv[-3]),
                patch("sys.stderr"),
                self.assertRaises(SystemExit),
            ):
                CORE.build_parser().parse_args(argv)

    def test_a_submission_record_is_reported_wherever_it_does_not_belong(self) -> None:
        path = self.make_task("AR-0001")
        meta, body = CORE.read_task(path)
        cases = (
            ({"status": "open", "submitted_by": "worker-a"}, "submission recorded outside review"),
            (
                {
                    "status": "in_progress",
                    "owner": "worker-a",
                    "claim_expires": "2099-01-01T00:00:00+00:00",
                    "submitted_by": "worker-a",
                },
                "submission recorded outside review",
            ),
            ({"status": "done", "submitted_by": "worker-a"}, "submission recorded outside review"),
            ({"status": "in_review"}, "awaiting review without a recorded submitter"),
            (
                {"status": "in_review", "submitted_by": ""},
                "awaiting review without a recorded submitter",
            ),
            (
                {"status": "in_review", "submitted_by": 7},
                "awaiting review without a recorded submitter",
            ),
            (
                {"status": "in_review", "submitted_by": "worker-a", "owner": "worker-a"},
                "inactive task retains claim",
            ),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                fields = dict(meta)
                fields.update({"owner": "", "claim_expires": "", "submitted_by": ""})
                fields.update(changes)
                CORE.write_task(path, fields, body)
                self.refresh_views()
                self.assertIn(f"AR-0001: {message}", "\n".join(CORE.validate()))

    def test_work_awaiting_review_is_distinct_in_both_generated_views(self) -> None:
        self.make_task(
            "AR-0001",
            status="in_progress",
            owner="worker-a",
            claim_expires="2099-01-01T00:00:00+00:00",
        )
        self.make_task("AR-0002", status="in_review", submitted_by="worker-b")
        self.make_task("AR-0003", status="done")
        tasks = CORE.all_tasks()
        current = CORE.render_current(tasks)
        status = CORE.render_status_view(tasks)
        self.assertEqual([], CORE.validate())
        current_sections = {block.splitlines()[0]: block for block in current.split("\n## ")[1:]}
        status_sections = {block.splitlines()[0]: block for block in status.split("\n### ")[1:]}
        self.assertIn("AR-0002", current_sections["In Review"])
        self.assertIn("Submitted by worker-b", current_sections["In Review"])
        self.assertNotIn("AR-0002", current_sections["In Progress"])
        self.assertNotIn("AR-0002", current_sections["Done"])
        self.assertIn("AR-0002", status_sections["In review (1)"])
        self.assertIn("Submitted by worker-b", status_sections["In review (1)"])
        self.assertNotIn("AR-0002", status_sections["In progress (1)"])
        self.assertNotIn("AR-0002", status_sections["Done (1)"])
        self.assertIn(
            "| **In review** | Submitted by its worker, awaiting a coordinator decision | 1 |",
            status,
        )
        self.assertIn('AR_0002["AR-0002 - In review"]:::status_in_review', status)
        self.assertIn("classDef status_in_review fill:#ad1457", status)

    def test_an_owner_column_is_never_borrowed_by_a_task_that_is_not_in_review(self) -> None:
        """The submitted-by cell belongs to review alone, so no other row can be forged by it."""
        self.assertEqual("", CORE.submission_label({"status": "done", "submitted_by": "worker-a"}))
        self.assertEqual("", CORE.submission_label({"status": "in_review"}))
        self.assertEqual(
            "Submitted by worker-a",
            CORE.submission_label({"status": "in_review", "submitted_by": "worker-a"}),
        )

    def test_the_owner_column_neutralises_worker_supplied_text_in_both_views(self) -> None:
        """``--owner`` and ``submitted_by`` are worker supplied and reach two generated tables.

        The schema bounds their length and nothing else, so both cells carry text a worker
        chose straight into Markdown. A cell that kept a bare pipe would forge a column, and
        one that kept a newline would forge a row.
        """
        hostile = "w|x `c` [z]%<b>\nsecond"
        self.make_task(
            "AR-0001",
            status="in_progress",
            owner=hostile,
            claim_expires="2099-01-01T00:00:00+00:00",
        )
        self.make_task("AR-0002", status="in_review", submitted_by=hostile)
        tasks = CORE.all_tasks()
        current = CORE.render_current(tasks)
        status = CORE.render_status_view(tasks)
        self.assertEqual([], CORE.validate())

        def cells(view: str, column: int) -> list[str]:
            """Return one column of the inventory rows, with escaped pipes taken out first."""
            rows = [line for line in view.splitlines() if re.match(r"\| P[0-4] \|", line)]
            self.assertEqual(2, len(rows))
            self.assertEqual(2, sum("second" in line for line in view.splitlines()))
            collected = []
            for row in rows:
                parts = [part.strip() for part in row.replace("\\|", "").strip("|").split("|")]
                self.assertEqual(5, len(parts))
                collected.append(parts[column])
            return collected

        for cell in cells(current, 4):
            self.assertNotIn("|", cell)
            self.assertIn("second", cell)
        self.assertEqual(2, current.count("w\\|x"))
        for cell in cells(status, 2):
            for character in ("|", "`", "[", "]", "%", "<", ">"):
                with self.subTest(character=character):
                    self.assertNotIn(character, cell)
            self.assertIn("&#124;", cell)
            self.assertIn("second", cell)

    def test_every_status_in_the_vocabulary_agrees_with_the_schema_and_renders(self) -> None:
        schema = json.loads(
            (Path(__file__).resolve().parent.parent / "schema/task-schema.json").read_text()
        )
        self.assertEqual(list(CORE.STATUSES), schema["properties"]["status"]["enum"])
        self.assertEqual(
            ("in_progress", "in_review"),
            tuple(value for value in CORE.STATUSES if value not in CORE.DECIDED),
        )
        extras: dict[str, dict[str, object]] = {
            "in_progress": {"owner": "worker-a", "claim_expires": "2099-01-01T00:00:00+00:00"},
            "in_review": {"submitted_by": "worker-b"},
        }
        for index, value in enumerate(CORE.STATUSES, start=1):
            self.make_task(f"AR-{index:04d}", status=value, **extras.get(value, {}))
        tasks = CORE.all_tasks()
        current = CORE.render_current(tasks)
        status = CORE.render_status_view(tasks)
        self.assertEqual([], CORE.validate())
        for value in CORE.STATUSES:
            with self.subTest(status=value):
                self.assertIn(f"## {value.replace('_', ' ').title()}", current)
                self.assertIn(f"classDef status_{value} fill:#", status)

    def test_a_milestone_cannot_close_over_work_that_is_still_awaiting_review(self) -> None:
        self.make_task("AR-0001", status="in_review", submitted_by="worker-a")
        with (
            patch.object(CORE, "commit", return_value=True),
            self.assertRaisesRegex(RuntimeError, "unfinished work"),
        ):
            CORE.cmd_complete_milestone("M0")

    def test_main_dispatches_the_worker_and_coordinator_review_transitions(self) -> None:
        cases = (
            ["handoffctl", "submit", "AR-0001", "--owner", "worker-a", "--note", "done"],
            [
                "handoffctl",
                "review",
                "AR-0001",
                "--reviewer",
                "coordinator",
                "--expected-revision",
                "1",
                "--status",
                "done",
                "--note",
                "accepted",
            ],
        )
        for argv in cases:
            with (
                self.subTest(command=argv[1]),
                patch.object(sys, "argv", argv),
                patch.object(CORE, "mutate") as called,
            ):
                self.assertEqual(0, CORE.main())
                called.assert_called_once()
                self.assertEqual(argv[1], called.call_args[0][1])

    def expire_args(
        self,
        revision: int,
        task_id: str = "AR-0001",
        cleared_by: str = "coordinator",
        note: str = "worker terminated by a provider rate limit",
    ) -> argparse.Namespace:
        """Build the argument namespace the expire transition consumes."""
        return argparse.Namespace(
            task=task_id,
            expected_revision=revision,
            cleared_by=cleared_by,
            note=note,
        )

    def make_abandoned(self, task_id: str = "AR-0001", owner: str = "dead-a") -> Path:
        """Create a task whose worker went away holding a claim that has since lapsed."""
        past = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=5)).replace(microsecond=0)
        return self.make_task(
            task_id,
            status="in_progress",
            owner=owner,
            claim_expires=past.isoformat(),
        )

    def make_live(self, task_id: str = "AR-0009", owner: str = "worker-live") -> Path:
        """Create a task held by a worker that is still alive."""
        soon = (dt.datetime.now(dt.UTC) + dt.timedelta(minutes=60)).replace(microsecond=0)
        return self.make_task(
            task_id,
            status="in_progress",
            owner=owner,
            claim_expires=soon.isoformat(),
        )

    def unrelated_update(self, task_id: str = "AR-0009", owner: str = "worker-live") -> None:
        """Mutate a task no expired claim touches, through the ordinary owned path."""
        meta, _ = CORE.read_task(CORE.TASKS / f"{task_id}-test.md")
        CORE.mutate(
            argparse.Namespace(
                task=task_id,
                owner=owner,
                expected_revision=meta["task_revision"],
                status=None,
                priority=None,
                summary=None,
                next_action=None,
                note="progress on work the abandoned claims do not touch",
            ),
            "update",
        )

    def test_an_abandoned_lease_deadlocks_every_transition_until_expire_clears_it(self) -> None:
        """The observed failure, end to end: three workers died at once and nothing could move.

        Before ``expire`` existed the only recorded recovery was a human editing the task
        files outside the tool, because ``validate()`` runs over every task inside every
        transaction and the two transitions that could have cleared a lapsed claim were
        refused by that very claim.
        """
        for identifier, owner in (("AR-0001", "dead-a"), ("AR-0002", "dead-b")):
            self.make_abandoned(identifier, owner)
        self.make_live()
        self.assertEqual(
            ["AR-0001: expired claim", "AR-0002: expired claim"],
            [value for value in CORE.validate() if value.endswith("expired claim")],
        )
        with patch.object(CORE, "commit", return_value=True):
            # Release and heartbeat are the only transitions that could have cleared one,
            # and each is refused by the other task's standing claim.
            with self.assertRaisesRegex(RuntimeError, "AR-0002: expired claim"):
                CORE.mutate(
                    argparse.Namespace(
                        task="AR-0001", owner="dead-a", status="open", note="worker died"
                    ),
                    "release",
                )
            with self.assertRaisesRegex(RuntimeError, "AR-0002: expired claim"):
                CORE.mutate(
                    argparse.Namespace(task="AR-0001", owner="dead-a", lease_minutes=60),
                    "heartbeat",
                )
            with self.assertRaisesRegex(RuntimeError, "AR-0001: expired claim"):
                self.unrelated_update()

            revision = CORE.read_task(CORE.TASKS / "AR-0001-test.md")[0]["task_revision"]
            CORE.mutate(self.expire_args(revision), "expire")
            meta, _ = CORE.read_task(CORE.TASKS / "AR-0001-test.md")
            self.assertEqual("open", meta["status"])
            self.assertEqual("", meta["owner"])
            self.assertEqual("", meta["claim_expires"])
            self.assertEqual([], CORE.claim_errors(meta, {}, {}, {}))

            # One repair does not lift the fence: the remaining claim still refuses.
            self.assertEqual(["AR-0002: expired claim"], CORE.validate())
            with self.assertRaisesRegex(RuntimeError, "AR-0002: expired claim"):
                self.unrelated_update()

            revision = CORE.read_task(CORE.TASKS / "AR-0002-test.md")[0]["task_revision"]
            CORE.mutate(self.expire_args(revision, task_id="AR-0002"), "expire")
            self.assertEqual([], CORE.validate())
            self.unrelated_update()
        self.assertEqual([], CORE.validate())

    def test_expire_is_the_only_transition_a_standing_expired_claim_excuses(self) -> None:
        """Every other transition stays fenced, so the repair is not the weakening."""
        self.make_abandoned()
        soon = (dt.datetime.now(dt.UTC) + dt.timedelta(minutes=60)).replace(microsecond=0)
        live: dict[str, object] = {
            "status": "in_progress",
            "owner": "worker-b",
            "claim_expires": soon.isoformat(),
        }
        cases: list[tuple[str, dict[str, object], argparse.Namespace]] = [
            (
                "claim",
                {"status": "open"},
                argparse.Namespace(task="AR-0002", owner="worker-b", lease_minutes=10),
            ),
            (
                "heartbeat",
                live,
                argparse.Namespace(task="AR-0002", owner="worker-b", lease_minutes=10),
            ),
            (
                "update",
                live,
                argparse.Namespace(
                    task="AR-0002",
                    owner="worker-b",
                    expected_revision=1,
                    status=None,
                    priority=None,
                    summary=None,
                    next_action=None,
                    note="recorded progress",
                ),
            ),
            (
                "release",
                live,
                argparse.Namespace(
                    task="AR-0002", owner="worker-b", status="open", note="handing back"
                ),
            ),
            (
                "submit",
                live,
                argparse.Namespace(task="AR-0002", owner="worker-b", note="submitted for review"),
            ),
            (
                "review",
                {"status": "in_review", "submitted_by": "worker-b"},
                argparse.Namespace(
                    task="AR-0002",
                    reviewer="coordinator",
                    expected_revision=1,
                    status="done",
                    note="accepted",
                ),
            ),
            (
                "promote",
                {"status": "planned"},
                argparse.Namespace(task="AR-0002", expected_revision=1, note="dependencies met"),
            ),
            (
                "resume",
                {"status": "blocked"},
                argparse.Namespace(task="AR-0002", expected_revision=1, note="unblocked"),
            ),
        ]
        for kind, fields, args in cases:
            with self.subTest(transition=kind):
                self.make_task("AR-0002", **fields)
                with (
                    patch.object(CORE, "commit", return_value=True),
                    patch.object(CORE, "dirty_state_paths", return_value=[]),
                    self.assertRaisesRegex(RuntimeError, "AR-0001: expired claim"),
                ):
                    CORE.mutate(args, kind)
                self.assertEqual(
                    fields["status"], CORE.read_task(CORE.TASKS / "AR-0002-test.md")[0]["status"]
                )

    def test_expire_is_still_fenced_by_every_error_that_is_not_an_expired_claim(self) -> None:
        """The tolerance is one message, not a mood: any other fault refuses the repair too."""
        self.make_abandoned()
        leak = self.root / "leak.md"
        cases: list[tuple[str, str, dict[str, object] | None, str]] = [
            ("privacy leak", "", None, "absolute Linux home path"),
            ("unknown field", "AR-0002", {"extra": "bad"}, "unknown field extra"),
            ("ungoverned series", "AR-1101", {}, "series 11: no milestone document"),
            # A claim fault that is not an expired claim must fence the repair as well.
            (
                "active without claim",
                "AR-0002",
                {"status": "in_progress", "owner": "worker-b", "claim_expires": ""},
                "AR-0002: active without claim",
            ),
            (
                "unreadable expiry",
                "AR-0002",
                {"status": "in_progress", "owner": "worker-b", "claim_expires": "not-a-time"},
                "AR-0002: invalid claim expiry",
            ),
        ]
        for label, task_id, fields, message in cases:
            with self.subTest(fault=label):
                if fields is None:
                    leak.write_text("/" + "home/example")
                else:
                    self.make_task(task_id, **fields)
                revision = CORE.read_task(CORE.TASKS / "AR-0001-test.md")[0]["task_revision"]
                with (
                    patch.object(CORE, "commit", return_value=True),
                    self.assertRaisesRegex(RuntimeError, re.escape(message)),
                ):
                    CORE.mutate(self.expire_args(revision), "expire")
                meta, _ = CORE.read_task(CORE.TASKS / "AR-0001-test.md")
                self.assertEqual("in_progress", meta["status"])
                self.assertEqual("dead-a", meta["owner"])
                self.assertEqual(revision, meta["task_revision"])
                leak.unlink(missing_ok=True)
                if task_id:
                    (CORE.TASKS / f"{task_id}-test.md").unlink()
                self.refresh_views()
        # With every other fault gone, the same repair on the same state now succeeds.
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(self.expire_args(1), "expire")
        self.assertEqual([], CORE.validate())

    def test_the_tolerance_covers_only_standing_expiries_and_only_for_expire(self) -> None:
        """Pin the helper's own boundary: which transition, and which messages.

        This calls the helper directly, so it says nothing about *when* the set is taken.
        The third axis, the moment, is pinned through the real transaction by
        ``test_an_expiry_that_lapses_during_the_transaction_is_not_excused``, and the
        whole-message anchoring by
        ``test_a_message_that_merely_contains_an_expired_claim_is_not_tolerated``.
        """
        self.make_abandoned("AR-0001", "dead-a")
        self.make_abandoned("AR-0002", "dead-b")
        self.make_task("AR-0003", status="in_progress", owner="worker-c", claim_expires="")
        (self.root / "leak.md").write_text("/" + "home/example")
        standing = CORE.validate()
        self.assertIn("AR-0003: active without claim", standing)
        self.assertIn("leak.md: absolute Linux home path", standing)
        self.assertEqual(
            frozenset({"AR-0001: expired claim", "AR-0002: expired claim"}),
            CORE.tolerated_expiry_errors("expire"),
        )
        for kind in ("claim", "heartbeat", "update", "release", "submit", "review", "promote"):
            with self.subTest(transition=kind):
                self.assertEqual(frozenset(), CORE.tolerated_expiry_errors(kind))

    def test_a_message_that_merely_contains_an_expired_claim_is_not_tolerated(self) -> None:
        """The tolerance matches a whole message, never a substring of a different fault.

        Anchoring is the whole of the message axis. A validation error that merely *contains*
        the tolerated text -- an unknown field named after it is enough -- is a different
        fault about a different task, and excusing it would commit a tree that ``doctor``
        then rejects: exactly the red-that-nothing-inside-the-process-can-clear this work
        exists to remove.
        """
        self.make_abandoned()
        hostile = "AR-0001: expired claim"
        self.make_task("AR-0002", **{hostile: "an unknown field named after the tolerated text"})
        contained = f"AR-0002-test.md: unknown field {hostile}"
        standing = CORE.validate()
        self.assertIn(hostile, standing)
        self.assertIn(contained, standing)
        self.assertIn(hostile, contained)
        self.assertNotEqual(hostile, contained)
        self.assertEqual(frozenset({hostile}), CORE.tolerated_expiry_errors("expire"))
        with (
            patch.object(CORE, "commit", return_value=True),
            self.assertRaisesRegex(RuntimeError, re.escape(contained)),
        ):
            CORE.mutate(self.expire_args(1), "expire")
        meta, _ = CORE.read_task(CORE.TASKS / "AR-0001-test.md")
        self.assertEqual("in_progress", meta["status"])
        self.assertEqual("dead-a", meta["owner"])
        self.assertEqual(1, meta["task_revision"])
        self.assertIn(contained, CORE.validate())

    def test_an_expiry_that_lapses_during_the_transaction_is_not_excused(self) -> None:
        """The set is taken before the write, so a lease that lapses mid-transaction fails closed.

        Recomputing it after the write would excuse an error that did not exist when the
        transition was authorised -- an error the transaction raced rather than one it was
        granted. The lapse is forced deterministically rather than waited for, so the test
        pins the ordering and not the clock.
        """
        self.make_abandoned("AR-0001", "dead-a")
        soon = (dt.datetime.now(dt.UTC) + dt.timedelta(minutes=60)).replace(microsecond=0)
        lapsing = self.make_task(
            "AR-0002", status="in_progress", owner="worker-b", claim_expires=soon.isoformat()
        )
        past = (dt.datetime.now(dt.UTC) - dt.timedelta(minutes=1)).replace(microsecond=0)
        self.assertEqual(
            frozenset({"AR-0001: expired claim"}), CORE.tolerated_expiry_errors("expire")
        )
        original = CORE.write_task

        def lapse_mid_transaction(path: Path, meta: dict[str, Any], body: str) -> None:
            """Let AR-0002's lease run out between the tolerated set and the validation."""
            original(path, meta, body)
            if path.name != "AR-0001-test.md":
                return
            other, other_body = CORE.read_task(lapsing)
            other["claim_expires"] = past.isoformat()
            original(lapsing, other, other_body)

        with (
            patch.object(CORE, "commit", return_value=True),
            patch.object(CORE, "write_task", side_effect=lapse_mid_transaction),
            self.assertRaisesRegex(RuntimeError, "AR-0002: expired claim"),
        ):
            CORE.mutate(self.expire_args(1), "expire")
        meta, _ = CORE.read_task(CORE.TASKS / "AR-0001-test.md")
        self.assertEqual("in_progress", meta["status"])
        self.assertEqual("dead-a", meta["owner"])
        self.assertEqual(1, meta["task_revision"])

    def test_expire_refuses_a_claim_that_has_not_lapsed(self) -> None:
        """A live lease is not an abandoned one; expire can never take work from a worker."""
        self.make_live("AR-0001", "worker-live")
        with (
            patch.object(CORE, "commit", return_value=True),
            self.assertRaisesRegex(RuntimeError, "does not hold an expired claim"),
        ):
            CORE.mutate(self.expire_args(1), "expire")
        meta, _ = CORE.read_task(CORE.TASKS / "AR-0001-test.md")
        self.assertEqual("in_progress", meta["status"])
        self.assertEqual("worker-live", meta["owner"])
        self.assertEqual(1, meta["task_revision"])

    def test_expire_refuses_a_claim_the_lease_check_cannot_read_as_expired(self) -> None:
        """Only the exact condition doctor calls an expired claim is repaired here."""
        cases = (
            ("", "active without claim"),
            ("not-a-time", "invalid claim expiry"),
            ("2000-01-01T00:00:00", "invalid claim expiry"),
        )
        for expiry, reported in cases:
            with self.subTest(claim_expires=expiry):
                self.make_task(
                    "AR-0001", status="in_progress", owner="dead-a", claim_expires=expiry
                )
                self.assertIn(f"AR-0001: {reported}", CORE.validate())
                with (
                    patch.object(CORE, "commit", return_value=True),
                    self.assertRaisesRegex(RuntimeError, "does not hold an expired claim"),
                ):
                    CORE.mutate(self.expire_args(1), "expire")
                meta, _ = CORE.read_task(CORE.TASKS / "AR-0001-test.md")
                self.assertEqual("in_progress", meta["status"])
                self.assertEqual("dead-a", meta["owner"])

    def test_expire_refuses_every_status_that_is_not_an_active_claim(self) -> None:
        for status in CORE.STATUSES:
            if status == "in_progress":
                continue
            with self.subTest(status=status):
                fields: dict[str, object] = {"status": status}
                if status == "in_review":
                    fields["submitted_by"] = "worker-a"
                self.make_task("AR-0001", **fields)
                with (
                    patch.object(CORE, "commit", return_value=True),
                    self.assertRaisesRegex(RuntimeError, "AR-0001 is not active"),
                ):
                    CORE.mutate(self.expire_args(1), "expire")
                self.assertEqual(
                    status, CORE.read_task(CORE.TASKS / "AR-0001-test.md")[0]["status"]
                )

    def test_expire_refuses_a_stale_revision(self) -> None:
        self.make_abandoned()
        with (
            patch.object(CORE, "commit", return_value=True),
            self.assertRaisesRegex(RuntimeError, "stale revision: expected 7, current 1"),
        ):
            CORE.mutate(self.expire_args(7), "expire")
        self.assertEqual("in_progress", CORE.read_task(CORE.TASKS / "AR-0001-test.md")[0]["status"])

    def test_clearing_an_abandoned_claim_requires_a_named_clearer_and_a_reason(self) -> None:
        """A silent repair is what turned the first recovery into an untraceable hand edit."""
        self.make_abandoned()
        cases = (
            ({"cleared_by": ""}, "must name who cleared it"),
            ({"cleared_by": "   "}, "must name who cleared it"),
            ({"note": ""}, "expiry note must not be empty"),
            ({"note": "   "}, "expiry note must not be empty"),
        )
        for changes, message in cases:
            with self.subTest(**changes):
                args = self.expire_args(1)
                for name, value in changes.items():
                    setattr(args, name, value)
                with (
                    patch.object(CORE, "commit", return_value=True),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    CORE.mutate(args, "expire")
                meta, _ = CORE.read_task(CORE.TASKS / "AR-0001-test.md")
                self.assertEqual("in_progress", meta["status"])
                self.assertEqual("dead-a", meta["owner"])
                self.assertEqual(1, meta["task_revision"])

    def test_clearing_an_abandoned_claim_records_who_whose_and_why(self) -> None:
        path = self.make_abandoned()
        expiry = CORE.read_task(path)[0]["claim_expires"]
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                self.expire_args(1, cleared_by="coordinator-1", note="rate limit killed the pool"),
                "expire",
            )
        body = CORE.read_task(path)[1]
        entries = CORE.EVIDENCE_ENTRY.findall(body)
        self.assertEqual(1, len(entries))
        self.assertIn("cleared by coordinator-1", body)
        self.assertIn("Abandoned claim of dead-a", body)
        self.assertIn(expiry, body)
        self.assertIn("rate limit killed the pool", body)
        self.assertEqual("# Test\n\n", CORE.task_description(body)[0])
        self.assertEqual([], CORE.validate())

    def test_doctor_reports_an_abandoned_claim_and_names_the_way_out_of_it(self) -> None:
        """A red check must still go red, and must say what clears it from inside the tool."""
        self.make_abandoned()
        with patch("builtins.print") as printed:
            self.assertEqual(1, CORE.cmd_doctor(live=False))
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("ERROR: AR-0001: expired claim", output)
        self.assertIn("handoffctl expire", output)
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(self.expire_args(1), "expire")
        with patch("builtins.print") as printed:
            self.assertEqual(0, CORE.cmd_doctor(live=False))
        self.assertNotIn(
            "handoffctl expire", "\n".join(str(call.args[0]) for call in printed.call_args_list)
        )

    def test_the_recovery_hint_is_keyed_on_an_expired_claim_and_nothing_else(self) -> None:
        self.make_task("AR-0001", status="open")
        (self.root / "leak.md").write_text("/" + "home/example")
        with patch("builtins.print") as printed:
            self.assertEqual(1, CORE.cmd_doctor(live=False))
        output = "\n".join(str(call.args[0]) for call in printed.call_args_list)
        self.assertIn("absolute Linux home path", output)
        self.assertNotIn("handoffctl expire", output)

    def test_expire_is_not_owner_authenticated_and_names_no_destination(self) -> None:
        """The owner of an abandoned claim is gone, so no identity of theirs is accepted."""
        self.assertIs(CORE.apply_expire, CORE.TRANSITIONS["expire"])
        parser = CORE.build_parser()
        args = parser.parse_args(
            [
                "expire",
                "AR-0001",
                "--expected-revision",
                "3",
                "--cleared-by",
                "coordinator",
                "--note",
                "worker gone",
            ]
        )
        self.assertEqual("coordinator", args.cleared_by)
        self.assertFalse(hasattr(args, "owner"))
        self.assertFalse(hasattr(args, "status"))
        for rejected in (["--owner", "dead-a"], ["--status", "done"]):
            with (
                self.subTest(argument=rejected[0]),
                patch.object(sys, "stderr", io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parser.parse_args(
                    [
                        "expire",
                        "AR-0001",
                        "--expected-revision",
                        "3",
                        "--cleared-by",
                        "coordinator",
                        "--note",
                        "worker gone",
                        *rejected,
                    ]
                )

    def test_a_cleared_task_returns_to_the_queue_and_cannot_be_expired_twice(self) -> None:
        self.make_abandoned()
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(self.expire_args(1), "expire")
            revision = CORE.read_task(CORE.TASKS / "AR-0001-test.md")[0]["task_revision"]
            with self.assertRaisesRegex(RuntimeError, "AR-0001 is not active"):
                CORE.mutate(self.expire_args(revision), "expire")
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-next", lease_minutes=10), "claim"
            )
        meta, _ = CORE.read_task(CORE.TASKS / "AR-0001-test.md")
        self.assertEqual("in_progress", meta["status"])
        self.assertEqual("worker-next", meta["owner"])
        self.assertEqual([], CORE.validate())

    def test_main_dispatches_the_expiry_recovery_transition(self) -> None:
        argv = [
            "handoffctl",
            "expire",
            "AR-0001",
            "--expected-revision",
            "1",
            "--cleared-by",
            "coordinator",
            "--note",
            "worker gone",
        ]
        with patch.object(sys, "argv", argv), patch.object(CORE, "mutate") as called:
            self.assertEqual(0, CORE.main())
        called.assert_called_once()
        self.assertEqual("expire", called.call_args[0][1])

    # --- Parity between the transaction gate and the published schema gate (AR-0015) ---

    def schema_of(self, name: str) -> dict[str, Any]:
        """Return one published schema, read from the source tree the tool ships with."""
        return cast(dict[str, Any], json.loads((CORE.SCHEMA / name).read_text()))

    @staticmethod
    def apply_changes(meta: dict[str, Any], changes: dict[str, object]) -> None:
        """Apply one fixture's changes, where ABSENT removes a field rather than setting it."""
        for name, value in changes.items():
            if value is ABSENT:
                meta.pop(name, None)
            else:
                meta[name] = value

    def plant_task(self, changes: dict[str, object]) -> dict[str, Any]:
        """Write a task carrying one fixture's defect, bypassing the transition that refuses it."""
        path = self.make_task("AR-0001")
        meta, body = CORE.read_task(path)
        self.apply_changes(meta, changes)
        CORE.write_task(path, meta, body)
        return cast(dict[str, Any], meta)

    def plant_milestone(self, changes: dict[str, object]) -> dict[str, Any]:
        """Write a milestone document carrying one fixture's defect."""
        self.make_task("AR-0001")
        path = self.root / "milestones" / "M0.md"
        meta, body = CORE.read_task(path)
        self.apply_changes(meta, changes)
        CORE.write_task(path, meta, body)
        return cast(dict[str, Any], meta)

    @staticmethod
    def expected_messages(
        document: str, path: str, keyword: str, bound: Any, changes: dict[str, object]
    ) -> list[str]:
        """Return the message the transaction gate must emit for one stated constraint."""
        if keyword == "required":
            return [
                f"{document}: missing {name}" for name, value in changes.items() if value is ABSENT
            ]
        if keyword == "additionalProperties":
            return [f"{document}: unknown field {name}" for name in changes]
        field = path.split(".")[-1].replace("[]", "[0]")
        return [f"{document}: {field} violates {keyword} {json.dumps(bound, sort_keys=True)}"]

    def check_constraint_fixtures(
        self,
        schema_name: str,
        document: str,
        fixtures: tuple[tuple[str, str, dict[str, object]], ...],
        plant: Any,
    ) -> None:
        """Prove each stated constraint is refused by validate() and by the schema gate alike."""
        schema = self.schema_of(schema_name)
        stated = stated_constraints(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        self.assertEqual(
            set(stated), {(path, keyword) for path, keyword, _ in fixtures}, schema_name
        )
        for path, keyword, changes in fixtures:
            with self.subTest(schema=schema_name, constraint=path, keyword=keyword):
                meta = plant(changes)
                errors = "\n".join(CORE.validate())
                for message in self.expected_messages(
                    document, path, keyword, stated[(path, keyword)], changes
                ):
                    self.assertIn(message, errors)
                self.assertTrue(list(validator.iter_errors(meta)), "schema gate accepted it")
                self.assertEqual(1, CORE.cmd_doctor(live=False))

    def test_every_task_constraint_the_schema_states_is_enforced_by_validate(self) -> None:
        """A fixture per stated bound, and a key set derived from the schema so none is missed."""
        with patch("builtins.print"):
            self.check_constraint_fixtures(
                "task-schema.json", "AR-0001-test.md", TASK_CONSTRAINT_FIXTURES, self.plant_task
            )

    def test_every_milestone_constraint_the_schema_states_is_enforced_by_validate(self) -> None:
        with patch("builtins.print"):
            self.check_constraint_fixtures(
                "milestone-schema.json",
                "M0.md",
                MILESTONE_CONSTRAINT_FIXTURES,
                self.plant_milestone,
            )

    def test_every_required_field_is_refused_one_at_a_time(self) -> None:
        """All sixteen, not the nine the hand-maintained tuple used to name."""
        required = self.schema_of("task-schema.json")["required"]
        self.assertEqual(16, len(required))
        for name in required:
            with self.subTest(field=name):
                self.plant_task({name: ABSENT})
                self.assertIn(f"AR-0001-test.md: missing {name}", CORE.validate())

    def test_update_refuses_an_over_long_value_naming_the_field_and_the_bound(self) -> None:
        """The defect this task exists to close: the write gate was the permissive one."""
        path = self.make_task()
        with patch.object(CORE, "commit", return_value=True):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim"
            )
            before = path.read_text()
            revision = CORE.read_task(path)[0]["task_revision"]
            for field, bound in (("next_action", 300), ("summary", 300)):
                with self.subTest(field=field):
                    args = argparse.Namespace(
                        task="AR-0001",
                        owner="worker-a",
                        expected_revision=revision,
                        status=None,
                        priority=None,
                        summary=None,
                        next_action=None,
                        note="over long",
                    )
                    setattr(args, field, "x" * (bound + 1))
                    with self.assertRaisesRegex(
                        RuntimeError, f"{field} violates maxLength {bound} \\(length {bound + 1}\\)"
                    ):
                        CORE.mutate(args, "update")
                    self.assertEqual(before, path.read_text())
                    self.assertEqual([], CORE.validate())

    def test_a_transition_refuses_an_over_long_title_it_did_not_write(self) -> None:
        """No transition names title, so the bound is proved where a transition meets one."""
        path = self.make_task()
        meta, body = CORE.read_task(path)
        meta["title"] = "t" * 121
        CORE.write_task(path, meta, body)
        self.refresh_views()
        with (
            patch.object(CORE, "commit", return_value=True),
            self.assertRaisesRegex(RuntimeError, "title violates maxLength 120 \\(length 121\\)"),
        ):
            CORE.mutate(
                argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim"
            )
        self.assertEqual("open", CORE.read_task(path)[0]["status"])

    def test_a_tree_that_passes_doctor_passes_the_published_schema_gate(self) -> None:
        """Doctor is only as strong as the weakest rule behind it; prove it is not weaker."""
        task_validator = Draft202012Validator(
            self.schema_of("task-schema.json"), format_checker=FormatChecker()
        )
        milestone_validator = Draft202012Validator(
            self.schema_of("milestone-schema.json"), format_checker=FormatChecker()
        )
        planted: list[tuple[str, dict[str, object]]] = [
            ("clean", {}),
            ("over-long next action", {"next_action": "x" * 301}),
            ("over-long title", {"title": "t" * 121}),
            ("unpatterned worktree key", {"worktree_key": "Game-Experiment"}),
            ("bare date", {"updated_at": "2026-09-08"}),
            ("boolean revision", {"task_revision": True}),
            ("valid alternative", {"worktree_key": "game-experiment-render"}),
            ("valid plan", {"plan": "../plans/AR-0001.md"}),
        ]
        (self.root / "plans").mkdir(exist_ok=True)
        (self.root / "plans" / "AR-0001.md").write_text("# AR-0001\n")
        for label, changes in planted:
            with self.subTest(tree=label), patch("builtins.print"):
                meta = self.plant_task(changes)
                healthy = CORE.cmd_doctor(live=False) == 0
                milestone = CORE.read_task(self.root / "milestones" / "M0.md")[0]
                accepted = not list(task_validator.iter_errors(meta)) and not list(
                    milestone_validator.iter_errors(milestone)
                )
                self.assertFalse(healthy and not accepted, "doctor passed what the schema rejects")

    def test_a_schema_construct_the_transaction_cannot_enforce_is_refused_loudly(self) -> None:
        """A bound the enforcer does not implement must stop the tree, never be ignored."""
        self.make_task("AR-0001")
        staged = self.root / "schema"
        staged.mkdir()
        schema = self.schema_of("task-schema.json")
        schema["properties"]["title"]["minItems"] = 1
        schema["properties"]["depends_on"]["items"]["contentEncoding"] = "base64"
        schema["properties"]["updated_at"]["format"] = "email"
        schema["properties"]["branch"]["type"] = "null"
        schema["allOf"][0]["not"] = {}
        # A type stated as a list is legal draft 2020-12 and unhashable in Python: this must
        # produce a refusal, not a TypeError out of validate() inside every transaction.
        schema["properties"]["observed_branch"]["type"] = ["string", "null"]
        # Inside a conditional branch, where an unenforced bound is easiest to miss: the
        # branch subschema is where the claim and review rules live.
        schema["allOf"][0]["then"]["properties"]["owner"]["multipleOf"] = 2
        schema["allOf"][1]["else"]["properties"]["submitted_by"]["format"] = "hostname"
        schema["dependentRequired"] = {}
        (staged / "task-schema.json").write_text(json.dumps(schema))
        (staged / "milestone-schema.json").write_text(
            (CORE.SCHEMA / "milestone-schema.json").read_text()
        )
        with patch.object(CORE, "SCHEMA", staged):
            errors = "\n".join(CORE.validate())
            for message in (
                "task-schema.json: title states unsupported constraint minItems",
                "task-schema.json: depends_on[] states unsupported constraint contentEncoding",
                "task-schema.json: updated_at states unsupported format email",
                "task-schema.json: branch states unsupported type null",
                "task-schema.json: unsupported conditional keyword not",
                "task-schema.json: unsupported schema keyword dependentRequired",
                "task-schema.json: observed_branch states unsupported type ['string', 'null']",
                "task-schema.json then: owner states unsupported constraint multipleOf",
                "task-schema.json else: submitted_by states unsupported format hostname",
            ):
                self.assertIn(message, errors)
            with (
                patch.object(CORE, "commit", return_value=True),
                self.assertRaisesRegex(RuntimeError, "unsupported schema keyword"),
            ):
                CORE.mutate(
                    argparse.Namespace(task="AR-0001", owner="worker-a", lease_minutes=10), "claim"
                )

    def test_a_conditional_states_which_half_applies_and_invents_no_other(self) -> None:
        """A branch the schema does not state must be skipped, and a stated one must fire."""
        self.make_task("AR-0001")
        staged = self.root / "conditional-schema"
        staged.mkdir()
        schema = self.schema_of("task-schema.json")
        schema["allOf"].append(
            {
                "if": {"properties": {"status": {"const": "blocked"}}, "required": ["status"]},
                "then": {"properties": {"priority": {"const": "P0"}}},
            }
        )
        (staged / "task-schema.json").write_text(json.dumps(schema))
        (staged / "milestone-schema.json").write_text(
            (CORE.SCHEMA / "milestone-schema.json").read_text()
        )
        document = CORE.TASKS / "AR-0001-test.md"
        with patch.object(CORE, "SCHEMA", staged):
            self.assertEqual([], CORE.validate())
            meta, body = CORE.read_task(document)
            meta["status"] = "blocked"
            CORE.write_task(document, meta, body)
            self.refresh_views()
            self.assertIn('AR-0001-test.md: priority violates const "P0"', CORE.validate())

    def test_an_unreadable_published_schema_refuses_the_tree(self) -> None:
        self.make_task("AR-0001")
        empty = self.root / "no-schema"
        empty.mkdir()
        with patch.object(CORE, "SCHEMA", empty):
            errors = CORE.validate()
        self.assertIn("task-schema.json: unreadable published schema (FileNotFoundError)", errors)
        broken = self.root / "broken-schema"
        broken.mkdir()
        (broken / "task-schema.json").write_text("{")
        (broken / "milestone-schema.json").write_text("{}")
        with patch.object(CORE, "SCHEMA", broken):
            errors = CORE.validate()
        self.assertIn("task-schema.json: unreadable published schema (JSONDecodeError)", errors)

    def test_the_constraint_helpers_reject_what_a_permissive_mutant_would_accept(self) -> None:
        """Pin each predicate directly, so a fail-open rewrite of one cannot hide behind another."""
        self.assertTrue(CORE._json_type_error(True, "integer"))
        self.assertFalse(CORE._json_type_error(3, "integer"))
        self.assertTrue(CORE._json_type_error(3, "string"))
        self.assertTrue(CORE._json_type_error("x", "array"))
        self.assertTrue(CORE._json_type_error([], "object"))
        self.assertFalse(CORE._json_type_error({}, "object"))
        for value in ("2026-09-08", "2026-09-08 12:00:00+00:00", "2026-09-08T12:00:00", "x"):
            with self.subTest(value=value):
                self.assertTrue(CORE._format_error(value, "date-time"))
        self.assertFalse(CORE._format_error("2026-09-08T12:00:00+00:00", "date-time"))
        self.assertFalse(CORE._format_error("2026-09-08T12:00:00Z", "date-time"))
        self.assertTrue(CORE._format_error("2026-02-30T12:00:00Z", "date-time"))
        self.assertFalse(CORE._format_error(7, "date-time"))
        self.assertTrue(CORE._format_error("anything", "uri"))
        self.assertTrue(CORE._length_error("xx", 1, longer=True))
        self.assertFalse(CORE._length_error(7, 1, longer=True))
        self.assertFalse(CORE._length_error("xx", "1", longer=True))
        self.assertTrue(CORE._length_error("", 1, longer=False))
        # Both bounds are inclusive, and both boundaries are pinned. An exclusive comparison
        # here refuses a one-character title, summary or milestone label and a 20-character
        # claim_expires, all of which the schema accepts -- the transaction refusing what the
        # schema allows, which is the same divergence this task closes, facing the other way.
        self.assertFalse(CORE._length_error("x", 1, longer=False))
        self.assertFalse(CORE._length_error("x" * 20, 20, longer=False))
        self.assertFalse(CORE._length_error("x", 1, longer=True))
        self.assertFalse(CORE._length_error("x" * 300, 300, longer=True))
        self.assertTrue(CORE._unique_error([{"a": 1}, {"a": 1}], True))
        self.assertFalse(CORE._unique_error([{"a": 1}, {"a": 2}], True))
        self.assertFalse(CORE._unique_error([1, 1], False))
        self.assertFalse(CORE._unique_error("aa", True))
        self.assertTrue(CORE._pattern_error("nope", "^AR-[0-9]{4}$"))
        self.assertFalse(CORE._pattern_error(5, "^AR-[0-9]{4}$"))
        # JSON Schema states pattern as an unanchored search. Every pattern the published
        # schemas state today happens to be anchored, so a match or fullmatch here would pass
        # the fixtures while refusing documents the schema gate accepts -- the same divergence
        # this task closes, reopened in the other direction by an unanchored pattern added later.
        self.assertFalse(CORE._pattern_error("xxAR-0001", "AR-[0-9]{4}"))
        self.assertFalse(CORE._pattern_error("AR-0001xx", "^AR-[0-9]{4}"))
        # The one place the transaction gate is deliberately stricter than the schema on a
        # mechanical bound. Draft 2020-12 counts a number with zero fractional part as an
        # integer, so 1.0 satisfies "type": "integer"; a revision counter that is a float
        # would be written back as 2.0 and fence subsequent transitions on a float forever,
        # so it is refused here. Pinned so the difference stays a decision, not a drift.
        self.assertTrue(CORE._json_type_error(1.0, "integer"))
        self.assertFalse(CORE._json_type_error(1, "integer"))
        self.assertTrue(CORE._json_equal(1, 1))
        self.assertTrue(CORE._json_equal(True, True))
        self.assertFalse(CORE._json_equal(True, 1))
        self.assertFalse(CORE._json_equal(0, False))
        self.assertTrue(CORE._json_equal(1, 1.0))
        self.assertTrue(CORE.SCHEMA_CONSTRAINTS["const"](True, 1))
        self.assertFalse(CORE.SCHEMA_CONSTRAINTS["const"](1, 1))
        self.assertTrue(CORE.SCHEMA_CONSTRAINTS["enum"](True, [1, 2]))
        self.assertFalse(CORE.SCHEMA_CONSTRAINTS["enum"]("P0", ["P0"]))
        self.assertFalse(CORE.SCHEMA_CONSTRAINTS["enum"]("P0", "P0"))
        self.assertTrue(CORE._minimum_error(0, 1))
        self.assertFalse(CORE._minimum_error(True, 1))
        self.assertFalse(CORE._minimum_error("0", 1))
        self.assertFalse(CORE._minimum_error(0, "1"))

    def test_compiled_bytecode_is_outside_the_state_file_size_cap(self) -> None:
        """The gate measures state, and a .pyc is a build artefact of the suite that precedes it.

        CI runs the fault tests and then doctor in the same job, so the tests write
        tests/__pycache__ and doctor then walks it. The cap exists to stop a state file
        growing without bound; a .pyc crossing it reports a defect in the tree that is not
        one, and would have stopped whoever next added a few hundred lines of tests.
        """
        cache = self.root / "tests" / "__pycache__"
        cache.mkdir(parents=True)
        (cache / "test_handoffctl.cpython-312.pyc").write_bytes(b"\x00" * 200_001)
        self.assertEqual([], [e for e in CORE.privacy_errors() if "exceeds" in e])
        # The cap itself still bites outside __pycache__, so the exclusion is narrow.
        (self.root / "tests" / "big.txt").write_text("x" * 200_001)
        self.assertIn("tests/big.txt: state file exceeds 200 KiB", CORE.privacy_errors())

    def plant_leak(self, directory: Path, name: str = "note.md") -> Path:
        """Plant one unmistakable private reference and return the file holding it."""
        directory.mkdir(parents=True, exist_ok=True)
        leak = directory / name
        leak.write_text("deployed from /ho" + "me/agent/checkout\n")
        return leak

    def test_every_exclusion_is_matched_relative_to_the_root_and_never_absolutely(self) -> None:
        """Pin the defect: an excluded name ABOVE the root must suppress nothing inside it.

        The walk used to test ``path.parts`` on the absolute path and compute the relative
        path afterwards, so any excluded name in any ancestor skipped the whole tree. Worker
        checkouts live at ``workspaces/AR-NNNN``, so that was every file of every worktree and
        the gate reported clean over an empty set. The loop is over the constant rather than
        over ``workspaces`` alone: a new exclusion is pinned the day it is added.
        """
        for excluded in sorted(CORE.PRIVACY_EXCLUSIONS):
            with self.subTest(excluded=excluded):
                checkout = self.root / "above" / excluded / "checkout"
                self.plant_leak(checkout)
                CORE.ROOT = checkout
                self.assertEqual([Path("note.md")], CORE.privacy_walk())
                self.assertEqual(["note.md: absolute Linux home path"], CORE.privacy_errors())

    def test_a_worktree_under_workspaces_is_examined_and_the_shared_checkout_still_skips_it(
        self,
    ) -> None:
        """Both directions of the exclusion, over one tree that is both root and workspace.

        Scoped to the root, a worktree examines its own files -- the same set the shared
        checkout would examine for identical content -- while the shared checkout still skips
        `workspaces/` whole, which is what keeps entire product checkouts out of every
        transaction.
        """
        shared = self.root / "shared"
        worktree = shared / "workspaces" / "AR-0001"
        for base in (shared, worktree):
            (base / "tasks").mkdir(parents=True)
            (base / "tasks" / "AR-0001-test.md").write_text("# Test\n")
            self.plant_leak(base)

        CORE.ROOT = shared
        from_shared = CORE.privacy_walk()
        self.assertEqual([Path("note.md"), Path("tasks/AR-0001-test.md")], from_shared)
        # One error, not two: the leak planted inside the worktree is out of scope from here.
        self.assertEqual(["note.md: absolute Linux home path"], CORE.privacy_errors())

        CORE.ROOT = worktree
        from_worktree = CORE.privacy_walk()
        self.assertNotEqual([], from_worktree)
        self.assertEqual(from_shared, from_worktree)
        self.assertEqual(["note.md: absolute Linux home path"], CORE.privacy_errors())

    def test_an_excluded_directory_is_pruned_rather_than_walked_and_filtered(self) -> None:
        """The exclusion has to be cheap as well as correct, so the tree is never descended.

        A product checkout under `workspaces/` is thousands of files, and `doctor` runs inside
        every transaction. Filtering after the fact would enumerate them all; pruning means the
        walk never enters. A dangling symlink is included here because `os.walk` reports one as
        a file and the size check must not follow it.
        """
        buried = self.root / "workspaces" / "AR-0001" / "deep"
        buried.mkdir(parents=True)
        self.plant_leak(buried)
        (self.root / "dangling.md").symlink_to(self.root / "absent.md")
        visited: list[str] = []
        walk = os.walk

        def spy(top: Any, **options: Any) -> Any:
            for entry in walk(top, **options):
                visited.append(entry[0])
                yield entry

        with patch.object(CORE.os, "walk", spy):
            files = CORE.privacy_walk()
        self.assertEqual([], [top for top in visited if "workspaces" in Path(top).parts])
        self.assertNotIn(Path("workspaces/AR-0001/deep/note.md"), files)
        self.assertIn(Path("dangling.md"), files)
        self.assertEqual([], CORE.privacy_errors())

    def test_a_worktree_git_marker_file_is_excluded_like_a_git_directory(self) -> None:
        """`.git` is a file in a worktree, and it holds the absolute path of the real one."""
        (self.root / ".git").write_text("gitdir: /ho" + "me/agent/repo/.git/worktrees/AR-0001\n")
        self.assertNotIn(Path(".git"), CORE.privacy_walk())
        self.assertEqual([], CORE.privacy_errors())

    def test_a_walk_that_examines_nothing_is_an_error_in_itself(self) -> None:
        """The decision recorded in AR-0018: an empty walk is a defect, not a clean report.

        Relative scoping is what stops the walk going vacuous, and this floor is the check on
        the checker: whatever empties the set -- a misconfigured root, an exclusion that
        swallows the tree, a regression in either -- the gate says so instead of passing. It
        catches total vacuity only, which is the shape the old defect took in every worktree.
        """
        vacuous = self.root / "vacuous"
        (vacuous / "workspaces" / "AR-0001").mkdir(parents=True)
        self.plant_leak(vacuous / "workspaces" / "AR-0001")
        CORE.ROOT = vacuous
        self.assertEqual([], CORE.privacy_walk())
        self.assertEqual(
            ["privacy walk examined no file: the root is empty or excluded entirely"],
            CORE.privacy_errors(),
        )

    def test_a_changed_top_level_type_is_refused_rather_than_silently_unenforced(self) -> None:
        """schema_object_errors skips the document type; the support net must not skip it too.

        read_task already refuses non-object front matter, which is why the document type is
        not re-checked. That reasoning holds only while the schema says object: a schema
        stating array refuses every document, so an unenforced top-level type would accept
        everything the schema gate rejects.
        """
        self.make_task("AR-0001")
        for declared in ("array", "string"):
            with self.subTest(type=declared):
                staged = self.root / f"top-{declared}"
                staged.mkdir()
                schema = self.schema_of("task-schema.json")
                schema["type"] = declared
                (staged / "task-schema.json").write_text(json.dumps(schema))
                (staged / "milestone-schema.json").write_text(
                    (CORE.SCHEMA / "milestone-schema.json").read_text()
                )
                with patch.object(CORE, "SCHEMA", staged):
                    self.assertIn(
                        f"task-schema.json: states unsupported document type {declared}",
                        CORE.validate(),
                    )
                    with patch("builtins.print"):
                        self.assertEqual(1, CORE.cmd_doctor(live=False))

    def test_the_schema_object_form_of_additional_properties_is_refused(self) -> None:
        """Only the boolean form is enforced, so the schema-object form must not pass quietly."""
        self.make_task("AR-0001")
        staged = self.root / "extra-schema"
        staged.mkdir()
        schema = self.schema_of("task-schema.json")
        schema["additionalProperties"] = {"type": "string"}
        (staged / "task-schema.json").write_text(json.dumps(schema))
        (staged / "milestone-schema.json").write_text(
            (CORE.SCHEMA / "milestone-schema.json").read_text()
        )
        with patch.object(CORE, "SCHEMA", staged):
            self.assertIn(
                "task-schema.json: states unsupported additionalProperties {'type': 'string'}",
                CORE.validate(),
            )

    def test_a_value_at_a_minimum_length_is_accepted_by_both_gates(self) -> None:
        """minLength is inclusive: the transaction must accept exactly what the schema does."""
        validator = Draft202012Validator(
            self.schema_of("task-schema.json"), format_checker=FormatChecker()
        )
        boundaries: list[dict[str, object]] = [
            {"title": "t", "summary": "s", "next_action": "n"},
            {"status": "in_progress", "owner": "w", "claim_expires": "2099-01-01T00:00:00+00"},
            {"status": "in_review", "submitted_by": "w"},
        ]
        for changes in boundaries:
            with self.subTest(changes=sorted(changes)):
                meta = self.plant_task(changes)
                self.assertEqual([], list(validator.iter_errors(meta)), "fixture is not valid")
                complaints = [e for e in CORE.validate() if "minLength" in e or "maxLength" in e]
                self.assertEqual([], complaints)

    def staged_schema(self, **changes: Any) -> Path:
        """Publish a modified task schema beside an untouched milestone schema.

        The support net reads what the repository publishes, so a construct is tested the way
        it would arrive: by editing the published file, not by calling the reader directly.
        """
        staged = self.root / f"staged-schema-{len(list(self.root.glob('staged-schema-*')))}"
        staged.mkdir()
        schema = self.schema_of("task-schema.json")
        schema.update(changes)
        (staged / "task-schema.json").write_text(json.dumps(schema))
        (staged / "milestone-schema.json").write_text(
            (CORE.SCHEMA / "milestone-schema.json").read_text()
        )
        return staged

    def support_errors_for(self, **changes: Any) -> list[str]:
        """Return what validate() says about one modified published schema."""
        self.make_task("AR-0001")
        with patch.object(CORE, "SCHEMA", self.staged_schema(**changes)):
            return cast(list[str], CORE.validate())

    def test_a_boolean_subschema_under_items_is_refused_rather_than_ignored(self) -> None:
        """``"items": false`` refuses every array element and this reader enforces neither form.

        Draft 2020-12 allows a boolean wherever a subschema may appear. Left unrefused it is
        the original divergence restored by a one-word edit: a real validator rejects every
        task that declares a dependency while the transaction commits them all.
        """
        properties = self.schema_of("task-schema.json")["properties"]
        properties["depends_on"]["items"] = False
        errors = self.support_errors_for(properties=properties)
        self.assertIn("task-schema.json: depends_on[] states unsupported subschema false", errors)

    def test_a_boolean_property_subschema_is_refused_rather_than_raising(self) -> None:
        """The same construct one level up used to raise TypeError out of every transaction."""
        properties = self.schema_of("task-schema.json")["properties"]
        properties["owner"] = False
        errors = self.support_errors_for(properties=properties)
        self.assertIn("task-schema.json: owner states unsupported subschema false", errors)

    def test_a_bound_whose_shape_the_checker_cannot_apply_is_refused(self) -> None:
        """A supported keyword is only enforced when its bound is the shape the checker reads.

        Each of these is silently unenforced or raises inside ``SCHEMA_CONSTRAINTS``, which is
        the same fault class as an unimplemented keyword and is refused the same way.
        """
        cases: tuple[tuple[str, object, str], ...] = (
            ("maxLength", "300", '"300"'),
            ("maxLength", True, "true"),
            ("minLength", 1.5, "1.5"),
            ("minimum", "3", '"3"'),
            ("pattern", "[", '"["'),
            ("pattern", 5, "5"),
            ("uniqueItems", "yes", '"yes"'),
            ("enum", {"a": 1}, '{"a": 1}'),
        )
        for keyword, bound, rendered in cases:
            with self.subTest(keyword=keyword, bound=bound):
                properties = self.schema_of("task-schema.json")["properties"]
                properties["title"] = {"type": "string", keyword: bound}
                errors = self.support_errors_for(properties=properties)
                self.assertIn(
                    f"task-schema.json: title states unsupported {keyword} bound {rendered}",
                    errors,
                )

    def test_a_keyword_container_of_the_wrong_shape_is_refused_before_it_is_walked(self) -> None:
        """``properties``, ``required`` and ``allOf`` are walked, so their shape is checked.

        The enforcement pass walks them too. Refusing here and then enforcing anyway would
        raise from inside the transaction, so ``schema_gate_errors`` enforces only a schema
        this net accepted.
        """
        cases: tuple[tuple[str, object, str], ...] = (
            ("properties", False, "task-schema.json: states unsupported properties false"),
            ("required", "id", 'task-schema.json: states unsupported required "id"'),
            ("required", [1], "task-schema.json: states unsupported required [1]"),
            ("allOf", {}, "task-schema.json: states unsupported allOf {}"),
            ("allOf", [True], "task-schema.json: states unsupported allOf branch true"),
            (
                "allOf",
                [{"if": {"properties": {}}, "then": False}],
                "task-schema.json then: states unsupported subschema false",
            ),
        )
        for keyword, value, expected in cases:
            with self.subTest(keyword=keyword, value=value):
                self.assertIn(expected, self.support_errors_for(**{keyword: value}))

    def test_an_unsupported_schema_stops_the_tree_without_being_enforced(self) -> None:
        """doctor must refuse, and must do it by reporting rather than by raising."""
        self.make_task("AR-0001")
        with (
            patch.object(CORE, "SCHEMA", self.staged_schema(properties=False)),
            patch("builtins.print"),
        ):
            self.assertEqual(1, CORE.cmd_doctor(live=False))

    def test_every_supported_keyword_states_the_shape_of_its_bound(self) -> None:
        """The net stays total as the reader grows: a new constraint needs a shape rule.

        ``type`` and ``format`` are checked by name against the values this reader implements,
        and ``const`` accepts any JSON value by definition. Every other supported keyword
        reads its bound, so every other supported keyword must declare what it can read.
        """
        self.assertEqual(
            {"const", "format", "type"},
            set(CORE.SCHEMA_CONSTRAINTS) - set(CORE.SCHEMA_BOUND_SHAPES),
        )
        self.assertEqual(set(), set(CORE.SCHEMA_BOUND_SHAPES) - set(CORE.SCHEMA_CONSTRAINTS))

    def test_the_milestone_label_at_its_minimum_length_is_accepted_by_both_gates(self) -> None:
        """The same inclusive-boundary pin the task fields get, through validate() itself."""
        validator = Draft202012Validator(
            self.schema_of("milestone-schema.json"), format_checker=FormatChecker()
        )
        for label in ("L", "x" * 80):
            with self.subTest(label=label):
                meta = self.plant_milestone({"label": label})
                self.assertEqual([], list(validator.iter_errors(meta)), "fixture is not valid")
                complaints = [e for e in CORE.validate() if "Length" in e]
                self.assertEqual([], complaints)

    def test_the_generic_layer_reports_a_nested_array_item_by_its_index(self) -> None:
        self.plant_task({"depends_on": ["AR-0002", "nope"]})
        self.assertIn(
            'AR-0001-test.md: depends_on[1] violates pattern "^AR-[0-9]{4}$"',
            CORE.validate(),
        )


class RealRepositoryTest(unittest.TestCase):
    """The state-branch guard and product heads, against a real repository: reading real Git
    state is their whole job, so a mocked ``run`` would test only the mock."""

    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.repo = Path(self.temp.name) / "product"
        self.saved_root = CORE.ROOT
        git = ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid"]
        CORE.run(["git", "init", "-q", "-b", "main", str(self.repo)])
        (self.repo / "coordination").mkdir()
        (self.repo / "coordination" / "README.md").write_text("fixture\n")
        CORE.run([*git, "-C", str(self.repo), "add", "."])
        CORE.run([*git, "-C", str(self.repo), "commit", "-q", "--no-gpg-sign", "-m", "init"])
        CORE.ROOT = self.repo / "coordination"

    def tearDown(self) -> None:
        CORE.ROOT = self.saved_root
        self.temp.cleanup()

    def commit(self, relative: str, text: str) -> str:
        """Write one file, commit it unsigned, and return the new head."""
        target = self.repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        git = ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid"]
        CORE.run([*git, "-C", str(self.repo), "add", "--", relative])
        CORE.run([*git, "-C", str(self.repo), "commit", "-q", "--no-gpg-sign", "-m", relative])
        return str(CORE.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"]).stdout.strip())

    def test_product_head_skips_commits_that_touch_only_coordination(self) -> None:
        product = self.commit("src/lib.rs", "fn main() {}\n")
        state = self.commit("coordination/STATUS.md", "generated\n")
        self.assertNotEqual(product, state)
        self.assertEqual(product, CORE.product_head(self.repo, "HEAD"))
        # The view it feeds is therefore unchanged by the commit that records it.
        self.commit("coordination/CURRENT.md", "generated\n")
        self.assertEqual(product, CORE.product_head(self.repo, "HEAD"))
        self.assertEqual(product, CORE.product_head(self.repo / "coordination", "HEAD"))

    def test_product_head_keeps_a_merge_on_the_first_parent_line(self) -> None:
        git = ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid"]
        CORE.run(["git", "-C", str(self.repo), "switch", "-q", "-c", "feature/x"])
        self.commit("src/lib.rs", "fn main() {}\n")
        CORE.run(["git", "-C", str(self.repo), "switch", "-q", "main"])
        CORE.run(
            [
                *git,
                "-C",
                str(self.repo),
                "merge",
                "-q",
                "--no-ff",
                "--no-gpg-sign",
                "-m",
                "merge",
                "feature/x",
            ]
        )
        merge = CORE.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"]).stdout.strip()
        self.commit("coordination/STATUS.md", "generated\n")
        self.assertEqual(merge, CORE.product_head(self.repo, "HEAD"))

    def test_product_head_reports_an_unknown_ref_as_given(self) -> None:
        unknown = "f" * 40
        self.assertEqual(unknown, CORE.product_head(self.repo, unknown))

    def test_canonical_checkout_on_main_passes(self) -> None:
        CORE.require_state_branch()

    def test_another_branch_is_refused(self) -> None:
        CORE.run(["git", "-C", str(self.repo), "switch", "-q", "-c", "feature/x"])
        with self.assertRaisesRegex(RuntimeError, "commit only to main.*feature/x"):
            CORE.require_state_branch()

    def test_detached_head_is_refused(self) -> None:
        CORE.run(["git", "-C", str(self.repo), "switch", "-q", "--detach"])
        with self.assertRaisesRegex(RuntimeError, "detached HEAD"):
            CORE.require_state_branch()

    def test_linked_worktree_is_refused_even_on_main(self) -> None:
        linked = Path(self.temp.name) / "AR-0101"
        CORE.run(["git", "-C", str(self.repo), "switch", "-q", "-c", "parking"])
        CORE.run(["git", "-C", str(self.repo), "worktree", "add", "-q", str(linked), "main"])
        CORE.ROOT = linked / "coordination"
        with self.assertRaisesRegex(RuntimeError, "not in a linked worktree"):
            CORE.require_state_branch()


if __name__ == "__main__":
    unittest.main()
