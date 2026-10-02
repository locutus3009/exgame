#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 11: one offline, network-free check of three repository properties.

**Licence headers.** Every tracked first-party source must carry
`SPDX-License-Identifier: MIT` near its top. AR-0005 added them; this check is
what makes them stay.

**Local Markdown links.** Every relative link in a tracked Markdown file must
resolve to a file that exists, and every `#anchor` must resolve to a heading in
the target. This is the mechanism against the failure that produced
`docs/superpowers/`: a document nobody is required to update rots silently, and
a dangling link is the earliest visible symptom.

The link property is **not reimplemented here**. `check_docs.py` (gate 15,
AR-0010) already implements exactly this rule, over the same tracked set, and
`docs/QUALITY_GATES.md` asks for one implementation rather than two, so this
module imports `check_docs.check_links` and reports its findings. Gate 15 keeps
the reachability half; the link half is invoked from here, once per run.

**Workflow immutability.** Every GitHub Action reference must be a full 40-hex
commit id, never a tag or a branch, because a tag can be moved under you. And
while AR-0008's workflows are dormant, none may trigger on push, pull request
or schedule: a dormant workflow that quietly starts running is not dormant.

Usage::

    repository_policy.py [--base REF] [--head REF] [--repo PATH]

`--base` and `--head` are accepted because `docs/QUALITY_GATES.md` names them
in the gate's command line. They are reported, not used to narrow the check:
every property here is enforced over the whole tracked tree, which is strictly
stronger than enforcing it over one range, and is what makes "AR-0005 added the
headers; this check is what makes them stay" true.

Exit status is 0 when every property holds and 1 when any fails. 2 is an
environment error.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_docs import check_links, tracked_markdown  # noqa: E402

# First-party source extensions that must carry the SPDX header. Data,
# configuration and Markdown are excluded: an SPDX line in a `.json` file is
# not valid JSON, and a licence header on every Markdown file would be noise.
SOURCE_SUFFIXES = (".rs", ".py", ".sh", ".glsl")

# How far into a file the header may be. A shebang, a `#!`-line and an encoding
# comment can precede it; a licence header further down than this is one nobody
# reads.
HEADER_LINES = 5

SPDX = "SPDX-License-Identifier: MIT"

# A dormant workflow may be started by hand and by nothing else. AR-0008 owns
# the workflows; this list is the definition of "dormant" that gate 11
# enforces, and widening it is a deliberate change to `docs/QUALITY_GATES.md`,
# not an implementation detail.
FORBIDDEN_TRIGGERS = ("push", "pull_request", "pull_request_target", "schedule")

FULL_COMMIT_ID = re.compile(r"^[0-9a-f]{40}$")


class Environment(Exception):
    """A problem with the environment, not a finding about the repository."""


def tracked(root: Path, *patterns: str) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", *patterns],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise Environment("git ls-files failed")
    return sorted(name for name in out.stdout.split("\0") if name)


def check_licence_headers(root: Path) -> list[str]:
    failures: list[str] = []
    for name in tracked(root):
        path = root / name
        if not name.endswith(SOURCE_SUFFIXES) or path.is_symlink():
            continue
        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                head = [next(handle, "") for _ in range(HEADER_LINES)]
        except OSError as error:
            failures.append(f"{name}: cannot read: {error.strerror}")
            continue
        if not any(SPDX in line for line in head):
            failures.append(f"{name}: no {SPDX} in the first {HEADER_LINES} lines")
    return failures


def workflow_files(root: Path) -> list[str]:
    return [
        name
        for name in tracked(root, ".github/workflows/*.yml", ".github/workflows/*.yaml")
        if name.startswith(".github/workflows/")
    ]


def _uses_references(node: object) -> list[str]:
    """Every `uses:` value anywhere in a parsed workflow."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "uses" and isinstance(value, str):
                found.append(value)
            else:
                found += _uses_references(value)
    elif isinstance(node, list):
        for item in node:
            found += _uses_references(item)
    return found


def _triggers(document: dict[str, object]) -> list[str]:
    # YAML 1.1 reads a bare `on` as the boolean True, which is why the key is
    # looked up both ways. A workflow written as `"on":` parses as the string.
    raw = document.get("on", document.get(True))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if isinstance(raw, dict):
        return [str(key) for key in raw]
    return []


def check_workflows(root: Path) -> list[str]:
    names = workflow_files(root)
    if not names:
        return []
    try:
        import yaml  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment dependent
        raise Environment(
            f"{len(names)} workflow file(s) to check and PyYAML is not installed; "
            "this gate fails closed rather than skipping them"
        ) from error

    failures: list[str] = []
    for name in names:
        try:
            document = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            failures.append(f"{name}: is not parseable YAML: {type(error).__name__}")
            continue
        if not isinstance(document, dict):
            failures.append(f"{name}: is not a workflow mapping")
            continue

        for reference in _uses_references(document):
            if reference.startswith("./"):
                continue  # a local action in this repository, pinned by the commit itself
            if reference.startswith("docker://"):
                if "@sha256:" not in reference:
                    failures.append(f"{name}: container action not pinned by digest: {reference}")
                continue
            _, separator, version = reference.partition("@")
            if not separator or not FULL_COMMIT_ID.match(version):
                failures.append(
                    f"{name}: action reference is mutable, a full 40-character commit id "
                    f"is required: {reference}"
                )

        for trigger in _triggers(document):
            if trigger in FORBIDDEN_TRIGGERS:
                failures.append(
                    f"{name}: dormant workflows must not trigger on {trigger}"
                )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="reported, not used to narrow")
    parser.add_argument("--head", default="HEAD", help="reported, not used to narrow")
    parser.add_argument("--repo", default=None, help="repository root (default: this one)")
    args = parser.parse_args()

    try:
        start = Path(args.repo) if args.repo else Path(__file__).resolve().parent
        top = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
        if top.returncode != 0:
            raise Environment("not inside a git repository")
        root = Path(top.stdout.strip())

        licence = check_licence_headers(root)
        markdown = tracked_markdown(root)
        links = check_links(root, markdown)
        workflows = check_workflows(root)
    except Environment as error:
        print(f"repository policy: {error}", file=sys.stderr)
        return 2

    failures = (
        [f"licence header: {item}" for item in licence]
        + [f"markdown link: {item}" for item in links]
        + [f"workflow: {item}" for item in workflows]
    )
    print(
        f"repository policy: {len(markdown)} tracked Markdown file(s), "
        f"{len(workflow_files(root))} workflow(s), whole tree "
        f"(range {args.base}..{args.head} reported only), {len(failures)} failures"
    )
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
