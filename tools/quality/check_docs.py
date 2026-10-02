#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Documentation integrity checks: local Markdown links, and index completeness.

Two independent checks over the tracked Markdown of this repository.

`links`
    Every relative Markdown link in a tracked `.md` file must resolve to a file
    that exists, and every `#anchor` must resolve to a heading in the target
    file. This is the property gate 11 specifies
    (docs/QUALITY_GATES.md#11-repository-policy) and is implemented here so that
    it runs before `tools/quality/` exists.

`index`
    Every governed document must be reachable from the architecture index by
    following links, and every link leaving a governed document must resolve.
    This is gate 15 (docs/QUALITY_GATES.md#15-architecture-index-completeness).

Exit status is 0 when every selected check passes and 1 when any fails. Any
other status is a defect in this script, not a finding.

Offline, no network, no dependencies outside the standard library.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# The architecture index: the single root from which every governed document
# must be reachable.
INDEX = "docs/architecture/overview.md"

# Directories whose every `.md` file is a governed document, plus individual
# files that are governed. A governed document carries an obligation: it is
# indexed, and the index entry is checked.
GOVERNED_DIRS = ("docs/architecture/", "docs/process/")
GOVERNED_FILES: tuple[str, ...] = ()

# Fenced code blocks are not prose; links inside them are examples.
FENCE = re.compile(r"^\s*(```|~~~)")

# Inline links `[text](target)` and reference definitions `[id]: target`.
# The target stops at whitespace so that `[t](path "title")` yields `path`.
INLINE_LINK = re.compile(r"\[[^\]]*\]\(\s*<?([^)>\s]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REF_DEF = re.compile(r"^\s{0,3}\[([^\]^]+)\]:\s*<?([^>\s]+)>?")

HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
EXPLICIT_ANCHOR = re.compile(r"<a\s+[^>]*(?:name|id)\s*=\s*[\"']([^\"']+)[\"']", re.I)

SKIP_SCHEMES = ("http://", "https://", "mailto:", "ftp://", "#!")


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(out.stdout.strip())


def tracked_markdown(root: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z", "*.md"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    names = [n for n in out.stdout.split("\0") if n]
    # A symlink (AGENTS.md -> CLAUDE.md) is the same content under a second
    # name; checking it twice would double-report every finding.
    return sorted(n for n in names if not (root / n).is_symlink())


def strip_fences(text: str) -> list[tuple[int, str]]:
    """Return (line number, line) for every line outside a fenced code block."""
    lines: list[tuple[int, str]] = []
    fence: str | None = None
    for number, line in enumerate(text.splitlines(), start=1):
        match = FENCE.match(line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
            continue
        if fence is None:
            lines.append((number, line))
    return lines


def links_in(root: Path, name: str) -> list[tuple[int, str]]:
    """Every link target in `name`, as (line number, raw target)."""
    text = (root / name).read_text(encoding="utf-8")
    found: list[tuple[int, str]] = []
    for number, line in strip_fences(text):
        # Inline code spans hold example paths, not links; drop them first.
        prose = re.sub(r"`[^`]*`", "", line)
        for target in INLINE_LINK.findall(prose):
            found.append((number, target))
        ref = REF_DEF.match(prose)
        if ref:
            found.append((number, ref.group(2)))
    return found


def slug(heading: str) -> str:
    """GitHub's heading slug: lowercase, punctuation dropped, spaces to dashes."""
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_~]", "", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def anchors_of(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for _, line in strip_fences(text):
        match = HEADING.match(line)
        if match:
            base = slug(match.group(2))
            if not base:
                continue
            count = seen.get(base, 0)
            anchors.add(base if count == 0 else f"{base}-{count}")
            seen[base] = count + 1
        for explicit in EXPLICIT_ANCHOR.findall(line):
            anchors.add(explicit)
    return anchors


def resolve(root: Path, name: str, target: str) -> tuple[Path | None, str | None, str | None]:
    """Resolve a link target.

    Returns (path, anchor, error). `path` is None when the link is external or
    is an anchor into the same file, in which case the caller uses `name`.
    """
    if target.startswith(SKIP_SCHEMES) or "://" in target.split("#", 1)[0]:
        return None, None, None
    path_part, _, anchor = target.partition("#")
    anchor = anchor or None
    if not path_part:
        return root / name, anchor, None
    if path_part.startswith("/"):
        candidate = root / path_part.lstrip("/")
    else:
        candidate = (root / name).parent / path_part
    try:
        candidate = candidate.resolve()
        candidate.relative_to(root.resolve())
    except (OSError, ValueError):
        return None, None, f"escapes the repository: {target}"
    if not candidate.exists():
        return None, None, f"no such file: {target}"
    return candidate, anchor, None


def check_one(root: Path, name: str, target: str) -> tuple[Path | None, str | None]:
    """Resolve one link fully, anchor included. Returns (path, error)."""
    path, anchor, error = resolve(root, name, target)
    if error:
        return None, error
    if path is None:
        return None, None
    if anchor is not None and path.suffix.lower() == ".md":
        if anchor not in anchors_of(path):
            return path, f"no such heading: {target}"
    return path, None


def check_links(root: Path, names: list[str]) -> list[str]:
    failures: list[str] = []
    for name in names:
        for number, target in links_in(root, name):
            _, error = check_one(root, name, target)
            if error:
                failures.append(f"{name}:{number}: {error}")
    return failures


def is_governed(name: str) -> bool:
    return name.startswith(GOVERNED_DIRS) or name in GOVERNED_FILES


def check_index(root: Path, names: list[str]) -> list[str]:
    failures: list[str] = []
    governed = {n for n in names if is_governed(n)}
    if INDEX not in governed:
        return [f"the index {INDEX} does not exist"]

    reached = {INDEX}
    queue = [INDEX]
    while queue:
        name = queue.pop()
        for number, target in links_in(root, name):
            path, error = check_one(root, name, target)
            if error:
                failures.append(f"{name}:{number}: index entry does not resolve: {error}")
            if path is None:
                continue
            relative = str(path.resolve().relative_to(root.resolve()))
            if relative in governed and relative not in reached:
                reached.add(relative)
                queue.append(relative)

    for name in sorted(governed - reached):
        failures.append(f"{name}: governed document is not reachable from {INDEX}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "check",
        nargs="?",
        default="all",
        choices=("all", "links", "index"),
        help="which check to run (default: all)",
    )
    args = parser.parse_args()

    root = repo_root()
    names = tracked_markdown(root)

    failures: list[str] = []
    if args.check in ("all", "links"):
        found = check_links(root, names)
        print(f"links: {len(names)} tracked Markdown files, {len(found)} failures")
        failures += found
    if args.check in ("all", "index"):
        found = check_index(root, names)
        governed = sum(1 for n in names if is_governed(n))
        print(f"index: {governed} governed documents, {len(found)} failures")
        failures += found

    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
