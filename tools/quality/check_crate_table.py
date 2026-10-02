#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 14: the entry point's crate table must name exactly the workspace crates.

`CLAUDE.md` is the entry point an agent or a new reader starts from. Its crate
table is the mechanism that stops it drifting: adding a crate without
documenting it, or removing one without undocumenting it, fails here.

The check is about *existence*, not prose. It cannot detect that a description
became wrong, only that a name went missing, and the correctness of each
crate's stated role stays a review obligation.

Specification (`docs/QUALITY_GATES.md` gate 14), implemented literally:

1. `cargo metadata --no-deps --format-version 1` at the repository root; take
   `packages[].name`. Nothing is excluded -- `experiments/*` members are
   workspace members and belong in the table.
2. The first Markdown table in `CLAUDE.md`: the first maximal run of
   consecutive lines beginning with `|`. **Discard its first two lines** -- the
   header row and the `| --- | --- |` delimiter -- by position, not by
   pattern-matching the delimiter. Keeping them would yield `Crate` and `---`
   as crate names and fail on a correct tree.
3. From each remaining row take the first cell, strip whitespace and backticks,
   split on commas, and take each part's final `/`-separated segment, so that
   both `newton` and `crates/newton` are accepted.
4. Compare the two sets, report the two differences separately, and exit
   non-zero if either is non-empty.

`README.md` also carries a crate table. It is a summary and is not
authoritative; only `CLAUDE.md` is checked.

Usage::

    check_crate_table.py [--repo PATH] [--entry-point CLAUDE.md]

Exit status is 0 when the two sets are equal and 1 when they differ. 2 is an
environment error: no repository, no entry point, no table, or a `cargo
metadata` that will not run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def workspace_crates(root: Path) -> set[str]:
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise SystemExit2(f"cargo metadata exited {out.returncode}")
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit2(f"cargo metadata did not produce JSON: {error.msg}") from error
    return {package["name"] for package in data["packages"]}


class SystemExit2(Exception):
    """An environment error, distinct from a finding about the table."""


def first_table(text: str) -> list[str]:
    """The first maximal run of consecutive lines beginning with `|`."""
    run: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            run.append(line)
        elif run:
            break
    return run


def table_crates(entry: Path) -> set[str]:
    run = first_table(entry.read_text(encoding="utf-8"))
    if len(run) < 3:
        raise SystemExit2(
            f"{entry.name} has no crate table: the first Markdown table has "
            f"{len(run)} line(s), and a table with rows needs at least three"
        )
    names: set[str] = set()
    for row in run[2:]:  # by position: line 1 is the header, line 2 the delimiter
        cells = row.split("|")
        first = cells[1] if len(cells) > 1 else ""
        for part in first.strip().strip("`").split(","):
            name = part.strip().strip("`").strip()
            if name:
                names.add(name.rsplit("/", 1)[-1])
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=None, help="repository root (default: this one)")
    parser.add_argument("--entry-point", default="CLAUDE.md", help="the file holding the table")
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
            raise SystemExit2("not inside a git repository")
        root = Path(top.stdout.strip())
        entry = root / args.entry_point
        if not entry.is_file():
            raise SystemExit2(f"no entry point at {args.entry_point}")
        metadata = workspace_crates(root)
        documented = table_crates(entry)
    except SystemExit2 as error:
        print(f"crate table: {error}", file=sys.stderr)
        return 2

    undocumented = sorted(metadata - documented)
    unknown = sorted(documented - metadata)

    print(
        f"crate table: {len(metadata)} workspace crate(s), "
        f"{len(documented)} named in {args.entry_point}, "
        f"{len(undocumented) + len(unknown)} failures"
    )
    for name in undocumented:
        print(
            f"FAIL {name}: a workspace crate that {args.entry_point} does not name",
            file=sys.stderr,
        )
    for name in unknown:
        print(
            f"FAIL {name}: named in {args.entry_point} but not a workspace crate",
            file=sys.stderr,
        )
    return 1 if (undocumented or unknown) else 0


if __name__ == "__main__":
    sys.exit(main())
