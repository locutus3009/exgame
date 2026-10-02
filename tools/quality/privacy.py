#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Privacy patterns for text that leaves this machine.

Commit messages, task notes and the gate report are all public artefacts. This
module holds the one list of things that must never appear in them, so that
gate 12 (`check_commits.py`) and the gate runner's self-check of its own report
apply the same rules.

The list is deliberately a *superset* of the coordinator's `handoffctl
check-commits` list, which is documented in `docs/DEVELOPMENT.md` as partial:

* the coordinator covers only the ``10`` and ``127`` blocks; the rules here
  cover all three RFC 1918 blocks -- the ``10/8``, ``172.16/12`` and
  ``192.168/16`` ranges -- plus loopback, as gate 12 of
  `docs/QUALITY_GATES.md` requires;
* the coordinator has no hostname pattern; this one rejects the usual private
  suffixes;
* the private-key pattern here also catches armoured PGP private key blocks.

Some literals are written as concatenations. That is not obfuscation: a scanner
of this kind flags its own source otherwise, and a file that cannot pass the
check it implements is a file someone will eventually weaken.

Run directly to scan files::

    python3 privacy.py FILE [FILE ...]

Exit status is 0 when nothing is found and 1 when anything is, so it can be
used as a check. Findings are printed with the offending fragment redacted --
printing it in full would move the private text into the log that is quoting
it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PRIVATE: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile("/" + "home/"), "absolute Linux home path"),
    (re.compile(r"[A-Za-z]:\\Users\\", re.I), "absolute Windows user path"),
    # RFC 1918 plus loopback. The 172.16/12 range stops at the 172.31 prefix,
    # which is why its second octet is enumerated rather than left as \d+.
    (
        re.compile(
            r"\b(?:10\.(?:\d{1,3}\.){2}\d{1,3}"
            r"|127\.(?:\d{1,3}\.){2}\d{1,3}"
            r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
            r"|192\.168\.\d{1,3}\.\d{1,3})\b"
        ),
        "private or loopback IP address",
    ),
    (
        re.compile(r"\b[a-z0-9][a-z0-9-]*\.(?:local|internal|lan|intranet|corp)\b", re.I),
        "private hostname",
    ),
    (
        re.compile(r"\b(?:password|passwd|token|secret|api[_-]?key)\s*[:=]\s*[^\s<]+", re.I),
        "possible credential",
    ),
    (
        re.compile(r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY(?: BLOCK)?-----"),
        "private key block",
    ),
    (re.compile(r"claude\.ai/" + r"code/session", re.I), "private agent session reference"),
    (re.compile(r"\bses" + r"sion_[A-Za-z0-9]{16,}\b", re.I), "private agent session identifier"),
    (
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
        "session-like UUID",
    ),
)


def redact(fragment: str) -> str:
    """A finding names its kind and its shape, never its content."""
    return f"<{len(fragment)} characters redacted>"


def findings(text: str) -> list[tuple[str, str]]:
    """Every private thing in `text`, as (description, redacted fragment)."""
    found: list[tuple[str, str]] = []
    for pattern, description in PRIVATE:
        for match in pattern.finditer(text):
            found.append((description, redact(match.group(0))))
    return found


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: privacy.py FILE [FILE ...]", file=sys.stderr)
        return 2
    failures = 0
    for name in argv:
        path = Path(name)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            print(f"FAIL {name}: cannot read: {error.strerror}", file=sys.stderr)
            failures += 1
            continue
        for description, fragment in findings(text):
            print(f"FAIL {name}: {description}: {fragment}", file=sys.stderr)
            failures += 1
    print(f"privacy: {len(argv)} file(s) scanned, {failures} findings")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
