#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 12: commit signature, sign-off and message privacy, in this repository.

Over the introduced range `base..head` of the **product** repository, for every
commit:

1. **Non-empty range.** An empty range fails. A gate that is a no-op on zero
   commits is the failure this check was written to avoid, so the emptiness is
   itself a finding rather than a silent pass.
2. **Sign-off matches the author exactly.** The author is `%an <%ae>`; the
   message's trailer block is parsed with `git interpret-trailers --parse`; a
   literal `Signed-off-by: <that exact string>` must be among the trailers.
   Not case-insensitively, not as a substring, not the address alone.
3. **Signature verified against an explicit allowed-key set.** This project
   signs with GPG. The allowed *public* keys are `config/allowed-keys.asc`;
   they are imported into a throwaway GNUPGHOME, so neither the ambient user
   keyring nor a hosting provider's branch rule can make a commit pass. Both a
   zero exit from `git verify-commit` and an allowed fingerprint are required.
   The fingerprint compared is `%GP`, the **primary** key's, not `%GF`, the
   signing key's: they are the same today because the project signs with a
   primary key that has no subkeys, and matching on the primary is what lets a
   subkey rotation happen without editing the allowed set.
4. **Message privacy.** The rules are in `privacy.py`, a superset of the
   coordinator's partial list.

This is not the coordinator's `handoffctl check-commits`. That command performs
no signature and no sign-off checking, and hardcodes the coordinator repository
as its git root, so from a product worktree it examines the wrong history.

Nothing hooks this into `git push`. It is a gate the runner invokes and a
reviewer reads.

Usage::

    check_commits.py [--base REF] [--head REF] [--allowed-keys PATH] [--repo PATH]

`--base` defaults to `origin/main` and `--head` to `HEAD`. On a post-merge run,
where `origin/main..HEAD` is empty by construction, name the range that was
introduced -- for a merge commit `M`, `--base M^1 --head M`.

Exit status is 0 when every commit passes and 1 when any fails. 2 is a usage or
environment error: an unknown revision, a missing key file, an unusable GPG.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from privacy import findings  # noqa: E402


class Environment(Exception):
    """A problem with the environment, not a finding about a commit."""


def git(root: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full = dict(os.environ)
    if env:
        full.update(env)
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
        env=full,
    )


def resolve(root: Path, ref: str) -> str:
    out = git(root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    if out.returncode != 0 or not out.stdout.strip():
        raise Environment(f"unknown revision: {ref}")
    return out.stdout.strip()


def import_allowed_keys(home: Path, keyfile: Path) -> tuple[set[str], list[str]]:
    """Import the allowed public keys into a throwaway GNUPGHOME.

    Returns the set of primary fingerprints and any notes worth reporting.

    `gpg --import` returns 2 in a sandbox where `gpg-agent` cannot start, even
    when every key imports correctly, so the exit code alone cannot decide this.
    It is not ignored either: the fingerprints the file *offers* are compared
    against the fingerprints actually in the throwaway keyring, and a
    difference fails. A genuinely failed import cannot become a silent pass.
    """
    notes: list[str] = []
    env = {"GNUPGHOME": str(home)}

    offered = subprocess.run(
        ["gpg", "--batch", "--no-autostart", "--with-colons", "--show-keys", str(keyfile)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    )
    if offered.returncode != 0:
        raise Environment(f"cannot read the allowed-key file: {keyfile.name}")
    expected = _primary_fingerprints(offered.stdout)
    if not expected:
        raise Environment(f"the allowed-key file contains no public key: {keyfile.name}")

    imported = subprocess.run(
        ["gpg", "--batch", "--no-autostart", "--import", str(keyfile)],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    )

    listed = subprocess.run(
        ["gpg", "--batch", "--no-autostart", "--with-colons", "--list-keys"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, **env},
    )
    present = _primary_fingerprints(listed.stdout)

    if imported.returncode != 0:
        if present != expected:
            raise Environment(
                f"gpg --import exited {imported.returncode} and the keyring does not hold "
                f"every offered key ({len(present)} of {len(expected)})"
            )
        notes.append(
            f"gpg --import exited {imported.returncode}; every offered key is nonetheless "
            "present in the throwaway keyring (the sandbox gpg-agent case)"
        )
    elif present != expected:
        raise Environment(
            f"gpg --import exited 0 but the keyring holds {len(present)} of "
            f"{len(expected)} offered keys"
        )
    return present, notes


def _primary_fingerprints(colons: str) -> set[str]:
    """Primary-key fingerprints from `--with-colons` output.

    A `fpr` record belongs to the record type that preceded it, so only the one
    directly after a `pub` line is a primary fingerprint; the ones after `sub`
    lines are subkeys and are deliberately not collected.
    """
    fingerprints: set[str] = set()
    previous = ""
    for line in colons.splitlines():
        fields = line.split(":")
        kind = fields[0]
        if kind == "fpr" and previous == "pub" and len(fields) > 9:
            fingerprints.add(fields[9].upper())
        if kind in ("pub", "sub", "uid", "sec", "ssb"):
            previous = kind
    return fingerprints


def signoff_trailers(root: Path, rev: str) -> list[str]:
    message = git(root, "show", "-s", "--format=%B", rev)
    if message.returncode != 0:
        raise Environment(f"cannot read the message of {rev[:12]}")
    parsed = subprocess.run(
        ["git", "interpret-trailers", "--parse"],
        input=message.stdout,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(root),
    )
    if parsed.returncode != 0:
        raise Environment(f"git interpret-trailers failed on {rev[:12]}")
    return [line for line in parsed.stdout.splitlines() if line.strip()]


def check_commit(root: Path, rev: str, home: Path, allowed: set[str]) -> list[str]:
    failures: list[str] = []
    short = rev[:12]

    author = git(root, "show", "-s", "--format=%an <%ae>", rev).stdout.strip()
    trailers = signoff_trailers(root, rev)
    wanted = f"Signed-off-by: {author}"
    if wanted not in trailers:
        present = [t for t in trailers if t.lower().startswith("signed-off-by:")]
        if not present:
            failures.append(f"{short}: no Signed-off-by trailer for author {author}")
        else:
            failures.append(
                f"{short}: Signed-off-by does not match the author exactly "
                f"(author {author}; trailers {'; '.join(present)})"
            )

    env = {"GNUPGHOME": str(home)}
    verified = git(root, "verify-commit", rev, env=env)
    fingerprint = git(root, "show", "-s", "--format=%GP", rev, env=env).stdout.strip().upper()
    status = git(root, "show", "-s", "--format=%G?", rev, env=env).stdout.strip()
    if verified.returncode != 0:
        failures.append(
            f"{short}: signature does not verify against the allowed keys "
            f"(git verify-commit exit {verified.returncode}, status {status or 'none'})"
        )
    elif not fingerprint:
        failures.append(f"{short}: verified signature reports no primary fingerprint")
    elif fingerprint not in allowed:
        # Defence in depth. The throwaway keyring holds exactly the allowed
        # keys, so a signature by an unlisted key normally fails one step
        # earlier, at `git verify-commit`, for want of a public key. This
        # branch is what still rejects it if that ever stops being true.
        failures.append(
            f"{short}: good signature by a key that is not in the allowed set "
            f"(primary fingerprint {fingerprint})"
        )

    message = git(root, "show", "-s", "--format=%B", rev).stdout
    for description, fragment in findings(message):
        failures.append(f"{short}: message privacy: {description}: {fragment}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="range start (default: origin/main)")
    parser.add_argument("--head", default="HEAD", help="range end (default: HEAD)")
    parser.add_argument("--allowed-keys", default=None, help="allowed public keys (armoured)")
    parser.add_argument("--repo", default=None, help="repository to examine (default: this one)")
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

        keyfile = Path(args.allowed_keys) if args.allowed_keys else root / "config" / "allowed-keys.asc"
        if not keyfile.is_file():
            raise Environment(f"no allowed-key file at {keyfile.relative_to(root) if keyfile.is_relative_to(root) else keyfile.name}")
        if not shutil.which("gpg"):
            raise Environment("gpg is not installed")

        base = resolve(root, args.base)
        head = resolve(root, args.head)

        listing = git(root, "rev-list", "--reverse", f"{base}..{head}")
        if listing.returncode != 0:
            raise Environment(f"cannot list {args.base}..{args.head}")
        revisions = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
    except Environment as error:
        print(f"commits: {error}", file=sys.stderr)
        return 2

    if not revisions:
        print(f"commits: 0 commits in {args.base}..{args.head}", file=sys.stderr)
        print(
            "FAIL empty range: this gate must examine at least one commit; "
            "name the introduced range explicitly on a post-merge run",
            file=sys.stderr,
        )
        return 1

    home = Path(tempfile.mkdtemp(prefix="gate-gnupg-"))
    home.chmod(0o700)
    try:
        try:
            allowed, notes = import_allowed_keys(home, keyfile)
        except Environment as error:
            print(f"commits: {error}", file=sys.stderr)
            return 2
        for note in notes:
            print(f"commits: note: {note}")

        failures: list[str] = []
        for rev in revisions:
            try:
                failures += check_commit(root, rev, home, allowed)
            except Environment as error:
                print(f"commits: {error}", file=sys.stderr)
                return 2
    finally:
        shutil.rmtree(home, ignore_errors=True)

    print(
        f"commits: {len(revisions)} commit(s) in {args.base}..{args.head}, "
        f"{len(allowed)} allowed key(s), {len(failures)} failures"
    )
    for failure in failures:
        print(f"FAIL {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
