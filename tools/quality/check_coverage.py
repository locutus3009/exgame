#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate 7: measured coverage against the floors in the tool manifest.

Reads the JSON that `cargo llvm-cov --summary-only --json` produced, reads the
floors from `config/quality-tools.json`, and compares them.

**Line and region coverage only.** On the stable toolchain `cargo-llvm-cov`
reports line and region coverage and **not** branch coverage, so no branch
number is printed here whatever the manifest says. A percentage from this gate
is not comparable to a branch-aware floor from another repository.

**Coverage establishes exercised lines, not correctness.** Do not lower a
floor, hide production code from the denominator, or refresh a baseline in
order to pass.

**Null floors are not a pass.** A null floor means the gate is *not yet armed*:
this tool prints the measured numbers, says so, and exits 3, which the runner
records as a distinct non-passing state. Exiting 0 there would be the silent
pass the whole milestone exists to prevent. AR-0009 armed the floors from a
measured baseline; the not-armed path stays because a later edit can null them
again, and because a half-armed manifest is not a pass either.

**A critical package must be justified, not merely listed.** Each entry in
`critical_packages` is an object carrying `name`, `measured_lines` (the
percentage measured when the floor was set), `measured_lines_total` (the
denominator that percentage was over) and `reason` (one line saying why the
crate is critical). An entry missing any of them, or written as a bare string,
is a manifest error and exits 2 -- listing a crate is cheap, and the manifest is
where the grounds for the choice have to survive. An entry may also carry
`floor_lines`, its own floor; `critical_lines` applies to any entry that does
not.

**A floor is checked against its own recorded baseline.** Requiring
`measured_lines` and then never reading it made it decoration: every floor could
be set to zero, with the manifest's own measured figure printed one line above
the pass. So each floor must lie within one slack of the baseline recorded
beside it, and `canonical_floor` below derives from that baseline the slackest
floor the rule permits. A floor under it exits 2, and a floor above its own
baseline exits 2 as an aspiration rather than a measurement.

What this can and cannot do is worth being exact about. It makes a *silent*
slackening impossible: lowering a floor alone now fails, so the only way to
lower one is to lower the recorded baseline with it, and that is a claim about
the world sitting in the diff where a reviewer reads it. It cannot decide
whether that claim is honest, because re-measuring after a coverage loss you
accept and re-measuring to launder a failure are the same edit. The gate makes
the edit visible and legible; judging it is review's job, and no threshold here
can take that over.

**The slack is a line count, not only a percentage.** Half a point of a
499-line crate is three lines: one deleted tested helper reds the gate, which is
a false red whose obvious remedy is lowering the floor -- pressure toward
exactly the slackening above. So the slack is `max(half a point, SLACK_LINES
lines)`, which leaves large crates where they were and gives small ones room to
be refactored. `SLACK_LINES` is 10 because that is roughly one small function
with its body: below it the gate measures churn rather than regression. It is a
judgement about what deserves the name regression, not a measurement, and it is
deliberately the only tunable here.

Usage::

    check_coverage.py --coverage-json FILE [--manifest FILE] [--repo PATH]

Exit status: 0 armed and met, 1 armed and below a floor, 3 not yet armed,
2 an environment error or a manifest that contradicts itself.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

NOT_ARMED = 3

#: The slack between a measured baseline and the floor set from it, as a
#: percentage. Half a point is the historical value and stays the minimum.
HALF_POINT = 0.5

#: ...and as an absolute number of covered lines, which is what makes the slack
#: scale-invariant. Half a point of the 17730-line workspace is 89 lines and of
#: the 499-line `peano` is 3, and a rule that reds on three lines is measuring
#: churn. See the module docstring for why this is 10 and why it is a judgement.
SLACK_LINES = 10

#: Floors are published rounded down to a half point, so the rule that derives
#: them has to round the same way or every published floor would look slack by
#: up to this much.
FLOOR_STEP = 0.5


def slack_percent(total_lines: float) -> float:
    """The permitted distance from a baseline to its floor, in percentage points."""
    if total_lines <= 0:
        return HALF_POINT
    return max(HALF_POINT, 100.0 * SLACK_LINES / total_lines)


def canonical_floor(measured: float, total_lines: float) -> float:
    """The slackest floor the rule permits for `measured` over `total_lines`.

    One rule for the workspace and for every crate: back off by the slack, then
    round down to a publishable half point. A recorded floor at or above this is
    within the rule; below it, the floor has been loosened past what its own
    baseline licenses.
    """
    backed_off = measured - slack_percent(total_lines)
    return math.floor(backed_off / FLOOR_STEP) * FLOOR_STEP


class Environment(Exception):
    """A problem with the environment, not a finding about coverage."""


def percent(summary: dict[str, object], kind: str) -> float:
    block = summary.get(kind)
    if not isinstance(block, dict):
        raise Environment(f"the coverage JSON has no {kind} summary")
    count = float(block.get("count", 0) or 0)
    covered = float(block.get("covered", 0) or 0)
    if count == 0:
        return 0.0
    return 100.0 * covered / count


def critical_packages(floors: dict[str, object]) -> list[dict[str, object]]:
    """The critical set, with each entry checked for its grounds.

    A crate is in this set because someone decided it is load-bearing, and the
    manifest is where that decision has to be readable later. So an entry must
    carry the number it was measured at and one line of why, and an entry that
    does not is rejected here rather than silently accepted: a list of bare
    names would record the outcome of the choice and lose the reasoning, which
    is the part a reviewer needs.
    """
    raw = floors.get("critical_packages") or []
    if not isinstance(raw, list):
        raise Environment("critical_packages must be a list of objects")
    entries: list[dict[str, object]] = []
    for index, item in enumerate(raw):
        where = f"critical_packages[{index}]"
        if not isinstance(item, dict):
            raise Environment(
                f"{where} is not an object: each critical package must be recorded as "
                "{'name': ..., 'measured_lines': ..., 'reason': ...}, because a bare name "
                f"gives no measured figure and no grounds for inclusion (got {item!r})"
            )
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise Environment(f"{where} has no usable 'name'")
        missing = []
        measured = item.get("measured_lines")
        if not isinstance(measured, (int, float)) or isinstance(measured, bool):
            missing.append("measured_lines (the coverage measured when the floor was set)")
        total = item.get("measured_lines_total")
        if not isinstance(total, (int, float)) or isinstance(total, bool) or total <= 0:
            missing.append(
                "measured_lines_total (the line count that percentage was over, which is what "
                "makes the slack scale with the crate)"
            )
        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            missing.append("reason (one line on why this crate is critical)")
        if missing:
            raise Environment(
                f"critical package {name} is listed without " + " and without ".join(missing)
            )
        floor = item.get("floor_lines")
        if floor is not None and (not isinstance(floor, (int, float)) or isinstance(floor, bool)):
            raise Environment(f"critical package {name} has a non-numeric floor_lines")
        entries.append(
            {
                "name": name,
                "measured_lines": float(measured),
                "measured_lines_total": float(total),
                "reason": reason,
                "floor_lines": floor,
            }
        )
    return entries


def check_floor_against_baseline(
    what: str, floor: float, measured: float, total_lines: float
) -> str | None:
    """Reject a floor that its own recorded baseline does not license.

    Returns a message, or None when the floor is within the rule. Two ways to
    fail. Below `canonical_floor` the floor has been loosened further than the
    slack allows, which is the slackening this check exists to catch. Above the
    baseline it is an aspiration: a floor no measurement has ever met, which is
    red on arrival and tells a reader the opposite of what the number claims.
    """
    permitted = canonical_floor(measured, total_lines)
    if floor > measured:
        return (
            f"{what} floor {floor:.2f}% is above its own recorded baseline "
            f"{measured:.2f}%: a floor is set from what was measured, not from what "
            f"was hoped for"
        )
    if floor < permitted - 1e-9:
        slack = slack_percent(total_lines)
        lines = slack * total_lines / 100.0
        return (
            f"{what} floor {floor:.2f}% is below {permitted:.2f}%, the slackest floor its "
            f"recorded baseline of {measured:.2f}% licenses (slack {slack:.2f} points, "
            f"{lines:.0f} of {total_lines:.0f} lines). Lowering a floor without lowering the "
            f"baseline beside it is how a floor stops constraining anything; if coverage really "
            f"was lost, re-measure and move both, so that the claim is in the diff"
        )
    return None


def package_directories(root: Path) -> dict[str, str]:
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0:
        raise Environment(f"cargo metadata exited {out.returncode}")
    data = json.loads(out.stdout)
    directories: dict[str, str] = {}
    for package in data["packages"]:
        directory = Path(package["manifest_path"]).parent
        try:
            directories[package["name"]] = str(directory.relative_to(root))
        except ValueError:
            directories[package["name"]] = str(directory)
    return directories


def package_lines(coverage: dict[str, object], directory: str) -> tuple[float, float]:
    """(covered, count) of lines in files under `directory`.

    The match is on the path *relative to the repository root*, not on an
    absolute prefix: the coverage JSON records the absolute paths of whichever
    worktree measured it, and those are not the paths `cargo metadata` reports
    unless the two ran in the same tree. Comparing the relative part is what
    makes a report measured in the runner's throwaway worktree still
    attributable to a package.
    """
    covered = count = 0.0
    inside = f"/{directory.strip('/')}/"
    for entry in coverage.get("files", []) or []:
        filename = str(entry.get("filename", ""))
        if inside not in filename and not filename.startswith(inside.lstrip("/")):
            continue
        lines = entry.get("summary", {}).get("lines", {})
        covered += float(lines.get("covered", 0) or 0)
        count += float(lines.get("count", 0) or 0)
    return covered, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--coverage-json", required=True, help="cargo llvm-cov --json output")
    parser.add_argument("--manifest", default=None, help="config/quality-tools.json")
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
        manifest_path = Path(args.manifest) if args.manifest else root / "config" / "quality-tools.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report = json.loads(Path(args.coverage_json).read_text(encoding="utf-8"))
        data = report.get("data") or []
        if not data:
            raise Environment("the coverage JSON holds no data")
        coverage = data[0]
        totals = coverage.get("totals")
        if not isinstance(totals, dict):
            raise Environment("the coverage JSON holds no totals")
        floors = manifest.get("coverage", {})
    except Environment as error:
        print(f"coverage: {error}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError) as error:
        print(f"coverage: cannot read the inputs: {error}", file=sys.stderr)
        return 2

    try:
        workspace_lines = percent(totals, "lines")
        workspace_regions = percent(totals, "regions")
        workspace_total_lines = float(totals["lines"].get("count", 0) or 0)
    except Environment as error:
        print(f"coverage: {error}", file=sys.stderr)
        return 2

    print(
        f"coverage: workspace lines {workspace_lines:.2f}%, "
        f"regions {workspace_regions:.2f}% (line and region only; "
        f"branch coverage is not available on the pinned stable toolchain)"
    )

    try:
        critical = critical_packages(floors)
    except Environment as error:
        print(f"coverage: {error}", file=sys.stderr)
        return 2

    critical_measured: dict[str, float] = {}
    if critical:
        try:
            directories = package_directories(root)
        except Environment as error:
            print(f"coverage: {error}", file=sys.stderr)
            return 2
        for entry in critical:
            name = entry["name"]
            if name not in directories:
                print(f"coverage: manifest names a critical package that does not exist: {name}", file=sys.stderr)
                return 2
            covered, count = package_lines(coverage, directories[name])
            value = 100.0 * covered / count if count else 0.0
            critical_measured[name] = value
            print(
                f"coverage: critical package {name} lines {value:.2f}% "
                f"(baseline recorded in the manifest: {float(entry['measured_lines']):.2f}%)"
            )

    workspace_floor = floors.get("workspace_lines")
    critical_floor = floors.get("critical_lines")

    if workspace_floor is None and critical_floor is None:
        print(
            "coverage: NOT ARMED - both floors are null in the manifest, so nothing was "
            "compared. AR-0009 set them from a measured baseline; a manifest that nulls "
            "them again disarms this gate, and a disarmed gate is not a pass.",
            file=sys.stderr,
        )
        return NOT_ARMED

    failures: list[str] = []
    if workspace_floor is None or critical_floor is None:
        missing = "workspace_lines" if workspace_floor is None else "critical_lines"
        print(
            f"coverage: NOT ARMED - {missing} is null while the other floor is set; "
            "a half-armed gate is not a pass.",
            file=sys.stderr,
        )
        return NOT_ARMED
    if critical_floor is not None and not critical:
        print(
            "coverage: critical_lines is set but critical_packages is empty, so the floor "
            "would apply to nothing",
            file=sys.stderr,
        )
        return 2

    # Before comparing anything against a floor, check that the floors are the
    # ones the manifest's own baselines license. A floor that has drifted from
    # its baseline is not a stricter or looser opinion about coverage, it is a
    # manifest that contradicts itself, and reporting a pass or a fail computed
    # from it would dress a contradiction up as a verdict.
    contradictions: list[str] = []
    measured_workspace = floors.get("measured_workspace")
    if (
        not isinstance(measured_workspace, dict)
        or not isinstance(measured_workspace.get("lines"), (int, float))
        or not isinstance(measured_workspace.get("lines_total"), (int, float))
        or measured_workspace.get("lines_total", 0) <= 0
    ):
        print(
            "coverage: the manifest sets workspace_lines but does not record both "
            "measured_workspace.lines and measured_workspace.lines_total to have set it from; "
            "a floor whose baseline is not written down cannot be checked against one",
            file=sys.stderr,
        )
        return 2
    if abs(float(measured_workspace["lines_total"]) - workspace_total_lines) > 0.5:
        print(
            f"coverage: NOTE the workspace is {workspace_total_lines:.0f} lines but the baseline "
            f"was taken over {float(measured_workspace['lines_total']):.0f}. The slack the floor "
            f"was derived from is the recorded size, deliberately, so that a crate growing "
            f"cannot tighten a floor nobody edited -- but the further these drift the less the "
            f"floor describes the tree it is guarding."
        )
    problem = check_floor_against_baseline(
        "workspace",
        float(workspace_floor),
        float(measured_workspace["lines"]),
        float(measured_workspace["lines_total"]),
    )
    if problem:
        contradictions.append(problem)

    for entry in critical:
        floor = entry["floor_lines"]
        floor = float(critical_floor) if floor is None else float(floor)
        problem = check_floor_against_baseline(
            f"critical package {entry['name']}",
            floor,
            entry["measured_lines"],
            entry["measured_lines_total"],
        )
        if problem:
            contradictions.append(problem)

    if contradictions:
        for problem in contradictions:
            print(f"coverage: {problem}", file=sys.stderr)
        return 2

    # A baseline the tree has since climbed well past is not a failure -- adding
    # tests must never turn this gate red -- but it does mean the floor is
    # holding the tree to an old standard while reading as if it held it to the
    # current one, so it is said out loud rather than left for someone to notice.
    for entry in critical:
        name = entry["name"]
        drift = critical_measured[name] - entry["measured_lines"]
        if drift > slack_percent(entry["measured_lines_total"]):
            print(
                f"coverage: NOTE {name} now measures {critical_measured[name]:.2f}%, "
                f"{drift:.2f} points above the baseline of {entry['measured_lines']:.2f}% its "
                f"floor was set from. The floor still holds, but it is stale; re-measure and "
                f"raise both so the floor constrains the tree as it is."
            )

    if workspace_lines < float(workspace_floor):
        failures.append(
            f"workspace lines {workspace_lines:.2f}% is below the floor {float(workspace_floor):.2f}%"
        )
    for entry in critical:
        name = entry["name"]
        value = critical_measured[name]
        floor = entry["floor_lines"]
        floor = float(critical_floor) if floor is None else float(floor)
        if value < floor:
            failures.append(
                f"critical package {name} lines {value:.2f}% is below the floor {floor:.2f}%"
            )

    for failure in failures:
        print(f"FAIL coverage: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
