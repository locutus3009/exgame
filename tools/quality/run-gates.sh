#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The hermetic gate runner. One command runs every product gate against an
# exact commit and emits a structured report.
#
#   tools/quality/run-gates.sh <REVISION> [options]
#
# This is the authority for this project, because no hosted runner can build
# this workspace: crates/newton/build.rs opens a Vulkan compute device at build
# time and has no fallback. The runner therefore has to recover by construction
# what hosted CI normally supplies -- a clean environment, an exact revision, a
# pinned toolchain, and a report someone else can check.
#
# How it runs, and why:
#
#   Hermetic.  It refuses a dirty tree and refuses an unknown revision. It
#   builds in a throwaway worktree checked out at the exact commit, with its own
#   CARGO_TARGET_DIR, so no stale build artefact and nothing in the developer's
#   working tree can change the answer. The worktree and the target directory
#   are removed on success and on failure alike.
#
#   Invoked by a reviewer.  The runner exists so that someone who did not write
#   a change can confirm the claims made about it. An implementer running it on
#   their own branch is a useful pre-check and is not evidence.
#
#   Every gate, every run.  Each gate's exit code and duration are captured and
#   the run continues, so one run reports everything that is wrong.
#
#   Fail closed.  A missing tool, an unexpected version, a digest mismatch or a
#   gate that cannot run is a non-passing result, never a skip. A coverage floor
#   that is still null in the manifest is reported as *not armed* and is not
#   counted as a pass. The run is green only when every gate is `pass`.
#
#   A pass has a ledger entry behind it.  On the honest path every command a
#   gate runs goes through `run_step`, which records it in an execution ledger
#   -- the argv digest, and the exit status the command returned rather than one
#   the dispatch chose -- after it returns. A gate's published digest, duration
#   and step table are read out of that ledger, and `build_report.py` refuses to
#   publish a status the ledger does not support: `pass` needs at least one
#   entry for the gate with every entry at 0, `not_armed` needs an entry
#   returning the code it names, and `not_applicable` and `not_implemented` need
#   no entry at all. The rule is over entries, not over executions -- see the
#   paragraph below for the difference, which is the whole of what this buys.
#   It exists because it was done:
#   review short-circuited gates 2, 3, 4 and 7 in the dispatch, left the command
#   templates and `--list-gates` untouched, published a plausible argv digest for
#   argv nobody invoked, and got a 15/15 green report with gate 7 *upgraded*
#   from a blocking `not_armed` to `pass` by being disabled.
#
#   What that does not buy, because a second review measured it. The dispatch
#   has no *variable* left in which to say what ran -- but `record_execution` is
#   a top-level function in this file, reachable from any arm, so disabling a
#   gate is still a one-line change in one arm of one `case`, in one file, with
#   `build_report.py` untouched. Demonstrated over gates 2, 3, 4 and 12: green
#   board, every fixture and assertion green, forged records identical in every
#   published field to honest ones. What changed is that such an arm must now
#   *name* a command, a digest and an exit status in the executor's format
#   rather than assert a bare status -- more for a reviewer to read, not more
#   for an attacker to write. The rest is unchanged and is written out in
#   `docs/QUALITY_GATES.md`: a runner is executed by the person invoking it,
#   nothing inside it can vouch for a runner the reviewer chose to trust, and
#   this file is the one to read, because one line here is still enough.
#
#   A function of the revision, not of the machine.  Every gate answers a
#   question about the named commit. Gate 9 is the one that had to be made to:
#   `gitleaks git` scans every ref it can see and the throwaway worktree shares
#   the invoking repository's object store, so it used to answer a question
#   about the reviewer's local branches. It now scans a repository holding this
#   revision's history and nothing else.
#
#   Explicit about whose checkers ran.  Gates 7 and 10 to 15 execute a checker
#   out of the revision under test, so that the report is a function of that
#   revision and of nothing else, and so that a change to a checker is exercised
#   by the run that reviews it. The price is that a revision is graded by its own
#   instruments. The report therefore carries the sha256 of every checker file
#   executed and the policy files it read, and a digest -- over worktree bytes,
#   not index entries -- of every revision-controlled input that can change a
#   verdict on both sides of the boundary: the checkers, the dependency policy,
#   the advisory ignore list, the lint severities in the workspace manifest and
#   in every member manifest, the formatting and clippy configuration, the
#   analyzer configuration and cargo's own configuration. When they differ the
#   run is not self-certifying and says so as a blocking reason, naming the
#   paths. See `runner.provenance` in the report.
#
#   A second opinion, advisory.  The four offline checkers are run again from
#   the base commit against the same revision, and a disagreement is reported.
#   It never blocks: a checker can disagree with its predecessor by being
#   stricter, by shedding a false positive, or by being weakened.
#
# Options:
#   --base REF     range start for the commit and policy gates (default:
#                  origin/main). On a post-merge run, where origin/main..HEAD is
#                  empty by construction, name the introduced range: for a merge
#                  commit M, --base M^1 <M>.
#   --report PATH  where to write the JSON report (default: ./gate-report.json)
#   --log-dir DIR  keep each gate's output in DIR (default: discarded; the
#                  report never carries raw output). Put it outside the
#                  repository, or the next run will see a dirty tree.
#   --list-gates   print the gate list as JSON and exit. This is the
#                  enumeration AR-0007's suite must read, so that a gate added
#                  here without a fixture is itself a failure.
#   --argv-digest ARG...
#                  print the digest this runner would publish for that argv and
#                  exit. The report's `argv_sha256` fields are the only evidence
#                  it carries of *what* was run, so the function that computes
#                  them is reachable from outside: a reader can recompute a
#                  published digest from the argv the report says produced it,
#                  and a fixture can prove that two argv differing only in where
#                  the argument boundaries fall do not collide. Must be the last
#                  option; everything after it is the argv. Folds the repository
#                  root out exactly as a live run does, so it has to find that
#                  root: outside a git repository it exits 2 rather than print a
#                  digest computed without the substitution.
set -uo pipefail

# These scripts are run from a throwaway worktree; leaving byte-cache directories
# behind in it is noise, and in a developer's tree it is an untracked file that
# the next run would report as a dirty tree.
export PYTHONDONTWRITEBYTECODE=1

readonly SEP=$'\x1f'

# The gates, in execution order. id|name|what the report calls it.
# Gate 10 installs the analyzers gates 8 and 9 need, so it runs first even
# though it is numbered later in docs/QUALITY_GATES.md. The cheap offline gates
# run before the compiling ones so that a reviewer watching the terminal learns
# the most in the first minute; every gate runs regardless of what fails.
readonly GATE_ORDER=(10 1 5 6 8 9 11 12 14 15 2 3 4 7 13)

gate_name() {
    case "$1" in
        1)  echo "formatting" ;;
        2)  echo "lints" ;;
        3)  echo "tests" ;;
        4)  echo "documentation build" ;;
        5)  echo "dependency policy" ;;
        6)  echo "advisories" ;;
        7)  echo "coverage floors" ;;
        8)  echo "workflow linting" ;;
        9)  echo "secret scanning" ;;
        10) echo "external analyzer installation" ;;
        11) echo "repository policy" ;;
        12) echo "commit signature, sign-off and message privacy" ;;
        13) echo "negative-fixture suite" ;;
        14) echo "entry-point crate-table consistency" ;;
        15) echo "architecture-index completeness" ;;
        *)  echo "unknown" ;;
    esac
}

gate_command_template() {
    case "$1" in
        1)  echo "cargo fmt --all -- --check" ;;
        2)  echo "cargo clippy --locked --workspace --all-targets -- -D warnings" ;;
        3)  echo "cargo test --locked --workspace" ;;
        4)  echo "RUSTDOCFLAGS=-D warnings cargo doc --locked --workspace --no-deps" ;;
        5)  echo "cargo deny --locked check" ;;
        6)  echo "cargo audit --deny warnings" ;;
        7)  echo "cargo llvm-cov --locked --workspace --summary-only --json; tools/quality/check_coverage.py" ;;
        8)  echo "env -u SHELLCHECK_OPTS <bin>/actionlint-with-shellcheck -config-file .github/actionlint.yaml; <bin>/zizmor --pedantic ." ;;
        9)  echo "env -u GITLEAKS_CONFIG <bin>/gitleaks git --redact --no-banner --log-opts=\"--full-history <commit>\" <isolated scan repository>" ;;
        10) echo "tools/quality/install-external-tools.sh --bin-dir <bin>" ;;
        11) echo "tools/quality/repository_policy.py --base <base> --head <head>" ;;
        12) echo "tools/quality/check_commits.py --base <base> --head <head>" ;;
        13) echo "tools/quality/test_failure_paths.py --bin-dir <bin>" ;;
        14) echo "tools/quality/check_crate_table.py" ;;
        15) echo "tools/quality/check_docs.py index" ;;
        *)  echo "unknown" ;;
    esac
}

die() { printf 'run-gates: %s\n' "$*" >&2; exit 2; }
say() { printf 'run-gates: %s\n' "$*" >&2; }

# `--list-gates` is an interface, not a convenience: AR-0007's suite reads it to
# discover the gates, so that a gate added here without a fixture is itself a
# failure. It was assembled with printf and a command template grew a pair of
# quotation marks, which made the output invalid JSON for a whole review cycle
# while every gate still passed -- an interface claim with nothing checking it,
# which is the defect this project keeps finding in other people's work. It is
# now built by a JSON encoder, so no template can break the syntax, and every
# run parses it before doing anything else.
list_gates() {
    local id spec=""
    for id in "${GATE_ORDER[@]}"; do
        spec+="$id$SEP$(gate_name "$id")$SEP$(gate_command_template "$id")"$'\n'
    done
    printf '%s' "$spec" | python3 -c '
import json, sys

gates = []
for line in sys.stdin.read().splitlines():
    if not line:
        continue
    gate_id, name, command = line.split("\x1f")
    gates.append({"id": int(gate_id), "name": name, "command": command})
print(json.dumps(gates, indent=2))
'
}

# Parse it, and check it enumerates exactly what the run will execute. Fails
# closed: a runner that cannot describe itself to the suite that consumes the
# description has no business grading anything.
check_gate_enumeration() {
    python3 -c '
import json, sys

expected = [int(value) for value in sys.argv[2].split()]
try:
    gates = json.loads(sys.argv[1])
except json.JSONDecodeError as error:
    sys.exit("--list-gates is not valid JSON: %s" % error)
found = [gate["id"] for gate in gates]
if found != expected:
    sys.exit("--list-gates enumerates %s, the run executes %s" % (found, expected))
for gate in gates:
    if not gate.get("name") or not gate.get("command"):
        sys.exit("--list-gates entry %s carries no name or no command" % gate.get("id"))
' "$(list_gates)" "${GATE_ORDER[*]}"
}

# Defined here rather than beside the gates, because `--argv-digest` is
# answered during argument parsing and a function has to exist before it is
# called. The path substitutions below tolerate the throwaway directories not
# existing yet, which is exactly the state that option runs in.
# An argv digest with the run's throwaway paths folded out, so that two runs of
# the same revision produce the same digest and no absolute path reaches the
# report.
#
# The separator is a NUL byte written straight into the pipe. It used to be
# accumulated in a shell variable -- `out+="$part"$'\0'` -- and a bash variable
# cannot hold a NUL, so the separator was discarded and the digest was taken
# over the arguments *concatenated with nothing between them*: `${#x}` is 0 for
# `x=$'\0'`, and argv `a b` hashed identically to argv `ab`
# (fb8e20fc2e4c3f24...). The boundaries between arguments are the one thing
# this digest exists to pin down -- two people ran "the same command" only if
# the boundaries fell in the same places -- so a digest that cannot see them
# attests less than it claims. `printf` into a pipe is not subject to that
# limit; only variables and command substitution drop the byte.
#
# Every digest this runner publishes changes as a result of the fix. A digest
# recorded before it and one recorded after are not comparable and never were
# over the same bytes; no comparison already on record is invalidated, because
# both sides of each was computed the same way. `--argv-digest` recomputes one
# from outside the runner, so a reader can check a published digest against the
# argv the report says produced it.
#
# The path substitutions are guarded on the variable being non-empty. An empty
# search pattern is not a no-op in bash -- `${part//""/X}` inserts X between
# every character -- and these variables are unset until the throwaway
# directory exists, which is after `--argv-digest` has to work.
argv_digest() {
    local part
    {
        for part in "$@"; do
            [ -n "${wt:-}" ]     && part="${part//$wt/<worktree>}"
            [ -n "${target:-}" ] && part="${part//$target/<target>}"
            [ -n "${bin:-}" ]    && part="${part//$bin/<bin>}"
            [ -n "${work:-}" ]   && part="${part//$work/<work>}"
            [ -n "${root:-}" ]   && part="${part//$root/<repo>}"
            printf '%s\0' "$part"
        done
        true
    } | sha256sum | cut -d' ' -f1
}

# ---------------------------------------------------------------- arguments --

revision=""
base_ref="origin/main"
report_path=""
log_dir=""

while [ $# -gt 0 ]; do
    case "$1" in
        --list-gates) list_gates; exit 0 ;;
        # `root` is resolved here rather than left to the assignment below the
        # loop. It used to be unset at this point, so `--argv-digest` skipped
        # the `<repo>` substitution that a live run applies and could return a
        # different digest for the same argv -- which would make the digest
        # exactly not recomputable from outside, the one thing this option is
        # for. No gate's argv contains the repository root today, so nothing
        # published was affected; the property that made it harmless was not
        # one anything checked, and the runner assertion
        # `argv-digest-folds-out-the-repository-root` now does.
        --argv-digest)
            shift
            here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
            root="$(git -C "$here" rev-parse --show-toplevel 2>/dev/null)" \
                || die "not inside a git repository, so <repo> cannot be folded out"
            argv_digest "$@"
            exit 0 ;;
        --base)       base_ref="${2:?--base needs a value}"; shift 2 ;;
        --report)     report_path="${2:?--report needs a value}"; shift 2 ;;
        --log-dir)    log_dir="${2:?--log-dir needs a value}"; shift 2 ;;
        # Bounded by the first line of code rather than by a line number:
        # the header outgrew a hardcoded `3,70p` the first time it was extended,
        # and silently truncated its own option list.
        -h|--help)    sed -n '3,/^set -uo pipefail/p' "$0" | sed '$d'; exit 0 ;;
        -*)           die "unknown option: $1" ;;
        *)
            [ -z "$revision" ] || die "more than one revision given: $revision and $1"
            revision="$1"; shift ;;
    esac
done

[ -n "$revision" ] || die "usage: run-gates.sh <REVISION> [--base REF] [--report PATH]"

check_gate_enumeration \
    || die "the gate enumeration is not machine-readable, and AR-0007's suite reads it"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(git -C "$here" rev-parse --show-toplevel 2>/dev/null)" || die "not inside a git repository"
[ -n "$report_path" ] || report_path="$PWD/gate-report.json"
mkdir -p "$(dirname "$report_path")" || die "cannot create the report directory"
report_path="$(cd "$(dirname "$report_path")" && pwd)/$(basename "$report_path")"

# ------------------------------------------------------------ preconditions --

for required in git python3 curl tar sha256sum gpg cargo rustup vulkaninfo; do
    command -v "$required" >/dev/null 2>&1 || die "missing required command: $required (fail closed)"
done

manifest="$root/config/quality-tools.json"
[ -f "$manifest" ] || die "no tool manifest at config/quality-tools.json"

manifest_field() {
    python3 - "$manifest" "$@" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2:]:
    data = data[key]
print("" if data is None else data)
PY
}

toolchain="$(manifest_field rust toolchain)" || die "cannot read rust.toolchain from the manifest"
[ -n "$toolchain" ] || die "the manifest names no toolchain"

# The tree must be clean. The one exception is the report this very run is
# about to write, which is otherwise an untracked file in the repository.
report_relative=""
case "$report_path" in
    "$root"/*) report_relative="${report_path#"$root"/}" ;;
esac
dirty="$(git -C "$root" status --porcelain --untracked-files=all)"
if [ -n "$report_relative" ]; then
    dirty="$(printf '%s\n' "$dirty" | grep -v -x -F "?? $report_relative" || true)"
fi
dirty="$(printf '%s\n' "$dirty" | sed '/^$/d')"
if [ -n "$dirty" ]; then
    say "the working tree is not clean; a gate run must describe an exact commit"
    printf '%s\n' "$dirty" | sed 's/^/run-gates:   /' >&2
    exit 2
fi

commit="$(git -C "$root" rev-parse --verify --quiet "${revision}^{commit}")" \
    || die "unknown revision: $revision"
base_commit="$(git -C "$root" rev-parse --verify --quiet "${base_ref}^{commit}")" \
    || die "unknown base revision: $base_ref"

# The pinned toolchain must exist and be the one the manifest names.
rustup toolchain list 2>/dev/null | grep -q "^${toolchain}-" \
    || die "the pinned toolchain $toolchain is not installed"
pinned_file="$root/rust-toolchain.toml"
if [ -f "$pinned_file" ]; then
    grep -q "\"${toolchain}\"" "$pinned_file" \
        || die "rust-toolchain.toml and the manifest disagree about the toolchain"
fi
export RUSTUP_TOOLCHAIN="$toolchain"
rustc_version="$(rustc --version 2>/dev/null | awk '{print $2}')"
cargo_version="$(cargo --version 2>/dev/null | awk '{print $2}')"
[ "$rustc_version" = "$toolchain" ] || die "rustc reports $rustc_version, the manifest pins $toolchain"
[ "$cargo_version" = "$toolchain" ] || die "cargo reports $cargo_version, the manifest pins $toolchain"

check_version() {  # name, observed, expected
    [ "$2" = "$3" ] || die "$1 reports $2, the manifest pins $3 (unexpected version fails the run)"
}
deny_version="$(cargo deny --version 2>/dev/null | awk '{print $NF}')"
audit_version="$(cargo audit --version 2>/dev/null | awk '{print $NF}')"
llvmcov_version="$(cargo llvm-cov --version 2>/dev/null | awk '{print $NF}')"
check_version cargo-deny "$deny_version" "$(manifest_field rust cargo_deny)"
check_version cargo-audit "$audit_version" "$(manifest_field rust cargo_audit)"
check_version cargo-llvm-cov "$llvmcov_version" "$(manifest_field rust cargo_llvm_cov)"

# A graphics device is a precondition, not a gate: every gate that compiles the
# workspace needs one, and the runner refuses rather than skipping when there is
# none.
device_json="$(vulkaninfo --summary 2>/dev/null | python3 -c '
import hashlib, json, sys

# rembrandt::GpuAccelerator::new() ranks devices discrete > integrated >
# virtual > cpu > other and takes the maximum with Rust`s max_by_key, which
# returns the *last* maximal element. This mirrors that rule, so the reported
# device is the one the build will open.
RANK = {
    "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU": 4,
    "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU": 3,
    "PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU": 2,
    "PHYSICAL_DEVICE_TYPE_CPU": 1,
}


def fingerprint(value):
    return "sha256:" + hashlib.sha256((value or "").encode("utf-8")).hexdigest()


devices, name, kind, driver = [], None, None, None
for line in sys.stdin:
    line = line.strip()
    if line.startswith("deviceName"):
        name = line.split("=", 1)[1].strip()
    elif line.startswith("deviceType"):
        kind = line.split("=", 1)[1].strip()
    elif line.startswith("driverInfo"):
        driver = line.split("=", 1)[1].strip()
        if name:
            devices.append({"name": name, "type": kind, "driver": driver})
            name = kind = driver = None
selected = None
best = -1
for device in devices:
    rank = RANK.get(device["type"] or "", 0)
    if rank >= best:          # >=, because max_by_key keeps the last maximum
        best, selected = rank, device
print(json.dumps({
    "api": "vulkan",
    "device_count": len(devices),
    # What makes a run meaningful is that a compute device of a given class was
    # present, not which retail part it was. The model string is recorded as a
    # fingerprint so that two runs can be compared without the report naming the
    # hardware -- a verbatim model beside an exact kernel release narrows the
    # host much further than the hashed hostname beside it admits.
    "enumerated": [
        {"type": d["type"], "fingerprint": fingerprint(d["name"])} for d in devices
    ],
    "selected": fingerprint(selected["name"]) if selected else None,
    "selected_type": (selected or {}).get("type"),
    "driver": (selected or {}).get("driver"),
    "selection_rule": "mirrors rembrandt::GpuAccelerator::new(): highest device type rank, last maximum wins",
    "note": (
        "Device models are fingerprints, not names. The driver version stays in "
        "plain text because it changes the answer; the model does not, only its "
        "class does. A hash of a short well-known string is guessable: this keeps "
        "the hardware out of the artefact, it does not make the class a secret."
    ),
}))
')"
[ -n "$device_json" ] || die "cannot enumerate Vulkan devices"
if printf '%s' "$device_json" | grep -q '"selected": null'; then
    die "no Vulkan device: this workspace cannot be built, and refusing is the only honest result"
fi
say "graphics device: $(printf '%s' "$device_json" | python3 -c '
import json, sys
device = json.load(sys.stdin)
print(device["selected_type"], device["selected"][:19] + "...")')"

# ------------------------------------------------------- throwaway worktree --

work="$(mktemp -d "${TMPDIR:-/tmp}/gate-run-XXXXXX")" || die "cannot create a work directory"
# This directory holds the gate records the report is assembled from, the
# analyzer binaries gate 10 installs and gate 9's scan repository, and the
# revision's own build scripts run as this user while the gates are running --
# a build script given only CARGO_TARGET_DIR was watched to locate `gates.jsonl`
# and rewrite a recorded gate. `mktemp -d` already creates it 0700 whatever the
# umask, so the mode below changes nothing and is here to state the property
# rather than inherit it. Neither stops the code under test, which runs as this
# user; that residual is stated in the report and is not fixed by either line.
chmod 700 "$work" || die "cannot restrict the work directory"
wt="$work/worktree"
target="$work/target"
bin="$work/analyzers"
records="$work/gates.jsonl"
# The execution ledger: one line per command, written after that command
# returned and from the status it returned. It is a separate file from the gate
# records on purpose -- the records hold what the dispatch concluded, the ledger
# holds what was executed, and `build_report.py` publishes a gate only when the
# two agree.
#
# On an honest tree `run_step` is the only writer. That is a property of the
# tree, not an invariant of the mechanism: `record_execution` below is a
# top-level function and any arm of the dispatch can call it in one line. See
# its comment.
executed="$work/executed.jsonl"
: > "$records"
: > "$executed"
: > "$work/shadowing.txt"
mkdir -p "$target" "$bin"

cleanup() {
    local status=$?
    if [ -d "$wt" ]; then
        git -C "$root" worktree remove --force "$wt" >/dev/null 2>&1 || rm -rf "$wt"
    fi
    git -C "$root" worktree prune >/dev/null 2>&1 || true
    rm -rf "$work"
    return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

git -C "$root" worktree add --detach --quiet "$wt" "$commit" \
    || die "cannot create a worktree at $commit"
say "worktree at ${commit:0:12} created; target directory is private to this run"

export CARGO_TARGET_DIR="$target"

# ------------------------------------------------------------------- gates --

# The ledger writer, and the honest caller's contract: `run_step` calls this
# after the command has returned, passing the status the command returned rather
# than one the dispatch chose. Every published claim about *what ran* is read
# out of this file.
#
# What that does NOT establish, because it was measured. This is a top-level
# shell function and nothing restricts who calls it, so an arm of the dispatch
# can write a ledger entry for a command it never ran, in one line:
#
#     2)  record_execution 2 "gate 2 (lints)" "<the honest digest>" 0 54.4
#         gate_status="pass" ;;
#
# Review did exactly that over gates 2, 3, 4 and 12 and got a green board, with
# the forged records identical in every published field to honest ones. So read
# `run_step is the only writer` as a fact about an honest tree and not as an
# invariant the ledger enforces -- it is precisely the fact a one-line edit to
# this file removes. A `FUNCNAME` guard here was tried by review and defeated
# for no extra line, by shadowing `run_step` in a subshell around the call, so
# it is deliberately absent rather than shipped as though it helped.
record_execution() {  # gate id, label, argv digest, captured exit, duration
    python3 - "$executed" "$@" <<'LEDGER'
import json, sys
path, gate, label, digest, code, duration = sys.argv[1:7]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "gate": int(gate),
        "label": label,
        "argv_sha256": digest,
        "exit_code": int(code),
        "duration_seconds": float(duration),
    }) + "\n")
LEDGER
}

# What the dispatch passes here, and what it does not.
#
# It passes a verdict -- the status and the exit code -- because some verdicts
# are genuinely the dispatch's to make: gate 10 fails an installer that returned
# 0 but left binaries reporting the wrong version, and gate 9 refuses before
# running anything when its scan repository did not come out isolated.
#
# It does not pass the argv digest, the duration or the per-step table. Those
# are read out of the execution ledger here. Scope that precisely, because an
# earlier version of this comment did not: what an arm short-circuited to
# `gate_status="pass"` can no longer do is publish a digest or a duration
# *through this function*, because they are not parameters of it. It can still
# publish both, by calling `record_execution` above -- which takes the digest
# and the duration as arguments -- in the same one line that sets the status.
# Review demonstrated that and got a green board.
#
# So what this half of AR-0017 buys is not that a fabricated digest became
# impossible. It is that fabricating one is no longer separable from the rest:
# an arm that wants a pass must name a command, a digest and an exit status in
# the ledger's format, and `build_report.py` checks the status it claims against
# the ledger and refuses when the two disagree. That is more for a reviewer to
# read. It is not more for an attacker to write.
record() {  # id, status, exit, note, configuration, scripts, inputs
    python3 - "$records" "$executed" "$@" <<'RECORD'
import json, sys
path, ledger, gate, status, code, note, configuration, scripts, inputs = sys.argv[1:10]
gate = int(gate)
steps = []
with open(ledger, encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            entry = json.loads(line)
            if entry["gate"] == gate:
                steps.append(entry)
duration = round(sum(step["duration_seconds"] for step in steps), 3)
# Joined with `+` where a gate ran several commands, which is the spelling the
# multi-step gates already used. Empty when the gate has no entry, which
# build_report.py turns into an explicit `no command was executed for this gate`.
digest = "+".join(step["argv_sha256"] for step in steps)
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "id": gate,
        "status": status,
        "exit_code": int(code),
        "duration_seconds": duration,
        # Derived, never asserted. Empty when the gate has no entry: the digest
        # of an argv nobody invoked reads as evidence of a command and is not.
        "argv_sha256": digest,
        "commands_executed": len(steps),
        # This gate's slice of the ledger, published so that a reader who has
        # only this file can see which commands the ledger records, in which
        # order, and with which exit status -- and so that a gate which stops
        # running changes the stable digest WHEN NO LEDGER ENTRY IS FORGED FOR
        # IT. Before AR-0017 nothing about a skipped gate reached that digest,
        # its only trace being a duration, and durations are excluded from it as
        # wall-clock noise. The qualification is the whole of what this buys: an
        # arm calling `record_execution` with the label, digest and exit an
        # honest run would have written produces a byte-identical digest at a
        # fixed revision. See `record_execution`.
        "steps": [
            {
                "label": step["label"],
                "argv_sha256": step["argv_sha256"],
                "exit_code": step["exit_code"],
                "duration_seconds": step["duration_seconds"],
            }
            for step in steps
        ],
        "note": note,
        "configuration": configuration,
        "scripts": json.loads(scripts),
        "inputs": json.loads(inputs),
    }) + "\n")
RECORD
}

# The exact checker files this run executed, by content. Gates 7 and 10 to 15
# execute a script out of the tree under test, so a reader who has the diff in
# front of them can tell whether the script that returned `pass` is the script
# they reviewed. See the provenance section of the report for why they run from
# there and not from the invoking checkout.
files_json() {  # relative paths inside the worktree
    local rel spec=""
    for rel in "$@"; do
        if [ -f "$wt/$rel" ]; then
            spec+="$rel$SEP$(sha256sum "$wt/$rel" | cut -d' ' -f1)"$'\n'
        else
            spec+="$rel$SEP"$'\n'
        fi
    done
    printf '%s' "$spec" | python3 -c '
import json, sys

entries = []
for line in sys.stdin.read().splitlines():
    if not line:
        continue
    path, _, digest = line.partition("\x1f")
    entries.append({
        "path": path,
        "sha256": ("sha256:" + digest) if digest else None,
        "source": "revision under test" if digest else "absent from the revision under test",
    })
print(json.dumps(entries))
'
}

# The subcommands the gates invoke through cargo. An `[alias]` in the
# revision's .cargo/config.toml that shadows one of these replaces the tool the
# report says ran -- and cargo announces it, in a warning that used to be
# discarded with the rest of the raw output.
readonly GATE_SUBCOMMANDS=" fmt clippy test doc deny audit llvm-cov "

detect_alias_shadowing() {  # label, log
    local name
    while IFS= read -r name; do
        case "$GATE_SUBCOMMANDS" in
            *" $name "*) printf '%s%s%s\n' "$1" "$SEP" "$name" >> "$work/shadowing.txt" ;;
        esac
    done < <(sed -n 's/.*user-defined alias `\([^`]*\)`.*shadowing an external subcommand.*/\1/p' \
        "$2" 2>/dev/null | LC_ALL=C sort -u)
}

step_exit=0
step_digest=""
step_duration=0
# The gate whose command is being run, set by `run_gate` before it dispatches.
# `run_step` needs it to file its ledger entry under the right gate; deriving it
# from the label would make the ledger depend on a display string.
current_gate=0

run_step() {  # runs one command in the worktree; sets step_exit/digest/duration
    local label="$1"; shift
    local started ended log
    step_digest="$(argv_digest "$@")"
    log="$work/step.log"
    say "--- $label"
    started="$(date +%s.%N)"
    ( cd "$wt" && "$@" ) 2>&1 | tee "$log"
    step_exit="${PIPESTATUS[0]}"
    ended="$(date +%s.%N)"
    step_duration="$(python3 -c "print(f'{float('$ended') - float('$started'):.3f}')")"
    detect_alias_shadowing "$label" "$log"
    # Written here, from `$step_exit`, which came from PIPESTATUS and not from
    # anything the dispatch chose. This is the honest path into the ledger, and
    # the ledger is the whole basis on which the report may later say a command
    # ran -- but it is not the only path into it, because `record_execution` is
    # callable from the dispatch directly. Its comment says what that costs.
    record_execution "$current_gate" "$label" "$step_digest" "$step_exit" "$step_duration"
    if [ -n "$log_dir" ]; then
        mkdir -p "$log_dir"
        cp "$log" "$log_dir/${label// /-}.log"
    fi
    say "--- $label exited $step_exit in ${step_duration}s"
}

declare -A GATE_STATUS=()

# What an arm of the dispatch may still set. `gate_digest` and `gate_duration`
# are deliberately absent: they were the two fields a short-circuited arm used
# to fabricate, and both are now derived from the execution ledger in `record`.
gate_status=""
gate_exit=0
gate_note=""
gate_config=""
gate_scripts="[]"
gate_inputs="[]"

simple_gate() {  # id, then the command
    local id="$1"; shift
    run_step "gate $id ($(gate_name "$id"))" "$@"
    gate_exit="$step_exit"
    [ "$gate_exit" -eq 0 ] && gate_status="pass" || gate_status="fail"
}

# The revision-controlled *data* each gate reads: the policy that decides it,
# as distinct from the script that applies the policy. Recording only the
# scripts left `"scripts": []` on gates 1, 5, 6 and 8, which a reader takes at
# face value as "nothing from the revision decided this" -- while deny.toml
# decides gate 5 outright. Every path here is also in the provenance comparison;
# this says which gate each one answers for.
gate_policy_inputs() {
    case "$1" in
        1)  files_json .rustfmt.toml rustfmt.toml ;;
        2)  # A member manifest's `[lints]` table overrides the workspace policy --
            # `[lints.clippy] needless_return = "allow"` in one crate takes the
            # gate from 101 to 0 -- so every member manifest decides this gate,
            # not only the workspace one.
            # shellcheck disable=SC2046
            files_json Cargo.toml $(git -C "$wt" ls-files -- '*/Cargo.toml') \
                clippy.toml .clippy.toml .cargo/config.toml .cargo/config ;;
        3)  files_json Cargo.toml .cargo/config.toml .cargo/config ;;
        4)  files_json Cargo.toml .cargo/config.toml .cargo/config ;;
        5)  files_json deny.toml ;;
        6)  files_json .cargo/audit.toml ;;
        7)  files_json config/quality-tools.json .cargo/config.toml .cargo/config ;;
        8)  files_json .github/actionlint.yaml .github/actionlint.yml ;;
        10) files_json config/quality-tools.json ;;
        12) files_json config/allowed-keys.asc ;;
        *)  printf '[]' ;;
    esac
}

workflow_count() {
    git -C "$wt" ls-files '.github/workflows/*.yml' '.github/workflows/*.yaml' | wc -l
}

run_gate() {
    local id="$1"
    current_gate="$id"
    gate_status=""; gate_exit=0; gate_note=""
    gate_config=""; gate_scripts="[]"; gate_inputs="$(gate_policy_inputs "$id")"
    case "$id" in
    1)  simple_gate 1 cargo fmt --all -- --check ;;
    2)  simple_gate 2 cargo clippy --locked --workspace --all-targets -- -D warnings ;;
    3)
        simple_gate 3 cargo test --locked --workspace
        # An excluded test must not disappear. The binaries are already built,
        # so listing the ignored ones costs almost nothing, and the report names
        # them. Whether each exclusion is justified and carries a tracking AR is
        # a review obligation; this only makes them impossible to overlook.
        run_step "gate 3 (excluded test inventory)" cargo test --locked --workspace -- --list --ignored
        if [ "$step_exit" -eq 0 ]; then
            grep -E '^[^ ]+: test$' "$work/step.log" | sed 's/: test$//' | sort -u > "$work/ignored.txt" || true
        else
            gate_status="fail"
            gate_exit="$step_exit"
            gate_note="the test run passed but the ignored-test inventory could not be listed"
        fi
        ;;
    4)  simple_gate 4 env RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps ;;
    5)  simple_gate 5 cargo deny --locked check ;;
    6)  simple_gate 6 cargo audit --deny warnings ;;
    7)
        run_step "gate 7 (coverage measurement)" \
            cargo llvm-cov --locked --workspace --summary-only --json --output-path "$work/coverage.json"
        if [ "$step_exit" -ne 0 ]; then
            gate_exit="$step_exit"; gate_status="fail"
            gate_note="cargo llvm-cov failed; no coverage numbers were produced"
        else
            gate_scripts="$(files_json tools/quality/check_coverage.py)"
            run_step "gate 7 (floor check)" \
                ./tools/quality/check_coverage.py --coverage-json "$work/coverage.json"
            gate_exit="$step_exit"
            case "$gate_exit" in
                0) gate_status="pass" ;;
                3) gate_status="not_armed"
                   gate_note="the manifest's coverage floors are null; AR-0009 sets them from measurement. Measured numbers are in the coverage section of this report and are not a pass." ;;
                *) gate_status="fail" ;;
            esac
        fi
        ;;
    8)
        if [ "$(workflow_count)" -eq 0 ]; then
            gate_status="not_applicable"; gate_exit=0
            gate_note="this revision contains no workflow file, so neither analyzer had an input. AR-0008 owns .github/workflows/; until it lands this gate cannot pass and is not counted as one."
        else
            local config=()
            [ -f "$wt/.github/actionlint.yaml" ] && config=(-config-file .github/actionlint.yaml)
            # `env -u SHELLCHECK_OPTS`, like gate 9's `env -u GITLEAKS_CONFIG`:
            # the shellcheck binary reads that variable for extra flags, so an
            # exported `SHELLCHECK_OPTS=-e SC2086` silences rules in the real
            # lint while the launcher's probe -- which fires on a different
            # rule -- stays green. Ambient environment must not decide what this
            # gate checked.
            run_step "gate 8 (actionlint)" env -u SHELLCHECK_OPTS \
                "$bin/actionlint-with-shellcheck" "${config[@]}"
            gate_exit="$step_exit"
            run_step "gate 8 (zizmor)" "$bin/zizmor" --pedantic .
            [ "$gate_exit" -eq 0 ] && gate_exit="$step_exit"
            [ "$gate_exit" -eq 0 ] && gate_status="pass" || gate_status="fail"
            # This is how the gate was configured, not why it did or did not
            # pass. Reported separately so that it can never be read out as a
            # failure reason.
            if [ "${#config[@]}" -eq 0 ]; then
                gate_config="no .github/actionlint.yaml in this revision; actionlint ran with its defaults"
            else
                gate_config="actionlint ran with .github/actionlint.yaml from this revision"
            fi
        fi
        ;;
    9)
        # gitleaks in `git` mode scans every ref it can see, and the throwaway
        # worktree shares the invoking repository's object store, so scanning
        # from inside it answers a question about the reviewer's machine rather
        # than about the revision. A token on an unrelated local branch fails an
        # innocent commit, and two reviewers with different local refs get
        # different reports -- and so different stable digests -- for the same
        # commit, which is precisely the reproducibility the stable digest
        # exists to assert. Demonstrated twice: a worktree whose entire history
        # is one commit, in a repository holding a token on a sibling branch,
        # reported `2 commits scanned` and one leak; and a reviewer's real run
        # failed this gate on a token that lived only on their own scratch
        # branch.
        #
        # So the scan gets a repository of its own containing this revision's
        # history and nothing else: an empty repository, one fetch of the
        # commit, HEAD pointed at it, and no refs at all. There `git log --all`
        # is the revision's ancestry by construction, whatever gitleaks chooses
        # to pass to git, and `--log-opts` names the same range a second time so
        # that neither mechanism is load-bearing alone. The isolation is checked
        # before the scan, and the gate fails closed if it did not take.
        local scan="$work/scan" expected_commits scan_commits scan_refs
        rm -rf "$scan"
        if ! ( git init -q "$scan" \
                && git -C "$scan" fetch -q --no-tags "$root" "$commit" \
                && git -C "$scan" update-ref --no-deref HEAD "$commit" ); then
            gate_status="fail"; gate_exit=2
            gate_note="the isolated scan repository could not be built, so a scan would have seen refs that are no part of this revision; refusing is the only honest result"
        else
            expected_commits="$(git -C "$root" rev-list --count "$commit")"
            scan_commits="$(git -C "$scan" rev-list --count --all)"
            scan_refs="$(git -C "$scan" for-each-ref | wc -l)"
            if [ "$scan_commits" != "$expected_commits" ] || [ "$scan_refs" -ne 0 ]; then
                gate_status="fail"; gate_exit=2
                gate_note="the scan repository holds $scan_commits commits and $scan_refs refs where this revision has $expected_commits commits and must have no other ref; the isolation did not take and the gate refuses to scan the wrong history"
            else
                # GITLEAKS_CONFIG is ambient, not revision-controlled, and it
                # replaces the rule set wholesale: pointed at a permissive file
                # it returns `no leaks found` on a leaky history while this
                # report claims the built-in rules ran. Unset for the gate, so
                # that the claim below is true rather than usually true.
                simple_gate 9 env -u GITLEAKS_CONFIG "$bin/gitleaks" git --redact --no-banner \
                    --log-opts="--full-history $commit" "$scan"
                gate_config="scanned in a repository holding this revision's $expected_commits commits and no other ref, so the answer is a function of the revision and not of the local refs of whoever ran it. gitleaks' built-in rules: that repository has no checkout, so a .gitleaks.toml in the revision is not read and cannot relax the scan."
            fi
        fi
        ;;
    10)
        gate_scripts="$(files_json tools/quality/install-external-tools.sh)"
        simple_gate 10 ./tools/quality/install-external-tools.sh --bin-dir "$bin"
        if [ "$gate_exit" -eq 0 ]; then
            local observed expected tool
            for tool in actionlint zizmor gitleaks; do
                expected="$(manifest_field external "$tool" version)"
                observed="$("$bin/$tool" --version 2>/dev/null | tr -d '\r' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
                if [ "$observed" != "$expected" ]; then
                    gate_status="fail"; gate_exit=1
                    gate_note="$tool reports $observed, the manifest pins $expected"
                fi
            done
        fi
        ;;
    11)
        # repository_policy.py does `from check_docs import check_links,
        # tracked_markdown`, so check_docs.py decides half this gate's verdict
        # and belongs in the record; privacy.py is never imported here and does
        # not. A provenance record that is wrong is worse than none.
        gate_scripts="$(files_json tools/quality/repository_policy.py tools/quality/check_docs.py)"
        simple_gate 11 ./tools/quality/repository_policy.py --base "$base_commit" --head "$commit"
        ;;
    12)
        gate_scripts="$(files_json tools/quality/check_commits.py tools/quality/privacy.py)"
        simple_gate 12 ./tools/quality/check_commits.py --base "$base_commit" --head "$commit"
        ;;
    13)
        gate_scripts="$(files_json tools/quality/test_failure_paths.py)"
        if [ -x "$wt/tools/quality/test_failure_paths.py" ]; then
            # The table goes to the run's throwaway directory and is copied out
            # afterwards, the way run_step already handles step logs. Naming
            # $log_dir in the argv would put it into gate 13's recorded digest,
            # and $log_dir is the flag a reviewer passes to obtain this very
            # table -- so the digest would vary across the runs that produce it.
            # Copying afterwards also fixes where a relative --log-dir lands: it
            # resolves against the invoking directory like every other artefact,
            # not against the throwaway worktree the gate command runs in, where
            # the table would be silently discarded with the run.
            simple_gate 13 ./tools/quality/test_failure_paths.py --bin-dir "$bin" \
                --json "$work/negative-fixture-table.json"
            if [ -n "$log_dir" ] && [ -f "$work/negative-fixture-table.json" ]; then
                mkdir -p "$log_dir"
                cp "$work/negative-fixture-table.json" "$log_dir/negative-fixture-table.json"
            fi
        else
            gate_status="not_implemented"; gate_exit=0
            gate_note="tools/quality/test_failure_paths.py does not exist in this revision. AR-0007 builds it; the gate is enumerated here so that its absence is visible and is not counted as a pass."
        fi
        ;;
    14)
        gate_scripts="$(files_json tools/quality/check_crate_table.py)"
        simple_gate 14 ./tools/quality/check_crate_table.py
        ;;
    15)
        gate_scripts="$(files_json tools/quality/check_docs.py)"
        simple_gate 15 ./tools/quality/check_docs.py index
        ;;
    *)  die "no such gate: $id" ;;
    esac
    GATE_STATUS[$id]="$gate_status"
    record "$id" "$gate_status" "$gate_exit" \
        "$gate_note" "$gate_config" "$gate_scripts" "$gate_inputs"
}

for id in "${GATE_ORDER[@]}"; do
    run_gate "$id"
done

# ------------------------------------------------------------------ report --

coverage_json="$work/coverage.json"
[ -f "$coverage_json" ] || coverage_json=""

analyzer_versions=""
for tool in actionlint zizmor gitleaks; do
    if [ -x "$bin/$tool" ]; then
        version="$("$bin/$tool" --version 2>/dev/null | tr -d '\r' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
        analyzer_versions+="$tool=$version"$'\n'
    fi
done

host_fingerprint="$(printf '%s' "$(uname -srm)$(hostname 2>/dev/null)$(nproc 2>/dev/null)" | sha256sum | cut -d' ' -f1)"

# ---------------------------------------------------------- second opinion --
#
# The gate checkers run from the revision under test, which is the only
# arrangement that makes the report a function of the revision -- and the price
# is that a revision is graded by its own instruments. The provenance digests
# tell a reader that the instruments moved. They cannot tell them whether the
# verdict moved with them, and answering that by reading the checker diff is
# work the runner can do first.
#
# So the four offline checkers are run a second time from the *base* commit,
# against this same revision, and any disagreement is reported. It costs tenths
# of a second: these four are pure Python over a git tree, with no build.
#
# Advisory, never blocking. A checker legitimately disagrees with its
# predecessor by getting stricter, by shedding a false positive, or by being
# weakened, and nothing here can tell those apart; that judgement is the
# reader's. What this removes is the reader having to notice the question.
second_opinion="$work/second-opinion.jsonl"
: > "$second_opinion"
opinion_root="$work/base-tools"
mkdir -p "$opinion_root"

base_checker_path() {  # basename -> its path at the base commit, or empty
    local hits
    hits="$(git -C "$root" ls-tree -r --name-only "$base_commit" \
        | grep -E "(^|/)$1\$" || true)"
    # Exactly one, or the answer is not well defined and the gate is reported
    # as unavailable rather than guessed at.
    [ "$(printf '%s\n' "$hits" | sed '/^$/d' | wc -l)" -eq 1 ] || return 0
    printf '%s' "$hits"
}

record_opinion() {  # id, revision status, base status, path, exit, base sum, revision sum, reason
    python3 - "$second_opinion" "$@" <<'SECOND'
import json, sys
path, gate, revision_status, base_status, base_path, code, base_sum, revision_sum, reason = \
    sys.argv[1:10]
# Only a verdict can agree or disagree with a verdict. `unavailable` and
# `error` are facts about the harness, not opinions about the revision, and
# reporting them as disagreements is how an advisory signal starts crying wolf.
verdict = base_status in ("pass", "fail")
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "id": int(gate),
        "revision_status": revision_status,
        "base_status": base_status,
        "base_path": base_path or None,
        "base_exit_code": int(code),
        "base_sha256": ("sha256:" + base_sum) if base_sum else None,
        "revision_sha256": ("sha256:" + revision_sum) if revision_sum else None,
        # A base checker byte-identical to the revision's cannot disagree with
        # it. Saying so is the difference between a comparison and a ritual.
        "identical_to_revision": (
            bool(base_sum) and bool(revision_sum) and base_sum == revision_sum
        ),
        "agrees": (base_status == revision_status) if verdict else None,
        "reason": reason or None,
    }) + "\n")
SECOND
}

opinion_paths=()
declare -A OPINION_OF=()
declare -A OPINION_REVISION_OF=()
for spec in "11:repository_policy.py" "12:check_commits.py" \
            "14:check_crate_table.py" "15:check_docs.py"; do
    found="$(base_checker_path "${spec#*:}")"
    OPINION_OF["${spec%%:*}"]="$found"
    # The revision's copy is the file the gate executed, which is not
    # necessarily where the base kept it: this branch moved check_docs.py out of
    # docs/, and looking for the revision's copy at the base's path found
    # nothing and reported a pure rename as a checker that had changed.
    OPINION_REVISION_OF["${spec%%:*}"]="tools/quality/${spec#*:}"
    [ -n "$found" ] && opinion_paths+=("$found")
done
if [ "${#opinion_paths[@]}" -gt 0 ]; then
    # Whole parent directories, so that a checker importing a sibling still can.
    mapfile -t opinion_dirs < <(printf '%s\n' "${opinion_paths[@]}" \
        | xargs -r -n1 dirname | LC_ALL=C sort -u)
    git -C "$root" archive "$base_commit" -- "${opinion_dirs[@]}" 2>/dev/null \
        | tar -x -C "$opinion_root" 2>/dev/null || true
fi

opinion_gate() {  # id, then any positional argument the gate passes
    local id="$1"; shift
    local rel="${OPINION_OF[$id]}" file help options status reason="" exit_code=0
    local base_sum="" rev_sum="" log="$work/opinion.log"
    local args=("$@")
    if [ -z "$rel" ] || [ ! -f "$opinion_root/$rel" ]; then
        record_opinion "$id" "${GATE_STATUS[$id]}" unavailable "" 0 "" "" \
            "the base commit carries no checker for this gate"
        return
    fi
    file="$opinion_root/$rel"
    base_sum="$(sha256sum "$file" | cut -d' ' -f1)"
    local mine="${OPINION_REVISION_OF[$id]}"
    [ -f "$wt/$mine" ] && rev_sum="$(sha256sum "$wt/$mine" | cut -d' ' -f1)"

    # The base's checker is invoked with the options *it* declares. Those used
    # to be discovered by grepping the file, which matches prose: a `--base`
    # named only in a docstring made this harness pass an option the script
    # rejects, and argparse's exit 2 was then recorded as a failing verdict and
    # printed as a disagreement on a passing gate. Ask the checker instead, and
    # read only the option lines of its help, never its description.
    if ! help="$( ( cd "$wt" && python3 "$file" --help ) 2>&1 )"; then
        record_opinion "$id" "${GATE_STATUS[$id]}" error "$rel" 0 "$base_sum" "$rev_sum" \
            "the base commit's checker does not answer --help, so its interface is unknown"
        return
    fi
    options="$(printf '%s\n' "$help" | grep -E '^[[:space:]]+-' || true)"
    case "$options" in *--base*) args+=(--base "$base_commit") ;; esac
    case "$options" in *--head*) args+=(--head "$commit") ;; esac
    case "$options" in *--repo*) args+=(--repo "$wt") ;; esac

    ( cd "$wt" && python3 "$file" "${args[@]}" ) > "$log" 2>&1
    exit_code=$?
    # These checkers document 0 for pass, 1 for findings and 2 for an
    # environment error. Only 0 and 1 are verdicts; anything else is this
    # harness failing to drive the checker, and must not read as a
    # disagreement.
    case "$exit_code" in
        0) status="pass" ;;
        1) status="fail" ;;
        *) status="error"
           if grep -qE 'unrecognized arguments|invalid choice|^usage:' "$log"; then
               reason="the base commit's checker rejected an argument this harness passed, so it never reached a verdict"
           else
               reason="the base commit's checker exited $exit_code without a verdict; these checkers use exit 2 for an environment error"
           fi ;;
    esac
    record_opinion "$id" "${GATE_STATUS[$id]}" "$status" "$rel" "$exit_code" \
        "$base_sum" "$rev_sum" "$reason"
}

say "--- second opinion: the base commit's checkers, against this revision"
opinion_gate 11
opinion_gate 12
opinion_gate 14
opinion_gate 15 index

# --------------------------------------------------------------- provenance --
#
# Who wrote the checkers that just ran.
#
# Gates 7 and 10 to 15 execute a script out of the throwaway worktree, that is,
# out of the revision under test; `build_report.py` and `privacy.py` execute
# from the checkout the runner was invoked from. That asymmetry is deliberate
# and is the only arrangement under which the report is a function of the
# revision alone -- run the checkers from the invoking checkout instead and the
# same commit yields different answers for two reviewers whose tooling differs,
# which is exactly the property the stable digest exists to assert. It also
# means a revision that guts its own checkers is graded by the gutted ones.
#
# So the runner does not hide the boundary, it publishes it: a per-file digest
# of every gate-deciding input on each side -- the checkers and the policies
# they read, hashed from the bytes in each worktree rather than from the blob
# ids in each index -- and the sha256 of every checker file actually executed.
# When the two sides differ the run is not self-certifying and says so as a
# blocking reason naming the paths, because the reviewer then has to read that
# diff themselves before any `pass` in this report means anything. The digests
# are computed in `build_report.py`, which is where the compared set is listed
# and justified path by path.
#
# One thing that comparison cannot do, said here as well as in the report: the
# workflow this runner prescribes -- invoke it from a checkout of the head under
# review -- makes the two sides the same tree, so `quality_tooling_matches` is
# true for an attacking revision as readily as for an honest one. It catches the
# reviewer who ran from somewhere else. In the prescribed case reading the diff
# of those paths is the whole obligation, and nothing here discharges it.
invoking_commit="$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)"

# The coverage floors quoted in the report must be the floors `check_coverage.py`
# actually compared against, and that script runs from the worktree.
report_manifest="$wt/config/quality-tools.json"
[ -f "$report_manifest" ] || report_manifest="$manifest"

GATE_RECORDS="$records" \
GATE_EXECUTED="$executed" \
GATE_COVERAGE="$coverage_json" \
GATE_IGNORED="$work/ignored.txt" \
GATE_DEVICE="$device_json" \
GATE_ANALYZERS="$analyzer_versions" \
GATE_COMMIT="$commit" \
GATE_BASE_REF="$base_ref" \
GATE_BASE_COMMIT="$base_commit" \
GATE_TOOLCHAIN="$toolchain" \
GATE_RUSTC="$rustc_version" \
GATE_CARGO="$cargo_version" \
GATE_DENY="$deny_version" \
GATE_AUDIT="$audit_version" \
GATE_LLVMCOV="$llvmcov_version" \
GATE_HOST="$host_fingerprint" \
GATE_MANIFEST="$report_manifest" \
GATE_INVOKING_COMMIT="$invoking_commit" \
GATE_ROOT="$root" \
GATE_WORKTREE="$wt" \
GATE_SECOND_OPINION="$second_opinion" \
GATE_SHADOWING="$(LC_ALL=C sort -u "$work/shadowing.txt" 2>/dev/null || true)" \
GATE_ORDER_LIST="${GATE_ORDER[*]}" \
GATE_REPORT="$report_path" \
python3 "$root/tools/quality/build_report.py"
report_build=$?
case "$report_build" in
    0) ;;
    # 3 is build_report.py saying a gate record is not supported by the
    # execution ledger. The report is written, names each one, and is not
    # green; the run continues to the summary so the reviewer sees it.
    3) say "a gate record is not supported by the execution ledger; the report names each one" ;;
    *) die "cannot write the report" ;;
esac

python3 "$root/tools/quality/privacy.py" "$report_path" >/dev/null \
    || die "the report itself trips the privacy rules and was not accepted"

digest="$(sha256sum "$report_path" | cut -d' ' -f1)"
stable="$(python3 -c '
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(report["stable_digest"])
' "$report_path")"

printf '\n'
python3 - "$report_path" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
width = max(len(g["name"]) for g in report["gates"])
status_width = max(len(g["status"]) for g in report["gates"]) + 2
for gate in sorted(report["gates"], key=lambda g: g["id"]):
    print(f"  gate {gate['id']:>2}  {gate['name']:<{width}}  {gate['status']:<{status_width}}"
          f"exit {gate['exit_code']:<4}{gate['duration_seconds']:>9.2f}s")
summary = report["summary"]
print()
for line in report["second_opinion"]["disagreements"]:
    print(f"  advisory: {line}")
for line in report["second_opinion"]["not_compared"]:
    print(f"  advisory: not compared, {line}")
for reason in report["blocking"]:
    print(f"  blocking: {reason}")
print(f"\n  result: {summary['result']}  "
      + "  ".join(f"{k}={v}" for k, v in summary["counts"].items()))
PY
printf '\nreport: %s\n' "$report_path"
printf 'report sha256 (file):   %s\n' "$digest"
printf 'report sha256 (stable): %s\n' "$stable"
printf '  the stable digest omits the wall-clock fields, so two runs of the same\n'
printf '  revision on this host must print the same value.\n'

result="$(python3 -c '
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["summary"]["result"])
' "$report_path")"
[ "$result" = "pass" ] && exit 0 || exit 1
