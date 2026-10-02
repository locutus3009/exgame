#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Gate 10: digest-verified installation of the external analyzers.
#
# Downloads actionlint, shellcheck, zizmor and gitleaks from the release assets
# named in config/quality-tools.json, verifies each archive against the sha256
# recorded there *before* extracting it, and installs the binaries into
# throwaway storage. Prints the directory holding them on stdout; everything
# else goes to stderr, so the caller can write
# `bin_dir="$(install-external-tools.sh)"`.
#
# The shellcheck pin is here for gate 8, not for a gate of its own. actionlint
# lints the shell inside every workflow `run:` block by delegating to it, and when
# that binary is not on PATH it disables the delegation *silently*: no warning,
# no note, exit 0. Gate 8 then reports a pass over shell nothing read. Pinning
# the binary is only half the repair, because a pin cannot prove the delegation
# happened -- so this script also installs `actionlint-with-shellcheck`, the
# launcher gate 8 invokes, and self-tests it here: it lints a workflow whose
# `run:` block carries a known shellcheck defect and requires the diagnostic
# back. Nothing that leaves this script can lint shell silently; if the probe
# does not fire, the install fails and no launcher is published.
#
# Fails closed. A digest mismatch aborts the run and is never a warning: an
# archive whose hash does not match the manifest is never unpacked and never
# executed. A missing archive, an unreachable release, an unknown architecture
# or a binary that will not report its version are all errors too.
#
# Usage:
#   install-external-tools.sh --bin-dir DIR [--cache-dir DIR] [--manifest FILE]
#
# --bin-dir     where the binaries are installed (created if absent). When
#               omitted a fresh mktemp directory is used and its path printed;
#               the caller owns removing it.
# --cache-dir   where downloaded archives are kept between runs. The digest is
#               re-verified on every run, cache hit or not, so a poisoned cache
#               entry fails exactly like a poisoned download.
set -euo pipefail

manifest=""
bin_dir=""
cache_dir="${GATE_TOOL_CACHE:-}"

die() { printf 'install-external-tools: %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bin-dir)   bin_dir="${2:?--bin-dir needs a value}"; shift 2 ;;
        --cache-dir) cache_dir="${2:?--cache-dir needs a value}"; shift 2 ;;
        --manifest)  manifest="${2:?--manifest needs a value}"; shift 2 ;;
        -h|--help)   sed -n '3,26p' "$0"; exit 0 ;;
        *)           die "unknown argument: $1" ;;
    esac
done

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(git -C "$here" rev-parse --show-toplevel)"
[ -n "$manifest" ] || manifest="$root/config/quality-tools.json"
[ -f "$manifest" ] || die "no manifest at the expected path"

# `xz` is needed because shellcheck ships .tar.xz where the other three ship
# .tar.gz; tar detects the compression, but only if it can find the decompressor.
for required in curl tar xz sha256sum python3; do
    command -v "$required" >/dev/null 2>&1 || die "missing required command: $required"
done

case "$(uname -s)/$(uname -m)" in
    Linux/x86_64)          platform="linux_x86_64" ;;
    Linux/aarch64|Linux/arm64) platform="linux_aarch64" ;;
    *) die "no manifest entry for this platform: $(uname -s)/$(uname -m)" ;;
esac

if [ -z "$bin_dir" ]; then
    bin_dir="$(mktemp -d "${TMPDIR:-/tmp}/gate-analyzers-XXXXXX")"
fi
mkdir -p "$bin_dir"
if [ -z "$cache_dir" ]; then
    cache_dir="${XDG_CACHE_HOME:-$HOME/.cache}/game-experiment-gate-tools"
fi
mkdir -p "$cache_dir"

# Fail closed, including part-way. A run that dies after actionlint is in place
# but before shellcheck is would otherwise leave behind exactly the arrangement
# this script exists to prevent: an actionlint that lints workflows and skips
# their shell without saying so. So a non-zero exit takes both the launcher and
# actionlint out of the bin directory, and gate 8 fails on a missing program
# rather than passing on an unlinted one.
#
# The EXIT trap alone does not run when the shell is killed by a signal, so an
# interrupted install was the one way to leave a bare actionlint behind. INT and
# TERM therefore re-raise as an exit with the conventional 128+signal status:
# non-zero, so the handler below removes both files, and still an exit, so the
# run stops instead of carrying on inside a handler that ignored the signal.
on_failure() {
    local status=$?
    if [ "$status" -ne 0 ]; then
        rm -f "$bin_dir/actionlint-with-shellcheck" "$bin_dir/actionlint"
    fi
    return "$status"
}
trap on_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Read one field out of the manifest without adding a jq dependency.
field() {
    python3 - "$manifest" "$@" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2:]:
    data = data[key]
print(data)
PY
}

install_one() {
    local tool="$1"
    local version repository asset sha url archive actual work found observed

    version="$(field external "$tool" version)"
    repository="$(field external "$tool" repository)"
    asset="$(field external "$tool" "$platform" asset)"
    sha="$(field external "$tool" "$platform" sha256)"

    case "$sha" in
        [0-9a-f]*) [ "${#sha}" -eq 64 ] || die "$tool: manifest sha256 is not 64 hex characters" ;;
        *) die "$tool: manifest sha256 is not lowercase hex" ;;
    esac

    url="https://github.com/$repository/releases/download/v$version/$asset"
    archive="$cache_dir/$tool-$version-$platform-$asset"

    if [ ! -f "$archive" ]; then
        printf 'install-external-tools: downloading %s %s\n' "$tool" "$version" >&2
        curl --fail --silent --show-error --location --proto '=https' --tlsv1.2 \
             --max-time 300 --output "$archive.part" "$url" \
            || { rm -f "$archive.part"; die "$tool: download failed"; }
        mv "$archive.part" "$archive"
    fi

    # Verify before extraction. Nothing below this line runs on an archive
    # whose hash does not match the manifest.
    actual="$(sha256sum "$archive" | cut -d' ' -f1)"
    if [ "$actual" != "$sha" ]; then
        rm -f "$archive"
        die "$tool: sha256 mismatch for $asset (manifest $sha, downloaded $actual) - refusing to extract"
    fi
    printf 'install-external-tools: %s %s digest verified\n' "$tool" "$version" >&2

    work="$(mktemp -d "$bin_dir/.unpack-XXXXXX")"
    # `-xf`, not `-xzf`: the compression is a property of the asset, not an
    # assumption this script gets to make. shellcheck ships .tar.xz.
    tar -xf "$archive" -C "$work" || { rm -rf "$work"; die "$tool: extraction failed"; }
    found="$(find "$work" -type f -name "$tool" -perm -u+x -print -quit)"
    [ -n "$found" ] || { rm -rf "$work"; die "$tool: no executable named $tool inside $asset"; }
    mv "$found" "$bin_dir/$tool"
    chmod 0755 "$bin_dir/$tool"
    rm -rf "$work"

    # The version is asserted here and not only in the runner's gate 10 check,
    # which iterates a list this script does not own. An unasserted version is
    # how a manifest pin degrades into a decoration.
    observed="$("$bin_dir/$tool" --version 2>&1 | tr -d '\r' \
        | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    [ "$observed" = "$version" ] \
        || die "$tool: installed binary reports '$observed', the manifest pins '$version'"
}

# --------------------------------------------------- gate 8's shell linter --
#
# Written into the bin directory rather than tracked, because it is meaningless
# outside one: it exists to bind actionlint to the shellcheck installed beside
# it. It is published only after every binary above is verified, so a failed
# install leaves gate 8 with no program to run instead of an unguarded one.
install_launcher() {
    cat > "$bin_dir/actionlint-with-shellcheck" <<'LAUNCHER'
#!/usr/bin/env bash
# Gate 8's shell linter: actionlint, with the pinned shellcheck beside it on
# PATH, and a refusal to lint anything if that delegation is not working.
#
# actionlint lints the shell in every `run:` block by handing it to shellcheck,
# and disables that rule without a diagnostic when the binary is not on PATH:
# no warning, no note, exit 0. A gate built on the bare binary therefore reports
# a pass over shell it never read, and the pass looks exactly like an honest
# one. Checking that a file called `shellcheck` exists is not enough either --
# it says nothing about whether actionlint used it.
#
# So this proves the delegation instead of assuming it. Before linting the
# caller's tree it lints a workflow whose `run:` block carries a defect that
# only shellcheck reports, and requires the diagnostic back. If it does not, the
# integration is inert -- absent, unexecutable, incompatible, or disabled by
# a future flag change -- and this exits 3 with a message saying that shell
# linting did not run. It never falls back to linting without it.
set -uo pipefail

bin_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$bin_dir:$PATH"

probe() {
    local dir output status
    dir="$(mktemp -d "${TMPDIR:-/tmp}/gate8-shellcheck-probe-XXXXXX")" || return 2
    mkdir -p "$dir/.github/workflows" || { rm -rf "$dir"; return 2; }
    # `shell: bash` is deliberate: it fixes the dialect so the probe does not
    # depend on how actionlint infers a shell. The defect is SC2086 on a
    # variable the script never assigns, so shellcheck's own dataflow analysis
    # cannot decide the expansion is safe and drop the diagnostic.
    cat > "$dir/.github/workflows/probe.yaml" <<'PROBE'
name: probe
on: workflow_dispatch
permissions: {}
jobs:
  probe:
    runs-on: ubuntu-latest
    steps:
      - run: |
          echo $PROBE_UNQUOTED_EXPANSION
        shell: bash
PROBE
    output="$("$bin_dir/actionlint" -no-color "$dir/.github/workflows/probe.yaml" 2>&1)"
    status=$?
    rm -rf "$dir"
    [ "$status" -eq 1 ] || return 1
    case "$output" in
        *SC2086*) return 0 ;;
    esac
    return 1
}

if [ "${1:-}" = "--self-test" ]; then
    if probe; then
        printf 'actionlint-with-shellcheck: shellcheck delegation verified\n' >&2
        exit 0
    fi
    printf 'actionlint-with-shellcheck: %s\n' \
        'the probe defect came back unreported, so actionlint is not delegating to shellcheck' >&2
    exit 3
fi

if ! probe; then
    printf 'actionlint-with-shellcheck: %s\n' \
        'SHELL LINTING DID NOT RUN. actionlint did not report the probe shell defect, so its shellcheck delegation is inert. Refusing to lint this revision: a pass here would certify shell that nothing read.' >&2
    exit 3
fi

exec "$bin_dir/actionlint" "$@"
LAUNCHER
    chmod 0755 "$bin_dir/actionlint-with-shellcheck"
}

for tool in shellcheck actionlint zizmor gitleaks; do
    install_one "$tool"
done

install_launcher
"$bin_dir/actionlint-with-shellcheck" --self-test \
    || die "actionlint does not report a shell defect its shellcheck delegation should catch; gate 8 would pass over unlinted shell and this install refuses to hand it a linter that cannot lint"

printf '%s\n' "$bin_dir"
