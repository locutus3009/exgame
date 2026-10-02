#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Differential fuzz of the transaction gate against a real draft 2020-12 validator.

The unit suite proves a fixture per constraint the published schemas state. That proves each
bound the fixture list names is enforced; it cannot prove the list is complete, which is the
failure mode this whole change exists to close -- a bound nobody thought to restate. This
harness closes the gap from the other side: it generates documents, puts each to both
``schema_object_errors`` and a real ``Draft202012Validator``, and reports every disagreement.

The direction that matters is *schema refuses and the transaction accepts*: that is state a
transition could commit and a later gate reject. The reverse is reported too, because the
transaction refusing what the schema allows is a divergence as well, just a safe one.

Two generators, because they miss different things. The mutation generator perturbs a valid
document and explores values; the structure-free generator builds documents from random key
subsets and never sees a valid one, which is what finds faults in required and
additionalProperties handling rather than in per-field bounds.

Not part of the gate sequence: it is slow and randomised. Run it when the schemas or the
enforcement change.

    python tests/fuzz_schema_parity.py [rounds]
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("handoffctl_fuzz", ROOT / "tools/handoffctl.py")
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import shape is fixed
    raise SystemExit("cannot load handoffctl")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules["handoffctl_fuzz"] = CORE
SPEC.loader.exec_module(CORE)

# Values chosen to sit on both sides of every bound the published schemas state: the length
# limits, the anchored and unanchored patterns, the enum members, the date-time format, and
# the boolean/integer and integer/float distinctions JSON Schema draws and Python does not.
VALUES: tuple[Any, ...] = (
    None,
    True,
    False,
    0,
    1,
    -1,
    2,
    1.0,
    1.5,
    "",
    " ",
    "  x",
    "\t",
    "x",
    "x" * 20,
    "x" * 80,
    "x" * 81,
    "x" * 120,
    "x" * 121,
    "x" * 160,
    "x" * 161,
    "x" * 300,
    "x" * 301,
    "AR-0001",
    "AR-1",
    "ar-0001",
    "M0",
    "M99",
    "M007",
    "M100",
    "open",
    "in_progress",
    "in_review",
    "done",
    "active",
    "complete",
    "retired",
    "bogus",
    "P0",
    "P9",
    "2026-09-09T00:00:00+00:00",
    "2026-09-09T00:00:00Z",
    "2026-09-09t00:00:00z",
    "2026-09-09",
    "2026-09-09 00:00:00+00:00",
    "2026-02-30T00:00:00Z",
    "2026-09-09T00:00:00.123+00:00",
    "2026-09-09T00:00:00",
    "a" * 40,
    "0" * 40,
    "z" * 40,
    "game-experiment",
    "game-experiment-x",
    "Game-Experiment",
    "../plans/AR-0001.md",
    "../plans/x.md",
    [],
    ["AR-0001"],
    ["AR-0001", "AR-0001"],
    ["x"],
    [1],
    [True],
    {},
    {"a": 1},
)

TASK_BASE: dict[str, Any] = {
    "schema_version": 1,
    "id": "AR-0001",
    "title": "t",
    "status": "open",
    "priority": "P1",
    "summary": "s",
    "next_action": "n",
    "task_revision": 1,
    "updated_at": "2026-09-09T00:00:00+00:00",
    "owner": "",
    "claim_expires": "",
    "worktree_key": "AR-0001",
    "branch": "b",
    "checkpoint_commit": "",
    "plan": "",
    "depends_on": [],
}
MILESTONE_BASE: dict[str, Any] = {
    "schema_version": 1,
    "id": "M0",
    "label": "L",
    "status": "active",
}


SEEDS = 4
# A run that generates almost nothing proves nothing. The floor is far below a useful run and
# far above what a broken generator produces, so it separates "no holes" from "no documents".
MINIMUM_DOCUMENTS = 1000


def mutated(rng: random.Random, base: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Perturb a valid document, so values are explored around every stated bound."""
    doc = dict(base)
    for _ in range(rng.randint(1, 4)):
        field = rng.choice(fields)
        if rng.random() < 0.12:
            doc.pop(field, None)
        else:
            doc[field] = rng.choice(VALUES)
    return doc


def structure_free(rng: random.Random, _base: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Build a document from a random subset of keys, never starting from a valid one.

    A mutation generator almost always holds the required fields and rarely adds an unknown
    one, so it exercises per-field bounds far more than it exercises `required` and
    `additionalProperties`. This one does the opposite.
    """
    chosen = rng.sample(fields, rng.randint(0, len(fields)))
    return {field: rng.choice(VALUES) for field in chosen}


GENERATORS = (mutated, structure_free)


def disagreements(schema_name: str, base: dict[str, Any], rounds: int) -> tuple[int, int, int]:
    """Return documents fuzzed, parity holes, and over-strict refusals for one schema."""
    schema = json.loads((ROOT / "schema" / schema_name).read_text())
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    fields = [*schema["properties"], "bogus_extra"]
    total = holes = strict = 0
    for index, generator in enumerate(GENERATORS):
        for seed in range(SEEDS):
            # A literal seed, not hash(): string hashing is salted per process, so a hash
            # seed would make a reported disagreement irreproducible on the next run.
            rng = random.Random(len(schema_name) * 1000 + index * 100 + seed)  # noqa: S311
            for _ in range(rounds):
                doc = generator(rng, base, fields)
                total += 1
                refused_by_schema = bool(list(validator.iter_errors(doc)))
                refused_by_gate = bool(CORE.schema_object_errors("d", doc, schema))
                if refused_by_schema and not refused_by_gate:
                    holes += 1
                    print(f"  HOLE {schema_name}: {json.dumps(doc, sort_keys=True)[:200]}")
                elif refused_by_gate and not refused_by_schema:
                    strict += 1
                    reported = CORE.schema_object_errors("d", doc, schema)[:1]
                    print(f"  STRICT {schema_name}: {reported}")
    return total, holes, strict


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
    total = holes = strict = 0
    schemas = (("task-schema.json", TASK_BASE), ("milestone-schema.json", MILESTONE_BASE))
    for name, base in schemas:
        counted, found, over = disagreements(name, base, rounds)
        total += counted
        holes += found
        strict += over
    expected = len(schemas) * len(GENERATORS) * SEEDS * rounds
    print(f"documents fuzzed: {total}")
    print(f"schema refuses and transaction accepts (parity holes): {holes}")
    print(f"transaction refuses and schema accepts (over-strict): {strict}")
    if total != expected or total < MINIMUM_DOCUMENTS:
        # Exit 0 must mean "the generators ran and found no hole", never "nothing ran". The
        # harness is evidence for a completeness claim, and evidence that can be satisfied by
        # generating nothing is the same fault class as a bound that is silently unenforced.
        print(
            f"FAIL: generated {total} documents, "
            f"expected {expected} and at least {MINIMUM_DOCUMENTS}"
        )
        return 1
    # Over-strict refusals are expected and bounded: draft 2020-12 counts a number with zero
    # fractional part as an integer and the transaction does not, for task_revision and
    # observed_dirty. That difference is deliberate, documented in PROVENANCE.md and pinned by
    # a unit test. A parity hole is never acceptable.
    return 1 if holes else 0


if __name__ == "__main__":
    raise SystemExit(main())
