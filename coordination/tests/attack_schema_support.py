#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Attack the support net: hostile constructs at every position of the published schemas.

``tests/fuzz_schema_parity.py`` fuzzes *documents* against a fixed schema. It cannot find the
fault this harness exists for, because that fault is in the *schema* direction: a construct
draft 2020-12 permits, that this reader neither enforces nor refuses, silently loses a bound
the published schema states -- which is the original divergence restored. The second shape of
the same fault is a construct that raises out of ``validate()`` instead of refusing, which
fences every transaction including ``release`` and ``expire``.

So this harness mutates the published schemas at every position the reader walks -- the
document object, each ``properties`` entry, each ``items`` entry, each ``allOf`` branch and
its ``if``/``then``/``else`` -- with keywords the reader does not implement, with booleans and
other non-objects where draft 2020-12 permits a subschema, and with bounds whose *value* is
not the shape the checker reads. Each mutant is put to both gates over the repository's real
documents, and a mutant is a fault unless the transaction either refuses it or agrees with a
real ``Draft202012Validator`` about every document.

    python tests/attack_schema_support.py [root]

Exit 1 names every fault. Not part of the gate sequence: it needs the quality environment.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("handoffctl_attack", ROOT / "tools/handoffctl.py")
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import shape is fixed
    raise SystemExit("cannot load handoffctl")
CORE = importlib.util.module_from_spec(SPEC)
sys.modules["handoffctl_attack"] = CORE
SPEC.loader.exec_module(CORE)

# Legal draft 2020-12 keywords this reader does not implement. Each one states a real bound,
# so silently ignoring any of them loses it.
UNIMPLEMENTED: tuple[tuple[str, Any], ...] = (
    ("not", {"type": "string"}),
    ("oneOf", [{"type": "string"}]),
    ("anyOf", [{"type": "string"}]),
    ("$ref", "#/$defs/x"),
    ("$defs", {"x": {"type": "string"}}),
    ("patternProperties", {"^x": {"type": "string"}}),
    ("prefixItems", [{"type": "string"}]),
    ("contains", {"type": "string"}),
    ("minItems", 5),
    ("maxItems", 0),
    ("multipleOf", 3),
    ("exclusiveMinimum", 0),
    ("maximum", 1),
    ("dependentRequired", {"id": ["owner"]}),
    ("dependentSchemas", {"id": {"required": ["zz"]}}),
    ("propertyNames", {"maxLength": 1}),
    ("unevaluatedProperties", False),
    ("minProperties", 99),
    ("maxProperties", 1),
    ("contentEncoding", "base64"),
    ("if", {"required": ["zz"]}),
    ("then", {"required": ["zz"]}),
    ("else", {"required": ["zz"]}),
)
# Draft 2020-12 permits a boolean wherever a subschema may appear; the rest are simply not
# schemas, and a reader that walks them raises rather than refusing.
NON_OBJECT: tuple[Any, ...] = (False, True, [], "x", 3, None)
# Bounds whose keyword is implemented but whose value the checker cannot read: silently
# unenforced, or an exception from inside the transaction.
BAD_BOUNDS: tuple[tuple[str, Any], ...] = (
    ("maxLength", "300"),
    ("maxLength", True),
    ("maxLength", -1),
    ("maxLength", 3.5),
    ("minLength", "1"),
    ("minimum", "3"),
    ("minimum", True),
    ("pattern", "["),
    ("pattern", 5),
    ("pattern", ["^a"]),
    ("uniqueItems", "yes"),
    ("uniqueItems", 1),
    ("enum", {"a": 1}),
    ("enum", "abc"),
    ("type", ["string", "null"]),
    ("type", "number"),
    ("type", True),
    ("format", "uri"),
    ("format", 1),
)


def walk(node: Any, pointer: str) -> Any:
    """Return the node one slash-separated pointer names."""
    for step in [part for part in pointer.split("/") if part]:
        node = node[int(step)] if step.isdigit() else node[step]
    return node


def place(schema: Any, pointer: str, value: Any) -> Any:
    """Return a copy of the schema with one position replaced."""
    mutant = copy.deepcopy(schema)
    steps = [part for part in pointer.split("/") if part]
    if not steps:
        return value
    parent = walk(mutant, "/".join(steps[:-1]))
    last = steps[-1]
    parent[int(last) if last.isdigit() else last] = value
    return mutant


def positions(schema: dict[str, Any]) -> list[tuple[str, str]]:
    """Return every position the reader walks, as (kind, pointer)."""
    found: list[tuple[str, str]] = [("object", "")]
    for field, rules in schema.get("properties", {}).items():
        found.append(("field", f"properties/{field}"))
        if isinstance(rules, dict) and "items" in rules:
            found.append(("field", f"properties/{field}/items"))
    for index, branch in enumerate(schema.get("allOf", [])):
        found.append(("branch", f"allOf/{index}"))
        found.extend(
            ("object", f"allOf/{index}/{name}") for name in ("if", "then", "else") if name in branch
        )
    return found


def mutants(schema: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return every (description, mutated schema) this attack puts to both gates."""
    built: list[tuple[str, Any]] = []
    for kind, pointer in positions(schema):
        at = pointer or "root"
        built.extend((f"{at} = {value!r}", place(schema, pointer, value)) for value in NON_OBJECT)
        node = walk(schema, pointer)
        built.extend(
            (f"{at} + {name}", place(schema, pointer, {**node, name: value}))
            for name, value in UNIMPLEMENTED
            if name not in node
        )
        if kind == "field":
            built.extend(
                (f"{at} {name} = {value!r}", place(schema, pointer, {**node, name: value}))
                for name, value in BAD_BOUNDS
            )
        if kind == "object":
            built.extend(container_mutants(schema, pointer, node))
    return built


def container_mutants(schema: dict[str, Any], pointer: str, node: Any) -> list[tuple[str, Any]]:
    """Return mutants of the containers the enforcer walks, which must not raise either."""
    at = pointer or "root"
    built: list[tuple[str, Any]] = [
        (f"{at} {name} = {value!r}", place(schema, pointer, {**node, name: value}))
        for name in ("properties", "required", "allOf")
        for value in NON_OBJECT
    ]
    built.append((f"{at} required = [1]", place(schema, pointer, {**node, "required": [1]})))
    built.extend(
        (f"{at} type = {value!r}", place(schema, pointer, {**node, "type": value}))
        for value in ("array", "string", True, ["object", "null"])
    )
    built.append(
        (
            f"{at} additionalProperties = schema",
            place(schema, pointer, {**node, "additionalProperties": {"type": "string"}}),
        )
    )
    return built


def transaction(schema: Any, documents: list[tuple[str, dict[str, Any]]]) -> tuple[str, list[str]]:
    """Put one mutant to the transaction gate, treating an exception as a verdict of its own."""
    try:
        support = CORE.schema_support_errors("s", schema)
        if support:
            return "refused", support
        errors: list[str] = []
        for name, meta in documents:
            errors.extend(CORE.schema_object_errors(name, meta, schema))
        return ("errors" if errors else "clean"), errors
    except Exception as error:  # the crash is exactly what is being hunted
        return "crash", [f"{type(error).__name__}: {error}"]


def published(schema: Any, documents: list[tuple[str, dict[str, Any]]]) -> str:
    """Put one mutant to a real validator: does it refuse a document, or refuse to load?"""
    try:
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        refused = any(list(validator.iter_errors(meta)) for _, meta in documents)
    except Exception:  # an unloadable schema is a refusal of its own
        return "unloadable"
    return "refuses" if refused else "accepts"


def attack(name: str, documents: list[tuple[str, dict[str, Any]]]) -> tuple[int, list[str]]:
    """Return the mutant count and every fault, for one published schema."""
    faults: list[str] = []
    built = mutants(CORE.schema_document(name))
    for label, mutant in built:
        verdict, reported = transaction(mutant, documents)
        real = published(mutant, documents)
        if verdict == "crash":
            faults.append(f"CRASH  {name} {label}: {reported[0]}")
        elif verdict == "clean" and real != "accepts":
            faults.append(f"SILENT {name} {label}: validator {real}, transaction clean")
    return len(built), faults


def main() -> int:
    tasks = [(path.name, meta) for path, meta, _ in CORE.all_tasks()]
    milestones = [(path.name, meta) for path, meta, _ in CORE.milestone_documents()]
    total = 0
    faults: list[str] = []
    for name, documents in (("task-schema.json", tasks), ("milestone-schema.json", milestones)):
        counted, found = attack(name, documents)
        total += counted
        faults.extend(found)
    print(f"schema mutants attacked: {total}")
    print(f"faults (silently unenforced, or raised out of the transaction): {len(faults)}")
    for line in faults:
        print(f"  {line}")
    return 1 if faults else 0


if __name__ == "__main__":
    raise SystemExit(main())
