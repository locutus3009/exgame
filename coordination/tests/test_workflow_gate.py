# SPDX-License-Identifier: MIT
"""Guard the two duplicated expressions in the coordination workflow.

The concurrency group suffix and the job condition in `.github/workflows/coordination.yml` encode
the same rule twice: a run that the job condition will skip must land in a different concurrency
group from a run that will verify something. Held in step by a comment alone, that pairing is a
hope. This parses both expressions out of the workflow and checks that one is the literal negation
of the other, so drifting either without the other fails the suite that runs inside the workflow
itself.

PyYAML is not in the quality group, so the two expressions are located in the raw file text
rather than by loading the document. Both locators raise instead of returning something empty,
which is what stops this from passing vacuously if the file is restructured.
"""

import re
import unittest
from pathlib import Path
from typing import cast

WORKFLOW = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "coordination.yml"
SELECTOR = " && 'skipped' || 'verified'"
LEXEME = re.compile(r"\(|\)|,|&&|\|\||!=|==|!|'[^']*'|[A-Za-z_][A-Za-z0-9_.]*")
NEGATED = {"and": "or", "or": "and", "eq": "ne", "ne": "eq"}

type Node = tuple[object, ...]


def group_condition(text: str) -> str:
    """Return the condition selecting between the skipped and verified concurrency groups."""
    lines = [entry for entry in text.splitlines() if entry.startswith("  group:")]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one concurrency group line, found {len(lines)}")
    line = lines[0]
    body = line[line.rindex("${{") + 3 : line.rindex("}}")].strip()
    if not body.endswith(SELECTOR):
        raise ValueError("the group expression no longer ends with the skipped/verified selector")
    return body[: -len(SELECTOR)]


def job_condition(text: str) -> str:
    """Return the job-level condition deciding whether the verifying job runs."""
    lines = text.splitlines()
    starts = [index for index, entry in enumerate(lines) if entry.strip() == "if: >-"]
    if len(starts) != 1:
        raise ValueError(f"expected exactly one job condition, found {len(starts)}")
    collected = []
    for entry in lines[starts[0] + 1 :]:
        if not entry.startswith(" " * 6):
            break
        collected.append(entry.strip())
    if not collected:
        raise ValueError("the job condition block is empty")
    return " ".join(collected)


def _parse_primary(tokens: list[str], pos: int) -> tuple[Node, int]:
    if tokens[pos] == "(":
        node, pos = _parse_or(tokens, pos + 1)
        if tokens[pos] != ")":
            raise ValueError("unbalanced parenthesis")
        return node, pos + 1
    name = tokens[pos]
    pos += 1
    if pos >= len(tokens) or tokens[pos] != "(":
        return ("value", name), pos
    arguments: list[Node] = []
    pos += 1
    while tokens[pos] != ")":
        if tokens[pos] == ",":
            pos += 1
            continue
        argument, pos = _parse_or(tokens, pos)
        arguments.append(argument)
    return ("call", name, *arguments), pos + 1


def _parse_compare(tokens: list[str], pos: int) -> tuple[Node, int]:
    node, pos = _parse_primary(tokens, pos)
    if pos >= len(tokens) or tokens[pos] not in ("==", "!="):
        return node, pos
    operator = "eq" if tokens[pos] == "==" else "ne"
    right, pos = _parse_primary(tokens, pos + 1)
    return (operator, node, right), pos


def _parse_unary(tokens: list[str], pos: int) -> tuple[Node, int]:
    if tokens[pos] != "!":
        return _parse_compare(tokens, pos)
    node, pos = _parse_unary(tokens, pos + 1)
    return ("not", node), pos


def _parse_and(tokens: list[str], pos: int) -> tuple[Node, int]:
    node, pos = _parse_unary(tokens, pos)
    while pos < len(tokens) and tokens[pos] == "&&":
        right, pos = _parse_unary(tokens, pos + 1)
        node = ("and", node, right)
    return node, pos


def _parse_or(tokens: list[str], pos: int) -> tuple[Node, int]:
    node, pos = _parse_and(tokens, pos)
    while pos < len(tokens) and tokens[pos] == "||":
        right, pos = _parse_and(tokens, pos + 1)
        node = ("or", node, right)
    return node, pos


def parse(text: str) -> Node:
    """Parse a GitHub Actions boolean expression into a comparable tree."""
    tokens = LEXEME.findall(text)
    if not tokens:
        raise ValueError("no tokens in expression")
    node, pos = _parse_or(tokens, 0)
    if pos != len(tokens):
        raise ValueError(f"trailing tokens after position {pos}")
    return node


def negate(node: Node) -> Node:
    """Push a negation through the tree, so two expressions can be compared as written."""
    head = str(node[0])
    if head == "not":
        return cast("Node", node[1])
    if head in ("and", "or"):
        return (NEGATED[head], negate(cast("Node", node[1])), negate(cast("Node", node[2])))
    if head in ("eq", "ne"):
        return (NEGATED[head], node[1], node[2])
    return ("not", node)


class WorkflowGateExpressions(unittest.TestCase):
    """The group suffix and the job condition must stay exact complements of each other."""

    def setUp(self) -> None:
        self.text = WORKFLOW.read_text()

    def test_group_suffix_is_the_negation_of_the_job_condition(self) -> None:
        skips = parse(group_condition(self.text))
        runs = parse(job_condition(self.text))
        self.assertEqual(skips, negate(runs))

    def test_both_expressions_are_the_reviewed_ones(self) -> None:
        self.assertEqual(
            " ".join(group_condition(self.text).split()),
            "(github.event_name == 'push' && (github.event.deleted == true"
            " || startsWith(github.event.head_commit.message, 'chore(state):')))",
        )
        self.assertEqual(
            " ".join(job_condition(self.text).split()),
            "github.event_name != 'push' || (github.event.deleted != true"
            " && !startsWith(github.event.head_commit.message, 'chore(state):'))",
        )

    def test_a_drifted_group_expression_is_caught(self) -> None:
        drifted = self.text.replace("github.event.deleted == true", "github.event.deleted != true")
        self.assertNotEqual(drifted, self.text)
        self.assertNotEqual(parse(group_condition(drifted)), negate(parse(job_condition(drifted))))

    def test_a_drifted_job_condition_is_caught(self) -> None:
        drifted = self.text.replace(
            "&& !startsWith(github.event.head_commit.message",
            "&& startsWith(github.event.head_commit.message",
        )
        self.assertNotEqual(drifted, self.text)
        self.assertNotEqual(parse(group_condition(drifted)), negate(parse(job_condition(drifted))))

    def test_a_missing_group_line_raises_rather_than_passing_vacuously(self) -> None:
        with self.assertRaises(ValueError):
            group_condition(self.text.replace("  group:", "  # group:"))

    def test_a_changed_selector_raises_rather_than_passing_vacuously(self) -> None:
        with self.assertRaises(ValueError):
            group_condition(self.text.replace(SELECTOR, " && 'a' || 'b'"))

    def test_a_missing_job_condition_raises_rather_than_passing_vacuously(self) -> None:
        with self.assertRaises(ValueError):
            job_condition(self.text.replace("if: >-", "if: |-"))

    def test_an_empty_job_condition_block_raises(self) -> None:
        with self.assertRaises(ValueError):
            job_condition("jobs:\n  verify:\n    if: >-\n    runs-on: ubuntu-24.04\n")


if __name__ == "__main__":
    unittest.main()
