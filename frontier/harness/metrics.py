"""Task metrics with bounded ``[0, 1]`` outputs."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalise(text: str) -> str:
    return " ".join(_tokens(text))


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def exact_match(pred: str, gold: str) -> float:
    return float(_normalise(pred) == _normalise(gold))


def token_f1(pred: str, gold: str) -> float:
    predicted, reference = _tokens(pred), _tokens(gold)
    if not predicted or not reference:
        return float(predicted == reference)
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return _bounded(2 * precision * recall / (precision + recall))


def _lcs_length(first: Sequence[str], second: Sequence[str]) -> int:
    previous = [0] * (len(second) + 1)
    for token in first:
        current = [0]
        for index, other in enumerate(second, start=1):
            current.append(
                previous[index - 1] + 1
                if token == other
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def rouge_l(pred: str, gold: str) -> float:
    predicted, reference = _tokens(pred), _tokens(gold)
    if not predicted or not reference:
        return float(predicted == reference)
    lcs = _lcs_length(predicted, reference)
    precision, recall = lcs / len(predicted), lcs / len(reference)
    return _bounded(2 * precision * recall / (precision + recall)) if lcs else 0.0


def bertscore(
    pred: str,
    gold: str,
    *,
    scorer: Callable[[str, str], float] | None = None,
) -> float:
    """Use an injected verified BERTScore implementation.

    The heavyweight BERTScore dependency is intentionally not imported at
    module import time.  Production wiring must inject the exact installed
    package call after verifying its signature; tests can inject a scorer.
    """

    if scorer is None:
        raise RuntimeError("BERTScore scorer is not configured")
    return _bounded(scorer(pred, gold))


def pass_at_1(pred: str, gold: str) -> float:
    """Run a code candidate against its supplied unit-test oracle."""

    return float(sandboxed_code_check(pred, gold))


def sandboxed_code_check(
    candidate: str, tests: str, *, timeout_s: float = 2.0
) -> bool:
    """Run candidate code and tests in an isolated temporary working directory.

    Imports and filesystem primitives are rejected before execution.  The
    subprocess has a timeout and a temporary cwd; this is a conservative unit
    test sandbox, not a general-purpose hostile-code execution boundary.
    """

    try:
        tree = ast.parse(candidate)
        ast.parse(tests)
    except SyntaxError:
        return False
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.With,
        ast.AsyncWith,
        ast.Delete,
    )
    test_tree = ast.parse(tests)
    if any(
        isinstance(node, forbidden)
        for candidate_tree in (tree, test_tree)
        for node in ast.walk(candidate_tree)
    ):
        return False
    blocked_calls = {"__import__", "compile", "eval", "exec", "open"}
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in blocked_calls
        for candidate_tree in (tree, test_tree)
        for node in ast.walk(candidate_tree)
    ):
        return False
    script = f"{candidate}\n\n{tests}\n"
    with tempfile.TemporaryDirectory(prefix="frontier-code-") as directory:
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""}
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", script],
                cwd=directory,
                env=environment,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                text=True,
            )
        except (subprocess.SubprocessError, OSError):
            return False
    return completed.returncode == 0
