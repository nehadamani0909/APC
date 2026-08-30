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


# Pure-computation standard library modules that correct HumanEval and MBPP
# solutions genuinely need.  An allow-list rather than a deny-list, so an
# unrecognised module is refused rather than admitted by omission.  ``sys`` is
# excluded deliberately: ``sys.exit(0)`` would make a failing candidate look
# like a pass.
ALLOWED_IMPORTS = frozenset(
    {
        "abc", "array", "bisect", "cmath", "collections", "copy", "dataclasses",
        "datetime", "decimal", "enum", "fractions", "functools", "heapq",
        "itertools", "json", "math", "numbers", "operator", "queue", "random",
        "re", "statistics", "string", "textwrap", "types", "typing",
        "unicodedata",
    }
)
_BLOCKED_CALLS = frozenset({"__import__", "compile", "eval", "exec", "open"})


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # A relative import has no module to resolve against here.
            roots.add((node.module or "").split(".")[0])
    return roots


def _is_statically_safe(tree: ast.AST) -> bool:
    if not _imported_roots(tree) <= ALLOWED_IMPORTS:
        return False
    return not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _BLOCKED_CALLS
        for node in ast.walk(tree)
    )


def _resource_limits(memory_bytes: int, cpu_seconds: int) -> Callable[[], None] | None:
    """Return a POSIX ``preexec_fn`` applying rlimits, or None on Windows."""

    try:
        import resource
    except ImportError:  # Windows has no resource module
        return None

    # Probed rather than referenced directly: the individual limits are not
    # all defined on every POSIX platform, and none of them exist on Windows.
    setrlimit = getattr(resource, "setrlimit", None)
    if setrlimit is None:
        return None

    def apply() -> None:
        limits = (
            ("RLIMIT_AS", memory_bytes),
            ("RLIMIT_CPU", cpu_seconds),
            ("RLIMIT_NPROC", 0),
        )
        for name, value in limits:
            limit = getattr(resource, name, None)
            if limit is not None:
                setrlimit(limit, (value, value))

    return apply


def sandboxed_code_check(
    candidate: str,
    tests: str,
    *,
    timeout_s: float = 5.0,
    memory_bytes: int = 512 * 1024 * 1024,
) -> bool:
    """Execute a candidate against its test harness under isolation.

    Layers, in order: a static screen (allow-listed imports, no ``eval`` /
    ``exec`` / ``open``), then a subprocess in isolated mode with a scrubbed
    environment, a temporary working directory, a wall-clock timeout, and --
    on POSIX -- address-space, CPU, and subprocess rlimits.

    This is a unit-test sandbox for model-generated code, not a boundary
    against a determined adversary: on Windows no rlimits are available, and
    nothing here blocks network syscalls directly (the import allow-list is
    what keeps ``socket``/``urllib`` out of reach).  Run untrusted grids in a
    container if that residual risk matters.
    """

    try:
        candidate_tree = ast.parse(candidate)
        test_tree = ast.parse(tests)
    except (SyntaxError, ValueError):
        return False
    if not all(_is_statically_safe(tree) for tree in (candidate_tree, test_tree)):
        return False

    script = f"{candidate}\n\n{tests}\n"
    with tempfile.TemporaryDirectory(prefix="frontier-code-") as directory:
        environment = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""}
        preexec = _resource_limits(memory_bytes, max(1, int(timeout_s)))
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", script],
                cwd=directory,
                env=environment,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                text=True,
                # None on Windows, where preexec_fn is unsupported anyway.
                preexec_fn=preexec,
            )
        except (subprocess.SubprocessError, OSError, ValueError):
            return False
    return bool(completed.returncode == 0)
