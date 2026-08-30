"""Sandboxed pass@1 execution for the HumanEval/MBPP code family."""

from frontier.harness.metrics import pass_at_1, sandboxed_code_check

HARNESS = "def check(candidate):\n    assert candidate(4) == 2\n\ncheck(square_root)\n"


def test_known_passing_solution() -> None:
    assert pass_at_1("def square_root(n):\n    return n // 2\n", HARNESS) == 1.0


def test_known_failing_solution() -> None:
    assert pass_at_1("def square_root(n):\n    return n + 100\n", HARNESS) == 0.0


def test_syntax_error_is_a_failure_not_a_crash() -> None:
    assert pass_at_1("def broken(:\n", HARNESS) == 0.0


def test_solutions_may_use_allowed_standard_library_imports() -> None:
    # Correct HumanEval/MBPP solutions routinely import these. Rejecting all
    # imports made most correct code score zero and would have made the code
    # family look far more compression-fragile than it is.
    candidate = (
        "import math\n"
        "from collections import Counter\n"
        "from typing import List\n"
        "def square_root(n):\n"
        "    counts = Counter([n])\n"
        "    return int(math.sqrt(n)) * max(counts.values())\n"
    )
    assert pass_at_1(candidate, HARNESS) == 1.0


def test_with_statements_are_allowed() -> None:
    candidate = (
        "from decimal import localcontext\n"
        "def square_root(n):\n"
        "    with localcontext() as ctx:\n"
        "        ctx.prec = 8\n"
        "        return n // 2\n"
    )
    assert pass_at_1(candidate, HARNESS) == 1.0


def test_dangerous_imports_are_refused() -> None:
    for module in ("os", "subprocess", "socket", "shutil", "urllib"):
        candidate = (
            f"import {module}\ndef square_root(n):\n    return n // 2\n"
        )
        assert sandboxed_code_check(candidate, HARNESS) is False


def test_sys_exit_cannot_fake_a_pass() -> None:
    # sys.exit(0) exits cleanly WITHOUT running the assertions, so the return
    # code alone cannot tell a pass from a skip. The success sentinel closes
    # that hole, which is what lets sys stay on the allow-list.
    candidate = "import sys\ndef square_root(n):\n    sys.exit(0)\n"
    assert sandboxed_code_check(candidate, HARNESS) is False


def test_sys_is_usable_for_computation() -> None:
    # Real MBPP solutions use sys.maxsize; banning sys outright cost two
    # correct solutions in the 500-problem test split.
    candidate = (
        "import sys\n"
        "def square_root(n):\n"
        "    return min(sys.maxsize, n // 2)\n"
    )
    assert pass_at_1(candidate, HARNESS) == 1.0


def test_hashlib_is_available() -> None:
    # HumanEval/162 (string_to_md5) needs it, and it is pure computation.
    candidate = (
        "import hashlib\n"
        "def square_root(n):\n"
        "    hashlib.md5(b'x').hexdigest()\n"
        "    return n // 2\n"
    )
    assert pass_at_1(candidate, HARNESS) == 1.0


def test_dynamic_execution_primitives_are_refused() -> None:
    for snippet in (
        "def square_root(n):\n    return eval('n // 2')\n",
        "def square_root(n):\n    exec('x = 1')\n    return n // 2\n",
        "def square_root(n):\n    open('x', 'w')\n    return n // 2\n",
        "def square_root(n):\n    __import__('os')\n    return n // 2\n",
    ):
        assert sandboxed_code_check(snippet, HARNESS) is False


def test_infinite_loop_times_out_and_fails() -> None:
    candidate = "def square_root(n):\n    while True:\n        pass\n"
    assert sandboxed_code_check(candidate, HARNESS, timeout_s=1.0) is False


def test_a_failing_assertion_inside_the_harness_is_a_failure() -> None:
    assert sandboxed_code_check("def square_root(n):\n    return 3\n", HARNESS) is False
