from frontier.harness.metrics import (
    exact_match,
    pass_at_1,
    rouge_l,
    sandboxed_code_check,
    token_f1,
)


def test_metrics_are_bounded() -> None:
    for metric in (exact_match, token_f1, rouge_l):
        assert 0.0 <= metric("The answer is 42.", "42") <= 1.0


def test_code_check_is_sandboxed_and_timed() -> None:
    assert sandboxed_code_check("def answer(): return 2", "assert answer() == 2")
    assert not sandboxed_code_check("import pathlib", "assert True")
    assert pass_at_1("def answer(): return 2", "assert answer() == 2") == 1.0
