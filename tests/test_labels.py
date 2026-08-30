from pathlib import Path

import numpy as np
import pandas as pd

from frontier.corpus.labels import (
    build_curves,
    monotone_safe_budget,
    naive_safe_budget,
    safety_indicator,
)
from frontier.predict.heads import BUDGETS


def test_monotone_safe_budget_ignores_a_lucky_low_point() -> None:
    # rho(0.2) is safe by luck, but rho(0.3) collapses, so the safe suffix
    # only starts at 0.5 (APC-04 §3.2).
    quality = np.array([[0.9, 0.1, 0.2, 0.9, 0.9, 0.9, 0.9]])
    assert naive_safe_budget(quality, 0.05)[0] == 0.2
    assert monotone_safe_budget(quality, 0.05)[0] == 0.5


def test_monotone_safe_budget_matches_naive_on_monotone_curves() -> None:
    quality = np.array([[0.1, 0.3, 0.5, 0.88, 0.9, 0.9, 0.9]])
    assert monotone_safe_budget(quality, 0.05)[0] == 0.5
    assert naive_safe_budget(quality, 0.05)[0] == 0.5


def test_fully_safe_and_fully_unsafe_curves() -> None:
    always = np.full((1, len(BUDGETS)), 0.9)
    assert monotone_safe_budget(always, 0.05)[0] == float(BUDGETS[0])
    never = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    assert monotone_safe_budget(never, 0.05)[0] == 1.0


def test_safety_indicator_is_relative_to_the_uncompressed_budget() -> None:
    quality = np.array([[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5]])
    indicator = safety_indicator(quality, 0.05)
    assert indicator[0, -1] == 1.0
    assert indicator[0, 0] == 0.0


def test_build_curves_averages_samples_per_budget() -> None:
    rows = [
        {
            "prompt_id": "p1",
            "family": "qa",
            "requested_b": float(budget),
            "quality": quality,
            "realised_r": float(budget),
            "T_out": 10.0,
            "T_in": 100.0,
            "sample_idx": sample,
        }
        for budget in BUDGETS
        for sample, quality in enumerate((0.0, 1.0))
    ]
    curves = build_curves(pd.DataFrame(rows))
    assert curves.prompt_ids == ("p1",)
    assert curves.families == ("qa",)
    # Graded rho is the mean over the k samples, not a binary collapse.
    assert np.allclose(curves.quality, 0.5)
    assert curves.quality.shape == (1, len(BUDGETS))
    assert curves.input_tokens.shape == (1, len(BUDGETS))


def test_output_store_round_trips_and_dedupes(tmp_path: Path) -> None:
    from frontier.harness.outputs import OutputStore

    store = OutputStore(tmp_path / "outputs.jsonl")
    store.put("aaa", "first text")
    store.put("aaa", "first text")
    store.put("bbb", "second text")
    assert len(store) == 2
    assert OutputStore(tmp_path / "outputs.jsonl").load() == {
        "aaa": "first text",
        "bbb": "second text",
    }


def _ap_frame() -> pd.DataFrame:
    """p1 answers identically everywhere; p2 drifts as the budget shrinks."""
    rows = []
    for budget in BUDGETS:
        for _sample in range(2):
            rows.append(
                {
                    "prompt_id": "p1",
                    "requested_b": float(budget),
                    "raw_output_hash": "same",
                }
            )
            drift = budget <= 0.4
            rows.append(
                {
                    "prompt_id": "p2",
                    "requested_b": float(budget),
                    "raw_output_hash": "other" if drift else "base",
                }
            )
    return pd.DataFrame(rows)


def test_answer_preservation_measures_drift_from_the_uncompressed_answer() -> None:
    from frontier.corpus.labels import answer_preservation

    outputs = {"same": "42", "base": "7", "other": "9"}
    values = answer_preservation(
        _ap_frame(), outputs, lambda text: text.strip(), prompt_ids=("p1", "p2")
    )
    # p1 never changes its answer: preserved at every budget.
    assert np.allclose(values[0], 1.0)
    # p2 keeps the uncompressed answer down to 0.5, then diverges.
    assert values[1][list(BUDGETS).index(1.0)] == 1.0
    assert values[1][list(BUDGETS).index(0.5)] == 1.0
    assert values[1][list(BUDGETS).index(0.2)] == 0.0


def test_answer_preservation_still_scores_prompts_the_model_always_fails() -> None:
    """The reason this metric exists (APC-04 §3.1.1)."""
    from frontier.corpus.labels import answer_preservation, monotone_safe_budget

    # Model is wrong everywhere, so graded quality is 0 at every budget and
    # b* collapses to the smallest budget -- a degenerate label.
    always_wrong = np.zeros((1, len(BUDGETS)))
    assert monotone_safe_budget(always_wrong, 0.05)[0] == float(BUDGETS[0])

    # Answer preservation still distinguishes "same wrong answer" from
    # "different wrong answer", so the prompt keeps carrying signal.
    frame = _ap_frame()
    values = answer_preservation(
        frame,
        {"same": "wrong", "base": "wrong-a", "other": "wrong-b"},
        lambda text: text.strip(),
        prompt_ids=("p2",),
    )
    assert values[0][list(BUDGETS).index(0.2)] == 0.0
    assert values[0][list(BUDGETS).index(1.0)] == 1.0
