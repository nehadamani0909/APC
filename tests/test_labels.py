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
