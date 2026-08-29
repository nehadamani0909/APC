import numpy as np
from pytest import approx

from frontier.eval.metrics import (
    cpgr,
    cpt,
    holm_bonferroni,
    minimum_detectable_effect,
    paired_bca,
    quality_cost_auc,
)


def test_metrics_and_bca_are_prompt_level() -> None:
    values = np.linspace(-0.1, 0.1, 20)
    interval = paired_bca(values, resamples=200)
    assert interval.low <= interval.estimate <= interval.high
    assert cpgr(0.8, 0.5, 0.9) == approx(0.75)
    assert cpt(np.array([1.0, 2.0]), np.array([0.9, 0.99]), 0.95) == 2.0
    assert quality_cost_auc(np.array([0.0, 1.0]), np.array([0.0, 1.0])) == 0.5


def test_holm_and_power() -> None:
    decisions = holm_bonferroni({"a": 0.001, "b": 0.2})
    assert decisions == {"a": True, "b": False}
    assert 0.024 < minimum_detectable_effect(1200) < 0.025
