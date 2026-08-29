import numpy as np

from frontier.predict.calibrate import expected_calibration_error, isotonic_calibrate
from frontier.predict.heads import (
    BUDGETS,
    CumulativeLogitHead,
    FrontierHeads,
    OutputLengthHead,
    RateAdherenceHead,
)
from frontier.predict.train import assert_disjoint, split_prompt_ids


def test_cumulative_logit_is_monotone_for_1000_inputs() -> None:
    features = np.random.default_rng(4).normal(size=(1000, 8))
    predictions = CumulativeLogitHead(8, seed=4).predict(features)
    assert np.all(predictions[:, :-1] <= predictions[:, 1:])


def test_all_heads_emit_budget_vectors() -> None:
    features = np.ones((3, 4))
    quality, adherence, output = FrontierHeads(
        CumulativeLogitHead(4),
        RateAdherenceHead(4),
        OutputLengthHead(4),
    ).predict(features)
    assert quality.shape == adherence.shape == output.shape == (3, len(BUDGETS))


def test_calibration_and_splits() -> None:
    splits = split_prompt_ids([f"p{i}" for i in range(100)], seed=3)
    assert_disjoint(splits)
    predictions = np.full((10, len(BUDGETS)), 0.5)
    labels = np.ones_like(predictions)
    calibrated = isotonic_calibrate(predictions, labels)
    assert np.all((0.0 <= calibrated.probabilities) & (calibrated.probabilities <= 1.0))
    assert expected_calibration_error(calibrated.probabilities, labels).shape == (
        len(BUDGETS),
    )
