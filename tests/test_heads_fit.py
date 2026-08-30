"""Fitting tests for the predictor heads.

The pre-existing head tests only exercised ``predict``.  These cover ``fit``,
which is what contributions C2 and C3 actually depend on.
"""

from typing import Any

import numpy as np

from frontier.predict.heads import (
    BUDGETS,
    BudgetClassifier,
    CumulativeLogitHead,
    FreeFormHead,
    OutputLengthHead,
    RateAdherenceHead,
)
from frontier.predict.scaling import FeatureScaler

Array = np.ndarray[Any, np.dtype[np.float64]]


def _curve_dataset(n: int = 240) -> tuple[Array, Array]:
    """Feature 0 controls where the safety curve turns on."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(n, 4))
    onset = features[:, 0]
    labels = np.zeros((n, len(BUDGETS)))
    for index in range(len(BUDGETS)):
        labels[:, index] = (BUDGETS[index] >= 0.5 + 0.25 * onset).astype(float)
    labels[:, -1] = 1.0
    return features, labels


def test_cumulative_logit_fit_learns_instance_specific_curves() -> None:
    features, labels = _curve_dataset()
    scaled = FeatureScaler.fit(features).transform(features)
    head = CumulativeLogitHead(features.shape[1], seed=0)
    head.fit(scaled, labels)
    # C3 requires the curve SHAPE to vary per instance, which is only
    # possible when the increment weights are non-zero.
    assert np.abs(head.increment_weights).max() > 1e-3
    predictions = head.predict(scaled)
    shapes = np.diff(predictions, axis=1)
    assert shapes.std(axis=0).max() > 1e-2


def test_cumulative_logit_stays_monotone_after_fitting() -> None:
    features, labels = _curve_dataset()
    scaled = FeatureScaler.fit(features).transform(features)
    head = CumulativeLogitHead(features.shape[1], seed=0)
    head.fit(scaled, labels)
    predictions = head.predict(scaled)
    assert np.all(predictions[:, :-1] <= predictions[:, 1:] + 1e-12)


def test_cumulative_logit_fit_reduces_error_and_is_deterministic() -> None:
    features, labels = _curve_dataset()
    scaled = FeatureScaler.fit(features).transform(features)
    before = CumulativeLogitHead(features.shape[1], seed=0)
    initial = float(np.abs(before.predict(scaled) - labels).mean())
    before.fit(scaled, labels)
    fitted = float(np.abs(before.predict(scaled) - labels).mean())
    assert fitted < initial

    repeat = CumulativeLogitHead(features.shape[1], seed=0)
    repeat.fit(scaled, labels)
    assert np.allclose(repeat.base_weights, before.base_weights)
    assert np.allclose(repeat.increment_weights, before.increment_weights)


def test_output_length_head_varies_with_budget() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(120, 3))
    # Output tokens grow as the budget shrinks: the compression-paradox
    # shape that contribution C2 depends on being representable.
    outputs = np.stack(
        [80.0 + 200.0 * (1.0 - budget) + 5.0 * features[:, 0] for budget in BUDGETS],
        axis=1,
    )
    head = OutputLengthHead(features.shape[1])
    head.fit(features, outputs)
    predictions = head.predict(features)
    assert predictions.shape == (120, len(BUDGETS))
    # A budget-independent head would make this spread zero and E9 unmeasurable.
    assert predictions.std(axis=1).min() > 1.0
    assert predictions[:, 0].mean() > predictions[:, -1].mean()


def test_free_form_and_classifier_heads_fit() -> None:
    features, labels = _curve_dataset()
    scaled = FeatureScaler.fit(features).transform(features)
    free_form = FreeFormHead(features.shape[1], seed=0)
    initial = float(np.abs(free_form.predict(scaled) - labels).mean())
    free_form.fit(scaled, labels)
    assert float(np.abs(free_form.predict(scaled) - labels).mean()) < initial

    classifier = BudgetClassifier(features.shape[1], seed=0)
    classifier.fit(scaled, labels)
    probabilities = classifier.predict(scaled)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_rate_adherence_recovers_a_systematic_shortfall() -> None:
    rng = np.random.default_rng(2)
    features = rng.normal(size=(150, 3))
    realised = np.clip(0.85 * BUDGETS[None, :] * np.ones((150, 1)), 0.0, 1.0)
    head = RateAdherenceHead(features.shape[1])
    head.fit(features, realised)
    predicted = head.predict(features)
    assert np.allclose(predicted, realised, atol=0.02)


def test_feature_scaler_handles_constant_columns() -> None:
    features = np.column_stack(
        [np.arange(10.0), np.full(10, 3.0), np.linspace(0.0, 1.0, 10)]
    )
    scaler = FeatureScaler.fit(features)
    scaled = scaler.transform(features)
    assert np.allclose(scaled[:, 1], 0.0)
    assert abs(float(scaled[:, 0].std()) - 1.0) < 1e-9
    restored = type(scaler).from_dict(scaler.to_dict())
    assert np.allclose(restored.transform(features), scaled)
