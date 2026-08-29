"""NumPy predictor heads for the quality, adherence, and output curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

BUDGETS = np.array((0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0), dtype=float)
Array = np.ndarray[Any, np.dtype[np.float64]]


def _softplus(values: Array) -> Array:
    return cast(Array, np.logaddexp(0.0, values))


def _sigmoid(values: Array) -> Array:
    return cast(Array, 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0))))


class CumulativeLogitHead:
    """H1: cumulative-logit quality head with monotone increments."""

    def __init__(self, feature_count: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.base_weights = rng.normal(0.0, 0.1, feature_count)
        self.increment_weights = rng.normal(0.0, 0.1, (len(BUDGETS) - 1, feature_count))
        self.base_bias = 0.0
        self.increment_bias = np.zeros(len(BUDGETS) - 1)

    def predict(self, features: Array) -> Array:
        base = features @ self.base_weights + self.base_bias
        increments = _softplus(
            features @ self.increment_weights.T + self.increment_bias
        )
        logits = np.concatenate(
            (base[:, None], base[:, None] + np.cumsum(increments, axis=1)), axis=1
        )
        return _sigmoid(logits)

    def fit(self, features: Array, labels: Array) -> None:
        """Fit a deterministic monotone approximation to curve labels.

        The production implementation can replace this optimizer, but the
        parameterization is already the ordinal cumulative-logit form: every
        increment is stored in unconstrained space and passed through
        ``softplus`` at prediction time.
        """
        if labels.shape[1] != len(BUDGETS):
            raise ValueError("one label column is required per budget")
        monotone = np.maximum.accumulate(labels, axis=1)
        logits = np.log(
            np.clip(monotone, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - monotone, 1e-4, 1.0)
        )
        design = np.column_stack((features, np.ones(len(features))))
        coefficients, *_ = np.linalg.lstsq(design, logits[:, 0], rcond=None)
        self.base_weights = coefficients[:-1]
        self.base_bias = float(coefficients[-1])
        mean_logits = logits.mean(axis=0)
        increments = np.maximum(1e-4, np.diff(mean_logits, prepend=mean_logits[0]))[1:]
        self.increment_bias = np.asarray(increments, dtype=float)
        self.increment_weights.fill(0.0)


class FreeFormHead:
    """Non-monotone H1 ablation with the same vector interface."""

    def __init__(self, feature_count: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.1, (len(BUDGETS), feature_count))
        self.bias = np.zeros(len(BUDGETS))

    def predict(self, features: Array) -> Array:
        return _sigmoid(features @ self.weights.T + self.bias)

    def fit(self, features: Array, labels: Array) -> None:
        design = np.column_stack((features, np.ones(len(features))))
        logits = np.log(
            np.clip(labels, 1e-4, 1.0 - 1e-4) / np.clip(1.0 - labels, 1e-4, 1.0)
        )
        coefficients, *_ = np.linalg.lstsq(design, logits, rcond=None)
        self.weights = coefficients[:-1].T
        self.bias = coefficients[-1]


class BudgetClassifier:
    """Direct K-way budget classifier ablation."""

    def __init__(self, feature_count: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.1, (len(BUDGETS), feature_count))
        self.bias = np.zeros(len(BUDGETS))

    def predict(self, features: Array) -> Array:
        logits = features @ self.weights.T + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return cast(Array, probabilities / probabilities.sum(axis=1, keepdims=True))

    def fit(self, features: Array, labels: Array) -> None:
        classes = np.argmax(labels, axis=1)
        design = np.column_stack((features, np.ones(len(features))))
        targets = np.eye(len(BUDGETS))[classes]
        coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
        self.weights = coefficients[:-1].T
        self.bias = coefficients[-1]


class RateAdherenceHead:
    """H2: L0-only regression of log(realised/requested) with inversion."""

    def __init__(self, feature_count: int) -> None:
        self.weights = np.zeros(feature_count)
        self.bias = 0.0

    def predict(self, features: Array) -> Array:
        multiplier = np.exp(np.clip(features @ self.weights + self.bias, -5.0, 5.0))
        return cast(Array, np.clip(multiplier[:, None] * BUDGETS[None, :], 0.0, 1.0))

    def fit(self, features: Array, realised: Array, requested: Array = BUDGETS) -> None:
        repeated = np.broadcast_to(requested, realised.shape)
        design = np.column_stack((features, np.ones(len(features))))
        coefficients, *_ = np.linalg.lstsq(
            design,
            np.log(np.maximum(realised / repeated, 1e-6)).mean(axis=1),
            rcond=None,
        )
        self.weights = coefficients[:-1]
        self.bias = float(coefficients[-1])

    def invert(self, features: Array, target_rate: float) -> Array:
        predictions = self.predict(features)
        indices = np.abs(predictions - target_rate).argmin(axis=1)
        return cast(Array, BUDGETS[indices])


class OutputLengthHead:
    """H3: positive output-token curve head."""

    def __init__(self, feature_count: int) -> None:
        self.weights = np.zeros(feature_count)
        self.bias = 0.0

    def predict(self, features: Array) -> Array:
        length = np.exp(np.clip(features @ self.weights + self.bias, -5.0, 20.0))
        return cast(
            Array,
            np.broadcast_to(length[:, None], (len(features), len(BUDGETS))).copy(),
        )

    def fit(self, features: Array, output_tokens: Array) -> None:
        design = np.column_stack((features, np.ones(len(features))))
        coefficients, *_ = np.linalg.lstsq(
            design, np.log(np.maximum(output_tokens, 1.0)).mean(axis=1), rcond=None
        )
        self.weights = coefficients[:-1]
        self.bias = float(coefficients[-1])


@dataclass
class FrontierHeads:
    """All three vector-emitting heads used by the predictor."""

    quality: CumulativeLogitHead
    adherence: RateAdherenceHead
    output_length: OutputLengthHead

    def predict(self, features: Array) -> tuple[Array, Array, Array]:
        return (
            self.quality.predict(features),
            self.adherence.predict(features),
            self.output_length.predict(features),
        )
