"""NumPy predictor heads for the quality, adherence, and output curves.

Every head emits one value per budget in a single forward pass; the budget is
never an input feature (APC-04 §5.3), which is what makes the predicted curve
coherent and lets H1 enforce monotonicity structurally.

All fits are full-batch and deterministic: the same inputs and the same seed
produce the same parameters on every run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

BUDGETS = np.array((0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0), dtype=float)
Array = np.ndarray[Any, np.dtype[np.float64]]


def _softplus(values: Array) -> Array:
    return cast(Array, np.logaddexp(0.0, values))


def _sigmoid(values: Array) -> Array:
    return cast(Array, 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0))))


def _check_design(features: Array, labels: Array) -> tuple[Array, Array]:
    values = np.asarray(features, dtype=float)
    targets = np.asarray(labels, dtype=float)
    if values.ndim != 2:
        raise ValueError("features must be a 2-D array")
    if len(values) != len(targets):
        raise ValueError("features and labels must have the same length")
    if len(values) == 0:
        raise ValueError("fitting requires at least one example")
    return cast(Array, values), cast(Array, targets)


def _ridge_solve(design: Array, targets: Array, l2: float) -> Array:
    """Deterministic ridge solution of ``design @ coef ~= targets``.

    Ridge rather than :func:`numpy.linalg.lstsq` so a constant or collinear
    feature column yields a bounded, reproducible coefficient instead of a
    rank-deficient minimum-norm solution.
    """

    gram = design.T @ design
    penalty = l2 * np.eye(gram.shape[0])
    # The intercept column is last and is deliberately left unpenalised.
    penalty[-1, -1] = 0.0
    return cast(Array, np.linalg.solve(gram + penalty, design.T @ targets))


class _Adam:
    """Fixed-step Adam over a list of parameter arrays."""

    def __init__(
        self, shapes: Sequence[tuple[int, ...]], learning_rate: float
    ) -> None:
        self.learning_rate = learning_rate
        self.moment = [np.zeros(shape, dtype=float) for shape in shapes]
        self.velocity = [np.zeros(shape, dtype=float) for shape in shapes]
        self.step_count = 0

    def step(self, params: list[Array], grads: list[Array]) -> list[Array]:
        self.step_count += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        bias1 = 1.0 - beta1**self.step_count
        bias2 = 1.0 - beta2**self.step_count
        updated: list[Array] = []
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            self.moment[index] = beta1 * self.moment[index] + (1 - beta1) * grad
            self.velocity[index] = beta2 * self.velocity[index] + (1 - beta2) * grad**2
            step = (self.moment[index] / bias1) / (
                np.sqrt(self.velocity[index] / bias2) + eps
            )
            updated.append(cast(Array, param - self.learning_rate * step))
        return updated


class CumulativeLogitHead:
    """H1: cumulative-logit quality head with monotone increments.

    ``s(x, b_j) = sigmoid(u(x) + sum_{i<=j} softplus(v_i(x)))``.  The
    increments are non-negative by construction, so the predicted curve is
    non-decreasing in the budget for every input — the structural fix for
    audit defect D1 and the monotone score CRC requires.
    """

    def __init__(self, feature_count: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.base_weights: Array = rng.normal(0.0, 0.1, feature_count)
        self.increment_weights: Array = rng.normal(
            0.0, 0.1, (len(BUDGETS) - 1, feature_count)
        )
        self.base_bias = 0.0
        self.increment_bias: Array = np.zeros(len(BUDGETS) - 1)

    def _forward(self, features: Array) -> tuple[Array, Array]:
        base = features @ self.base_weights + self.base_bias
        pre_activation = features @ self.increment_weights.T + self.increment_bias
        increments = _softplus(pre_activation)
        logits = np.concatenate(
            (base[:, None], base[:, None] + np.cumsum(increments, axis=1)), axis=1
        )
        return cast(Array, logits), cast(Array, pre_activation)

    def predict(self, features: Array) -> Array:
        logits, _ = self._forward(np.asarray(features, dtype=float))
        return _sigmoid(logits)

    def fit(
        self,
        features: Array,
        labels: Array,
        *,
        l2: float = 1e-3,
        epochs: int = 600,
        learning_rate: float = 0.05,
    ) -> None:
        """Minimise the cumulative-logit NLL against per-budget curve labels.

        ``labels`` holds one target per budget in ``[0, 1]``.  Soft targets
        (the graded quality rho) and hard targets (the binary safety
        indicator ``1[rho(b) >= rho(1) - epsilon]``) are both valid; the
        binary cross-entropy below handles either.
        """

        values, targets = _check_design(features, labels)
        if targets.ndim != 2 or targets.shape[1] != len(BUDGETS):
            raise ValueError("one label column is required per budget")
        targets = cast(Array, np.clip(targets, 0.0, 1.0))
        scale = float(targets.size)
        optimiser = _Adam(
            [
                self.base_weights.shape,
                (),
                self.increment_weights.shape,
                self.increment_bias.shape,
            ],
            learning_rate,
        )
        for _ in range(epochs):
            logits, pre_activation = self._forward(values)
            residual = (_sigmoid(logits) - targets) / scale
            # d_i enters z_j for every j >= i + 1, so each increment's
            # gradient is the residual suffix sum starting one column later.
            suffix = np.cumsum(residual[:, ::-1], axis=1)[:, ::-1]
            base_grad = suffix[:, 0]
            increment_grad = suffix[:, 1:] * _sigmoid(pre_activation)
            grads: list[Array] = [
                cast(Array, values.T @ base_grad + 2.0 * l2 * self.base_weights),
                cast(Array, np.asarray(base_grad.sum(), dtype=float)),
                cast(
                    Array,
                    increment_grad.T @ values + 2.0 * l2 * self.increment_weights,
                ),
                cast(Array, increment_grad.sum(axis=0)),
            ]
            params: list[Array] = [
                self.base_weights,
                cast(Array, np.asarray(self.base_bias, dtype=float)),
                self.increment_weights,
                self.increment_bias,
            ]
            updated = optimiser.step(params, grads)
            self.base_weights = updated[0]
            self.base_bias = float(updated[1])
            self.increment_weights = updated[2]
            self.increment_bias = updated[3]


class FreeFormHead:
    """Non-monotone H1 ablation (APC-06 §E4) with the same vector interface."""

    def __init__(self, feature_count: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weights: Array = rng.normal(0.0, 0.1, (len(BUDGETS), feature_count))
        self.bias: Array = np.zeros(len(BUDGETS))

    def predict(self, features: Array) -> Array:
        values = np.asarray(features, dtype=float)
        return _sigmoid(cast(Array, values @ self.weights.T + self.bias))

    def fit(
        self,
        features: Array,
        labels: Array,
        *,
        l2: float = 1e-3,
        epochs: int = 600,
        learning_rate: float = 0.05,
    ) -> None:
        """Independent per-budget logistic fit, matching H1's loss exactly.

        Using an identical loss and optimiser is what makes E4 a fair
        monotone-versus-free-form comparison rather than an optimiser
        comparison.
        """

        values, targets = _check_design(features, labels)
        if targets.ndim != 2 or targets.shape[1] != len(BUDGETS):
            raise ValueError("one label column is required per budget")
        targets = cast(Array, np.clip(targets, 0.0, 1.0))
        scale = float(targets.size)
        optimiser = _Adam([self.weights.shape, self.bias.shape], learning_rate)
        for _ in range(epochs):
            residual = (self.predict(values) - targets) / scale
            grads: list[Array] = [
                cast(Array, residual.T @ values + 2.0 * l2 * self.weights),
                cast(Array, residual.sum(axis=0)),
            ]
            updated = optimiser.step([self.weights, self.bias], grads)
            self.weights, self.bias = updated[0], updated[1]


class BudgetClassifier:
    """Direct K-way budget classifier ablation (APC-06 §E4)."""

    def __init__(self, feature_count: int, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.weights: Array = rng.normal(0.0, 0.1, (len(BUDGETS), feature_count))
        self.bias: Array = np.zeros(len(BUDGETS))

    def predict(self, features: Array) -> Array:
        values = np.asarray(features, dtype=float)
        logits = values @ self.weights.T + self.bias
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return cast(Array, probabilities / probabilities.sum(axis=1, keepdims=True))

    def fit(
        self,
        features: Array,
        labels: Array,
        *,
        l2: float = 1e-3,
        epochs: int = 600,
        learning_rate: float = 0.05,
    ) -> None:
        """Softmax cross-entropy over the argmax budget of each label row."""

        values, targets = _check_design(features, labels)
        if targets.ndim != 2 or targets.shape[1] != len(BUDGETS):
            raise ValueError("one label column is required per budget")
        one_hot = cast(Array, np.eye(len(BUDGETS))[np.argmax(targets, axis=1)])
        scale = float(len(values))
        optimiser = _Adam([self.weights.shape, self.bias.shape], learning_rate)
        for _ in range(epochs):
            residual = (self.predict(values) - one_hot) / scale
            grads: list[Array] = [
                cast(Array, residual.T @ values + 2.0 * l2 * self.weights),
                cast(Array, residual.sum(axis=0)),
            ]
            updated = optimiser.step([self.weights, self.bias], grads)
            self.weights, self.bias = updated[0], updated[1]


class RateAdherenceHead:
    """H2: L0-only regression of ``log(r / b)`` with inversion (contribution C4)."""

    def __init__(self, feature_count: int) -> None:
        self.weights: Array = np.zeros(feature_count)
        self.bias = 0.0

    def predict(self, features: Array) -> Array:
        values = np.asarray(features, dtype=float)
        multiplier = np.exp(np.clip(values @ self.weights + self.bias, -5.0, 5.0))
        return cast(Array, np.clip(multiplier[:, None] * BUDGETS[None, :], 0.0, 1.0))

    def fit(
        self,
        features: Array,
        realised: Array,
        requested: Array = BUDGETS,
        *,
        l2: float = 1e-3,
    ) -> None:
        values, targets = _check_design(features, realised)
        repeated = np.broadcast_to(requested, targets.shape)
        design = np.column_stack((values, np.ones(len(values))))
        ratio = np.log(np.maximum(targets / repeated, 1e-6)).mean(axis=1)
        coefficients = _ridge_solve(cast(Array, design), cast(Array, ratio), l2)
        self.weights = coefficients[:-1]
        self.bias = float(coefficients[-1])

    def invert(self, features: Array, target_rate: float) -> Array:
        predictions = self.predict(features)
        indices = np.abs(predictions - target_rate).argmin(axis=1)
        return cast(Array, BUDGETS[indices])


class OutputLengthHead:
    """H3: per-budget output-token curve head (contribution C2).

    The prediction varies with the budget.  A budget-independent output
    length would make the total-cost objective monotone in ``b`` by
    construction, so ``argmin cost s.t. safe`` could never diverge from
    ``min safe budget`` and experiment E9 could not measure anything.
    """

    def __init__(self, feature_count: int) -> None:
        self.weights: Array = np.zeros((len(BUDGETS), feature_count))
        self.bias: Array = np.zeros(len(BUDGETS))

    def predict(self, features: Array) -> Array:
        values = np.asarray(features, dtype=float)
        logits = values @ self.weights.T + self.bias
        return cast(Array, np.exp(np.clip(logits, -5.0, 20.0)))

    def fit(self, features: Array, output_tokens: Array, *, l2: float = 1e-3) -> None:
        """Log-normal regression of ``T_out`` independently per budget."""

        values, targets = _check_design(features, output_tokens)
        if targets.ndim != 2 or targets.shape[1] != len(BUDGETS):
            raise ValueError("one output-token column is required per budget")
        design = np.column_stack((values, np.ones(len(values))))
        logged = np.log(np.maximum(targets, 1.0))
        coefficients = _ridge_solve(cast(Array, design), cast(Array, logged), l2)
        self.weights = coefficients[:-1].T
        self.bias = coefficients[-1]


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
