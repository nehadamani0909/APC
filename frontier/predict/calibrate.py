"""H1 probability calibration on D_cal_a only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

Array = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class CalibrationResult:
    probabilities: Array
    method: str
    temperature: float


def _sigmoid(values: Array) -> Array:
    return cast(Array, 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0))))


def _binary_nll(probabilities: Array, labels: Array) -> float:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return float(
        -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
    )


def temperature_scale(predictions: Array, labels: Array) -> CalibrationResult:
    """Fit one temperature using D_cal_a predictions and labels."""

    logits = np.log(
        np.clip(predictions, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - predictions, 1e-6, 1.0)
    )
    candidates = np.linspace(0.25, 4.0, 151)
    scores = [
        _binary_nll(_sigmoid(logits / temperature), labels)
        for temperature in candidates
    ]
    temperature = float(candidates[int(np.argmin(scores))])
    return CalibrationResult(_sigmoid(logits / temperature), "temperature", temperature)


def isotonic_calibrate(predictions: Array, labels: Array) -> CalibrationResult:
    """Pool adjacent violators calibration, independently per budget.

    Use :meth:`IsotonicModel.fit` when the mapping must later be replayed on
    a different split; this helper returns in-sample calibrated values.
    """

    model = IsotonicModel.fit(predictions, labels)
    return CalibrationResult(model.transform(predictions), "isotonic", 1.0)


def expected_calibration_error(
    predictions: Array, labels: Array, bins: int = 10
) -> Array:
    result = np.zeros(predictions.shape[1])
    edges = np.linspace(0.0, 1.0, bins + 1)
    for budget in range(predictions.shape[1]):
        for index, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            column = predictions[:, budget]
            # Half-open bins, closed only at the top edge, so no prediction is
            # counted twice and the bin weights sum to exactly one.
            mask = (
                (column >= low) & (column <= high)
                if index == bins - 1
                else (column >= low) & (column < high)
            )
            if mask.any():
                result[budget] += mask.mean() * abs(
                    predictions[mask, budget].mean() - labels[mask, budget].mean()
                )
    return cast(Array, result)


@dataclass(frozen=True)
class IsotonicModel:
    """A fitted per-budget isotonic map that can be applied to new data.

    :func:`isotonic_calibrate` returns calibrated values for the calibration
    sample only.  CRC calibrates ``lambda`` on ``D_cal_b`` and reports on
    ``D_test``, so the mapping itself has to be persisted and replayed rather
    than refitted — refitting on the evaluation split would void the
    guarantee it exists to support.
    """

    thresholds: tuple[tuple[float, ...], ...]
    values: tuple[tuple[float, ...], ...]

    @classmethod
    def fit(cls, predictions: Array, labels: Array) -> IsotonicModel:
        thresholds: list[tuple[float, ...]] = []
        values: list[tuple[float, ...]] = []
        for budget in range(predictions.shape[1]):
            order = np.argsort(predictions[:, budget], kind="stable")
            knots = _pool_adjacent_violators(
                predictions[order, budget].astype(float).tolist(),
                labels[order, budget].astype(float).tolist(),
            )
            thresholds.append(tuple(edge for edge, _ in knots))
            values.append(tuple(value for _, value in knots))
        return cls(tuple(thresholds), tuple(values))

    def transform(self, predictions: Array) -> Array:
        if predictions.shape[1] != len(self.thresholds):
            raise ValueError("prediction budget count does not match the fitted model")
        calibrated = np.empty_like(np.asarray(predictions, dtype=float))
        for budget, (edges, fitted) in enumerate(
            zip(self.thresholds, self.values, strict=True)
        ):
            if not edges:
                calibrated[:, budget] = predictions[:, budget]
                continue
            index = np.searchsorted(np.asarray(edges, dtype=float),
                                    predictions[:, budget], side="right") - 1
            calibrated[:, budget] = np.asarray(fitted, dtype=float)[
                np.clip(index, 0, len(fitted) - 1)
            ]
        return cast(Array, calibrated)

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {
            "thresholds": [list(edges) for edges in self.thresholds],
            "values": [list(fitted) for fitted in self.values],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[list[float]]]) -> IsotonicModel:
        return cls(
            tuple(tuple(edges) for edges in payload["thresholds"]),
            tuple(tuple(fitted) for fitted in payload["values"]),
        )


def _pool_adjacent_violators(
    values: list[float], targets: list[float]
) -> list[tuple[float, float]]:
    """Return ``(left_edge, fitted_value)`` knots of the isotonic fit."""

    blocks: list[list[float]] = []
    for value, target in zip(values, targets, strict=True):
        blocks.append([value, target, 1.0])
        while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
            right, left = blocks.pop(), blocks.pop()
            weight = left[2] + right[2]
            blocks.append(
                [left[0], (left[1] * left[2] + right[1] * right[2]) / weight, weight]
            )
    return [(block[0], block[1]) for block in blocks]
