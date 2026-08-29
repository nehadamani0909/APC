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
    """Pool adjacent violators calibration, independently per budget."""

    calibrated = np.empty_like(predictions)
    for budget in range(predictions.shape[1]):
        order = np.argsort(predictions[:, budget])
        values = predictions[order, budget].astype(float).tolist()
        targets = labels[order, budget].astype(float).tolist()
        blocks: list[list[float]] = []
        for value, target in zip(values, targets, strict=True):
            blocks.append([value, target, 1.0])
            while len(blocks) >= 2 and blocks[-2][1] > blocks[-1][1]:
                right, left = blocks.pop(), blocks.pop()
                weight = left[2] + right[2]
                blocks.append(
                    [0.0, (left[1] * left[2] + right[1] * right[2]) / weight, weight]
                )
        fitted: list[float] = []
        for _, target, weight in blocks:
            fitted.extend([target] * int(weight))
        calibrated[order, budget] = fitted
    return CalibrationResult(calibrated, "isotonic", 1.0)


def expected_calibration_error(
    predictions: Array, labels: Array, bins: int = 10
) -> Array:
    result = np.zeros(predictions.shape[1])
    edges = np.linspace(0.0, 1.0, bins + 1)
    for budget in range(predictions.shape[1]):
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            mask = (predictions[:, budget] >= low) & (predictions[:, budget] <= high)
            if mask.any():
                result[budget] += mask.mean() * abs(
                    predictions[mask, budget].mean() - labels[mask, budget].mean()
                )
    return cast(Array, result)
