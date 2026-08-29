"""Inference-time policy implementing the APC-04 algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from frontier.select.crc import BUDGETS

Array = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class Selection:
    requested_b: float
    predicted_quality: float | None
    predicted_cost: float | None
    overhead_ms: float
    abstained: bool


@dataclass(frozen=True)
class PolicyPrediction:
    quality: Array
    cost: Array
    realised_rate: Array


@runtime_checkable
class FrontierPredictor(Protocol):
    def predict(self, x: str, q: str, task_family: str) -> PolicyPrediction: ...


@runtime_checkable
class Policy(Protocol):
    def select(self, x: str, q: str, task_family: str) -> Selection: ...


class RiskControlledPolicy:
    """Select by predicted total cost subject to calibrated safety."""

    def __init__(self, predictor: FrontierPredictor, lambda_hat: float) -> None:
        self.predictor = predictor
        self.lambda_hat = lambda_hat

    def select(self, x: str, q: str, task_family: str) -> Selection:
        prediction = self.predictor.predict(x, q, task_family)
        quality = np.asarray(prediction.quality, dtype=float)
        cost = np.asarray(prediction.cost, dtype=float)
        realised_rate = np.asarray(prediction.realised_rate, dtype=float)
        if quality.shape != (len(BUDGETS),) or cost.shape != (len(BUDGETS),):
            raise ValueError("predictor must emit one value per budget")
        feasible = np.flatnonzero(quality >= self.lambda_hat)
        abstained = len(feasible) == 0
        if abstained:
            index = len(BUDGETS) - 1
        else:
            index = int(feasible[np.argmin(cost[feasible])])
        selected = float(BUDGETS[index])
        # Invert adherence: request the budget predicted to land closest to
        # the selected realised-rate target. The target is b_hat, not the
        # model's prediction at an arbitrary request (r != b in general).
        target_rate = selected
        request_index = int(np.abs(realised_rate - target_rate).argmin())
        requested = 1.0 if abstained else float(BUDGETS[request_index])
        return Selection(
            requested,
            float(quality[index]),
            float(cost[index]),
            0.0,
            abstained,
        )
