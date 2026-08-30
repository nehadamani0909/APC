"""Conformal Risk Control calibration for budget selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

Array = np.ndarray[Any, np.dtype[np.float64]]
BUDGETS = np.array((0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0), dtype=float)


@dataclass(frozen=True)
class CRCResult:
    lambda_hat: float
    epsilon: float
    risk_curve: dict[float, float]


def clipped_degradation(baseline_quality: Array, selected_quality: Array) -> Array:
    return cast(Array, np.maximum(0.0, baseline_quality - selected_quality))


def select_budget(
    predicted_quality: Array,
    predicted_cost: Array,
    lambda_value: float,
) -> Array:
    """Select the cheapest predicted-cost budget above a quality threshold."""

    selected: list[float] = []
    for quality, cost in zip(predicted_quality, predicted_cost, strict=True):
        feasible = np.flatnonzero(quality >= lambda_value)
        index = (
            int(feasible[np.argmin(cost[feasible])])
            if len(feasible)
            else len(BUDGETS) - 1
        )
        selected.append(float(BUDGETS[index]))
    return cast(Array, np.asarray(selected, dtype=float))


def empirical_risk(
    predicted_quality: Array,
    predicted_cost: Array,
    true_quality: Array,
    lambda_value: float,
) -> float:
    selected = select_budget(predicted_quality, predicted_cost, lambda_value)
    indices = np.searchsorted(BUDGETS, selected)
    losses = clipped_degradation(
        true_quality[:, -1], true_quality[np.arange(len(true_quality)), indices]
    )
    return float(np.mean(losses))


def calibrate_crc(
    predicted_quality: Array,
    predicted_cost: Array,
    true_quality: Array,
    epsilon: float,
) -> CRCResult:
    """Implement ``inf{λ: n/(n+1) Rhat + 1/(n+1) <= ε}`` on calibration data.

    The CRC bound (Angelopoulos et al., ICLR 2024) requires the risk to be
    non-increasing in ``λ``.  Population risk is monotone here by
    construction — raising ``λ`` shrinks the feasible set toward ``{1.0}``
    (APC-04 §3.4) — but the *empirical* ``Rhat_n`` is a step function over a
    finite sample and need not be, which matters precisely because this
    project's premise is that measured quality curves are non-monotone
    (audit D1).  Taking the smallest admissible ``λ`` would then be free to
    land in a spurious low-risk pocket and silently void the guarantee.

    So admissibility is required to hold across the whole suffix ``λ' >= λ``,
    the same monotone-safe construction used for ``b*`` in APC-04 §3.2.  On
    genuinely monotone risk this returns exactly the classical threshold.
    """

    if len(predicted_quality) == 0:
        raise ValueError("CRC calibration requires at least one example")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    candidates = sorted({0.0, 1.0, *predicted_quality.ravel().tolist()})
    n = len(predicted_quality)
    risks = {
        float(lambda_value): empirical_risk(
            predicted_quality, predicted_cost, true_quality, float(lambda_value)
        )
        for lambda_value in candidates
    }
    ordered = sorted(risks)
    lambda_hat = 1.0
    suffix_admissible = True
    for lambda_value in reversed(ordered):
        bound = (n / (n + 1.0)) * risks[lambda_value] + 1.0 / (n + 1.0)
        if bound > epsilon:
            suffix_admissible = False
        if suffix_admissible:
            lambda_hat = lambda_value
    return CRCResult(lambda_hat, epsilon, risks)
