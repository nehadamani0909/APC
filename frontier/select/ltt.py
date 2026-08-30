"""Learn-then-Test tail-risk calibration (APC-04 §3.4, APC-06 §E5c).

CRC controls the *expected* loss.  Learn-then-Test answers the different
product question -- "rarely bad" rather than "good on average" -- by
controlling ``P(L > tau) <= delta``.

LTT is a multiple-testing procedure: each lambda on the grid is its own
null hypothesis, each gets a valid p-value, and family-wise error is
controlled across them.  A single bound applied to a swept grid, taking the
first threshold that passes, controls nothing -- it is the classic
multiple-comparisons error, and it matters here precisely because the risk
curve need not be monotone.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from frontier.predict.heads import BUDGETS
from frontier.select.crc import Array, clipped_degradation, select_budget


def tail_losses(
    predicted_quality: Array,
    predicted_cost: Array,
    true_quality: Array,
    lambda_value: float,
) -> Array:
    """Per-example degradation loss at one threshold."""

    selected = select_budget(predicted_quality, predicted_cost, lambda_value)
    indices = np.searchsorted(BUDGETS, selected)
    return clipped_degradation(
        true_quality[:, -1], true_quality[np.arange(len(true_quality)), indices]
    )


def hoeffding_bentkus_p_value(violations: int, n: int, tau_rate: float) -> float:
    """Valid p-value for ``H0: P(L > tau) > tau_rate`` on ``n`` Bernoulli draws.

    Uses the Hoeffding bound; conservative but valid without distributional
    assumptions, which is what a distribution-free guarantee requires.
    """

    if n <= 0:
        return 1.0
    observed = violations / n
    if observed >= tau_rate:
        return 1.0
    return float(math.exp(-2.0 * n * (tau_rate - observed) ** 2))


def calibrate_ltt(
    predicted_quality: Array,
    predicted_cost: Array,
    true_quality: Array,
    *,
    tau: float,
    delta: float,
    max_violation_rate: float | None = None,
) -> float:
    """Smallest lambda whose tail-risk hypothesis survives Holm correction.

    Returns the smallest admissible threshold, or ``1.0`` (abstain
    everywhere) when no hypothesis is rejected.
    """

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    n = len(predicted_quality)
    if n == 0:
        raise ValueError("LTT calibration requires at least one example")
    target = delta if max_violation_rate is None else max_violation_rate

    candidates = sorted({0.0, 1.0, *predicted_quality.ravel().tolist()})
    p_values: dict[float, float] = {}
    for lambda_value in candidates:
        losses = tail_losses(
            predicted_quality, predicted_cost, true_quality, lambda_value
        )
        violations = int((losses > tau).sum())
        p_values[lambda_value] = hoeffding_bentkus_p_value(violations, n, target)

    # Holm step-down over the lambda grid: sort ascending by p-value and
    # reject while p_(i) <= delta / (m - i + 1).
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    rejected: list[float] = []
    for index, (lambda_value, p_value) in enumerate(ordered):
        if p_value <= delta / (len(ordered) - index):
            rejected.append(lambda_value)
        else:
            break
    return min(rejected) if rejected else 1.0


def ltt_report(
    predicted_quality: Array,
    predicted_cost: Array,
    true_quality: Array,
    *,
    tau: float,
    delta: float,
) -> dict[str, Any]:
    lambda_hat = calibrate_ltt(
        predicted_quality, predicted_cost, true_quality, tau=tau, delta=delta
    )
    losses = tail_losses(predicted_quality, predicted_cost, true_quality, lambda_hat)
    return {
        "lambda_hat": lambda_hat,
        "tau": tau,
        "delta": delta,
        "empirical_tail_rate": float((losses > tau).mean()),
        "n": int(len(predicted_quality)),
    }


__all__ = [
    "calibrate_ltt",
    "hoeffding_bentkus_p_value",
    "ltt_report",
    "tail_losses",
]
