"""Learn-then-Test tail-risk calibration helper."""

from __future__ import annotations

import math

from frontier.select.crc import Array, empirical_risk


def calibrate_ltt(
    predicted_quality: Array,
    predicted_cost: Array,
    true_quality: Array,
    *,
    tau: float,
    delta: float,
) -> float:
    """Return the first threshold whose Hoeffding upper bound controls tails."""

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    candidates = sorted({0.0, 1.0, *predicted_quality.ravel().tolist()})
    n = len(predicted_quality)
    for threshold in candidates:
        selected_risk = empirical_risk(
            predicted_quality, predicted_cost, true_quality, threshold
        )
        # A bounded-loss upper bound is used as a conservative tail proxy.
        upper = selected_risk + math.sqrt(math.log(1.0 / delta) / (2.0 * n))
        if upper <= tau:
            return float(threshold)
    return 1.0
