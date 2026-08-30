"""CRC calibration under NON-monotone empirical risk.

The pre-existing CRC tests set ``truth = prediction`` with identical rows, so
empirical risk was monotone by arithmetic and the calibration assertion
(``lambda_hat >= 0.0``) could not fail.  These construct the case the
guarantee actually has to survive.
"""

from typing import Any

import numpy as np

from frontier.select.crc import BUDGETS, calibrate_crc, empirical_risk

Array = np.ndarray[Any, np.dtype[np.float64]]

EPSILON = 0.05


def _pocketed_calibration_set(
    n: int = 50,
) -> tuple[Array, Array, Array]:
    """Risk is low at tiny lambda, high in the middle, low again at the top.

    The cheapest budget happens to be safe, every middle budget is unsafe,
    and the uncompressed budget is safe. Selecting the smallest admissible
    lambda therefore lands in a spurious low-risk pocket.
    """
    predicted_quality = np.tile(
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]), (n, 1)
    )
    predicted_cost = np.tile(np.arange(1.0, len(BUDGETS) + 1.0), (n, 1))
    true_quality = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (n, 1)
    )
    return predicted_quality, predicted_cost, true_quality


def test_empirical_risk_really_is_non_monotone_here() -> None:
    quality, cost, truth = _pocketed_calibration_set()
    risks = [empirical_risk(quality, cost, truth, value) for value in (0.05, 0.35, 0.7)]
    assert risks[0] == 0.0
    assert risks[1] == 1.0
    assert risks[2] == 0.0


def test_calibration_skips_the_spurious_low_lambda_pocket() -> None:
    quality, cost, truth = _pocketed_calibration_set()
    result = calibrate_crc(quality, cost, truth, EPSILON)

    # The pocket at lambda <= 0.1 satisfies the bound in isolation, so the
    # naive "smallest admissible lambda" rule would select it.
    n = len(quality)
    naive_admissible = [
        value
        for value, risk in result.risk_curve.items()
        if (n / (n + 1.0)) * risk + 1.0 / (n + 1.0) <= EPSILON
    ]
    assert min(naive_admissible) <= 0.1

    # The suffix rule must not: every lambda above the chosen one has to
    # satisfy the bound too.
    assert result.lambda_hat == 0.7
    for value, risk in result.risk_curve.items():
        if value >= result.lambda_hat:
            bound = (n / (n + 1.0)) * risk + 1.0 / (n + 1.0)
            assert bound <= EPSILON


def test_calibration_controls_risk_on_a_held_out_split() -> None:
    quality, cost, truth = _pocketed_calibration_set(n=100)
    calibration = calibrate_crc(quality[:50], cost[:50], truth[:50], EPSILON)
    held_out = empirical_risk(
        quality[50:], cost[50:], truth[50:], calibration.lambda_hat
    )
    assert held_out <= EPSILON


def test_monotone_risk_still_yields_the_classical_threshold() -> None:
    quality = np.tile(BUDGETS, (40, 1))
    cost = np.tile(np.arange(len(BUDGETS), dtype=float), (40, 1))
    result = calibrate_crc(quality, cost, quality.copy(), EPSILON)

    # Risk decreases strictly with lambda here, so the suffix rule must agree
    # exactly with the classical "smallest admissible lambda" infimum.
    risks = list(result.risk_curve.items())
    ordered = [risk for _, risk in sorted(risks)]
    assert all(
        first >= second
        for first, second in zip(ordered, ordered[1:], strict=False)
    )

    n = len(quality)
    classical = min(
        value
        for value, risk in risks
        if (n / (n + 1.0)) * risk + 1.0 / (n + 1.0) <= EPSILON
    )
    assert result.lambda_hat == classical
