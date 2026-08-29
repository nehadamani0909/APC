import numpy as np

from frontier.select.crc import BUDGETS, empirical_risk


def test_empirical_risk_is_nonincreasing_in_lambda() -> None:
    predicted = np.tile(BUDGETS, (100, 1))
    costs = np.tile(np.arange(len(BUDGETS), dtype=float), (100, 1))
    truth = predicted.copy()
    risks = [
        empirical_risk(predicted, costs, truth, threshold) for threshold in BUDGETS
    ]
    assert all(first >= second for first, second in zip(risks, risks[1:], strict=False))
