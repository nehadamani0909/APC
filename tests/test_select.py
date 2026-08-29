import numpy as np

from frontier.select.crc import BUDGETS, calibrate_crc, empirical_risk
from frontier.select.policy import PolicyPrediction, RiskControlledPolicy


def test_crc_empirical_risk_is_nonincreasing_on_aligned_data() -> None:
    quality = np.tile(BUDGETS, (40, 1))
    costs = np.tile(np.arange(len(BUDGETS), dtype=float), (40, 1))
    result = np.tile(BUDGETS, (40, 1))
    lambdas = (0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)
    risks = [empirical_risk(quality, costs, result, value) for value in lambdas]
    assert all(first >= second for first, second in zip(risks, risks[1:], strict=False))
    assert calibrate_crc(quality, costs, result, 0.05).lambda_hat >= 0.0


class FakePredictor:
    def predict(self, x: str, q: str, task_family: str) -> PolicyPrediction:
        return PolicyPrediction(
            quality=np.array([0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 1.0]),
            cost=np.array([3.0, 2.0, 8.0, 6.0, 5.0, 4.0, 10.0]),
            realised_rate=np.array([0.22, 0.31, 0.39, 0.51, 0.64, 0.81, 1.0]),
        )


def test_policy_selects_cheapest_feasible_without_target_llm() -> None:
    selection = RiskControlledPolicy(FakePredictor(), 0.95).select("x", "q", "qa")
    assert selection.requested_b == 0.8
    assert not selection.abstained
