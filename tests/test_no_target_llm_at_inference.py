import numpy as np

from frontier.select.policy import PolicyPrediction, RiskControlledPolicy


class PredictorOnly:
    def predict(self, x: str, q: str, task_family: str) -> PolicyPrediction:
        return PolicyPrediction(
            quality=np.array([0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 1.0]),
            cost=np.array([3.0, 2.0, 8.0, 6.0, 5.0, 4.0, 10.0]),
            realised_rate=np.array([0.22, 0.31, 0.39, 0.51, 0.64, 0.81, 1.0]),
        )


def test_select_uses_predictor_only() -> None:
    selection = RiskControlledPolicy(PredictorOnly(), 0.95).select("x", "q", "qa")
    assert selection.requested_b == 0.8
    assert not selection.abstained
