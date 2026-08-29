from frontier.predict.frontier import FrontierPredictor


def test_frontier_predictor_emits_all_budget_heads_without_target_call() -> None:
    prediction = FrontierPredictor().predict("context text " * 20, "question", "qa")
    assert prediction.quality.shape == (7,)
    assert prediction.cost.shape == (7,)
    assert prediction.realised_rate.shape == (7,)
