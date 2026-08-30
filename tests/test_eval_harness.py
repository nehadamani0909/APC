"""E3 evaluation harness, tuned baselines, and Learn-then-Test."""

from typing import Any

import numpy as np
import pytest

from frontier.corpus.labels import CurveLabels
from frontier.corpus.text import PromptText
from frontier.eval.harness import (
    compare_policies,
    cost_matrix,
    evaluate_policy,
    headline_metrics,
)
from frontier.harness.prices import ModelPrice
from frontier.predict.heads import BUDGETS
from frontier.select.baselines import (
    FixedBudgetPolicy,
    LookupOraclePolicy,
    PerFamilyFixedPolicy,
    tune_global_budget,
    tune_per_family_budgets,
)
from frontier.select.ltt import calibrate_ltt, hoeffding_bentkus_p_value, tail_losses

Array = np.ndarray[Any, np.dtype[np.float64]]


def _curves(n: int = 24) -> CurveLabels:
    rng = np.random.default_rng(0)
    quality = np.tile(np.linspace(0.2, 1.0, len(BUDGETS)), (n, 1))
    quality += rng.normal(0.0, 0.01, quality.shape)
    families = tuple("code" if index % 2 else "qa" for index in range(n))
    return CurveLabels(
        tuple(f"p{index}" for index in range(n)),
        families,
        np.clip(quality, 0.0, 1.0),
        np.tile(BUDGETS, (n, 1)),
        np.tile(np.linspace(200.0, 100.0, len(BUDGETS)), (n, 1)),
        np.tile(np.linspace(300.0, 1000.0, len(BUDGETS)), (n, 1)),
    )


def _texts(curves: CurveLabels) -> dict[str, PromptText]:
    return {
        pid: PromptText("some context here", "a query")
        for pid in curves.prompt_ids
    }


def test_tuned_global_budget_is_not_a_constant() -> None:
    # A curve that only becomes safe at 0.65 must tune to 0.65, not to a
    # hardcoded 0.5 -- B2b is the bar Gate 3 is defined against.
    quality = np.tile(np.array([0.1, 0.2, 0.3, 0.4, 0.97, 0.98, 1.0]), (10, 1))
    assert tune_global_budget(quality, 0.05) == 0.65
    assert tune_global_budget(quality, 0.9) == 0.2


def test_per_family_tuning_differs_between_families() -> None:
    fragile = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (6, 1))
    robust = np.tile(np.array([1.0] * 7), (6, 1))
    quality = np.vstack([fragile, robust])
    families = ["code"] * 6 + ["summ"] * 6
    budgets = tune_per_family_budgets(quality, families, 0.05)
    assert budgets["code"] == 1.0
    assert budgets["summ"] == 0.2
    # The per-family lookup is what B2b actually uses.
    policy = PerFamilyFixedPolicy(budgets)
    assert policy.select("x", "q", "code").requested_b == 1.0
    assert policy.select("x", "q", "summ").requested_b == 0.2
    # An unseen family falls back rather than raising mid-evaluation.
    assert policy.select("x", "q", "unseen").requested_b == 1.0


def test_evaluate_policy_reads_the_corpus_at_the_chosen_budget() -> None:
    curves = _curves()
    costs = cost_matrix(curves, curves.input_tokens, ModelPrice(1.0, 2.0))
    outcome = evaluate_policy(
        "B1@0.2", FixedBudgetPolicy(0.2), curves, costs, _texts(curves)
    )
    assert outcome.requested_b.tolist() == [0.2] * len(curves)
    # Quality must be the value the corpus measured at b=0.2, not a proxy.
    assert np.allclose(outcome.quality, curves.quality[:, 0])
    assert np.allclose(outcome.usd, costs[:, 0])


def test_oracle_selects_per_prompt_and_is_flagged_as_an_upper_bound() -> None:
    curves = _curves(n=4)
    costs = cost_matrix(curves, curves.input_tokens, ModelPrice(1.0, 2.0))
    budgets = {"p0": 0.2, "p1": 0.5, "p2": 0.8, "p3": 1.0}
    outcome = evaluate_policy(
        "ORACLE", LookupOraclePolicy(budgets), curves, costs, _texts(curves)
    )
    assert outcome.requested_b.tolist() == [0.2, 0.5, 0.8, 1.0]
    assert outcome.upper_bound is True


def test_comparison_pairs_over_prompts_and_excludes_upper_bounds() -> None:
    curves = _curves()
    costs = cost_matrix(curves, curves.input_tokens, ModelPrice(1.0, 2.0))
    texts = _texts(curves)
    outcomes = [
        evaluate_policy("B2b", FixedBudgetPolicy(0.2), curves, costs, texts),
        evaluate_policy(
            "ORACLE (upper bound)",
            LookupOraclePolicy({pid: 1.0 for pid in curves.prompt_ids}),
            curves,
            costs,
            texts,
        ),
        evaluate_policy("OURS", FixedBudgetPolicy(1.0), curves, costs, texts),
    ]
    comparisons = {row.policy: row for row in compare_policies(outcomes)}
    ours_vs_b2b = comparisons["B2b"]
    assert ours_vs_b2b.quality_delta is not None
    # b=1.0 scores strictly above b=0.2 on this curve.
    assert ours_vs_b2b.quality_delta.estimate > 0.0
    assert ours_vs_b2b.p_value is not None
    # The upper bound is reported but never enters the multiplicity correction.
    assert comparisons["ORACLE (upper bound)"].p_value is None
    assert comparisons["ORACLE (upper bound)"].upper_bound is True


def test_headline_metrics_expose_a_negative_cpgr() -> None:
    curves = _curves()
    costs = cost_matrix(curves, curves.input_tokens, ModelPrice(1.0, 2.0))
    texts = _texts(curves)
    outcomes = [
        evaluate_policy("B0", FixedBudgetPolicy(1.0), curves, costs, texts),
        evaluate_policy("B2b", FixedBudgetPolicy(0.8), curves, costs, texts),
        evaluate_policy(
            "ORACLE (upper bound)", FixedBudgetPolicy(1.0), curves, costs, texts
        ),
        # OURS deliberately worse than B2b: Gate 3 failing must be visible.
        evaluate_policy("OURS", FixedBudgetPolicy(0.2), curves, costs, texts),
    ]
    metrics = headline_metrics(outcomes)
    assert metrics["CPGR"] < 0.0


def test_cost_matrix_uses_both_input_and_output_tokens() -> None:
    curves = _curves(n=2)
    costs = cost_matrix(curves, curves.input_tokens, ModelPrice(1.0, 10.0))
    expected = (
        curves.input_tokens * 1.0 / 1e6 + curves.output_tokens * 10.0 / 1e6
    )
    assert np.allclose(costs, expected)


def test_ltt_p_value_is_one_when_violations_exceed_the_target() -> None:
    assert hoeffding_bentkus_p_value(50, 100, 0.1) == 1.0
    assert hoeffding_bentkus_p_value(0, 100, 0.5) < 0.01


def test_ltt_controls_the_tail_and_abstains_when_nothing_is_admissible() -> None:
    n = 200
    predicted = np.tile(np.linspace(0.2, 1.0, len(BUDGETS)), (n, 1))
    cost = np.tile(np.arange(len(BUDGETS), dtype=float), (n, 1))
    truth = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (n, 1))

    lambda_hat = calibrate_ltt(predicted, cost, truth, tau=0.5, delta=0.1)
    losses = tail_losses(predicted, cost, truth, lambda_hat)
    assert float((losses > 0.5).mean()) <= 0.1

    # An impossible target must abstain (lambda = 1.0), not silently pass.
    assert calibrate_ltt(predicted, cost, truth, tau=0.0, delta=1e-6) == 1.0


def test_ltt_rejects_invalid_parameters() -> None:
    values = np.tile(BUDGETS, (5, 1))
    with pytest.raises(ValueError):
        calibrate_ltt(values, values, values, tau=0.1, delta=0.0)
    with pytest.raises(ValueError):
        calibrate_ltt(values, values, values, tau=2.0, delta=0.1)
