from frontier.eval.policies import render_t3, smoke_table
from frontier.select.baselines import all_policies
from frontier.select.policy import Policy


def test_all_baselines_implement_one_policy_protocol() -> None:
    policies = all_policies()
    assert {"B0", "B2b", "B5a", "B5b", "B5c", "B7", "OURS"}.issubset(policies)
    assert all(isinstance(policy, Policy) for policy in policies.values())


def test_smoke_evaluator_labels_oracles_and_emits_ci() -> None:
    table = render_t3(smoke_table())
    assert "ORACLE (upper bound)" in table
    assert "95% paired bootstrap CI" in table
