"""P8 baseline policies sharing the frozen :class:`Policy` interface."""

from __future__ import annotations

import gzip
import random
import re
from collections.abc import Callable, Mapping

from frontier.select.policy import Policy, RiskControlledPolicy, Selection

BUDGETS = (0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)


def _selection(budget: float, *, overhead_ms: float = 0.0) -> Selection:
    return Selection(budget, None, None, overhead_ms, False)


def _length(x: str) -> int:
    return max(1, len(x.split()))


class FixedBudgetPolicy:
    def __init__(self, budget: float) -> None:
        if budget not in BUDGETS:
            raise ValueError(f"unsupported budget: {budget}")
        self.budget = budget

    def select(self, x: str, q: str, task_family: str) -> Selection:
        return _selection(self.budget)


class LengthPolicy:
    """B3: map context length quantiles to a budget, without labels."""

    def select(self, x: str, q: str, task_family: str) -> Selection:
        index = min(len(BUDGETS) - 1, _length(x) // 32)
        return _selection(BUDGETS[index])


class GzipRedundancyPolicy:
    """B3b: retain more text when gzip finds less redundancy."""

    def select(self, x: str, q: str, task_family: str) -> Selection:
        raw = max(1, len(x.encode("utf-8")))
        ratio = len(gzip.compress(x.encode("utf-8"))) / raw
        index = min(len(BUDGETS) - 1, max(0, round(ratio * (len(BUDGETS) - 1))))
        return _selection(BUDGETS[index])


class RandomBudgetPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select(self, x: str, q: str, task_family: str) -> Selection:
        return _selection(self.rng.choice(BUDGETS))


class RandomTokenDropPolicy(RandomBudgetPolicy):
    """B6: random-drop operating point, represented by its requested rate."""


class PointRatePolicy:
    """B5a: token-granularity point estimate, AdaComp-style."""

    def select(self, x: str, q: str, task_family: str) -> Selection:
        # Longer contexts receive a lower point rate; this is intentionally a
        # transparent, fitted-model-free adapter until corpus labels exist.
        rate = 1.0 if _length(x) < 16 else max(0.2, 1.0 - _length(x) / 160.0)
        budget = min(BUDGETS, key=lambda candidate: abs(candidate - rate))
        return _selection(budget)


def _overlap(x: str, q: str) -> float:
    context = set(re.findall(r"\w+", x.lower()))
    query = set(re.findall(r"\w+", q.lower()))
    return len(context & query) / max(1, len(query))


class QuerySelectPolicy:
    """B5b: query-aware variable rate using lexical token-level relevance."""

    def select(self, x: str, q: str, task_family: str) -> Selection:
        rate = 0.2 + 0.8 * min(1.0, _overlap(x, q))
        budget = min(BUDGETS, key=lambda candidate: abs(candidate - rate))
        return _selection(budget)


class ThresholdAdaptivePolicy:
    """B5c: cheap statistic threshold, standing in for attention mass."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def select(self, x: str, q: str, task_family: str) -> Selection:
        statistic = _overlap(x, q)
        budget = 1.0 if statistic >= self.threshold else 0.4
        return _selection(budget)


class ModelRoutingPolicy:
    """B7: route to a cheaper model; protocol has no model field.

    The frozen Selection contract can only express the uncompressed operating
    point. Evaluators must attach the selected model externally by policy name.
    """

    def select(self, x: str, q: str, task_family: str) -> Selection:
        return _selection(1.0)


class CachingSensitivityPolicy(FixedBudgetPolicy):
    """B8: fixed operating point used in cache-hit sensitivity sweeps."""


class OraclePolicy:
    """Oracle upper bound using an injected held-out safe-budget function."""

    upper_bound = True

    def __init__(self, budget_fn: Callable[[str, str, str], float]) -> None:
        self.budget_fn = budget_fn

    def select(self, x: str, q: str, task_family: str) -> Selection:
        budget = self.budget_fn(x, q, task_family)
        if budget not in BUDGETS:
            raise ValueError("oracle returned an unsupported budget")
        return _selection(budget)


class NoisyOraclePolicy(OraclePolicy):
    """Oracle-noisy upper-bound diagnostic with injected noisy labels."""

    upper_bound = True


def all_policies(
    *,
    ours: Policy | None = None,
    oracle_budget: Callable[[str, str, str], float] | None = None,
) -> Mapping[str, Policy]:
    if oracle_budget is None:
        def oracle_budget(x: str, q: str, family: str) -> float:
            return 1.0
    return {
        "B0": FixedBudgetPolicy(1.0),
        **{f"B1@{budget:g}": FixedBudgetPolicy(budget) for budget in BUDGETS},
        "B2a": FixedBudgetPolicy(0.5),
        "B2b": FixedBudgetPolicy(0.5),
        "B2c": FixedBudgetPolicy(0.5),
        "B3": LengthPolicy(),
        "B3b": GzipRedundancyPolicy(),
        "B4": RandomBudgetPolicy(),
        "B5a": PointRatePolicy(),
        "B5b": QuerySelectPolicy(),
        "B5c": ThresholdAdaptivePolicy(),
        "B6": RandomTokenDropPolicy(),
        "B7": ModelRoutingPolicy(),
        "B8": CachingSensitivityPolicy(0.5),
        "ORACLE (upper bound)": OraclePolicy(oracle_budget),
        "ORACLE-noisy (upper bound)": NoisyOraclePolicy(oracle_budget),
        "OURS": ours if ours is not None else FixedBudgetPolicy(0.5),
    }


__all__ = ["RiskControlledPolicy", "all_policies"]
