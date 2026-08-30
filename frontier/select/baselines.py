"""P8 baseline policies sharing the frozen :class:`Policy` interface."""

from __future__ import annotations

import gzip
import random
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from frontier.select.policy import Policy, RiskControlledPolicy, Selection

Array = np.ndarray[Any, np.dtype[np.float64]]
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


class PerFamilyFixedPolicy:
    """B2b/B2c: a validation-tuned fixed budget, looked up by task family.

    Gate 3 is defined against B2b, so this is the bar the method has to
    clear. It must be genuinely tuned on held-out data rather than pinned to
    a constant, or the comparison is meaningless.
    """

    def __init__(self, budgets: Mapping[str, float], fallback: float = 1.0) -> None:
        self.budgets = dict(budgets)
        self.fallback = fallback

    def select(self, x: str, q: str, task_family: str) -> Selection:
        return _selection(self.budgets.get(task_family, self.fallback))


def tune_global_budget(
    quality: Array, epsilon: float = 0.05
) -> float:
    """B2a: the smallest budget whose *mean* quality stays within epsilon.

    # DECISION: APC-06 §5 says "best fixed budget (validation-tuned)" without
    # naming the criterion. The population analogue of the monotone-safe b*
    # (APC-04 §3.2) is used here -- the smallest budget that, together with
    # every larger budget, keeps mean quality within epsilon of uncompressed.
    # Tuning the baseline under the same safety rule the method uses keeps
    # the comparison apples-to-apples, and sweeping epsilon traces the curve
    # that E3 compares.
    """

    means = quality.mean(axis=0)
    safe = means >= means[-1] - epsilon
    suffix = np.minimum.accumulate(safe[::-1])[::-1]
    for index in range(len(BUDGETS)):
        if suffix[index]:
            return float(BUDGETS[index])
    return float(BUDGETS[-1])


def tune_per_family_budgets(
    quality: Array, families: Sequence[str], epsilon: float = 0.05
) -> dict[str, float]:
    """B2b: :func:`tune_global_budget` applied within each task family."""

    budgets: dict[str, float] = {}
    for family in sorted(set(families)):
        mask = np.asarray([name == family for name in families], dtype=bool)
        if mask.any():
            budgets[family] = tune_global_budget(quality[mask], epsilon)
    return budgets


class LookupOraclePolicy:
    """Oracle upper bound: the true b* for each prompt, from held-out labels.

    Labelled an upper bound everywhere it is reported (APC-06 §5). It sees
    the answer, so it is a headroom measurement, never a competitor.
    """

    upper_bound = True

    def __init__(self, budget_by_prompt: Mapping[str, float]) -> None:
        self.budget_by_prompt = dict(budget_by_prompt)
        self._current: str | None = None

    def for_prompt(self, prompt_id: str) -> LookupOraclePolicy:
        self._current = prompt_id
        return self

    def select(self, x: str, q: str, task_family: str) -> Selection:
        if self._current is None:
            raise RuntimeError("oracle needs for_prompt() before select()")
        return _selection(self.budget_by_prompt.get(self._current, 1.0))


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
    global_budget: float = 0.5,
    family_budgets: Mapping[str, float] | None = None,
    family_model_budgets: Mapping[str, float] | None = None,
) -> Mapping[str, Policy]:
    """Every baseline behind the one frozen ``Policy`` interface.

    The tuned budgets are passed in rather than computed here so that tuning
    always happens on a validation split the caller controls, never on the
    split the results are reported from.
    """

    if oracle_budget is None:

        def oracle_budget(x: str, q: str, family: str) -> float:
            return 1.0

    return {
        "B0": FixedBudgetPolicy(1.0),
        **{f"B1@{budget:g}": FixedBudgetPolicy(budget) for budget in BUDGETS},
        "B2a": FixedBudgetPolicy(global_budget),
        "B2b": PerFamilyFixedPolicy(family_budgets or {}, global_budget),
        "B2c": PerFamilyFixedPolicy(
            family_model_budgets or family_budgets or {}, global_budget
        ),
        "B3": LengthPolicy(),
        "B3b": GzipRedundancyPolicy(),
        "B4": RandomBudgetPolicy(),
        "B5a": PointRatePolicy(),
        "B5b": QuerySelectPolicy(),
        "B5c": ThresholdAdaptivePolicy(),
        "B6": RandomTokenDropPolicy(),
        "B7": ModelRoutingPolicy(),
        "B8": CachingSensitivityPolicy(global_budget),
        "ORACLE (upper bound)": OraclePolicy(oracle_budget),
        "ORACLE-noisy (upper bound)": NoisyOraclePolicy(oracle_budget),
        "OURS": ours if ours is not None else FixedBudgetPolicy(global_budget),
    }


__all__ = [
    "LookupOraclePolicy",
    "PerFamilyFixedPolicy",
    "RiskControlledPolicy",
    "all_policies",
    "tune_global_budget",
    "tune_per_family_budgets",
]
