"""E3 policy comparison against a real corpus (APC-06 §4).

Every policy -- baselines, oracle, and ours -- goes through the one frozen
``Policy.select`` path and is scored by looking up the outcome the corpus
actually measured at the budget it chose.  Any baseline needing a special
evaluation path is a baseline you will be accused of handicapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from frontier.corpus.labels import CurveLabels
from frontier.corpus.text import PromptText
from frontier.eval.metrics import (
    ConfidenceInterval,
    cpgr,
    cpt,
    holm_bonferroni,
    paired_bca,
    quality_cost_auc,
)
from frontier.harness.prices import ModelPrice
from frontier.predict.heads import BUDGETS
from frontier.select.baselines import LookupOraclePolicy
from frontier.select.policy import Policy

Array = np.ndarray[Any, np.dtype[np.float64]]


def cost_matrix(
    curves: CurveLabels,
    input_tokens: Array,
    price: ModelPrice,
) -> Array:
    """USD per (prompt, budget) from measured tokens and a price row.

    Costing a locally generated corpus at a chosen price row is deliberate:
    a local model is zero-priced, which would make every cost comparison
    degenerate, and E9 explicitly sweeps the c_out/c_in ratio (APC-06 §11).
    The price row used is recorded in the report.
    """

    usd_in = input_tokens * price.input_usd_per_million / 1_000_000.0
    usd_out = curves.output_tokens * price.output_usd_per_million / 1_000_000.0
    return cast(Array, usd_in + usd_out)


@dataclass(frozen=True)
class PolicyOutcome:
    """Per-prompt outcomes for one policy, kept unaggregated for pairing."""

    policy: str
    quality: Array
    usd: Array
    requested_b: Array
    upper_bound: bool = False

    @property
    def mean_quality(self) -> float:
        return float(self.quality.mean())

    @property
    def mean_usd(self) -> float:
        return float(self.usd.mean())

    @property
    def mean_budget(self) -> float:
        return float(self.requested_b.mean())


def _budget_index(budget: float) -> int:
    return int(np.abs(BUDGETS - budget).argmin())


def evaluate_policy(
    name: str,
    policy: Policy,
    curves: CurveLabels,
    costs: Array,
    texts: Mapping[str, PromptText],
) -> PolicyOutcome:
    """Score one policy by looking up what the corpus measured at its choice."""

    quality: list[float] = []
    usd: list[float] = []
    chosen: list[float] = []
    for row, (prompt_id, family) in enumerate(
        zip(curves.prompt_ids, curves.families, strict=True)
    ):
        text = texts.get(prompt_id)
        context = text.context if text else ""
        query = text.query if text else ""
        if isinstance(policy, LookupOraclePolicy):
            policy.for_prompt(prompt_id)
        selection = policy.select(context, query, family)
        index = _budget_index(selection.requested_b)
        quality.append(float(curves.quality[row, index]))
        usd.append(float(costs[row, index]))
        chosen.append(float(BUDGETS[index]))
    return PolicyOutcome(
        name,
        cast(Array, np.asarray(quality, dtype=float)),
        cast(Array, np.asarray(usd, dtype=float)),
        cast(Array, np.asarray(chosen, dtype=float)),
        upper_bound=bool(getattr(policy, "upper_bound", False)),
    )


@dataclass(frozen=True)
class Comparison:
    """One baseline's paired comparison against the method."""

    policy: str
    mean_quality: float
    mean_usd: float
    mean_budget: float
    quality_delta: ConfidenceInterval | None
    usd_delta: ConfidenceInterval | None
    p_value: float | None
    rejected: bool
    upper_bound: bool


def _paired_p_value(differences: Array) -> float:
    """Two-sided paired bootstrap p-value for a zero mean difference."""

    if len(differences) < 2 or float(np.std(differences)) == 0.0:
        return 1.0
    rng = np.random.default_rng(0)
    centred = differences - differences.mean()
    resamples = np.asarray(
        [
            centred[rng.integers(0, len(centred), len(centred))].mean()
            for _ in range(2_000)
        ]
    )
    observed = abs(float(differences.mean()))
    return float((np.abs(resamples) >= observed).mean())


def compare_policies(
    outcomes: Sequence[PolicyOutcome],
    *,
    method: str = "OURS",
    alpha: float = 0.05,
) -> list[Comparison]:
    """Paired BCa deltas against the method, with Holm across the family.

    The unit of analysis is the prompt, and comparisons are paired, so the
    bootstrap resamples prompts rather than (prompt, budget) cells -- which
    would be pseudo-replication (APC-06 §14).
    """

    by_name = {outcome.policy: outcome for outcome in outcomes}
    if method not in by_name:
        raise ValueError(f"method {method!r} is not among the evaluated policies")
    ours = by_name[method]

    raw_p: dict[str, float] = {}
    deltas: dict[str, tuple[ConfidenceInterval, ConfidenceInterval]] = {}
    for outcome in outcomes:
        if outcome.policy == method:
            continue
        quality_delta = ours.quality - outcome.quality
        usd_delta = ours.usd - outcome.usd
        if len(quality_delta) >= 2:
            deltas[outcome.policy] = (
                paired_bca(quality_delta, resamples=10_000),
                paired_bca(usd_delta, resamples=10_000),
            )
            # Upper bounds are reported, never tested against.
            if not outcome.upper_bound:
                raw_p[outcome.policy] = _paired_p_value(quality_delta)
    decisions = holm_bonferroni(raw_p, alpha) if raw_p else {}

    results: list[Comparison] = []
    for outcome in outcomes:
        pair = deltas.get(outcome.policy)
        results.append(
            Comparison(
                outcome.policy,
                outcome.mean_quality,
                outcome.mean_usd,
                outcome.mean_budget,
                pair[0] if pair else None,
                pair[1] if pair else None,
                raw_p.get(outcome.policy),
                decisions.get(outcome.policy, False),
                outcome.upper_bound,
            )
        )
    return results


def headline_metrics(
    outcomes: Sequence[PolicyOutcome],
    *,
    method: str = "OURS",
    baseline: str = "B2b",
    oracle: str = "ORACLE (upper bound)",
    targets: Sequence[float] = (0.95, 0.98, 0.99),
) -> dict[str, float]:
    """CPGR, CPT(x%), and quality-vs-cost AUC (APC-06 §4)."""

    by_name = {outcome.policy: outcome for outcome in outcomes}
    missing = [name for name in (method, baseline, oracle) if name not in by_name]
    if missing:
        raise ValueError(f"missing policies for headline metrics: {missing}")
    ours, fixed, upper = by_name[method], by_name[baseline], by_name[oracle]

    uncompressed = by_name.get("B0")
    reference = uncompressed.mean_quality if uncompressed else 1.0
    costs = np.asarray([outcome.mean_usd for outcome in outcomes], dtype=float)
    qualities = np.asarray([outcome.mean_quality for outcome in outcomes], dtype=float)

    metrics: dict[str, float] = {
        "CPGR": cpgr(ours.mean_quality, fixed.mean_quality, upper.mean_quality),
        "quality_cost_auc": quality_cost_auc(costs, qualities),
    }
    for target in targets:
        metrics[f"CPT@{int(target * 100)}"] = cpt(
            costs, qualities, reference * target
        )
    return metrics
