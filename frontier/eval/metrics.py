"""Evaluation scalars and paired uncertainty estimates for P9."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

Array = np.ndarray[Any, np.dtype[Any]]


def cpgr(ours: float, fixed: float, oracle: float) -> float:
    """Compression performance gap recovered against the B2b-to-oracle gap.

    Deliberately not clipped at zero. Gate 3 is defined as beating B2b, so a
    negative CPGR -- the policy doing WORSE than the per-family tuned fixed
    budget -- is the single most important value this can take, and clipping
    it to 0.0 would report a gate failure as a neutral result.
    """
    denominator = oracle - fixed
    if denominator == 0.0:
        return 0.0
    return float((ours - fixed) / denominator)


def cpt(cost: Array, quality: Array, target: float) -> float:
    """Minimum cost at or above a fraction of uncompressed quality."""
    feasible = cost[quality >= target]
    return float(feasible.min()) if len(feasible) else float("nan")


def quality_cost_auc(cost: Array, quality: Array) -> float:
    order = np.argsort(cost)
    return float(np.trapz(quality[order], cost[order]))


def regret(actual: float, optimum: float) -> float:
    return float(actual - optimum)


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    low: float
    high: float


def paired_bca(
    values: Array,
    statistic: str = "mean",
    *,
    resamples: int = 10_000,
    seed: int = 0,
) -> ConfidenceInterval:
    """BCa interval for a prompt-level paired statistic.

    ``values`` contains one paired difference per prompt. Keeping the unit of
    analysis at prompt level prevents pseudo-replication across budgets.
    """
    if statistic != "mean":
        raise ValueError("only the paired mean statistic is supported")
    if len(values) < 2:
        raise ValueError("BCa requires at least two prompt-level values")
    values = np.asarray(values, dtype=float)
    estimate = float(values.mean())
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [
            values[rng.integers(0, len(values), len(values))].mean()
            for _ in range(resamples)
        ]
    )
    proportion = float(np.mean(bootstrap < estimate))
    proportion = float(np.clip(proportion, 1e-6, 1.0 - 1e-6))
    z0 = NormalDist().inv_cdf(proportion)
    jackknife = np.asarray(
        [(values.sum() - value) / (len(values) - 1) for value in values]
    )
    centered = jackknife.mean() - jackknife
    denominator = 6.0 * np.sum(centered**2) ** 1.5
    acceleration = (
        float(np.sum(centered**3) / denominator) if denominator > 0.0 else 0.0
    )
    normal = NormalDist()
    bounds = []
    for alpha in (0.025, 0.975):
        z_alpha = normal.inv_cdf(alpha)
        adjusted = normal.cdf(
            z0 + (z0 + z_alpha) / (1.0 - acceleration * (z0 + z_alpha))
        )
        bounds.append(float(np.quantile(bootstrap, np.clip(adjusted, 0.0, 1.0))))
    return ConfidenceInterval(estimate, bounds[0], bounds[1])


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Return Holm step-down rejection decisions across baseline comparisons."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    decisions: dict[str, bool] = {name: False for name in p_values}
    for index, (name, p_value) in enumerate(ordered):
        if p_value <= alpha / (len(ordered) - index):
            decisions[name] = True
        else:
            break
    return decisions


def minimum_detectable_effect(n: int, standard_deviation: float = 0.3) -> float:
    """80%-power, two-sided normal approximation used in APC-06 §14."""
    if n <= 0 or standard_deviation < 0.0:
        raise ValueError("n must be positive and standard deviation non-negative")
    return float((1.96 + 0.84) * standard_deviation / np.sqrt(n))
