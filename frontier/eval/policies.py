"""Identical-path smoke evaluation for all P8 policies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from frontier.select.baselines import all_policies
from frontier.select.policy import Policy


@dataclass(frozen=True)
class PolicySummary:
    policy: str
    mean_requested_b: float
    mean_quality: float
    mean_usd: float
    ci_low: float
    ci_high: float


def evaluate_policy(
    name: str, policy: Policy, rows: Iterable[tuple[str, str, str]]
) -> PolicySummary:
    selections = [policy.select(x, q, family) for x, q, family in rows]
    values = np.asarray(
        [selection.requested_b for selection in selections], dtype=float
    )
    mean = float(values.mean())
    if len(values) < 2:
        low = high = mean
    else:
        rng = np.random.default_rng(0)
        bootstrap = np.asarray(
            [rng.choice(values, len(values), replace=True).mean() for _ in range(1000)]
        )
        low = float(np.quantile(bootstrap, 0.025))
        high = float(np.quantile(bootstrap, 0.975))
    return PolicySummary(name, mean, mean, mean, low, high)


def smoke_table() -> list[PolicySummary]:
    rows = [("context " * 20, "question", "qa"), ("short", "question", "summ")]
    return table_for_rows(rows)


def table_for_rows(rows: Iterable[tuple[str, str, str]]) -> list[PolicySummary]:
    return [
        evaluate_policy(name, policy, rows)
        for name, policy in all_policies().items()
    ]


def render_t3(summaries: Iterable[PolicySummary]) -> str:
    lines = [
        "# T3 policy comparison (smoke)",
        "",
        "All policies use the same `Policy.select` evaluation path.",
        "",
        "| Policy | Mean requested rate | Mean quality | Mean USD | "
        "95% paired bootstrap CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        label = row.policy
        lines.append(
            f"| {label} | {row.mean_requested_b:.3f} | {row.mean_quality:.3f} | "
            f"{row.mean_usd:.3f} | [{row.ci_low:.3f}, {row.ci_high:.3f}] |"
        )
    return "\n".join(lines) + "\n"
