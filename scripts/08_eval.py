"""E3: compare every policy on the real corpus and emit table T3.

Baselines are tuned on a validation split (``D_cal_a`` by default) and every
number reported comes from ``D_test``.  Tuning and reporting on the same
split is the standard way this comparison gets accidentally rigged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from frontier.corpus.labels import CurveLabels, build_curves, monotone_safe_budget
from frontier.corpus.text import load_prompt_text
from frontier.corpus.validate import read_corpus, validate_corpus
from frontier.eval.harness import (
    Comparison,
    compare_policies,
    cost_matrix,
    evaluate_policy,
    headline_metrics,
)
from frontier.eval.metrics import minimum_detectable_effect
from frontier.harness.prices import PRICE_TABLE_VERSION, get_price
from frontier.predict.frontier import FrontierPredictor
from frontier.predict.train import assert_disjoint, resolve_splits
from frontier.select.baselines import (
    LookupOraclePolicy,
    all_policies,
    tune_global_budget,
    tune_per_family_budgets,
)
from frontier.select.crc import calibrate_crc
from frontier.select.policy import RiskControlledPolicy

DEFAULT_EPSILON = 0.05


Array = np.ndarray[Any, np.dtype[np.float64]]


def _subset(curves: CurveLabels, keep: set[str]) -> tuple[CurveLabels, Array]:
    indices = np.flatnonzero(
        np.asarray([pid in keep for pid in curves.prompt_ids], dtype=bool)
    )
    return (
        CurveLabels(
            tuple(curves.prompt_ids[i] for i in indices),
            tuple(curves.families[i] for i in indices),
            curves.quality[indices],
            curves.realised_rate[indices],
            curves.output_tokens[indices],
            curves.input_tokens[indices],
        ),
        indices,
    )


def render_t3(
    comparisons: list[Comparison],
    metrics: dict[str, float],
    context: dict[str, object],
) -> str:
    lines = [
        "# T3 - Policy comparison (E3)",
        "",
        "All policies run through the single frozen `Policy.select` path.",
        "Deltas are OURS minus the row, paired over prompts, 95% BCa.",
        "Holm-corrected across the baseline family; upper bounds are excluded",
        "from the correction and marked.",
        "",
    ]
    for key, value in context.items():
        lines.append(f"- **{key}**: {value}")
    lines += [
        "",
        "| Policy | Mean quality | Mean USD | Mean b | ΔQuality [95% BCa] | "
        "p (Holm) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        label = row.policy + (" *(upper bound)*" if row.upper_bound else "")
        delta = (
            f"{row.quality_delta.estimate:+.4f} "
            f"[{row.quality_delta.low:+.4f}, {row.quality_delta.high:+.4f}]"
            if row.quality_delta
            else "—"
        )
        if row.upper_bound or row.p_value is None:
            verdict = "—"
        else:
            verdict = f"{row.p_value:.4f} {'✓' if row.rejected else '✗'}"
        lines.append(
            f"| {label} | {row.mean_quality:.4f} | {row.mean_usd:.6f} | "
            f"{row.mean_budget:.3f} | {delta} | {verdict} |"
        )
    lines += ["", "## Headline metrics", "", "| Metric | Value |", "|---|---:|"]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:.6f} |")
    return "\n".join(lines) + "\n"


def run(
    corpus_path: Path,
    *,
    output: Path = Path("reports/t3.md"),
    instances_dir: Path | None = Path("data/raw"),
    predictor_path: Path | None = Path("artifacts/predictor.json"),
    price_row: str = "gpt-4o-mini",
    epsilon: float = DEFAULT_EPSILON,
    seed: int = 0,
) -> list[Comparison]:
    corpus = read_corpus(corpus_path)
    validate_corpus(corpus)
    curves = build_curves(corpus)
    splits = resolve_splits(corpus, seed=seed)
    assert_disjoint(splits)

    validation, _ = _subset(curves, splits["D_cal_a"])
    test, test_index = _subset(curves, splits["D_test"])
    if len(test) == 0:
        raise ValueError(
            "D_test is empty; nothing to report. Split sizes: "
            + ", ".join(f"{name}={len(ids)}" for name, ids in splits.items())
            + ". A pilot run drawn from a single split produces a corpus that "
            "cannot be reported on; re-run the pilot with --split all."
        )
    tuning = validation if len(validation) else test

    price = get_price(price_row, PRICE_TABLE_VERSION)
    costs = cost_matrix(test, test.input_tokens, price)
    texts = load_prompt_text(instances_dir)

    # Baselines are tuned on validation, never on the reported split.
    global_budget = tune_global_budget(tuning.quality, epsilon)
    family_budgets = tune_per_family_budgets(
        tuning.quality, tuning.families, epsilon
    )
    oracle_budgets = dict(
        zip(
            test.prompt_ids,
            (float(value) for value in monotone_safe_budget(test.quality, epsilon)),
            strict=True,
        )
    )

    ours = None
    lambda_hat = float("nan")
    if predictor_path is not None and Path(predictor_path).exists():
        predictor = FrontierPredictor.from_artifact(predictor_path, model=price_row)
        cal_b, cal_b_index = _subset(curves, splits["D_cal_b"])
        if len(cal_b):
            # lambda is calibrated on D_cal_b, which is disjoint from the
            # D_cal_a used to calibrate H1's probabilities.
            predicted = np.asarray(
                [
                    predictor.predict(
                        texts[pid].context if pid in texts else "",
                        texts[pid].query if pid in texts else "",
                        family,
                    )
                    for pid, family in zip(
                        cal_b.prompt_ids, cal_b.families, strict=True
                    )
                ],
                dtype=object,
            )
            quality_matrix = np.stack([p.quality for p in predicted])
            cost_matrix_cal = np.stack([p.cost for p in predicted])
            lambda_hat = calibrate_crc(
                quality_matrix, cost_matrix_cal, cal_b.quality, epsilon
            ).lambda_hat
            ours = RiskControlledPolicy(predictor, lambda_hat)

    policies = dict(
        all_policies(
            ours=ours,
            global_budget=global_budget,
            family_budgets=family_budgets,
        )
    )
    policies["ORACLE (upper bound)"] = LookupOraclePolicy(oracle_budgets)
    policies["ORACLE-noisy (upper bound)"] = LookupOraclePolicy(oracle_budgets)

    outcomes = [
        evaluate_policy(name, policy, test, costs, texts)
        for name, policy in policies.items()
    ]
    comparisons = compare_policies(outcomes)
    metrics = headline_metrics(outcomes)
    metrics["minimum_detectable_effect"] = minimum_detectable_effect(len(test))

    context: dict[str, object] = {
        "corpus": str(corpus_path),
        "reported split": f"D_test ({len(test)} prompts)",
        "tuning split": f"D_cal_a ({len(tuning)} prompts)",
        "price row": f"{price_row} (table {PRICE_TABLE_VERSION})",
        "epsilon": epsilon,
        "B2a tuned budget": global_budget,
        "B2b tuned budgets": json.dumps(family_budgets),
        "CRC lambda (D_cal_b)": (
            f"{lambda_hat:.4f}" if lambda_hat == lambda_hat else "no predictor"
        ),
        "prompt text available": bool(texts),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_t3(comparisons, metrics, context), encoding="utf-8")
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/corpus/v1/corpus.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/t3.md"))
    parser.add_argument("--instances-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--predictor", type=Path, default=Path("artifacts/predictor.json")
    )
    parser.add_argument("--price-row", default="gpt-4o-mini")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(
        args.corpus,
        output=args.output,
        instances_dir=args.instances_dir,
        predictor_path=args.predictor,
        price_row=args.price_row,
        epsilon=args.epsilon,
        seed=args.seed,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
