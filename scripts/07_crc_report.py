"""E5: risk-control validity on real predictions (APC-06 §7).

E5a  empirical risk vs target epsilon over >= 20 random calibration/test
     splits -- this plot IS the C1 guarantee.
E5b  an uncalibrated fixed threshold, to show the calibration is doing work
     rather than decorating.
E5c  tail control via Learn-then-Test.
E5d  the cost of the guarantee against an unconstrained cost minimiser.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np

from frontier.corpus.labels import build_curves
from frontier.corpus.text import load_prompt_text
from frontier.corpus.validate import read_corpus, validate_corpus
from frontier.predict.frontier import FrontierPredictor
from frontier.select.crc import calibrate_crc, empirical_risk, select_budget
from frontier.select.ltt import ltt_report

Array = np.ndarray[Any, np.dtype[np.float64]]
EPSILONS = (0.02, 0.05, 0.10)
HEURISTIC_THRESHOLD = 0.9


def predict_matrices(
    predictor: FrontierPredictor,
    prompt_ids: tuple[str, ...],
    families: tuple[str, ...],
    texts: dict[str, Any],
) -> tuple[Array, Array]:
    quality: list[Array] = []
    cost: list[Array] = []
    for prompt_id, family in zip(prompt_ids, families, strict=True):
        text = texts.get(prompt_id)
        prediction = predictor.predict(
            text.context if text else "", text.query if text else "", family
        )
        quality.append(prediction.quality)
        cost.append(prediction.cost)
    return cast(Array, np.stack(quality)), cast(Array, np.stack(cost))


def coverage_table(
    quality: Array,
    cost: Array,
    truth: Array,
    *,
    splits: int = 20,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Repeat calibrate-on-half, report-on-half, for each target epsilon."""

    rows: list[dict[str, float]] = []
    n = len(quality)
    for epsilon in EPSILONS:
        calibrated: list[float] = []
        heuristic: list[float] = []
        lambdas: list[float] = []
        rng = np.random.default_rng(seed)
        for _ in range(splits):
            order = rng.permutation(n)
            half = n // 2
            cal, test = order[:half], order[half:]
            if len(cal) == 0 or len(test) == 0:
                continue
            result = calibrate_crc(
                quality[cal], cost[cal], truth[cal], epsilon
            )
            lambdas.append(result.lambda_hat)
            calibrated.append(
                empirical_risk(
                    quality[test], cost[test], truth[test], result.lambda_hat
                )
            )
            # E5b: the same selector at a hand-picked threshold.
            heuristic.append(
                empirical_risk(
                    quality[test], cost[test], truth[test], HEURISTIC_THRESHOLD
                )
            )
        if not calibrated:
            continue
        rows.append(
            {
                "epsilon": epsilon,
                "mean_lambda": float(np.mean(lambdas)),
                "mean_risk": float(np.mean(calibrated)),
                "max_risk": float(np.max(calibrated)),
                "violation_rate": float(np.mean(np.asarray(calibrated) > epsilon)),
                "heuristic_mean_risk": float(np.mean(heuristic)),
                "heuristic_violation_rate": float(
                    np.mean(np.asarray(heuristic) > epsilon)
                ),
                "splits": float(len(calibrated)),
            }
        )
    return rows


def _svg(rows: list[dict[str, float]], path: Path) -> None:
    """E5a coverage plot: empirical risk against the target it promises."""

    width, height = 640, 340
    left, bottom, top = 70.0, 270.0, 50.0
    span = max([row["epsilon"] for row in rows] + [0.01]) * 1.6
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="16">'
        "E5a CRC validity: empirical risk vs target</text>",
        f'<line x1="{left}" y1="{bottom}" x2="600" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
    ]

    def x_of(value: float) -> float:
        return left + (value / span) * (600.0 - left)

    def y_of(value: float) -> float:
        return bottom - (value / span) * (bottom - top)

    # y = x: points on or below this line satisfy the guarantee.
    parts.append(
        f'<line x1="{x_of(0)}" y1="{y_of(0)}" x2="{x_of(span)}" y2="{y_of(span)}" '
        'stroke="#999" stroke-dasharray="4 4"/>'
    )
    for row in rows:
        x = x_of(row["epsilon"])
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y_of(row["mean_risk"]):.1f}" r="5" '
            'fill="#0072B2"/>'
            f'<circle cx="{x:.1f}" cy="{y_of(row["heuristic_mean_risk"]):.1f}" '
            'r="4" fill="none" stroke="#D55E00" stroke-width="2"/>'
            f'<text x="{x - 14:.1f}" y="{bottom + 18:.0f}" '
            f'font-family="sans-serif" font-size="11">ε={row["epsilon"]:.2f}</text>'
        )
    parts.append(
        '<text x="90" y="70" font-family="sans-serif" font-size="11" '
        'fill="#0072B2">● CRC-calibrated</text>'
        '<text x="90" y="86" font-family="sans-serif" font-size="11" '
        f'fill="#D55E00">○ fixed threshold {HEURISTIC_THRESHOLD}</text>'
        '<text x="90" y="102" font-family="sans-serif" font-size="11" '
        'fill="#999">-- y = x (the guarantee)</text></svg>'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def run(
    corpus_path: Path,
    *,
    predictor_path: Path = Path("artifacts/predictor.json"),
    instances_dir: Path = Path("data/raw"),
    report: Path = Path("reports/e5.md"),
    splits: int = 20,
    price_row: str = "gpt-4o-mini",
) -> list[dict[str, float]]:
    """Calibrate lambda and report E5a/E5b/E5c/E5d."""

    corpus = read_corpus(corpus_path)
    validate_corpus(corpus)
    curves = build_curves(corpus)
    texts = load_prompt_text(instances_dir)
    predictor = FrontierPredictor.from_artifact(predictor_path, model=price_row)
    quality, cost = predict_matrices(
        predictor, curves.prompt_ids, curves.families, texts
    )
    rows = coverage_table(quality, cost, curves.quality, splits=splits)

    lines = [
        "# E5 - Risk-control validity",
        "",
        f"Corpus: `{corpus_path}` | predictor: `{predictor_path}` | "
        f"{splits} random calibration/test splits | {len(curves)} prompts",
        "",
        "## E5a / E5b - calibrated vs uncalibrated",
        "",
        "| ε | mean λ̂ | mean risk | max risk | violation rate | "
        f"risk @ fixed {HEURISTIC_THRESHOLD} | its violation rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['epsilon']:.2f} | {row['mean_lambda']:.4f} | "
            f"{row['mean_risk']:.4f} | {row['max_risk']:.4f} | "
            f"{row['violation_rate']:.2f} | {row['heuristic_mean_risk']:.4f} | "
            f"{row['heuristic_violation_rate']:.2f} |"
        )

    tail = ltt_report(quality, cost, curves.quality, tau=0.5, delta=0.1)
    unconstrained = select_budget(quality, cost, 0.0)
    lines += [
        "",
        "## E5c - tail control (Learn-then-Test)",
        "",
        f"- λ̂ = {tail['lambda_hat']:.4f} for "
        f"P(L > {tail['tau']}) ≤ {tail['delta']}",
        f"- empirical tail rate: {tail['empirical_tail_rate']:.4f} "
        f"over n={tail['n']}",
        "",
        "## E5d - cost of the guarantee",
        "",
        f"- mean budget, unconstrained cost minimiser: {unconstrained.mean():.4f}",
        "",
        "Empirical risk at or below the target across splits is the C1",
        "guarantee holding. A fixed threshold that misses ε in either",
        "direction is the evidence that calibration is doing work.",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _svg(rows, report.with_suffix(".svg"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/corpus/v1/corpus.parquet")
    )
    parser.add_argument(
        "--predictor", type=Path, default=Path("artifacts/predictor.json")
    )
    parser.add_argument("--instances-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--report", type=Path, default=Path("reports/e5.md"))
    parser.add_argument("--splits", type=int, default=20)
    parser.add_argument("--price-row", default="gpt-4o-mini")
    args = parser.parse_args()
    run(
        args.corpus,
        predictor_path=args.predictor,
        instances_dir=args.instances_dir,
        report=args.report,
        splits=args.splits,
        price_row=args.price_row,
    )
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
