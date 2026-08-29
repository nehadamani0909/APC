"""Gate 1 analysis: within-family frontier heterogeneity beyond sampling noise."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

BUDGETS = (0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)
EPSILONS = (0.02, 0.05, 0.10)


def _quality_map(frame: pd.DataFrame) -> dict[tuple[str, float], float]:
    grouped = frame.groupby(["prompt_id", "requested_b"])["quality"].mean()
    return {
        (str(prompt), round(float(rate), 2)): float(value)
        for (prompt, rate), value in grouped.items()
    }


def _budget_label(values: dict[float, float], epsilon: float, naive: bool) -> float:
    baseline = values[1.0]
    safe = [rate for rate in BUDGETS if values[rate] >= baseline - epsilon]
    if naive:
        return min(safe)
    for rate in safe:
        if all(
            values[larger] >= baseline - epsilon for larger in BUDGETS if larger >= rate
        ):
            return rate
    return 1.0


def _bootstrap_variance(
    values: list[float], seed: int = 0
) -> tuple[float, float, float]:
    if len(values) < 2:
        return 0.0, 0.0, 0.0
    variance = float(pd.Series(values).var(ddof=1))
    generator = random.Random(seed)
    samples = [
        float(pd.Series(generator.choices(values, k=len(values))).var(ddof=1))
        for _ in range(1000)
    ]
    return (
        variance,
        float(pd.Series(samples).quantile(0.025)),
        float(pd.Series(samples).quantile(0.975)),
    )


def _mean_ci(values: list[float], seed: int = 0) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = float(pd.Series(values).mean())
    generator = random.Random(seed)
    samples = [
        float(pd.Series(generator.choices(values, k=len(values))).mean())
        for _ in range(1000)
    ]
    return mean, float(pd.Series(samples).quantile(0.025)), float(
        pd.Series(samples).quantile(0.975)
    )


def _labels(frame: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    values = _quality_map(frame)
    records: list[dict[str, object]] = []
    for prompt_id in sorted(frame["prompt_id"].astype(str).unique()):
        prompt_values = {rate: values[(prompt_id, rate)] for rate in BUDGETS}
        family = str(
            frame.loc[frame["prompt_id"].astype(str) == prompt_id, "family"].iloc[0]
        )
        records.append(
            {
                "prompt_id": prompt_id,
                "family": family,
                "b_star": _budget_label(prompt_values, epsilon, False),
                "b_star_naive": _budget_label(prompt_values, epsilon, True),
            }
        )
    return pd.DataFrame.from_records(records)


def _noise_floor(frame: pd.DataFrame, epsilon: float) -> float:
    values: list[float] = []
    for prompt_id, prompt_frame in frame.groupby("prompt_id"):
        halves = []
        for parity in (0, 1):
            half = prompt_frame[prompt_frame["sample_idx"] % 2 == parity]
            quality = _quality_map(half)
            if len(quality) != len(BUDGETS):
                continue
            halves.append(
                _budget_label(
                    {rate: quality[(str(prompt_id), rate)] for rate in BUDGETS},
                    epsilon,
                    False,
                )
            )
        if len(halves) == 2:
            values.append((halves[0] - halves[1]) ** 2 / 2.0)
    return float(pd.Series(values).mean()) if values else 0.0


def write_gate1_report(ledger_path: str | Path, report_path: str | Path) -> None:
    frame = pd.read_json(ledger_path, lines=True)
    labels_by_epsilon = {epsilon: _labels(frame, epsilon) for epsilon in EPSILONS}
    report = [
        "# Gate 1 report",
        "",
        f"Rows: {len(frame)}",
        "",
        "> Provenance: this run uses the deterministic offline pilot target and "
        "> fixture-style prompts to validate the P3 harness path. Its PASS/FAIL "
        "> result is not evidence about a real target model; rerun with the pinned "
        "> local model before making the project go/no-go decision.",
        "",
    ]
    report.append("## Gate test")
    report.append("")
    passed_any = False
    for epsilon, labels in labels_by_epsilon.items():
        report.append(f"### ε = {epsilon:.2f}")
        report.append("")
        for family, family_labels in labels.groupby("family"):
            values = family_labels["b_star"].tolist()
            variance, ci_low, ci_high = _bootstrap_variance(values)
            noise = _noise_floor(frame[frame["family"] == family], epsilon)
            passed = ci_low > 2.0 * noise
            passed_any = passed_any or passed
            report.append(
                f"- {family}: Var[b*]={variance:.6f}; bootstrap CI="
                f"[{ci_low:.6f}, {ci_high:.6f}]; σ²_noise={noise:.6f}; "
                f"threshold={2.0 * noise:.6f}; **{'PASS' if passed else 'FAIL'}**"
            )
        disagreement = float((labels["b_star"] != labels["b_star_naive"]).mean())
        report.append(f"- E1b non-monotonicity disagreement: {disagreement:.4f}")
        report.append("")
    report.append(f"## Overall Gate 1 result: **{'PASS' if passed_any else 'FAIL'}**")
    report.append("")
    report.append("## E1c rate adherence")
    report.append("")
    for key, group in frame.groupby(["family", "backend", "requested_b"]):
        mean, low, high = _mean_ci(group["realised_r"].tolist())
        report.append(
            f"- {key}: mean realised rate={mean:.4f} "
            f"(95% bootstrap CI [{low:.4f}, {high:.4f}])"
        )
    report.append("")
    report.append("## E1d output expansion")
    report.append("")
    baseline = frame[frame["requested_b"] == 1.0].groupby("prompt_id")["T_out"].mean()
    expanded = frame.copy()
    expanded["expansion"] = expanded.apply(
        lambda row: float(row["T_out"]) / float(baseline[str(row["prompt_id"])] or 1.0),
        axis=1,
    )
    for key, group in expanded.groupby(["family", "requested_b"]):
        mean, low, high = _mean_ci(group["expansion"].tolist())
        report.append(
            f"- {key}: mean T_out(b)/T_out(1)={mean:.4f} "
            f"(95% bootstrap CI [{low:.4f}, {high:.4f}])"
        )
    report.append("")
    report.append("## Figure 1")
    report.append("")
    figure_path = Path(report_path).with_name("figure1.svg")
    _write_spaghetti_svg(frame, figure_path)
    report.append(f"![Figure 1](./{figure_path.name})")
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")


def _write_spaghetti_svg(frame: pd.DataFrame, path: Path) -> None:
    families = sorted(frame["family"].astype(str).unique())
    width, height = 900, 180 * max(1, len(families))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
    ]
    for family_index, family in enumerate(families):
        subset = frame[frame["family"] == family]
        x0, y0 = 40, family_index * 180 + 30
        lines.append(f'<text x="{x0}" y="{y0}" font-size="16">{family}</text>')
        for _, prompt in subset.groupby("prompt_id"):
            means = prompt.groupby("requested_b")["quality"].mean()
            points = []
            for index, rate in enumerate(BUDGETS):
                x = x0 + 70 + index * 110
                y = y0 + 120 - int(float(means.get(rate, 0.0)) * 100)
                points.append(f"{x},{y}")
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                'stroke="#4472c4" stroke-opacity="0.25"/>'
            )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    write_gate1_report(args.ledger, args.report)


if __name__ == "__main__":
    main()
