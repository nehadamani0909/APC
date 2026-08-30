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


def _noise_floor(frame: pd.DataFrame, epsilon: float) -> float | None:
    """Split-half estimate of the sampling-noise variance in ``b*``.

    Returns ``None`` when it cannot be estimated -- fewer than two sample
    indices, or budgets missing from a half.  Returning 0.0 there would be
    read as "there is no noise", and the gate would then compare the observed
    variance against a threshold of zero and look decisive when nothing was
    measured.  The gate is the project's go/no-go; it must not silently
    degrade.
    """
    values: list[float] = []
    rng = random.Random(0)
    for prompt_id, prompt_frame in frame.groupby("prompt_id"):
        sample_indices = sorted(prompt_frame["sample_idx"].unique().tolist())
        if len(sample_indices) < 2:
            continue
        for _ in range(100):
            first_indices = set(rng.sample(sample_indices, len(sample_indices) // 2))
            halves = []
            for selected in (first_indices, set(sample_indices) - first_indices):
                half = prompt_frame[prompt_frame["sample_idx"].isin(selected)]
                quality = _quality_map(half)
                if len(quality) != len(BUDGETS):
                    break
                halves.append(
                    _budget_label(
                        {rate: quality[(str(prompt_id), rate)] for rate in BUDGETS},
                        epsilon,
                        False,
                    )
                )
            if len(halves) == 2:
                values.append((halves[0] - halves[1]) ** 2 / 2.0)
    return float(pd.Series(values).mean()) if values else None


def check_validity(frame: pd.DataFrame) -> list[str]:
    """Preconditions the gate needs before its verdict means anything.

    Gate 1 asks whether compression tolerance varies across instances. Two
    situations make that unanswerable no matter how much data is collected,
    and both look like an ordinary FAIL if they are not checked:

    1. A prompt the model already fails uncompressed has no compression
       tolerance to measure. ``rho(x,1) = 0`` makes every budget satisfy
       ``rho(b) >= 0 - epsilon``, so ``b*`` collapses to the smallest budget
       for a reason that has nothing to do with compression.
    2. If compression appears to *improve* aggregate quality, the signal is
       below the noise floor -- removing context cannot systematically help.

    APC-05 §8 pre-commits the project to abandoning the method paper on a
    FAIL, so reporting one from an uninterpretable measurement is the most
    expensive mistake this module can make.
    """

    problems: list[str] = []
    quality = frame.assign(b=frame["requested_b"].round(4)).pivot_table(
        index="prompt_id", columns="b", values="quality", aggfunc="mean"
    )
    if 1.0 not in quality.columns:
        return ["no uncompressed (b=1.0) rows: rho(x,1) is undefined"]
    baseline = quality[1.0]
    unsolvable = int((baseline <= 0.0).sum())
    if unsolvable:
        share = unsolvable / len(baseline)
        problems.append(
            f"{unsolvable}/{len(baseline)} prompts ({share:.0%}) have "
            "rho(x,1) = 0: the target model never solves them uncompressed, "
            "so their b* is degenerate (every budget is trivially 'safe')"
        )
    solvable = int((baseline > 0.0).sum())
    if solvable < 10:
        problems.append(
            f"only {solvable} prompts are solvable uncompressed; too few to "
            "estimate within-family Var[b*]"
        )
    means = quality.mean()
    best_budget = float(means.idxmax())
    if best_budget < 1.0 and means.max() - means[1.0] > 0.02:
        problems.append(
            f"mean quality peaks at b={best_budget:g} ({means.max():.3f}), "
            f"above uncompressed ({means[1.0]:.3f}): compression cannot "
            "systematically improve accuracy, so the curves are noise-dominated"
        )
    return problems


def write_gate1_report(ledger_path: str | Path, report_path: str | Path) -> None:
    frame = pd.read_json(ledger_path, lines=True)
    validity = check_validity(frame)
    labels_by_epsilon = {epsilon: _labels(frame, epsilon) for epsilon in EPSILONS}
    report = [
        "# Gate 1 report",
        "",
        f"Rows: {len(frame)}",
        "",
        "> **Provenance.** Gate 1 is the project's go/no-go (APC-05 §2).",
        "> A PASS or FAIL here is only evidence about the target model,",
        "> compressor, and prompts that actually produced this ledger.",
        "> Check the corpus provenance before treating it as the decision.",
        "",
    ]
    if validity:
        report += [
            "## ⚠ Validity preconditions NOT met",
            "",
            "The gate's verdict is **not interpretable** on this data:",
            "",
            *[f"- {problem}" for problem in validity],
            "",
            "Everything below is reported for diagnosis only. Do not act on",
            "it, and in particular do not treat it as the APC-05 §8 FAIL that",
            "pre-commits the project to the corpus/negative-result paper.",
            "",
        ]
    report.append("## Gate test")
    report.append("")
    passed_all = True
    any_inconclusive = False
    for epsilon, labels in labels_by_epsilon.items():
        report.append(f"### ε = {epsilon:.2f}")
        report.append("")
        for family, family_labels in labels.groupby("family"):
            values = family_labels["b_star"].tolist()
            variance, ci_low, ci_high = _bootstrap_variance(values)
            noise = _noise_floor(frame[frame["family"] == family], epsilon)
            if noise is None:
                # Not a FAIL: a FAIL is a scientific claim that the
                # heterogeneity is within noise. This is "the test did not
                # run", which must never be reported as a verdict.
                any_inconclusive = True
                report.append(
                    f"- {family}: Var[b*]={variance:.6f}; bootstrap CI="
                    f"[{ci_low:.6f}, {ci_high:.6f}]; σ²_noise=**not estimable** "
                    f"(needs k>=2 samples per prompt at every budget); "
                    f"**INCONCLUSIVE**"
                )
                continue
            passed = ci_low > 2.0 * noise
            passed_all = passed_all and passed
            report.append(
                f"- {family}: Var[b*]={variance:.6f}; bootstrap CI="
                f"[{ci_low:.6f}, {ci_high:.6f}]; σ²_noise={noise:.6f}; "
                f"threshold={2.0 * noise:.6f}; **{'PASS' if passed else 'FAIL'}**"
            )
        gaps = (labels["b_star"] - labels["b_star_naive"]).abs().tolist()
        disagreement = float((labels["b_star"] != labels["b_star_naive"]).mean())
        gap_mean, gap_low, gap_high = _mean_ci(gaps)
        report.append(
            f"- E1b non-monotonicity disagreement: {disagreement:.4f}; "
            f"mean absolute budget gap={gap_mean:.4f} "
            f"(95% bootstrap CI [{gap_low:.4f}, {gap_high:.4f}])"
        )
        report.append("")
    if validity:
        verdict = "INCONCLUSIVE (validity preconditions not met)"
        note = (
            "The target model, not the compression, is the limiting factor "
            "here. Re-run with a model that solves the task uncompressed "
            "before reading any verdict from this gate."
        )
    elif any_inconclusive:
        verdict = "INCONCLUSIVE"
        note = (
            "At least one family could not have its noise floor estimated, so "
            "the gate did not run there. This is not a FAIL: a FAIL asserts "
            "that heterogeneity is within sampling noise, which was not "
            "measured. Re-run with k>=2 samples per prompt at every budget."
        )
    elif passed_all:
        verdict = "PASS"
        note = "Within-family Var[b*] exceeds twice the sampling-noise floor."
    else:
        verdict = "FAIL"
        note = (
            "Within-family Var[b*] does not exceed the noise floor. APC-05 §8 "
            "pre-commits to the C5 corpus/negative-result paper on this "
            "outcome; do not proceed on a borderline result."
        )
    report.append(f"## Overall Gate 1 result: **{verdict}**")
    report.append("")
    report.append(note)
    report.append("")
    report.append("## E1c rate adherence")
    report.append("")
    grouped = frame.assign(requested_b=frame["requested_b"].round(4))
    for key, group in grouped.groupby(["family", "backend", "requested_b"]):
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
    expanded["requested_b"] = expanded["requested_b"].round(4)
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
