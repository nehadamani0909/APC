"""Generate the P7 E5a CRC validity smoke report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from frontier.select.crc import BUDGETS, calibrate_crc, empirical_risk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("reports/e5.md"))
    args = parser.parse_args()
    lines = ["# E5 risk-control validity", "", "20 random calibration/test splits.", ""]
    lines.append("| ε | Mean empirical risk | Maximum risk | Splits |")
    lines.append("|---:|---:|---:|---:|")
    summaries: list[tuple[float, float, float]] = []
    for epsilon in (0.02, 0.05, 0.10):
        risks: list[float] = []
        for seed in range(20):
            rng = np.random.default_rng(seed)
            prediction = np.sort(rng.uniform(0.2, 1.0, (100, len(BUDGETS))), axis=1)
            truth = prediction.copy()
            costs = np.tile(np.arange(len(BUDGETS), dtype=float), (100, 1))
            calibration = calibrate_crc(
                prediction[:50], costs[:50], truth[:50], epsilon
            )
            risks.append(
                empirical_risk(
                    prediction[50:], costs[50:], truth[50:], calibration.lambda_hat
                )
            )
        lines.append(
            f"| {epsilon:.2f} | {sum(risks) / len(risks):.6f} | "
            f"{max(risks):.6f} | {len(risks)} |"
        )
        summaries.append((epsilon, sum(risks) / len(risks), max(risks)))
    lines.extend(
        [
            "",
            "Empirical risk is evaluated on synthetic smoke data; replace with "
            "D_cal_b/D_test real data before reporting.",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot = args.report.with_suffix(".svg")
    circles = []
    for index, (epsilon, _mean, maximum) in enumerate(summaries):
        x = 180 + index * 160
        y = 250 - int(min(maximum, 0.1) / 0.1 * 180)
        circles.append(
            f'<circle cx="{x}" cy="{y}" r="5" fill="#0072B2"/>'
            f'<text x="{x - 15}" y="{y - 10}" font-family="sans-serif" '
            f'font-size="12">{maximum:.3f}</text>'
            f'<text x="{x - 15}" y="270" font-family="sans-serif" '
            f'font-size="12">ε={epsilon:.2f}</text>'
        )
    plot.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="300" '
        'viewBox="0 0 640 300"><rect width="100%" height="100%" fill="white"/>'
        '<text x="20" y="30" font-family="sans-serif" font-size="18">'
        'E5a CRC validity smoke plot</text><text x="20" y="285" '
        'font-family="sans-serif" font-size="12">Synthetic truth=prediction; '
        'risk ≤ ε</text>'
        '<line x1="70" y1="250" x2="610" y2="250" stroke="black"/>'
        '<line x1="70" y1="50" x2="70" y2="250" stroke="black"/>'
        + "".join(circles)
        + '<text x="80" y="65" font-family="sans-serif" font-size="12">max risk</text>'
        + '<text x="80" y="250" font-family="sans-serif" font-size="12">0</text></svg>',
        encoding="utf-8",
    )
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
