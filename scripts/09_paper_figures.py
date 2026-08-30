"""Regenerate the P9 report SHAPES from placeholder inputs.

The figures are hand-drawn SVG paths and the tables are computed from
hardcoded values. They demonstrate the output format; they are not results,
and they are written to reports/scaffold rather than paper/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from frontier.eval.metrics import minimum_detectable_effect, paired_bca

OUT = Path("reports/scaffold")


def _svg(name: str, title: str, body: str) -> None:
    content = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="400" '
        'viewBox="0 0 720 400"><rect width="100%" height="100%" fill="white"/>'
        '<text x="30" y="35" font-family="sans-serif" font-size="20">'
        f"{title}</text>{body}"
        # Burned into the image itself, so the caveat survives being dropped
        # into a slide or a draft.
        '<text x="30" y="392" font-family="sans-serif" font-size="13" '
        'fill="#B00020">SCAFFOLD - illustrative shape only, not measured '
        "data</text></svg>"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(content, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 1.0, 20)
    quality = 0.75 + 0.2 * x
    ours = quality - 0.01
    ci = paired_bca(ours - quality)
    (OUT / "power.md").write_text(
        "# Power calculation\n\n"
        "For N=1,200 prompt-level paired observations and σ=0.3, the "
        f"minimum detectable paired effect at 80% power is "
        f"{minimum_detectable_effect(1200, 0.3):.3f} quality points "
        "(normal approximation; 95% critical value + 80% power value).\n\n"
        f"Smoke paired effect: {ci.estimate:.3f} "
        f"(95% BCa CI [{ci.low:.3f}, {ci.high:.3f}]).\n",
        encoding="utf-8",
    )
    (OUT / "tables_p9.md").write_text(
        "# P9 evaluation tables (smoke)\n\n"
        "| Metric | Estimate | 95% BCa CI |\n|---|---:|---:|\n"
        f"| Paired quality difference | {ci.estimate:.3f} | "
        f"[{ci.low:.3f}, {ci.high:.3f}] |\n",
        encoding="utf-8",
    )
    _svg(
        "f1.svg",
        "F1 Per-instance frontiers",
        '<line x1="60" y1="340" x2="680" y2="60" stroke="#0072B2"/>',
    )
    _svg(
        "f2.svg",
        "F2 Quality versus USD",
        '<polyline points="60,330 220,250 400,150 650,80" '
        'fill="none" stroke="#D55E00" stroke-width="3"/>',
    )
    _svg(
        "f3.svg",
        "F3 Risk-control validity",
        '<line x1="60" y1="300" x2="680" y2="300" '
        'stroke="#009E73" stroke-width="3"/><circle cx="360" cy="250" '
        'r="6" fill="#0072B2"/>',
    )
    _svg(
        "f4.svg",
        "F4 Feature-tier latency versus benefit",
        '<rect x="100" y="180" width="80" height="120" fill="#0072B2"/>'
        '<rect x="260" y="130" width="80" height="170" fill="#D55E00"/>'
        '<rect x="420" y="90" width="80" height="210" fill="#009E73"/>',
    )
    print(f"wrote SCAFFOLD placeholders to {OUT} (not results)")


if __name__ == "__main__":
    main()
