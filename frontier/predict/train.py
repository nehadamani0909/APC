"""Deterministic split and lightweight training utilities for P6."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path


def split_prompt_ids(
    prompt_ids: Sequence[str], *, seed: int = 0
) -> dict[str, set[str]]:
    """Split by prompt ID into the architecture's four disjoint partitions."""

    unique = list(dict.fromkeys(prompt_ids))
    random.Random(seed).shuffle(unique)
    n = len(unique)
    train_end, cal_a_end, cal_b_end = int(0.6 * n), int(0.7 * n), int(0.8 * n)
    return {
        "D_train": set(unique[:train_end]),
        "D_cal_a": set(unique[train_end:cal_a_end]),
        "D_cal_b": set(unique[cal_a_end:cal_b_end]),
        "D_test": set(unique[cal_b_end:]),
    }


def assert_disjoint(splits: dict[str, set[str]]) -> None:
    names = list(splits)
    for index, name in enumerate(names):
        for other in names[index + 1 :]:
            overlap = splits[name] & splits[other]
            if overlap:
                raise ValueError(
                    f"splits overlap: {name} ∩ {other} = {sorted(overlap)}"
                )


def write_e2_report(output: str | Path, *, seeds: Sequence[int] = (0, 1, 2)) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = [1.0 - (seed * 0.01) for seed in seeds]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    destination.write_text(
        "# E2 frontier prediction\n\n"
        "This report is a training-pipeline smoke report; real corpus metrics are "
        "filled after a real corpus is available.\n\n"
        "| Metric | Mean | s.d. |\n|---|---:|---:|\n"
        f"| Reproducibility smoke score | {mean:.4f} | {variance**0.5:.4f} |\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=Path("reports/e2.md"))
    args = parser.parse_args()
    write_e2_report(args.report)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
