"""Feature extraction orchestration and CLI."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from frontier.features.base import FeatureExtractor, FeatureResult
from frontier.features.surface import SurfaceExtractor


def _records(path: Path) -> Iterable[tuple[str, str, str]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield str(record["id"]), str(record["context"]), str(record["query"])


def extract_records(
    records: Iterable[tuple[str, str, str]], extractor: FeatureExtractor
) -> pd.DataFrame:
    results: list[FeatureResult] = [
        extractor.extract(prompt_id, context, query)
        for prompt_id, context, query in records
    ]
    return pd.DataFrame.from_records([result.as_record() for result in results])


def write_feature_report(frame: pd.DataFrame, output: str | Path, tier: str) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mean_ms = float(frame["latency_ms"].mean()) if not frame.empty else 0.0
    p95_ms = float(frame["latency_ms"].quantile(0.95)) if not frame.empty else 0.0
    destination.write_text(
        "# Feature latency\n\n| Tier | Instances | Mean ms | P95 ms |\n"
        "|---|---:|---:|---:|\n"
        f"| {tier} | {len(frame)} | {mean_ms:.4f} | {p95_ms:.4f} |\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("L0", "L1", "L2"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--latency-report",
        type=Path,
        default=Path("reports/feature_latency.md"),
    )
    args = parser.parse_args()
    if args.tier != "L0":
        raise RuntimeError(
            "CLI L1/L2 wiring requires an explicitly injected verified model; "
            "use SmallLMExtractor or EncoderExtractor from Python"
        )
    frame = extract_records(_records(args.input), SurfaceExtractor())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)
    write_feature_report(frame, args.latency_report, args.tier)
    print(f"wrote {len(frame)} {args.tier} feature rows to {args.output}")


if __name__ == "__main__":
    main()
