"""Corpus validation, parquet export, and T2 datasheet statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from frontier.corpus.schema import CORPUS_METADATA_FIELDS
from frontier.harness.ledger import KEY_FIELDS, LEDGER_FIELDS, validate_ledger

CORPUS_FIELDS = (*LEDGER_FIELDS, *CORPUS_METADATA_FIELDS)


def _corpus_file(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    parquet = sorted(candidate.glob("*.parquet"))
    if parquet:
        return parquet[0]
    jsonl = sorted(candidate.glob("*.jsonl"))
    if jsonl:
        return jsonl[0]
    raise FileNotFoundError(f"No parquet or JSONL corpus found under {candidate}")


def read_corpus(path: str | Path) -> pd.DataFrame:
    """Load a corpus file or the first supported file in a corpus directory."""

    source = _corpus_file(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    return pd.read_json(source, lines=True)


def validate_corpus(
    path_or_frame: str | Path | pd.DataFrame,
    *,
    api_spend_cap_usd: float | None = None,
) -> bool:
    """Validate schema, leakage controls, ledger integrity, and spend cap."""

    frame = (
        path_or_frame
        if isinstance(path_or_frame, pd.DataFrame)
        else read_corpus(path_or_frame)
    )
    missing = [field for field in CORPUS_FIELDS if field not in frame.columns]
    problems: list[str] = []
    if missing:
        problems.append(f"missing fields: {missing}")
    if frame.empty:
        problems.append("corpus is empty")
    fields_present = [field for field in CORPUS_FIELDS if field in frame]
    if not frame.empty and frame.loc[:, fields_present].isna().any().any():
        problems.append("NaN values present")
    if not frame.empty and frame.duplicated(subset=list(KEY_FIELDS)).any():
        problems.append("duplicate grid keys present")
    if not frame.empty:
        for identifier in ("prompt_id", "source_document_id"):
            split_counts = frame.groupby(identifier)["split"].nunique()
            if (split_counts > 1).any():
                problems.append(f"{identifier} appears in multiple splits")
    if not problems:
        try:
            validate_ledger(frame.loc[:, LEDGER_FIELDS])
        except ValueError as exc:
            problems.append(str(exc))
    if api_spend_cap_usd is not None and not frame.empty:
        total = float(frame["usd_total"].sum())
        if total > api_spend_cap_usd:
            problems.append(
                f"API spend cap exceeded: {total:.6f} > {api_spend_cap_usd:.6f}"
            )
    if problems:
        raise ValueError("Corpus validation failed: " + "; ".join(problems))
    return True


def export_parquet(frame: pd.DataFrame, output: str | Path) -> Path:
    """Validate and export a corpus DataFrame to parquet."""

    validate_corpus(frame)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, CORPUS_FIELDS].to_parquet(destination, index=False)
    return destination


def corpus_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Return the reproducible statistics used by table T2."""

    validate_corpus(frame)
    return {
        "rows": int(len(frame)),
        "prompts": int(frame["prompt_id"].nunique()),
        "families": int(frame["family"].nunique()),
        "budgets": int(frame["requested_b"].nunique()),
        "backends": int(frame["backend"].nunique()),
        "target_models": int(frame["target_model"].nunique()),
        "generations": int(len(frame)),
        "usd_total": float(frame["usd_total"].sum()),
        "gpu_hours": float(frame["gpu_seconds"].sum() / 3600.0),
    }


def write_stats_report(frame: pd.DataFrame, output: str | Path) -> Path:
    stats = corpus_stats(frame)
    lines = [
        "# Corpus statistics",
        "",
        "## T2",
        "",
        "| Measure | Value |",
        "|---|---:|",
    ]
    for key, value in stats.items():
        formatted = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {formatted} |")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--api-spend-cap-usd", type=float)
    parser.add_argument("--stats-report", type=Path)
    args = parser.parse_args()
    frame = read_corpus(args.path)
    validate_corpus(frame, api_spend_cap_usd=args.api_spend_cap_usd)
    if args.stats_report is not None:
        write_stats_report(frame, args.stats_report)
    print(f"valid corpus: {len(frame)} rows")


if __name__ == "__main__":
    main()
