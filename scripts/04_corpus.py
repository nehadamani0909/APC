"""Build and validate a corpus parquet from a run's ledger.

Attaches the real ``source_document_id`` and ``split`` from the normalised
instances, so the leakage checks in ``validate_corpus`` have something to
check.  Without that join every row carries ``split="unspecified"`` and the
disjointness guarantee is unverifiable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontier.corpus.build_full import export_from_ledger
from frontier.corpus.validate import (
    corpus_stats,
    read_corpus,
    validate_corpus,
    write_stats_report,
)
from frontier.harness.ledger import read_ledger


def instance_metadata(
    instances_dir: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return ``(source_document_id, split)`` keyed by prompt id."""

    documents: dict[str, str] = {}
    splits: dict[str, str] = {}
    for path in sorted(instances_dir.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            prompt_id = str(record["id"])
            meta = record.get("meta", {})
            documents[prompt_id] = str(meta.get("source_document_id", prompt_id))
            split = str(record.get("split") or meta.get("split") or "")
            if split:
                splits[prompt_id] = split
    return documents, splits


def write_datasheet(stats: dict[str, object], output: Path, tier: str) -> None:
    lines = [
        "# Compression Frontier Corpus - datasheet",
        "",
        f"Generated from a real run (tier {tier}).",
        "",
        "| Measure | Value |",
        "|---|---:|",
    ]
    for key, value in stats.items():
        formatted = f"{value:.6f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {formatted} |")
    lines += [
        "",
        "## Provenance",
        "",
        "- Splits are assigned by source document, so a document reused across",
        "  instances cannot straddle a split boundary.",
        "- Quality is the graded rho: the mean over the k sampled generations",
        "  at each budget.",
        "- USD is reconciled against provider-reported usage to within 1% by",
        "  `validate_ledger`; a locally generated corpus is zero-priced and is",
        "  costed at an explicit price row at analysis time.",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("data/corpus/v1/corpus.parquet")
    )
    parser.add_argument("--instances-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--stats-report", type=Path, default=Path("reports/corpus_stats.md")
    )
    parser.add_argument(
        "--datasheet", type=Path, default=Path("docs/corpus_datasheet.md")
    )
    parser.add_argument("--tier", default="A")
    parser.add_argument("--api-spend-cap-usd", type=float)
    args = parser.parse_args()

    documents, splits = instance_metadata(args.instances_dir)
    if not documents:
        print(
            f"warning: no instances under {args.instances_dir}; "
            "split and source-document metadata will be placeholders"
        )
    export_from_ledger(
        read_ledger(args.ledger),
        args.output,
        tier=args.tier,
        source_document=documents or None,
        split=splits or None,
    )
    corpus = read_corpus(args.output)
    validate_corpus(corpus, api_spend_cap_usd=args.api_spend_cap_usd)
    write_stats_report(corpus, args.stats_report)
    stats = corpus_stats(corpus)
    write_datasheet(stats, args.datasheet, args.tier)
    print(f"validated corpus: {len(corpus)} rows -> {args.output}")
    print(f"datasheet: {args.datasheet}")


if __name__ == "__main__":
    main()
