"""Materialize and validate a corpus parquet from an existing ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

from frontier.corpus.build_full import export_from_ledger
from frontier.corpus.validate import read_corpus, validate_corpus, write_stats_report
from frontier.harness.ledger import read_ledger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-report", type=Path, required=True)
    parser.add_argument("--tier", default="A")
    args = parser.parse_args()
    export_from_ledger(read_ledger(args.ledger), args.output, tier=args.tier)
    corpus = read_corpus(args.output)
    validate_corpus(corpus)
    write_stats_report(corpus, args.stats_report)
    print(f"validated corpus: {len(corpus)} rows")


if __name__ == "__main__":
    main()
