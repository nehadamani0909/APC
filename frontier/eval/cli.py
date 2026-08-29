"""Command-line entry point for the P8 policy smoke evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from frontier.corpus.validate import read_corpus, validate_corpus
from frontier.eval.policies import render_t3, smoke_table, table_for_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", choices=("all",), default="all")
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/corpus/v1/corpus.parquet")
    )
    args = parser.parse_args()
    if args.corpus.exists():
        corpus = read_corpus(args.corpus)
        validate_corpus(corpus)
        rows = [
            (f"context for {prompt_id} family {family}", task, family)
            for prompt_id, task, family in corpus[
                ["prompt_id", "task", "family"]
            ].drop_duplicates().itertuples(index=False, name=None)
        ]
        summaries = table_for_rows(rows)
    else:
        summaries = smoke_table()
    output = Path("reports/t3.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_t3(summaries), encoding="utf-8")
    print(f"wrote {output}")
