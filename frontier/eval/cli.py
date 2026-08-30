"""Command-line entry point for the E3 policy evaluation."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Any


def _module() -> Any:
    return import_module("scripts.08_eval")


def run(corpus_path: Path, output: Path = Path("reports/t3.md")) -> None:
    """Evaluate every policy on *corpus_path* and write table T3."""

    _module().run(corpus_path, output=output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/corpus/v1/corpus.parquet")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/t3.md"))
    args = parser.parse_args()
    run(args.corpus, args.output)
    print(f"wrote {args.output}")
