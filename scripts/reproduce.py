"""Rebuild all release reports from the committed corpus."""

import argparse
import shutil
from importlib import import_module
from pathlib import Path

from frontier.corpus.validate import read_corpus, validate_corpus, write_stats_report
from frontier.eval.cli import run as eval_run
from frontier.eval.p10 import generate_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    corpus_path = Path("data/corpus/v1/corpus.parquet")
    corpus = read_corpus(corpus_path)
    validate_corpus(corpus)
    write_stats_report(corpus, "reports/corpus_stats.md")
    train_module = import_module("scripts.06_train")
    train_module.train_from_corpus(corpus_path, Path("artifacts/predictor.json"))
    eval_run(corpus_path)
    paper_figures = import_module("scripts.09_paper_figures")
    paper_figures.main()
    generate_reports()
    figures = Path("paper/figures")
    tables = Path("paper/tables")
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    for source in Path("reports").glob("f*.svg"):
        shutil.copy2(source, figures / source.name)
    for source in (
        Path("reports") / "t3.md",
        Path("reports") / "tables_p9.md",
        *sorted(Path("reports").glob("t[4-9].md")),
    ):
        shutil.copy2(source, tables / source.name)
    if args.fast:
        print(
            "fast mode: reused the committed deterministic corpus and all "
            "pipeline stages"
        )


if __name__ == "__main__":
    main()
