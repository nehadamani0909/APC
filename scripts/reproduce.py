"""Rebuild all release reports from the committed corpus."""

from importlib import import_module
from pathlib import Path

from frontier.corpus.validate import read_corpus, validate_corpus, write_stats_report
from frontier.eval.cli import main as eval_main
from frontier.eval.p10 import generate_reports


def main() -> None:
    corpus_path = Path("data/corpus/v1/corpus.parquet")
    corpus = read_corpus(corpus_path)
    validate_corpus(corpus)
    write_stats_report(corpus, "reports/corpus_stats.md")
    eval_main()
    paper_figures = import_module("scripts.09_paper_figures")
    paper_figures.main()
    generate_reports()


if __name__ == "__main__":
    main()
