"""Rebuild the release reports from the committed corpus.

Scaffold output goes to ``reports/scaffold`` and is deliberately NOT copied
into ``paper/``: those numbers are placeholders, and a real BCa interval
around a placeholder reads exactly like a result.  ``paper/`` is populated
only from a real corpus and a trained predictor.
"""

import argparse
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
    train_module.train_from_corpus(
        corpus_path,
        Path("artifacts/predictor.json"),
        instances_dir=Path("data/raw"),
        report_path=Path("reports/e2_e4.md"),
    )
    eval_run(corpus_path)
    paper_figures = import_module("scripts.09_paper_figures")
    paper_figures.main()
    generate_reports()
    Path("paper/figures").mkdir(parents=True, exist_ok=True)
    Path("paper/tables").mkdir(parents=True, exist_ok=True)
    print(
        "Pipeline stages ran against the committed fixture corpus.\n"
        "Scaffold tables and figures are in reports/scaffold and are NOT "
        "results; paper/ is left empty until a real corpus and a trained "
        "predictor exist."
    )
    if args.fast:
        print("fast mode: reused the committed deterministic corpus")


if __name__ == "__main__":
    main()
