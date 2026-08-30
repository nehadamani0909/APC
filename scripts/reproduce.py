"""Run the pipeline end to end: corpus -> train -> calibrate -> evaluate.

``--fast`` uses a small sample so the whole chain runs in minutes.

Scaffold output goes to ``reports/scaffold`` and is never copied into
``paper/``: those numbers are placeholders, and a real BCa interval around a
placeholder reads exactly like a result.  ``paper/`` is populated only from a
real corpus and a trained predictor.
"""

from __future__ import annotations

import argparse
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any

from frontier.corpus.validate import read_corpus, validate_corpus, write_stats_report

REPORTS = Path("reports")


def _module(name: str) -> Any:
    return import_module(f"scripts.{name}")


def build_corpus_from_ledger(
    ledger: Path, corpus_path: Path, instances_dir: Path, tier: str
) -> None:
    corpus_module = _module("04_corpus")
    documents, splits = corpus_module.instance_metadata(instances_dir)
    from frontier.corpus.build_full import export_from_ledger
    from frontier.harness.ledger import read_ledger

    export_from_ledger(
        read_ledger(ledger),
        corpus_path,
        tier=tier,
        source_document=documents or None,
        split=splits or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/corpus/v1/corpus.parquet")
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        help="build the corpus from this ledger first (a real run)",
    )
    parser.add_argument("--instances-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--tier", default="A")
    parser.add_argument("--price-row", default="gpt-4o-mini")
    parser.add_argument("--skip-scaffold", action="store_true")
    args = parser.parse_args()

    if args.ledger is not None:
        print(f"[1/6] building corpus from {args.ledger}")
        build_corpus_from_ledger(
            args.ledger, args.corpus, args.instances_dir, args.tier
        )
    else:
        print("[1/6] using the existing corpus (pass --ledger to rebuild)")

    print("[2/6] validating corpus")
    corpus = read_corpus(args.corpus)
    validate_corpus(corpus)
    write_stats_report(corpus, REPORTS / "corpus_stats.md")

    print("[3/6] training predictor (D_train) and calibrating H1 (D_cal_a)")
    artifact = _module("06_train").train_from_corpus(
        args.corpus,
        Path("artifacts/predictor.json"),
        instances_dir=args.instances_dir,
        report_path=REPORTS / "e2_e4.md",
    )

    print("[4/6] E5 risk-control report (lambda on D_cal_b)")
    _module("07_crc_report").run(
        args.corpus,
        predictor_path=Path("artifacts/predictor.json"),
        instances_dir=args.instances_dir,
        report=REPORTS / "e5.md",
        splits=5 if args.fast else 20,
        price_row=args.price_row,
    )

    print("[5/6] E3 policy comparison (reported on D_test)")
    _module("08_eval").run(
        args.corpus,
        output=REPORTS / "t3.md",
        instances_dir=args.instances_dir,
        predictor_path=Path("artifacts/predictor.json"),
        price_row=args.price_row,
    )

    if not args.skip_scaffold:
        print("[6/6] regenerating placeholder report shapes (reports/scaffold)")
        _module("09_paper_figures").main()
        from frontier.eval.p10 import generate_reports

        generate_reports()
    else:
        print("[6/6] skipped scaffold placeholders")

    figures = Path("paper/figures")
    tables = Path("paper/tables")
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    # Only genuinely computed artefacts are promoted to paper/. Anything
    # still carrying the scaffold marker is held back: paper/ is where a
    # draft reaches for its tables, and a placeholder wrapped in a real
    # confidence interval is indistinguishable from a result.
    promoted: list[str] = []
    withheld: list[str] = []
    status = artifact.get("status")
    # A scaffold predictor makes OURS something other than the method, so
    # the whole E3 comparison stops being a result -- not just the training
    # report. Withhold everything rather than promoting a table whose
    # headline row is meaningless.
    predictor_is_scaffold = bool(status)
    for source, destination in (
        (REPORTS / "t3.md", tables / "t3.md"),
        (REPORTS / "e2_e4.md", tables / "e2_e4.md"),
        (REPORTS / "corpus_stats.md", tables / "t2.md"),
        (REPORTS / "e5.svg", figures / "f3.svg"),
    ):
        if not source.exists():
            continue
        marked = "SCAFFOLD" in source.read_text(encoding="utf-8", errors="ignore")
        if marked or predictor_is_scaffold:
            withheld.append(source.name)
            destination.unlink(missing_ok=True)
            continue
        shutil.copy2(source, destination)
        promoted.append(destination.name)

    print()
    print(f"promoted to paper/: {promoted or 'nothing'}")
    if withheld:
        print(f"withheld (not results): {withheld}")
    if status:
        print(f"NOTE: {status}")
    if args.fast:
        print("fast mode: small-sample run; not a scientific result")


if __name__ == "__main__":
    main()
