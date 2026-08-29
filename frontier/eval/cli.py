"""Command-line entry point for the P8 policy smoke evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from frontier.eval.policies import render_t3, smoke_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policies", choices=("all",), default="all")
    parser.parse_args()
    output = Path("reports/t3.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_t3(smoke_table()), encoding="utf-8")
    print(f"wrote {output}")
