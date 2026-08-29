"""Run a real/injected compressor over 20 contexts and all budget rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from frontier.compress.base import WhitespaceTokenizer
from frontier.compress.baselines import TruncateTailCompressor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ][:20]
    compressor = TruncateTailCompressor(WhitespaceTokenizer())
    for row in rows:
        for rate in (0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0):
            result = compressor.compress(str(row["context"]), str(row["query"]), rate)
            print(
                f"{row['id']} requested={rate:.2f} realised={result.realised_rate:.3f}"
            )


if __name__ == "__main__":
    main()
