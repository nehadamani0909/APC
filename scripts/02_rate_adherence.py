"""Print a requested-versus-realised rate table for deterministic P2 backends."""

from __future__ import annotations

import argparse
from pathlib import Path

from frontier.compress.base import WhitespaceTokenizer
from frontier.compress.baselines import RandomDropCompressor, TruncateTailCompressor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    tokenizer = WhitespaceTokenizer()
    backends = [
        TruncateTailCompressor(tokenizer, cache_dir=args.cache_dir),
        RandomDropCompressor(tokenizer, seed=0, cache_dir=args.cache_dir),
    ]
    context = "one two three four five six seven eight nine ten"
    print("backend\trequested_b\trealised_r\tcache_hit")
    for backend in backends:
        for rate in (1.0, 0.8, 0.65, 0.5, 0.4):
            result = backend.compress(context, "question", rate)
            print(
                f"{backend.name}\t{rate:.2f}\t{result.realised_rate:.4f}\t"
                f"{result.cache_hit}"
            )


if __name__ == "__main__":
    main()
