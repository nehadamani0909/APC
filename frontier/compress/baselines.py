"""Deterministic text baselines used by the P2 adapter tests."""

from __future__ import annotations

import random
from pathlib import Path

from frontier.compress.base import TargetTokenizer, TextCompressor, validate_rate


def _words(text: str) -> list[str]:
    return text.split()


class TruncateTailCompressor(TextCompressor):
    name = "truncate_tail"
    backend_version = "v1"
    model_version = "deterministic-v1"

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        validate_rate(rate)
        words = _words(ctx)
        return " ".join(words[: round(len(words) * rate)]), 0.0


class RandomDropCompressor(TextCompressor):
    name = "random_drop"
    backend_version = "v1"
    model_version = "deterministic-v1"

    def __init__(
        self,
        tokenizer: TargetTokenizer,
        *,
        seed: int = 0,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(tokenizer, cache_dir=cache_dir)
        self.seed = seed
        self.backend_version = f"v1-seed-{seed}"

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        validate_rate(rate)
        words = _words(ctx)
        keep = round(len(words) * rate)
        rng = random.Random(self.seed)
        selected = set(rng.sample(range(len(words)), keep)) if keep else set()
        return (
            " ".join(word for index, word in enumerate(words) if index in selected),
            0.0,
        )
