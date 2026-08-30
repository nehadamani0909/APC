"""Deterministic text baselines and controls (APC-04 §5.1)."""

from __future__ import annotations

import random
import re
from pathlib import Path

from frontier.compress.base import TargetTokenizer, TextCompressor, validate_rate

_WORD_RE = re.compile(r"\S+")


def _spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in _WORD_RE.finditer(text)]


class TruncateTailCompressor(TextCompressor):
    """Keep the head, drop the tail. The TAAC-equivalent control."""

    name = "truncate_tail"
    backend_version = "v2"
    model_version = "deterministic-v2"

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        validate_rate(rate)
        spans = _spans(ctx)
        keep = round(len(spans) * rate)
        if keep <= 0:
            return "", 0.0
        if keep >= len(spans):
            return ctx, 0.0
        # Slice the original string rather than rejoining split tokens, so
        # paragraph breaks and indentation survive. Rejoining with single
        # spaces silently reformats the prompt, which is a change the
        # experiment is not supposed to be making.
        return ctx[: spans[keep - 1][1]], 0.0


class TruncateHeadCompressor(TextCompressor):
    """Keep the tail, drop the head. Recency-biased control."""

    name = "truncate_head"
    backend_version = "v2"
    model_version = "deterministic-v2"

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        validate_rate(rate)
        spans = _spans(ctx)
        keep = round(len(spans) * rate)
        if keep <= 0:
            return "", 0.0
        if keep >= len(spans):
            return ctx, 0.0
        return ctx[spans[len(spans) - keep][0] :], 0.0


class RandomDropCompressor(TextCompressor):
    """Drop tokens uniformly at random: the null control for *which* tokens."""

    name = "random_drop"
    backend_version = "v2"
    model_version = "deterministic-v2"

    def __init__(
        self,
        tokenizer: TargetTokenizer,
        *,
        seed: int = 0,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(tokenizer, cache_dir=cache_dir)
        self.seed = seed
        self.backend_version = f"v2-seed-{seed}"

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        validate_rate(rate)
        words = ctx.split()
        keep = round(len(words) * rate)
        if keep <= 0:
            return "", 0.0
        rng = random.Random(self.seed)
        selected = set(rng.sample(range(len(words)), keep))
        # Interior deletions make surrounding whitespace meaningless, so this
        # control necessarily reflows the text. That is inherent to the
        # baseline, not an accident of implementation.
        kept = " ".join(
            word for index, word in enumerate(words) if index in selected
        )
        return kept, 0.0
