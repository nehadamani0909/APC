"""Black-box LongLLMLingua adapter."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from frontier.compress.base import TargetTokenizer, TextCompressor


class LongLLMLinguaCompressor(TextCompressor):
    name = "longllmlingua"
    backend_version = "injected"

    def __init__(
        self,
        tokenizer: TargetTokenizer,
        compressor: Callable[..., str] | None = None,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(tokenizer, cache_dir=cache_dir)
        self.compressor = compressor

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        if self.compressor is None:
            raise RuntimeError(
                "LongLLMLingua is not installed; inject a verified black-box compressor"
            )
        return self.compressor(ctx, query, rate), 0.0
