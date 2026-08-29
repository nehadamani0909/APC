"""Black-box LLMLingua-2 adapter."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from frontier.compress.base import TargetTokenizer, TextCompressor


class LLMLingua2Compressor(TextCompressor):
    name = "llmlingua2"
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
                "LLMLingua-2 is not installed; inject a verified black-box compressor"
            )
        return self.compressor(ctx, query, rate), 0.0
