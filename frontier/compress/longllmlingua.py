"""Black-box LongLLMLingua adapter."""

from __future__ import annotations

import importlib.metadata
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
        model_name: str = "NousResearch/Llama-2-7b-hf",
        device_map: str = "cuda",
    ) -> None:
        super().__init__(tokenizer, cache_dir=cache_dir)
        self.compressor = compressor
        self.model_name = model_name
        self.device_map = device_map
        try:
            version = importlib.metadata.version("llmlingua")
        except importlib.metadata.PackageNotFoundError:
            version = "uninstalled"
        self.backend_version = f"llmlingua-{version}"

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        if self.compressor is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as exc:
                raise RuntimeError(
                    "install the 'real' extra for LongLLMLingua"
                ) from exc
            engine = PromptCompressor(
                model_name=self.model_name, device_map=self.device_map
            )
            result = engine.compress_prompt([ctx], question=query or "", rate=rate)
            return str(result["compressed_prompt"]), 0.0
        return self.compressor(ctx, query, rate), 0.0
