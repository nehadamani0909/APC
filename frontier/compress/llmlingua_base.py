"""Shared engine management for the LLMLingua-family adapters.

The compressor stays a black box behind this adapter (APC-04 §1): we never
modify LLMLingua, which is what keeps the contribution orthogonal and makes
cross-backend transfer (E7) a free experiment.
"""

from __future__ import annotations

import importlib.metadata
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from frontier.compress.base import TargetTokenizer, TextCompressor


def default_device() -> str:
    """``cuda`` when a GPU is actually usable, otherwise ``cpu``."""

    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _package_version() -> str:
    try:
        return importlib.metadata.version("llmlingua")
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


class LLMLinguaAdapter(TextCompressor):
    """Base adapter that loads one PromptCompressor and reuses it."""

    use_llmlingua2 = False
    default_model = ""

    def __init__(
        self,
        tokenizer: TargetTokenizer,
        compressor: Callable[..., str] | None = None,
        *,
        cache_dir: str | Path | None = None,
        model_name: str | None = None,
        model_revision: str = "main",
        device_map: str | None = None,
        compress_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(tokenizer, cache_dir=cache_dir)
        self.compressor = compressor
        self.model_name = model_name or self.default_model
        self.model_revision = model_revision
        self.device_map = device_map or default_device()
        self.compress_kwargs = dict(compress_kwargs or {})
        self.backend_version = f"llmlingua-{_package_version()}"
        # Pins the checkpoint into every ledger row and every cache key.
        self.model_version = f"{self.model_name}@{self.model_revision}"
        self._engine: Any = None

    def _load_engine(self) -> Any:
        """Load the PromptCompressor once and keep it.

        Constructing it per call would reload the model on every one of the
        grid's cells, which is the difference between a feasible run and an
        impossible one.
        """

        if self._engine is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as exc:
                raise RuntimeError(
                    "install the 'real' extra to use the LLMLingua backends"
                ) from exc
            self._engine = PromptCompressor(
                model_name=self.model_name,
                device_map=self.device_map,
                use_llmlingua2=self.use_llmlingua2,
            )
        return self._engine

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        if self.compressor is not None:
            return self.compressor(ctx, query, rate), 0.0
        engine = self._load_engine()
        gpu_ms = 0.0
        if self.device_map.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
            started = time.perf_counter()
        result = engine.compress_prompt(
            [ctx],
            question=query or "",
            rate=rate,
            **self.compress_kwargs,
        )
        if self.device_map.startswith("cuda"):
            import torch

            torch.cuda.synchronize()
            gpu_ms = (time.perf_counter() - started) * 1000.0
        # The realised rate is deliberately NOT taken from the engine's own
        # report: it counts tokens in the compressor's tokenizer, and the
        # quantity that drives cost is the target model's token count.
        # TextCompressor measures it on the injected target tokenizer, and
        # the gap between requested and realised is contribution C4.
        return str(result["compressed_prompt"]), gpu_ms
