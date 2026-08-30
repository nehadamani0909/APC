"""Frozen contracts and cache support for compression backends."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TargetTokenizer(Protocol):
    """The target model tokenizer used to measure realised rates."""

    def encode(self, text: str) -> Sequence[Any]: ...


@dataclass(frozen=True)
class CompressedResult:
    compressed_text: str
    realised_rate: float
    wall_ms: float
    gpu_ms: float
    model_version: str
    cache_hit: bool = False


@runtime_checkable
class Compressor(Protocol):
    name: str
    backend_version: str

    def compress(
        self, ctx: str, query: str | None, rate: float
    ) -> CompressedResult: ...


def validate_rate(rate: float) -> None:
    if not 0.0 < rate <= 1.0:
        raise ValueError("rate must be in the interval (0, 1]")


def realised_rate(
    original: str, compressed: str, tokenizer: TargetTokenizer
) -> float:
    """Measure compressed/original length using the target tokenizer."""

    original_count = len(tokenizer.encode(original))
    if original_count == 0:
        # An empty context cannot be compressed; anything the backend emits
        # for it is an expansion, not a rate in (0, 1].
        return 1.0
    return len(tokenizer.encode(compressed)) / original_count


class CachedCompressor:
    """Content-addressed cache around a text compressor.

    A cache hit replays the timings measured when the entry was first
    written, and flags itself with ``cache_hit``.  ``compress_ms`` and
    ``gpu_seconds`` are cost inputs -- the ``c_comp`` term of the APC-03 C2
    objective and the E10 net-efficiency accounting -- so zeroing them would
    make compression look free on every re-run, and the build plan budgets
    for a full re-run.  Latency benchmarks should filter on ``cache_hit``
    instead.
    """

    def __init__(self, backend: Compressor, cache_dir: str | Path = "data/cache"):
        self.backend = backend
        self.cache_dir = Path(cache_dir)

    @property
    def name(self) -> str:
        return self.backend.name

    @property
    def backend_version(self) -> str:
        return self.backend.backend_version

    @property
    def model_version(self) -> str:
        return str(getattr(self.backend, "model_version", "unversioned"))

    def _path(self, ctx: str, query: str | None, rate: float) -> Path:
        # JSON-encoded so field boundaries are unambiguous, and keyed on the
        # model version too: two checkpoints of the same backend at the same
        # package version would otherwise collide.
        material = json.dumps(
            [
                ctx,
                query or "",
                self.name,
                f"{float(rate):.8f}",
                self.backend_version,
                self.model_version,
            ],
            separators=(",", ":"),
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def compress(
        self, ctx: str, query: str | None, rate: float
    ) -> CompressedResult:
        path = self._path(ctx, query, rate)
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                cached = CompressedResult(**json.load(handle))
            return CompressedResult(
                cached.compressed_text,
                cached.realised_rate,
                cached.wall_ms,
                cached.gpu_ms,
                cached.model_version,
                cache_hit=True,
            )
        if not isinstance(self.backend, TextCompressor):
            raise TypeError("CachedCompressor requires a TextCompressor backend")
        result = self.backend._compress_uncached(ctx, query, rate)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(result), handle, ensure_ascii=False, sort_keys=True)
        return result


class WhitespaceTokenizer:
    """Small deterministic tokenizer used only by offline tests/examples."""

    def encode(self, text: str) -> list[str]:
        return text.split()


class TextCompressor:
    """Base class for text-producing adapters with timing and rate measurement."""

    name = "text"
    backend_version = "v1"
    model_version = "unversioned"

    def __init__(
        self,
        tokenizer: TargetTokenizer,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self._cache = (
            CachedCompressor(self, cache_dir) if cache_dir is not None else None
        )

    def compress(
        self, ctx: str, query: str | None, rate: float
    ) -> CompressedResult:
        if self._cache is not None:
            return self._cache.compress(ctx, query, rate)
        return self._compress_uncached(ctx, query, rate)

    def _compress_uncached(
        self, ctx: str, query: str | None, rate: float
    ) -> CompressedResult:
        validate_rate(rate)
        started = time.perf_counter()
        if rate >= 1.0:
            # b = 1.0 is "no compression" (APC-04 §3.1), so it must be an
            # exact passthrough. rho(x, 1) is the baseline that every safety
            # label and the whole degradation loss are measured against; a
            # backend that reformats the text here -- even only normalising
            # whitespace -- corrupts that baseline and every label built on it.
            compressed, gpu_ms = ctx, 0.0
        else:
            compressed, gpu_ms = self._compress_text(ctx, query, rate)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return CompressedResult(
            compressed,
            realised_rate(ctx, compressed, self.tokenizer),
            wall_ms,
            gpu_ms,
            self.model_version,
        )

    def _compress_text(
        self, ctx: str, query: str | None, rate: float
    ) -> tuple[str, float]:
        raise NotImplementedError
