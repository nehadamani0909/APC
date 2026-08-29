"""Contracts shared by the feature-extraction tiers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class FeatureResult:
    prompt_id: str
    features: dict[str, float]
    latency_ms: float

    def as_record(self) -> dict[str, float | str]:
        return {
            "prompt_id": self.prompt_id,
            **self.features,
            "latency_ms": self.latency_ms,
        }


@runtime_checkable
class FeatureExtractor(Protocol):
    tier: str

    def extract(self, prompt_id: str, context: str, query: str) -> FeatureResult: ...


@runtime_checkable
class SmallLMScorer(Protocol):
    """A verified small-LM adapter; it must not call a target LLM."""

    def token_nll(
        self, text: str, *, condition: str | None = None
    ) -> Sequence[float]: ...


@runtime_checkable
class Encoder(Protocol):
    def encode(self, text: str) -> Sequence[float]: ...
