"""L2 frozen-encoder features around an injected encoder."""

from __future__ import annotations

import math
import time

from frontier.features.base import Encoder, FeatureResult


def _cosine(first: list[float], second: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(first, second, strict=True))
    denominator = math.sqrt(sum(a * a for a in first) * sum(b * b for b in second))
    return numerator / denominator if denominator else 0.0


class EncoderExtractor:
    tier = "L2"

    def __init__(self, encoder: Encoder) -> None:
        self.encoder = encoder

    def extract(self, prompt_id: str, context: str, query: str) -> FeatureResult:
        started = time.perf_counter()
        context_embedding = [float(value) for value in self.encoder.encode(context)]
        query_embedding = [float(value) for value in self.encoder.encode(query)]
        features = {
            "context_embedding_norm": math.sqrt(
                sum(value * value for value in context_embedding)
            ),
            "query_embedding_norm": math.sqrt(
                sum(value * value for value in query_embedding)
            ),
            "context_query_cosine": _cosine(context_embedding, query_embedding),
        }
        return FeatureResult(
            prompt_id,
            features,
            (time.perf_counter() - started) * 1000.0,
        )
