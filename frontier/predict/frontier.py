"""Concrete feature-to-frontier predictor used by the policy layer."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from frontier.features.surface import SurfaceExtractor
from frontier.harness.prices import PRICE_TABLE_VERSION, get_price
from frontier.predict.heads import (
    CumulativeLogitHead,
    OutputLengthHead,
    RateAdherenceHead,
)
from frontier.select.policy import PolicyPrediction

Array = np.ndarray[Any, np.dtype[np.float64]]
FEATURES = (
    "context_token_count",
    "sentence_count",
    "type_token_ratio",
    "bigram_repetition_rate",
    "gzip_compression_ratio",
    "digit_ratio",
    "code_token_ratio",
    "punctuation_ratio",
    "average_sentence_length",
    "query_token_count",
    "query_context_overlap",
)


class FrontierPredictor:
    """Run all three heads in one pass without calling a target LLM."""

    def __init__(self, *, model: str = "local-default", input_tokens: int = 0) -> None:
        self.extractor = SurfaceExtractor()
        self.quality = CumulativeLogitHead(len(FEATURES), seed=0)
        self.adherence = RateAdherenceHead(len(FEATURES))
        self.output_length = OutputLengthHead(len(FEATURES))
        self.model = model
        self.input_tokens = input_tokens
        self.price_table_version = PRICE_TABLE_VERSION

    def _features(self, x: str, q: str) -> Array:
        result = self.extractor.extract("inference", x, q)
        values = np.asarray([result.features[name] for name in FEATURES], dtype=float)
        return cast(Array, values[None, :])

    def predict(self, x: str, q: str, task_family: str) -> PolicyPrediction:
        features = self._features(x, q)
        quality = self.quality.predict(features)[0]
        realised = self.adherence.predict(features)[0]
        output = self.output_length.predict(features)[0]
        price = get_price(self.model, self.price_table_version)
        input_tokens = self.input_tokens or max(1, int(features[0, 0]))
        input_cost = price.input_usd_per_million * input_tokens * realised / 1_000_000
        cost = input_cost + price.output_usd_per_million * output / 1_000_000
        return PolicyPrediction(quality, cast(Array, cost), realised)
