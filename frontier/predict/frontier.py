"""Concrete feature-to-frontier predictor used by the policy layer.

Implements the inference path of APC-04 §6: features, one forward pass over
all three heads, and the true-cost objective from APC-03 C2.  It never calls
the target LLM -- that invariant is enforced by
``tests/test_no_target_llm_at_inference.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from frontier.features.surface import SurfaceExtractor
from frontier.harness.prices import PRICE_TABLE_VERSION, ModelPrice, get_price
from frontier.predict.calibrate import IsotonicModel
from frontier.predict.heads import (
    CumulativeLogitHead,
    OutputLengthHead,
    RateAdherenceHead,
)
from frontier.predict.scaling import FeatureScaler
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

    def __init__(
        self,
        *,
        model: str = "local-default",
        input_tokens: int = 0,
        price_table_version: str = PRICE_TABLE_VERSION,
        scaler: FeatureScaler | None = None,
        quality: CumulativeLogitHead | None = None,
        adherence: RateAdherenceHead | None = None,
        output_length: OutputLengthHead | None = None,
        temperature: float = 1.0,
        isotonic: IsotonicModel | None = None,
        compression_usd: float = 0.0,
        prediction_usd: float = 0.0,
    ) -> None:
        self.extractor = SurfaceExtractor()
        self.quality = quality or CumulativeLogitHead(len(FEATURES), seed=0)
        self.adherence = adherence or RateAdherenceHead(len(FEATURES))
        self.output_length = output_length or OutputLengthHead(len(FEATURES))
        self.scaler = scaler
        self.temperature = temperature
        self.isotonic = isotonic
        self.model = model
        self.input_tokens = input_tokens
        self.price_table_version = price_table_version
        # c_comp and c_pred from APC-04 §3.3. Measured overheads, not
        # estimates: without them the objective flatters the method by
        # ignoring the cost of running it.
        self.compression_usd = compression_usd
        self.prediction_usd = prediction_usd
        self.last_overhead_ms = 0.0

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        model: str = "local-default",
        input_tokens: int = 0,
        **overrides: Any,
    ) -> FrontierPredictor:
        """Load a predictor trained by ``scripts/06_train.py``."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("artifact") != "frontier-predictor":
            raise ValueError(f"{path} is not a frontier predictor artifact")
        heads = payload["heads"]
        count = len(payload["features"])

        quality = CumulativeLogitHead(count)
        quality.base_weights = np.asarray(heads["quality"]["base_weights"], dtype=float)
        quality.base_bias = float(heads["quality"]["base_bias"])
        quality.increment_weights = np.asarray(
            heads["quality"]["increment_weights"], dtype=float
        )
        quality.increment_bias = np.asarray(
            heads["quality"]["increment_bias"], dtype=float
        )

        adherence = RateAdherenceHead(count)
        adherence.weights = np.asarray(heads["adherence"]["weights"], dtype=float)
        adherence.bias = float(heads["adherence"]["bias"])

        output_length = OutputLengthHead(count)
        output_length.weights = np.asarray(
            heads["output_length"]["weights"], dtype=float
        )
        output_length.bias = np.asarray(heads["output_length"]["bias"], dtype=float)

        calibration = payload.get("calibration", {})
        isotonic = (
            IsotonicModel.from_dict(calibration["isotonic"])
            if "isotonic" in calibration
            else None
        )
        return cls(
            model=model,
            input_tokens=input_tokens,
            price_table_version=payload.get("provenance", {}).get(
                "price_table_version", PRICE_TABLE_VERSION
            ),
            scaler=FeatureScaler.from_dict(payload["scaler"]),
            quality=quality,
            adherence=adherence,
            output_length=output_length,
            temperature=float(calibration.get("temperature", 1.0)),
            isotonic=isotonic,
            **overrides,
        )

    def _features(self, x: str, q: str) -> Array:
        result = self.extractor.extract("inference", x, q)
        values = np.asarray([result.features[name] for name in FEATURES], dtype=float)
        raw = cast(Array, values[None, :])
        return self.scaler.transform(raw) if self.scaler is not None else raw

    def _calibrate(self, quality: Array) -> Array:
        if self.temperature != 1.0:
            logits = np.log(
                np.clip(quality, 1e-6, 1 - 1e-6)
                / np.clip(1.0 - quality, 1e-6, 1.0)
            )
            quality = cast(Array, 1.0 / (1.0 + np.exp(-logits / self.temperature)))
        return quality

    def predict(self, x: str, q: str, task_family: str) -> PolicyPrediction:
        started = time.perf_counter()
        features = self._features(x, q)
        quality = self._calibrate(self.quality.predict(features))[0]
        realised = self.adherence.predict(features)[0]
        output = self.output_length.predict(features)[0]
        price: ModelPrice = get_price(self.model, self.price_table_version)
        input_tokens = self.input_tokens or max(1, int(self.extractor_tokens(x)))
        # Cost(x, q, b) = c_in * T_in * r(b) + c_out * T_out(b) + c_comp + c_pred.
        # Non-monotone in b by construction, which is the whole point of C2:
        # aggressive compression can be safe and simultaneously dearer.
        input_cost = price.input_usd_per_million * input_tokens * realised / 1_000_000
        output_cost = price.output_usd_per_million * output / 1_000_000
        cost = input_cost + output_cost + self.compression_usd + self.prediction_usd
        self.last_overhead_ms = (time.perf_counter() - started) * 1000.0
        return PolicyPrediction(quality, cast(Array, cost), realised)

    def extractor_tokens(self, x: str) -> float:
        return float(self.extractor.extract("inference", x, "").features[
            "context_token_count"
        ])
