"""L1 feature extraction around an injected small-LM scorer."""

from __future__ import annotations

import math
import time
from statistics import fmean, pvariance

from frontier.features.base import FeatureResult, SmallLMScorer


def _perplexity(nlls: list[float]) -> float:
    return math.exp(fmean(nlls)) if nlls else 1.0


class SmallLMExtractor:
    tier = "L1"

    def __init__(self, scorer: SmallLMScorer) -> None:
        self.scorer = scorer

    def extract(self, prompt_id: str, context: str, query: str) -> FeatureResult:
        started = time.perf_counter()
        nlls = [float(value) for value in self.scorer.token_nll(context)]
        conditioned = [
            float(value)
            for value in self.scorer.token_nll(context, condition=query)
        ]
        ordered = sorted(nlls)
        features = {
            "nll_mean": fmean(nlls) if nlls else 0.0,
            "nll_variance": pvariance(nlls) if len(nlls) > 1 else 0.0,
            "nll_p10": ordered[max(0, int(len(ordered) * 0.10) - 1)]
            if ordered
            else 0.0,
            "nll_p50": ordered[max(0, int(len(ordered) * 0.50) - 1)]
            if ordered
            else 0.0,
            "nll_p90": ordered[max(0, int(len(ordered) * 0.90) - 1)]
            if ordered
            else 0.0,
            "perplexity": _perplexity(nlls),
            "contrastive_perplexity": _perplexity(conditioned) - _perplexity(nlls),
            "saliency_spread": (max(nlls) - min(nlls)) if nlls else 0.0,
        }
        return FeatureResult(
            prompt_id,
            features,
            (time.perf_counter() - started) * 1000.0,
        )
