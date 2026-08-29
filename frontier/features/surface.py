"""Fast, target-model-free L0 surface features."""

from __future__ import annotations

import gzip
import re
import time
from collections import Counter

from frontier.features.base import FeatureResult

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
CODE_RE = re.compile(r"(?:def |class |return |import |```|=>|::|\bself\b)")


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _ngram_repetition(tokens: list[str], n: int = 2) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


class SurfaceExtractor:
    tier = "L0"

    def extract(self, prompt_id: str, context: str, query: str) -> FeatureResult:
        started = time.perf_counter()
        tokens = _tokens(context)
        query_tokens = _tokens(query)
        context_words = context.split()
        compressed_bytes = len(gzip.compress(context.encode("utf-8")))
        raw_bytes = max(1, len(context.encode("utf-8")))
        context_counter = Counter(tokens)
        query_counter = Counter(query_tokens)
        overlap = sum((context_counter & query_counter).values())
        features = {
            "context_token_count": float(len(tokens)),
            "sentence_count": float(len(SENTENCE_RE.findall(context))),
            "type_token_ratio": len(set(tokens)) / max(1, len(tokens)),
            "bigram_repetition_rate": _ngram_repetition(tokens),
            "gzip_compression_ratio": compressed_bytes / raw_bytes,
            "digit_ratio": sum(char.isdigit() for char in context)
            / max(1, len(context)),
            "code_token_ratio": len(CODE_RE.findall(context))
            / max(1, len(context_words)),
            "punctuation_ratio": sum(
                not char.isalnum() and not char.isspace() for char in context
            )
            / max(1, len(context)),
            "average_sentence_length": len(tokens)
            / max(1, len(SENTENCE_RE.findall(context))),
            "query_token_count": float(len(query_tokens)),
            "query_context_overlap": overlap / max(1, len(query_tokens)),
        }
        latency_ms = (time.perf_counter() - started) * 1000.0
        return FeatureResult(prompt_id, features, latency_ms)
