import pandas as pd
from pytest import MonkeyPatch

from frontier.features.encoder import EncoderExtractor
from frontier.features.registry import extract_records
from frontier.features.smallm import SmallLMExtractor
from frontier.features.surface import SurfaceExtractor


def test_l0_includes_gzip_and_meets_latency_budget() -> None:
    extractor = SurfaceExtractor()
    result = extractor.extract("p", "Repeated context text. " * 20, "question")
    assert result.features["gzip_compression_ratio"] > 0.0
    assert result.latency_ms < 5.0


class FakeSmallLM:
    def token_nll(self, text: str, *, condition: str | None = None) -> list[float]:
        return [1.0, 2.0, 3.0] if condition is None else [2.0, 3.0, 4.0]


class FakeEncoder:
    def encode(self, text: str) -> list[float]:
        return [1.0, 0.0] if text == "ctx" else [0.0, 1.0]


def test_l1_and_l2_are_injected_and_bounded() -> None:
    l1 = SmallLMExtractor(FakeSmallLM()).extract("p", "ctx", "q")
    l2 = EncoderExtractor(FakeEncoder()).extract("p", "ctx", "q")
    assert l1.features["perplexity"] > 0.0
    assert -1.0 <= l2.features["context_query_cosine"] <= 1.0


def test_l0_registry_writes_prompt_keyed_frame() -> None:
    frame = extract_records([("p", "context", "query")], SurfaceExtractor())
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "prompt_id"] == "p"
    assert "latency_ms" in frame


def test_features_do_not_call_target_llm(monkeypatch: MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("target LLM called")

    monkeypatch.setattr("frontier.harness.models.APIBackend.generate", fail)
    SurfaceExtractor().extract("p", "context", "query")


def test_tiered_extractor_composes_and_sums_latency() -> None:
    from frontier.features.registry import TieredExtractor

    class Scorer:
        def token_nll(self, text: str, *, condition: str | None = None) -> list[float]:
            return [0.5, 1.5, 2.5]

    class Enc:
        def encode(self, text: str) -> list[float]:
            return [1.0, 0.0, 1.0]

    l0 = TieredExtractor()
    assert l0.tier == "L0"
    full = TieredExtractor(small_lm=Scorer(), encoder=Enc())
    assert full.tier == "L0+L1+L2"

    base = l0.extract("p", "some context here.", "a query")
    combined = full.extract("p", "some context here.", "a query")
    # L0 features survive composition, and the higher tiers add to them.
    assert set(base.features).issubset(set(combined.features))
    assert "nll_mean" in combined.features
    assert "context_query_cosine" in combined.features
    # The APC-04 5.2 latency budget applies to the whole stack.
    assert combined.latency_ms >= base.latency_ms
