"""Provider client behaviour, with no network and no credentials."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from frontier.harness.models import ProviderResponse
from frontier.harness.prices import cost_from_tokens
from frontier.harness.providers import (
    AnthropicClient,
    CachedProvider,
    OpenAIClient,
    RetryingClient,
    SpendCap,
    SpendCapExceeded,
)


class RecordingClient:
    """Captures the arguments the retry wrapper passes through."""

    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list[tuple[str, float, int]] = []
        self.fail_times = fail_times

    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        self.calls.append((prompt, temperature, seed))
        if len(self.calls) <= self.fail_times:
            raise TimeoutError("transient")
        return ProviderResponse(f"echo:{prompt}:{seed}", 10, 5)


def test_retrying_client_passes_prompt_temperature_and_seed_through() -> None:
    inner = RecordingClient()
    client = RetryingClient(inner)
    first = client.generate("alpha", temperature=0.7, seed=3)
    second = client.generate("beta", temperature=0.0, seed=4)
    # The wrapper previously discarded all three and re-invoked a fixed
    # thunk, so it could not vary the prompt across a grid.
    assert inner.calls == [("alpha", 0.7, 3), ("beta", 0.0, 4)]
    assert first.text == "echo:alpha:3"
    assert second.text == "echo:beta:4"


def test_retrying_client_retries_transient_failures() -> None:
    inner = RecordingClient(fail_times=2)
    client = RetryingClient(inner, max_retries=3, sleep=lambda _: None)
    assert client.generate("p", temperature=0.0, seed=0).text == "echo:p:0"
    assert len(inner.calls) == 3


def test_retrying_client_gives_up_after_max_retries() -> None:
    inner = RecordingClient(fail_times=99)
    client = RetryingClient(inner, max_retries=2, sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="failed after retries"):
        client.generate("p", temperature=0.0, seed=0)
    assert len(inner.calls) == 3


class CapExhaustedClient:
    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        raise SpendCapExceeded("spend cap exceeded")


def test_spend_cap_stop_is_not_retried() -> None:
    slept: list[float] = []
    client = RetryingClient(
        CapExhaustedClient(), max_retries=5, sleep=slept.append
    )
    with pytest.raises(SpendCapExceeded):
        client.generate("p", temperature=0.0, seed=0)
    # Retrying a budget stop would only burn the remaining allowance.
    assert slept == []


def test_spend_cap_reserves_then_settles_against_actual_usage() -> None:
    cap = SpendCap(1.0)
    cap.reserve(0.10)
    assert cap.spent_usd == pytest.approx(0.10)
    cap.settle(0.10, 0.03)
    assert cap.spent_usd == pytest.approx(0.03)
    cap.reserve(0.90)
    with pytest.raises(SpendCapExceeded):
        cap.reserve(0.10)


@dataclass
class _Usage:
    prompt_tokens: int
    completion_tokens: int


class FakeOpenAI:
    """Minimal stand-in for the installed OpenAI client surface."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}
        outer = self

        class _Completions:
            def create(self, **kwargs: Any) -> Any:
                outer.kwargs = kwargs
                message = type("M", (), {"content": "an answer"})()
                choice = type("C", (), {"message": message})()
                return type(
                    "R", (), {"choices": [choice], "usage": _Usage(1_000, 200)}
                )()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_openai_client_uses_provider_usage_and_settles_the_cap() -> None:
    cap = SpendCap(1.0)
    fake = FakeOpenAI()
    client = OpenAIClient(
        model="gpt-4o-mini", spend_cap=cap, client=fake, estimated_cost_usd=0.01
    )
    response = client.generate("prompt", temperature=0.7, seed=5)

    # Token counts must come from the response, never local re-tokenisation.
    assert (response.input_tokens, response.output_tokens) == (1_000, 200)
    assert fake.kwargs["temperature"] == 0.7
    assert fake.kwargs["seed"] == 5

    expected = cost_from_tokens("gpt-4o-mini", 1_000, 200)[2]
    assert cap.spent_usd == pytest.approx(expected)


def test_failed_call_releases_its_reservation() -> None:
    class Boom:
        chat = type(
            "Chat",
            (),
            {
                "completions": type(
                    "C",
                    (),
                    {"create": lambda self, **k: (_ for _ in ()).throw(OSError("net"))},
                )()
            },
        )()

    cap = SpendCap(1.0)
    client = OpenAIClient(
        model="gpt-4o-mini", spend_cap=cap, client=Boom(), estimated_cost_usd=0.25
    )
    with pytest.raises(OSError):
        client.generate("p", temperature=0.0, seed=0)
    assert cap.spent_usd == pytest.approx(0.0)


def test_anthropic_client_reads_usage_from_the_response() -> None:
    class FakeAnthropic:
        def __init__(self) -> None:
            outer = self
            self.kwargs: dict[str, Any] = {}

            class _Messages:
                def create(self, **kwargs: Any) -> Any:
                    outer.kwargs = kwargs
                    block = type("B", (), {"text": "claude answer"})()
                    usage = type("U", (), {"input_tokens": 12, "output_tokens": 7})()
                    return type("R", (), {"content": [block], "usage": usage})()

            self.messages = _Messages()

    fake = FakeAnthropic()
    client = AnthropicClient(
        model="gpt-4o-mini", spend_cap=SpendCap(1.0), client=fake
    )
    response = client.generate("p", temperature=0.5, seed=1)
    assert response.text == "claude answer"
    assert (response.input_tokens, response.output_tokens) == (12, 7)
    assert "seed" not in fake.kwargs


def test_provider_cache_keys_on_model_identity(tmp_path: Path) -> None:
    inner = RecordingClient()
    first = CachedProvider(inner, model="m", revision="r1", cache_dir=tmp_path)
    assert first.generate("p", temperature=0.0, seed=0).text == "echo:p:0"
    assert first.generate("p", temperature=0.0, seed=0).text == "echo:p:0"
    assert len(inner.calls) == 1

    # A different pinned revision must not reuse the old model's output.
    second = CachedProvider(inner, model="m", revision="r2", cache_dir=tmp_path)
    second.generate("p", temperature=0.0, seed=0)
    assert len(inner.calls) == 2

    # Different sample seeds are different cells of the k>1 grid.
    first.generate("p", temperature=0.0, seed=1)
    assert len(inner.calls) == 3
