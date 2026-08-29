from frontier.harness.models import APIBackend, ProviderResponse


class FakeProvider:
    def generate(
        self, prompt: str, *, temperature: float, seed: int
    ) -> ProviderResponse:
        assert prompt == "prompt"
        return ProviderResponse("answer", 10, 2)


def test_api_usage_and_cost_use_provider_counts() -> None:
    result = APIBackend(
        FakeProvider(), model="gpt-4o-mini", model_revision="pinned-revision"
    ).generate("prompt", temperature=0.7, seed=3)
    assert result.T_in == 10
    assert result.T_out == 2
    assert result.usd == 10 * 0.15 / 1_000_000 + 2 * 0.60 / 1_000_000
