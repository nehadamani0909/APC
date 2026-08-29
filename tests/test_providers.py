from pathlib import Path

from frontier.harness.models import ProviderResponse
from frontier.harness.providers import CachedProvider, RetryingClient, SpendCap


def test_retrying_client_enforces_spend_cap() -> None:
    client = RetryingClient(
        lambda: ProviderResponse("ok", 1, 1),
        spend_cap=SpendCap(0.01),
        estimated_cost_usd=0.01,
    )
    assert client.generate("p", temperature=0.0, seed=0).text == "ok"
    try:
        client.generate("p", temperature=0.0, seed=0)
    except RuntimeError as exc:
        assert "cap" in str(exc)
    else:
        raise AssertionError("second request should exceed cap")


def test_provider_cache_uses_identity(tmp_path: Path) -> None:
    calls = 0

    def request() -> ProviderResponse:
        nonlocal calls
        calls += 1
        return ProviderResponse("ok", 1, 1)

    client = CachedProvider(RetryingClient(request, spend_cap=SpendCap(1.0)), tmp_path)
    first = client.generate("p", temperature=0.0, seed=0, model="m", revision="r")
    second = client.generate("p", temperature=0.0, seed=0, model="m", revision="r")
    assert first == second
    assert calls == 1
