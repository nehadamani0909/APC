from frontier.harness.prices import cost_from_tokens, get_price


def test_versioned_price_lookup_and_cost() -> None:
    assert get_price("gpt-4o-mini", "v1").input_usd_per_million == 0.15
    assert cost_from_tokens("gpt-4o-mini", 1_000_000, 1_000_000) == (
        0.15,
        0.60,
        0.75,
    )
