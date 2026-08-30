"""Opt-in integration tests against real backends, models, and providers.

All skipped unless ``FRONTIER_RUN_INTEGRATION=1``; the API test additionally
needs a key and spends a few cents.  They exist so the claims that cannot be
checked offline -- rate adherence on a real compressor, and USD reconciling
against provider-reported usage -- have somewhere to live.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

RATES = (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2)
INSTANCES = Path("data/raw/gsm8k.jsonl")


@pytest.mark.integration
def test_llmlingua2_realised_rate_is_monotone_ish_and_bounded() -> None:
    """Realised rate must track requested rate on a real compressor."""

    pytest.importorskip("llmlingua")
    pytest.importorskip("transformers")
    if not INSTANCES.exists():
        pytest.skip("run scripts/03_fetch_data.py gsm8k first")

    import json

    from frontier.compress.base import WhitespaceTokenizer
    from frontier.compress.llmlingua2 import LLMLingua2Compressor

    rows = [
        json.loads(line)
        for line in INSTANCES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:5]
    backend = LLMLingua2Compressor(WhitespaceTokenizer(), device_map="cpu")

    for row in rows:
        realised = [
            backend.compress(row["context"], row["query"], rate).realised_rate
            for rate in RATES
        ]
        assert all(0.0 < value <= 1.0 for value in realised)
        # b = 1.0 must be an exact passthrough.
        assert realised[0] == pytest.approx(1.0)
        # Monotone-ish: allow small inversions, but the trend must hold.
        assert realised[0] > realised[-1]
        pairs = list(zip(realised, realised[1:], strict=False))
        inversions = sum(1 for higher, lower in pairs if lower > higher + 0.02)
        assert inversions <= 1, realised


@pytest.mark.integration
def test_openai_call_reconciles_usd_to_usage_within_one_percent() -> None:
    """One real call: the ledger's USD must match provider-reported usage."""

    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    pytest.importorskip("openai")

    from frontier.harness.ledger import GridRow
    from frontier.harness.models import APIBackend
    from frontier.harness.prices import PRICE_TABLE_VERSION, cost_from_tokens
    from frontier.harness.providers import OpenAIClient, SpendCap

    model = "gpt-4o-mini"
    cap = SpendCap(0.05)
    backend = APIBackend(
        OpenAIClient(model=model, spend_cap=cap, estimated_cost_usd=0.001),
        model=model,
        model_revision="integration-test",
    )
    result = backend.generate("Reply with the single word: ok", temperature=0.0)

    assert result.T_in > 0 and result.T_out > 0
    expected = cost_from_tokens(
        model, result.T_in, result.T_out, PRICE_TABLE_VERSION
    )[2]
    assert result.usd == pytest.approx(expected, rel=0.01)
    assert cap.spent_usd == pytest.approx(expected, rel=0.01)

    # The row the grid would write must validate.
    GridRow(
        prompt_id="integration",
        task="smoke",
        family="qa",
        backend="none",
        requested_b=1.0,
        realised_r=1.0,
        target_model=model,
        sample_idx=0,
        temperature=0.0,
        seed=0,
        quality=1.0,
        T_in=result.T_in,
        T_out=result.T_out,
        latency_ms=result.latency_s * 1000.0,
        compress_ms=0.0,
        usd_in=cost_from_tokens(model, result.T_in, 0)[0],
        usd_out=cost_from_tokens(model, 0, result.T_out)[1],
        usd_total=expected,
        gpu_seconds=0.0,
        raw_output_hash="0" * 64,
        code_version="integration",
        price_table_version=PRICE_TABLE_VERSION,
    )


@pytest.mark.integration
def test_local_hf_generation_reports_real_token_counts() -> None:
    """The local client must return the model's own token counts."""

    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from frontier.harness.providers import HFClient

    client = HFClient(
        model="Qwen/Qwen2.5-0.5B-Instruct", device="cpu", max_new_tokens=16
    )
    tokenizer = client.load_tokenizer()
    prompt = "Question: What is 2 + 2?\nAnswer:"
    response = client.generate(prompt, temperature=0.0, seed=0)

    assert response.input_tokens == len(tokenizer.encode(prompt))
    assert 0 < response.output_tokens <= 16
    assert response.text
