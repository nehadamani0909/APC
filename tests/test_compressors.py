from pathlib import Path

import pytest

from frontier.compress.base import WhitespaceTokenizer
from frontier.compress.baselines import RandomDropCompressor, TruncateTailCompressor
from frontier.compress.cpc import CPCCompressor
from frontier.compress.llmlingua2 import LLMLingua2Compressor
from frontier.compress.longllmlingua import LongLLMLinguaCompressor


def _truncate(ctx: str, query: str | None, rate: float) -> str:
    words = ctx.split()
    return " ".join(words[: round(len(words) * rate)])


def test_backends_measure_realised_rate_and_cache(tmp_path: Path) -> None:
    tokenizer = WhitespaceTokenizer()
    context = "one two three four five six seven eight nine ten"
    backends = [
        TruncateTailCompressor(tokenizer, cache_dir=tmp_path),
        RandomDropCompressor(tokenizer, seed=4, cache_dir=tmp_path),
        LLMLingua2Compressor(
            tokenizer, compressor=_truncate, cache_dir=tmp_path
        ),
        LongLLMLinguaCompressor(
            tokenizer, compressor=_truncate, cache_dir=tmp_path
        ),
    ]
    for backend in backends:
        first = backend.compress(context, "question", 0.5)
        second = backend.compress(context, "question", 0.5)
        assert 0.0 <= first.realised_rate <= 1.0
        assert second.cache_hit
        assert second.wall_ms == second.gpu_ms == 0.0


def test_twenty_prompts_and_five_rates_have_finite_realised_rates() -> None:
    tokenizer = WhitespaceTokenizer()
    backends = [
        TruncateTailCompressor(tokenizer),
        RandomDropCompressor(tokenizer, seed=4),
        LLMLingua2Compressor(tokenizer, compressor=_truncate),
        LongLLMLinguaCompressor(tokenizer, compressor=_truncate),
    ]
    for prompt_index in range(20):
        context = " ".join(f"word{prompt_index}_{index}" for index in range(20))
        for rate in (1.0, 0.8, 0.65, 0.5, 0.4):
            for backend in backends:
                result = backend.compress(context, "question", rate)
                assert result.realised_rate == result.realised_rate
                assert 0.0 <= result.realised_rate <= 1.0


def test_random_drop_is_reproducible() -> None:
    tokenizer = WhitespaceTokenizer()
    context = "one two three four five six"
    first = RandomDropCompressor(tokenizer, seed=7).compress(context, None, 0.5)
    second = RandomDropCompressor(tokenizer, seed=7).compress(context, None, 0.5)
    assert first.compressed_text == second.compressed_text


def test_cpc_is_explicitly_unavailable() -> None:
    with pytest.raises(RuntimeError, match="no usable released"):
        CPCCompressor(WhitespaceTokenizer()).compress("text", None, 0.5)
