"""Black-box LongLLMLingua adapter (query-aware, long-context)."""

from __future__ import annotations

from frontier.compress.llmlingua_base import LLMLinguaAdapter


class LongLLMLinguaCompressor(LLMLinguaAdapter):
    name = "longllmlingua"
    use_llmlingua2 = False
    # LongLLMLingua scores tokens with a causal LM, so unlike LLMLingua-2 it
    # genuinely needs a generative checkpoint -- and is impractical without a
    # GPU. It serves Tier D (cross-backend transfer, E7), not the main grid.
    default_model = "NousResearch/Llama-2-7b-hf"
