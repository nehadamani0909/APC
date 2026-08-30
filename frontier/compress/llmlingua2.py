"""Black-box LLMLingua-2 adapter (the default backend, APC-04 §5.1)."""

from __future__ import annotations

from frontier.compress.llmlingua_base import LLMLinguaAdapter


class LLMLingua2Compressor(LLMLinguaAdapter):
    name = "llmlingua2"
    use_llmlingua2 = True
    # LLMLingua-2 is a token-classification model (XLM-RoBERTa), not a causal
    # LM. The previous default paired a 7B Llama checkpoint with
    # use_llmlingua2=True, which is an incoherent combination and needs 13GB
    # of weights besides.
    #
    # NOTE: this checkpoint is trained on MeetingBank, so MeetingBank results
    # are in-domain for the compressor and belong in the paper as a control
    # rather than a headline (APC-05 §4.3).
    default_model = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
