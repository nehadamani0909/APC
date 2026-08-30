# Compression backends

The compressor is a black box behind an adapter (APC-04 §1). We never modify
LLMLingua; that is what keeps the contribution orthogonal and makes
cross-backend transfer (E7) a free experiment.

| Backend | Role | Model | Runs on |
|---|---|---|---|
| `llmlingua2` | default | `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | CPU or GPU |
| `longllmlingua` | query-aware, long context | a causal LM (Llama-2-7B class) | GPU in practice |
| `truncate_tail` | the TAAC-equivalent control | — | anywhere |
| `truncate_head` | recency-biased control | — | anywhere |
| `random_drop` | null control: does *which* tokens matter? | — | anywhere |
| `cpc` | raises; no usable released implementation | — | — |

## Invariants

- **`b = 1.0` is an exact passthrough** for every backend. `ρ(x,1)` is the
  baseline that every safety label and the CRC risk are measured against, so
  a backend that reformats the text there — even only normalising whitespace —
  corrupts every downstream label.
- **Realised rate is measured on the target model's tokenizer**, never the
  compressor's. The compressor reports rate in its own token space; the
  quantity that drives cost is the target's. That gap is contribution C4.
- **Truncation preserves the original whitespace** of the text it keeps.
  `random_drop` necessarily reflows, because it deletes interior tokens; that
  is inherent to the control, not an implementation accident.
- **Caching is content-addressed** on context, query, backend, rate, backend
  version, and model version. A cache hit replays the originally measured
  timings and sets `cache_hit`; `compress_ms` is a cost input, so zeroing it
  would make compression look free on every re-run.

## Note on LLMLingua-2's checkpoint

The default checkpoint is trained on MeetingBank. MeetingBank results are
therefore in-domain for the compressor and belong in the paper as a control,
never as a headline (APC-05 §4.3).
