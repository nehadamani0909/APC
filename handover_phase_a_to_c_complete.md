# Final handover: Phases A–C

## Status

Phases A, B, and C are complete at the implementation level. The repository
retains offline determinism and does not make scientific claims from the
fixture corpus.

## Phase A — benchmark ETL

Implemented in `frontier/data/etl.py`:

- Hugging Face `load_dataset` integration using the installed and verified
  signature.
- Dataset specifications for GSM8K, LongBench subsets, MeetingBank,
  HumanEval, MBPP, and ShareGPT.
- Normalization into the existing `Instance` contract with context/query
  separation and source-document IDs.
- Deterministic 60/10/10/20 partitions using prompt/source-document groups and
  `split_prompt_ids`.
- Raw normalized JSONL under `data/raw` and split JSONL under `data/splits`.
- Optional real integration test marked `integration` and skipped by default.

`default_tasks(source_dir=...)` now points tasks at normalized real JSONL files
when present and preserves fixture fallback behavior otherwise.

## Phase B — compressors

Implemented:

- LLMLingua-2 adapter using the verified installed signatures:
  `PromptCompressor(...)` and `compress_prompt(context, question, rate, ...)`.
- LongLLMLingua adapter through the same installed LLMLingua engine, with the
  backend version recorded from package metadata.
- Target-tokenizer measurement remains in `TextCompressor`, so realised rate
  is not inferred from the compressor tokenizer.
- Existing injected callable path remains available for offline tests.
- Deterministic `TruncateHeadCompressor` and `RandomDropCompressor` controls.
- `scripts/compress_smoke.py` for requested-versus-realised rate measurements.

The `real` optional extra installs `llmlingua`, Torch, Transformers, and the
provider SDKs. Model weights are not downloaded or committed.

## Phase C — target clients

Implemented in `frontier/harness/providers.py`:

- OpenAI chat-completions client using provider-reported
  `prompt_tokens`/`completion_tokens`.
- Anthropic messages client using provider-reported `input_tokens`/
  `output_tokens`.
- Environment-only API key loading.
- Spend-cap reservation, bounded retries, exponential backoff, and semaphore
  concurrency control.
- Disk request cache keyed by prompt, model, revision, temperature, and seed.
- Existing `APIBackend` remains the accounting boundary and reconciles usage
  against the versioned price table.

Verified installed SDK signatures:

- OpenAI `Completions.create(...)`.
- Anthropic `Messages.create(...)`.
- Hugging Face `datasets.load_dataset(...)`.

## Verification

```bash
uv run pytest
uv run ruff check .
uv run mypy frontier scripts tests
```

Current result: 43 passed, 1 integration test skipped; Ruff and mypy clean.

## Real-run prerequisites

The following are intentionally not executed in the offline environment:

1. Install the optional `real` extra.
2. Make model weights available locally for LLMLingua and the local target
   model.
3. Set `OPENAI_API_KEY` and/or `ANTHROPIC_API_KEY` only in the environment.
4. Set explicit model revisions, price-table version, temperature, seeds, and
   a non-zero spend cap.
5. Run `scripts/03_fetch_data.py` for each licensed dataset family.
6. Run `scripts/compress_smoke.py` before the full grid.

No API call, dataset download, model weight, or secret is part of this commit.

## Scope boundary

Phases D–I remain separate work. The current fast reproduction and fixture
corpus are plumbing validation only; they must not be presented as real
benchmark evidence.
