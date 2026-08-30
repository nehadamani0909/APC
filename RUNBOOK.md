# Frontier runbook

## 1. Environment

```bash
uv sync                      # base, offline
uv sync --extra dev          # + pytest, ruff, mypy
uv sync --extra real         # + datasets, llmlingua, openai, anthropic
```

The project pins **Python 3.12** in `.python-version`. This matters: `uv`
otherwise picks the newest interpreter on the machine, and `pyarrow<20`
publishes no wheels for 3.14+, so `uv sync` falls back to a source build and
fails. If you see a `cmake` error from pyarrow, check which interpreter uv
selected.

Verify:

```bash
uv run --extra dev pytest        # 96 passed, 2 skipped
uv run --extra dev ruff check .
uv run --extra dev mypy frontier scripts tests
```

Integration tests (network, credentials, model weights) are skipped unless
`FRONTIER_RUN_INTEGRATION=1` is set.

## 2. What is real and what is not

| Artifact | Status |
|---|---|
| `data/corpus/v1/corpus.parquet` | **fixture** — 200 synthetic prompts, a stub target model, truncation as the compressor |
| `artifacts/predictor.json` | trained, but stamped `SCAFFOLD` while features come from prompt metadata |
| `reports/scaffold/*` | **placeholder numbers**, never results |
| `paper/figures`, `paper/tables` | intentionally empty until a real run fills them |
| `docs/claims_ledger.md` | all 15 claims deferred; updated only by the researcher from held-out runs |

`scripts/reproduce.py --fast` exercises every stage against the fixture
corpus. It validates plumbing. It is not evidence.

## 3. Secrets

Read from the environment only; never written to the repo.

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## 4. Price table

`frontier/harness/prices.py` ships `local-default`, `gpt-4o-mini`, `gpt-4o`.
Anything else must be registered from the provider's **published** rates
before a run:

```python
from frontier.harness.prices import ModelPrice, register_price
register_price("claude-...", ModelPrice(input_usd_per_million=..., output_usd_per_million=...))
```

Prices are never guessed. The ledger's 1% reconciliation checks that the
ledger agrees with this table — not that the table agrees with the invoice —
so a wrong row corrupts every USD figure silently. Bump
`PRICE_TABLE_VERSION` when rates change rather than editing a row mid-project.

## 5. Fetching benchmarks

```bash
uv run python scripts/03_fetch_data.py gsm8k --output-dir data/raw --n 200
uv run python scripts/03_fetch_data.py hotpotqa --output-dir data/raw --n 200
uv run python scripts/03_fetch_data.py meetingbank --output-dir data/raw --n 200
uv run python scripts/03_fetch_data.py sharegpt --output-dir data/raw --n 200
```

Writes normalised JSONL to `data/raw/<name>.jsonl` (all splits, each row
carrying its own `split`) and per-split files to `data/splits/`. Splits are
60/10/10/20, assigned by **source document** so a document reused across
instances cannot straddle a boundary.

LongBench sub-family names (`multidoc_qa`, `summarisation`, …) are accepted as
aliases for their first config; pass a real config name (`hotpotqa`,
`gov_report`, `lcc`, …) to select one exactly.

Datasets and weights are gitignored and must never be committed.

## 6. Gate 1 — the decision that determines the paper

200 prompts stratified over 4 families × 7 budgets × k=5 ≈ **7,000
generations** (APC-06 §E1). Until this number exists you do not know whether
you are writing the method paper (C1+C2) or the C5 corpus paper.

**Without a GPU**, run it on `gpt-4o-mini` rather than a local model:

- ≈7,000 calls, long inputs and short outputs
- ≈34M input + ≈1M output tokens ⇒ **roughly $5–6**, under $3 on short families
- set a spend cap well below your tolerance and let it abort rather than trusting an estimate

`# DECISION:` APC-04 §3.1.1 specifies k=5 on a *local* model and k=1 at T=0 on
API models, because API cost was the constraint. At $6 that reasoning does not
apply, and the gate depends on k=5 for its noise floor — dropping to k=1 would
remove the test. Use a distinct seed per `sample_idx`; OpenAI's `seed` is
best-effort and at T=0.7 the variation is wanted.

Compression for the pilot runs on CPU: LLMLingua-2 is XLM-RoBERTa-large
(~560M), roughly 20–45 min for the pilot's ~1,400 compressions. **LongLLMLingua
needs a 7B causal LM and is not practical without a GPU** — it serves Tier D
(E7 backend transfer), not Gate 1.

Note that the default LLMLingua-2 checkpoint is trained on MeetingBank, making
MeetingBank an in-domain control rather than a headline (APC-05 §4.3).

Then:

```bash
uv run python -m frontier.eval.gate1 --ledger data/pilot/ledger.jsonl --report reports/gate1.md
```

## 7. Resuming, and the spend cap

The grid is idempotent. `GridRunner` rebuilds its completed set from both the
completion index and the ledger, so re-running the same cells is a no-op —
just re-issue the command. Failures are recorded to the failure index with
their cell coordinates and error type; the run continues.

The cap is a hard ceiling: it reserves an estimate before each call and settles
against provider-reported usage afterwards. Exceeding it raises
`SpendCapExceeded`, which is deliberately **not** retried. To raise it, change
the configured `spend_cap_usd` and note the change in the run metadata — do not
remove the cap.

**On Windows**, ledger and completion-index writes are lock-guarded because
concurrent appends are not atomic there. Do not bypass `GridRunner` to write
the ledger from multiple threads.

## 8. Full grid (needs a GPU)

Tier A is 6,000 prompts × 7 budgets × k=5 = 210,000 generations on a local
7–8B model, ≈150–220 GPU-hours (APC-05 §4.2), plus ≈25–40 for compression.
Tiers B/C are ~17k API calls at ≈$250–600. Renting a few A100-hours (≈$10–20)
is the cheapest way to get Tier A started from this machine; vLLM does not run
on Windows, so use Linux or WSL2 with a GPU passed through.

## 9. Training and calibration

```bash
uv run python scripts/06_train.py \
  --corpus data/corpus/v1/corpus.parquet \
  --instances-dir data/raw \
  --report reports/e2_e4.md
```

Fits on `D_train`, calibrates H1 on `D_cal_a` only. `D_cal_b` is reserved for
the CRC threshold and `D_test` for reporting. Calibrating and selecting λ̂ on
the same split voids the C1 guarantee — the splits are kept apart in code, and
should stay that way.

Without `--instances-dir` the artifact is stamped `SCAFFOLD`, because the
features would come from prompt metadata rather than real prompt text.
