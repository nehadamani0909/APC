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
uv run --extra dev pytest        # 134 passed, 5 skipped
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
uv run python scripts/03_fetch_data.py humaneval --output-dir data/raw --n 164
uv run python scripts/03_fetch_data.py mbpp --output-dir data/raw --n 200
```

Writes normalised JSONL to `data/raw/<name>.jsonl` (all splits, each row
carrying its own `split`) and per-split files to `data/splits/`. Splits are
60/10/10/20, assigned by **source document** so a document reused across
instances cannot straddle a boundary.

LongBench sub-family names (`multidoc_qa`, `summarisation`, …) are accepted as
aliases for their first config; pass a real config name (`hotpotqa`,
`gov_report`, `lcc`, …) to select one exactly.

**LongBench is distributed as a loading script**, so its specs set
`trust_remote_code=True` and `datasets` will execute that script. This is
opt-in per dataset rather than global, so it never silently applies to the
others. On Windows you will also see a `huggingface_hub` symlink warning
unless Developer Mode is on; it is harmless, and only costs disk space.

Datasets and weights are gitignored and must never be committed.

### Verified against live data

All six families fetched and checked: exact 60/10/10/20 ratios, no prompt id
or source document crossing a split, no empty fields, no empty splits.

| Family | n | Context words (min / median / max) | Notes |
|---|---:|---|---|
| `gsm8k` | 200 | 737 / 737 / 737 | 8-shot CoT block, constant by construction |
| `hotpotqa` | 120 | 1,258 / 10,260 / 12,680 | the long-context workhorse |
| `meetingbank` | 64 | 122 / 999 / 28,247 | in-domain for the LLMLingua-2 checkpoint |
| `humaneval` | 64 | 17 / 39 / 131 | short; see below |
| `mbpp` | 64 | 7 / 14 / 27 | very short; see below |
| `sharegpt` | 64 | 32 / 853 / 1,526 | ~22% of conversations skipped as malformed |

**The code families are short-prompt cases.** MBPP's compressible context is
the natural-language description alone -- a median of 14 words, so `b=0.2`
leaves about three. HumanEval is not much longer. Any "code cliff" measured
there is the APC-06 §E11.4 *short prompt* regime ("little to remove; overhead
dominates") rather than a compression-tolerance finding, and should be
reported as such. LongBench's `lcc` and `repobench-p` are the long-context
code subsets if a genuine code-compression result is wanted.

**Dataset quirks found by running the fetch:**

- LongBench ships a loading script, so its specs set `trust_remote_code`.
- HumanEval needs the namespaced id `openai/openai_humaneval`; modern
  `huggingface_hub` rejects a bare repo id.
- ShareGPT stores raw JSON that `datasets` cannot auto-detect, so the spec
  names `data_files` explicitly. It is scraped assistant output with unclear
  licensing -- check redistribution terms before shipping anything derived
  from it.

## 5b. Model downloads on a throttled connection

Unauthenticated Hugging Face downloads are rate-limited, and large weight
files are where that bites. Measured here: **0.21 MB/s** on the 3.1 GB
`Qwen2.5-1.5B-Instruct` weights, with `snapshot_download` stalling outright
twice. Small files and datasets were unaffected.

Set a token from a free account before pulling anything above ~1 GB:

```bash
export HF_TOKEN=...        # huggingface.co/settings/tokens
export HF_HOME=D:/hf-cache # keep weights off a small system drive
```

## 6. Gate 1 — the decision that determines the paper

200 prompts stratified over 4 families × 7 budgets × k=5 ≈ **7,000
generations** (APC-06 §E1). Until this number exists you do not know whether
you are writing the method paper (C1+C2) or the C5 corpus paper.

**Without a GPU**, run it on `gpt-4o-mini` rather than a local model:

≈7,000 calls, long inputs and short outputs. Estimated from **measured**
context lengths (above), 50 prompts per family, and the mean realised rate
across the seven budgets (≈0.55):

| Family | ≈ tokens/call | Calls | ≈ input tokens |
|---|---:|---:|---:|
| gsm8k | 700 | 1,750 | 1.2M |
| hotpotqa | 7,150 | 1,750 | 12.5M |
| meetingbank | 4,400 | 1,750 | 7.7M |
| sharegpt | 1,100 | 1,750 | 1.9M |
| **total** | | **7,000** | **≈23M** |

At `gpt-4o-mini` rates: ≈23M input ⇒ **$3.50**, plus ≈1M output ⇒ **$0.63**.
**Roughly $4–6.** Set the cap at $10 and let it abort rather than trusting the
estimate — long-context families dominate, so a mis-sized MeetingBank pull
moves the total more than anything else.

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
