# Reproduction

## Environment

```bash
uv sync --extra dev
uv run --extra dev pytest        # 117 passed, 5 skipped
uv run --extra dev ruff check .
uv run --extra dev mypy frontier scripts tests
```

Python 3.12 is pinned in `.python-version`; `pyarrow<20` has no wheels for
3.14+, so a newer interpreter forces a source build that fails.

## One command

```bash
uv run python scripts/reproduce.py --fast
```

Runs the full chain: validate corpus → train on `D_train` → calibrate H1 on
`D_cal_a` → CRC λ̂ on `D_cal_b` → E3 on `D_test` → figures.

To rebuild the corpus from a real run's ledger first:

```bash
uv run python scripts/reproduce.py --ledger data/pilot/stage_a.jsonl
```

## What gets promoted to `paper/`

Nothing, unless it is a genuine result. Promotion is refused when a report
carries the scaffold marker, and refused for *everything* when the predictor
artifact is marked `SCAFFOLD` — because `OURS` is then not the method, which
makes the headline row of T3 meaningless rather than merely provisional.

On the committed fixture corpus the pipeline correctly promotes nothing.

## Full pipeline, stage by stage

```bash
# 1. data
uv run python scripts/03_fetch_data.py gsm8k --output-dir data/raw --n 200

# 2. rate adherence (E1c / C4)
uv run python scripts/compress_smoke.py data/raw/gsm8k.jsonl \
    --backend llmlingua2 --target-model Qwen/Qwen2.5-0.5B-Instruct \
    --output reports/e1c_adherence.md

# 3. Gate 1 pilot (the go/no-go)
uv run python scripts/01_gate1_pilot.py --families gsm8k \
    --prompts-per-family 20 --samples 5 --compressor llmlingua2 \
    --ledger data/pilot/stage_a.jsonl --report reports/gate1.md

# 4. corpus
uv run python scripts/04_corpus.py --ledger data/pilot/stage_a.jsonl \
    --output data/corpus/v1/corpus.parquet --instances-dir data/raw

# 5-7. train, calibrate, evaluate
uv run python scripts/06_train.py --instances-dir data/raw
uv run python scripts/07_crc_report.py
uv run python scripts/08_eval.py
```

See `RUNBOOK.md` for costs, resuming, spend caps, and the GPU-less path.

## Determinism

Fits are full-batch with fixed seeds; splits are deterministic given a seed;
the grid is content-hash cached and idempotent. `docs/claims_ledger.md` is
updated only by the researcher, from held-out runs.
