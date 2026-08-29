# Frontier runbook

## Fast, offline smoke run

```bash
uv sync
uv run pytest
uv run python scripts/reproduce.py --fast
```

This exercises corpus validation, feature/evaluation/report generation, and
the release output layout without network access. The committed parquet is a
deterministic pilot artifact, not a real benchmark result.

## Real data run

Install the optional ETL dependencies and provide no secrets in files:

```bash
uv sync --extra real
uv run python scripts/03_fetch_data.py gsm8k --output-dir data/raw
```

Provider keys must be supplied through the relevant environment variables,
such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. Never commit them. Configure
model revisions, price-table version, temperature, seeds, and the USD spend
cap before running a real grid. The grid is resumable from its completed-cell
ledger; raise the cap only deliberately and record the change in the run
metadata.

## Budget and hardware

The deterministic pilot records 0 USD and 0 GPU-hours. Real cost and runtime
depend on prompt count, seven budgets, sample count, target model, compressor,
and hardware; obtain an estimate from a dry run before removing the cap.

## Outputs

Real runs should populate `data/raw`, `data/splits`, `data/corpus`,
`artifacts`, `paper/figures`, and `paper/tables`. The claims ledger is updated
only by the human researcher after checking held-out results.
