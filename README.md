# Adaptive Prompt Compression (APC)

APC is a research and evaluation workspace for risk-controlled, instance-adaptive prompt compression. It combines corpus preparation, feature extraction, predictor training, evaluation, and reproducibility artifacts.

## Quick start

The project targets Python 3.11 or newer. Install the package and development tools in an isolated environment:

```bash
pip install -e ".[dev]"
```

The optional `real` extra adds model and dataset integrations for experiments that require external providers.

## CLI entry points

- `frontier-validate` — validate corpus inputs
- `frontier-features` — compute registered features
- `frontier-train` — train the predictor
- `frontier-eval` — run evaluation workflows
- `frontier-crc-report` and `frontier-p10` — generate experiment reports

Configuration files are in `configs/`. Reproduction and operational guidance is in `REPRODUCE.md`, `RUNBOOK.md`, and `PAPER_GUIDE.md`; the numbered `APC_*.md` files form the project index and research plan.

## Quality checks

```bash
pytest
ruff check .
mypy frontier
```

Results and generated artifacts should be recorded with their configuration and input snapshot so experiments remain reproducible.

## Research architecture

```mermaid
flowchart LR
    Corpus[Prompt corpus] --> Validate[Corpus validation]
    Validate --> Features[Feature registry]
    Features --> Predictor[Instance-adaptive predictor]
    Predictor --> Policy[Risk-controlled compression policy]
    Policy --> Compressor[Compression backends]
    Compressor --> Evaluate[Evaluation and P10 reports]
    Evaluate --> Artifacts[Metrics, calibration, and paper artifacts]
```
