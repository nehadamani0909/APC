# Final handover: P0–P11

## Delivery status

The complete build-prompt sequence P0 through P11 is implemented and pushed
to GitHub on `main`.

- Final commit: `e5dea61`
- Remote: `https://github.com/nehadamani0909/APC.git`
- Working tree was clean after the release push.

## Implemented system

- P0: repository skeleton, Hydra configuration support, Structlog logging,
  append-only cost ledger, and versioned prices.
- P1: task, target-model, and metric adapters.
- P2: compressor interfaces, deterministic baselines, and cache support.
- P3: pilot grid and Gate 1 analysis.
- P4: corpus schema, validation, parquet export, and datasheet.
- P5: L0/L1/L2 feature extraction interfaces.
- P6: predictor heads, training scaffold, and calibration utilities.
- P7: CRC/LTT risk control and inference policy.
- P8: shared-path baseline policies and smoke T3 evaluator.
- P9: evaluation metrics, BCa bootstrap, multiplicity correction, power
  calculation, and vector figure generation.
- P10: ablation, transfer, cost-sensitivity, efficiency, and failure reports.
- P11: MIT release, artifacts, model card, claims ledger, cost summary, and
  reproduction command.

## Verification

The final repository passes:

```bash
uv run pytest
uv run ruff check .
uv run mypy frontier scripts tests
uv run python scripts/reproduce.py
```

At handover, the suite contained 37 passing tests. The reproduction command
validates the committed parquet corpus and regenerates corpus statistics, T3,
P9 artifacts, and P10 tables/failure analysis.

## Scientific status

The code path is complete, but the current released evidence is explicitly an
offline deterministic fixture/scaffold. The following are not publication or
production claims yet:

- real benchmark coverage;
- real neural compressor and target-model measurements;
- trained predictor quality;
- real calibration constants;
- final CPGR, transfer, efficiency, or failure results.

Before publication, replace the fixture pilot with licensed benchmark data,
run the real compressor/model grid, train on `D_train`, calibrate H1 on
`D_cal_a`, calibrate CRC on disjoint `D_cal_b`, and report held-out `D_test`
and shifted-family results. Update the claims ledger only from those runs.

## Gate status

Gate 1 was exercised by the deterministic pilot and recorded as passing, with
the report caveat that it is not a scientific conclusion until real model and
benchmark data are used.

No further build prompt remains. The next work is empirical execution and
claim validation, not additional repository scaffolding.
