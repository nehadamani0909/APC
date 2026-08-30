# Configuration

Each file pins everything a run must record (APC-04 §7): model id and
revision, compressor backend, price-table version, seeds, temperature, and
for API runs a hard USD spend cap.

| File | Purpose |
|---|---|
| `pilot.yaml` | Gate 1 pilot, local CPU model, zero spend |
| `pilot_api.yaml` | the same pilot on an API model, with a cap |
| `tiered.yaml` | the full tiered grid from APC-05 §4.1 |
| `train.yaml` | predictor fitting and calibration splits |
| `eval.yaml` | E3 evaluation and the price row costs are reported at |
| `compressors.yaml` | backend registry |

Values reading `set_me_explicitly` must be set deliberately before a run;
nothing supplies a default for a model revision or a spend cap.
