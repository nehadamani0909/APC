# Frontier model card

## Model

The released predictor artifact is the deterministic L0 scaffold in
`artifacts/predictor.json`. It emits the seven-budget interface required by
the policy layer. The repository does not claim a trained benchmark model:
real training requires corpus labels and the split discipline in APC-04 §5.4.

## Calibration domain

The calibration contract is `D_cal_a` for H1 probability calibration and
disjoint `D_cal_b` for CRC threshold selection. The checked-in constants are
placeholders for the offline fixture release and must be replaced before
deployment.

## Shift behaviour

P10 deliberately records positive shifted-family risk in T5 rather than
hiding it. Exchangeability is not assumed across task families or target
models; deployment requires in-domain recalibration and a fresh shift audit.

## Limitations and intended use

Use this release to reproduce the software pipeline and smoke reports. Do not
use the placeholder predictor or calibration constants to make production
safety claims.
