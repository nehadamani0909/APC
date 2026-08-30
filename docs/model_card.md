# Model card — FRONTIER predictor

## What it is

A three-head predictor that maps a prompt `(x, q)` to, for every budget
`b ∈ {1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2}`:

- **H1** `ŝ(b)` — probability the compressed prompt stays within `ε` of the
  uncompressed answer's quality. Monotone in `b` by construction.
- **H2** `r̂(b)` — realised compression rate, inverted at selection time.
- **H3** `T̂_out(b)` — output tokens, which makes the cost objective
  non-monotone in `b`.

It never calls the target LLM. That invariant is enforced by a unit test.

## Intended use

Selecting a compression rate per prompt, subject to a calibrated safety
threshold. It is a research artifact, not a production component.

## Training and calibration

| Split | Use |
|---|---|
| `D_train` (60%) | fit all heads |
| `D_cal_a` (10%) | calibrate H1's probabilities — **and nothing else** |
| `D_cal_b` (10%) | calibrate the CRC threshold λ̂ |
| `D_test` (20%) | everything reported |

Splits are assigned by **source document**, so a document reused across
instances cannot straddle a boundary. Calibrating H1 and selecting λ̂ on the
same split would void the C1 guarantee; they are kept apart in code.

## Inputs

L0 surface features by default (11 features, sub-millisecond). L1 small-LM
statistics and L2 encoder embeddings are supported via injection. Features
are standardised, and the constants ship with the artifact.

## Provenance recorded in every artifact

Git revision, corpus path and row count, price-table version, seed, `ε`,
feature tier, whether features came from real prompt text, the compressor
backends and target models present in the corpus, and split sizes.

## Limitations

- **Exchangeability fails across task families.** Calibrate per deployment
  domain; the risk violation under deliberate shift is measured and reported
  (E6c) rather than assumed away.
- **Label noise.** Graded quality is estimated from `k` samples, so its
  standard error is at best `1/(2√k)`. At `k=5` that is 0.224, which bounds
  how much heterogeneity can be claimed.
- **A `SCAFFOLD`-marked artifact is not a usable predictor.** It means
  features were derived from prompt metadata rather than real prompt text.
- The predictor is only as transferable as its corpus: a model, compressor,
  or family absent from training is out of distribution.

## Claims

`docs/claims_ledger.md` records claim status. It is updated only by the
researcher, from held-out runs.
