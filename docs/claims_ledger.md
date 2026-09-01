# Resolved claims ledger

Updated from `runs/real_v1` (630 cells, 30 instances, 3 families,
LLMLingua-2 `bert-base-multilingual-meetingbank`, target
`llama3.2:3B-Q4_K_M`, k=3, 2026-08-31). Superseded the previous version,
in which every row was `deferred` because the only corpus was an offline
fixture and a truncation-backed pilot.

A claim is **supported** only where this grid measures it. Scope is one
target model, one compressor checkpoint, and a 500–3500 token context
window; see `PAPER_GUIDE.md` §5.

| # | Claim | Experiment | Status |
|---:|---|---|---|
| 1 | Per-instance frontiers are heterogeneous beyond label noise | E1 | **not supported at this n.** `qa` is the one family with no validity problems (10/10 solvable): Var[b*]=0.0695 against a noise floor of 0.0782 — *below* the floor. An interpretable FAIL, not an uninterpretable one. `multidoc_qa` (9/10 solvable but ρ(x,1)=0.105) and `reason` (6/10 solvable) fail the preconditions |
| 2 | Curves are frequently non-monotone | E1b | **supported.** Naive and monotone-safe `b*` disagree on **20.0%** of 30 instances (mean gap 0.483). Aggregate quality peaks at b=0.8 (0.428) above uncompressed (0.372) |
| 3 | Cheap features predict frontier position | E2 | deferred: no predictor trained on this grid |
| 4 | Curve prediction beats classification | E4 | deferred |
| 5 | CRC attains target in-distribution risk | E5 | deferred: needs a predictor and populated `D_cal_a`/`D_cal_b` |
| 6 | OURS beats per-family fixed budget | E3 | deferred: no policy trained |
| 7 | OURS beats AdaComp-style point prediction | E3 | deferred |
| 8 | OURS beats threshold adaptation | E3 | deferred |
| 9 | Output expansion changes the optimal budget | E9 | **measured and NOT supported, in this regime.** Expansion is real — `qa` reaches **1.68×** at b=0.2 (95% CI [1.27, 2.14]) and `reason` **1.28×** (CI [1.09, 1.48]), both excluding 1.0 — but the cheapest safe budget equalled the smallest safe budget for **all 25** evaluable instances at every price ratio `c_out/c_in ∈ [1,5]`. Input reduction dominates ~40:1 (T_in 2405→498 vs T_out 30→40). See note below |
| 10 | Net cost and latency savings are positive | E10 | not evaluated: local inference is zero-priced |
| 11 | Adherence correction improves control | E2c | **the gap is supported; the correction is not built.** LLMLingua-2 systematically *over*-compresses at every budget, signed error −0.027 to −0.060, all 95% CIs excluding zero, n=90 per budget. Largest at b=0.65 |
| 12 | Policy transfers across target LLMs | E6a | deferred: one model |
| 13 | Policy transfers across task families | E6b | deferred |
| 14 | Policy transfers across compression backends | E7 | deferred: one backend |
| 15 | Shift behaviour is measured honestly | E6c | supported as a protocol; no guarantee claimed |

## The claim-9 result is a finding, not a null

The compression paradox reported by the TAAC series is not contradicted
here — it is **bounded**. Both are consistent, because the paradox needs
output tokens to be large relative to input tokens. TAAC measured it on code
generation, where outputs are long. This grid measures short-answer QA and
few-shot reasoning, where a 1.68× output expansion adds ~10 tokens against an
input reduction of ~1900.

That is a useful, publishable statement: *compression-induced output
expansion is real and measurable with a neural compressor, but it changes the
cost-optimal budget only when the output-to-input token ratio is high.* It
tells a practitioner exactly when to worry, and it is honest about when not
to.

Stating it requires reporting the negative alongside the positive. Reporting
only the 1.68× expansion, without the 0/25 divergence, would be the same
error in the opposite direction from the one the project set out to fix.
