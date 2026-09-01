# Findings — real grid

Ledger: `runs/real_v1/ledger.jsonl` · rows **630** · instances **30**
Families: multidoc_qa, qa, reason · backends: llmlingua2 · targets: llama3.2:latest

> Local inference is zero-priced, so USD is 0 by construction. Every
> monetary statement below is in **token-cost units** `T_in + (c_out/c_in)·T_out`
> and is therefore a claim about the price *ratio*, not about one
> vendor's bill.

## 1. Quality by budget

| backend | b | mean quality | n |
|---|---|---|---|
| llmlingua2 | 0.20 | 0.354 | 90 |
| llmlingua2 | 0.30 | 0.327 | 90 |
| llmlingua2 | 0.40 | 0.390 | 90 |
| llmlingua2 | 0.50 | 0.364 | 90 |
| llmlingua2 | 0.65 | 0.423 | 90 |
| llmlingua2 | 0.80 | 0.428 | 90 |
| llmlingua2 | 1.00 | 0.372 | 90 |

## 2. Gate 1 — within-family frontier heterogeneity

### backend: `llmlingua2`

**⚠ Validity preconditions NOT met — the verdict below is not interpretable:**

- 5/30 prompts (17%) have rho(x,1) = 0: the target model never solves them uncompressed, so their b* is degenerate (every budget is trivially 'safe')
- mean quality peaks at b=0.8 (0.428), above uncompressed (0.372): compression cannot systematically improve accuracy, so the curves are noise-dominated

Do not report this as a Gate 1 FAIL; it is an uninterpretable measurement, which is a different thing and has a different fix: a stronger target model, or instances the model can actually solve uncompressed.

| ε | Var[b*] | 95% CI | noise floor | 2×floor | passes |
|---|---|---|---|---|---|
| 0.02 | 0.1116 | [0.0796, 0.1335] | 0.0578 | 0.1157 | **False** |
| 0.05 | 0.1079 | [0.0738, 0.1325] | 0.0547 | 0.1094 | **False** |
| 0.1 | 0.0797 | [0.0435, 0.1097] | 0.0475 | 0.0950 | **False** |

Non-monotonicity: naive and monotone-safe `b*` disagree on **20.0%** of 30 instances (mean gap 0.483).

## 3. Rate adherence (C4) — requested vs realised

| backend | requested b | realised (mean) | sd | signed error | 95% CI | n |
|---|---|---|---|---|---|---|
| llmlingua2 | 0.20 | 0.173 | 0.012 | -0.027 | [-0.030, -0.025] | 90 |
| llmlingua2 | 0.30 | 0.262 | 0.016 | -0.038 | [-0.041, -0.035] | 90 |
| llmlingua2 | 0.40 | 0.354 | 0.021 | -0.046 | [-0.050, -0.042] | 90 |
| llmlingua2 | 0.50 | 0.444 | 0.022 | -0.056 | [-0.061, -0.052] | 90 |
| llmlingua2 | 0.65 | 0.590 | 0.032 | -0.060 | [-0.067, -0.053] | 90 |
| llmlingua2 | 0.80 | 0.751 | 0.041 | -0.049 | [-0.057, -0.041] | 90 |
| llmlingua2 | 1.00 | 1.000 | 0.000 | 0.000 | [0.000, 0.000] | 90 |

## 4. Output expansion (C2 mechanism)

| family | b | T_out(b)/T_out(1) | 95% CI | n |
|---|---|---|---|---|
| multidoc_qa | 0.20 | 1.033 | [0.889, 1.176] | 30 |
| multidoc_qa | 0.30 | 0.971 | [0.823, 1.131] | 30 |
| multidoc_qa | 0.40 | 1.042 | [0.921, 1.164] | 30 |
| multidoc_qa | 0.50 | 1.088 | [0.941, 1.231] | 30 |
| multidoc_qa | 0.65 | 1.030 | [0.892, 1.177] | 30 |
| multidoc_qa | 0.80 | 0.931 | [0.794, 1.072] | 30 |
| multidoc_qa | 1.00 | 1.000 | [0.864, 1.138] | 30 |
| qa | 0.20 | 1.681 | [1.268, 2.135] | 30 |
| qa | 0.30 | 1.648 | [1.293, 2.039] | 30 |
| qa | 0.40 | 1.591 | [1.170, 2.060] | 30 |
| qa | 0.50 | 1.289 | [0.985, 1.620] | 30 |
| qa | 0.65 | 0.985 | [0.833, 1.135] | 30 |
| qa | 0.80 | 0.966 | [0.819, 1.123] | 30 |
| qa | 1.00 | 1.000 | [0.911, 1.082] | 30 |
| reason | 0.20 | 1.275 | [1.086, 1.481] | 30 |
| reason | 0.30 | 1.353 | [1.135, 1.610] | 30 |
| reason | 0.40 | 1.294 | [1.095, 1.531] | 30 |
| reason | 0.50 | 1.261 | [1.090, 1.443] | 30 |
| reason | 0.65 | 1.257 | [1.087, 1.440] | 30 |
| reason | 0.80 | 1.227 | [1.021, 1.456] | 30 |
| reason | 1.00 | 1.000 | [0.924, 1.078] | 30 |

## 5. Cheapest safe budget ≠ smallest safe budget (C2 headline)

| c_out/c_in | instances | divergent | % | mean saving when divergent | 95% CI |
|---|---|---|---|---|---|
| 1 | 25 | 0 | 0.0% | 0.000 | [0.000, 0.000] |
| 2 | 25 | 0 | 0.0% | 0.000 | [0.000, 0.000] |
| 3 | 25 | 0 | 0.0% | 0.000 | [0.000, 0.000] |
| 4 | 25 | 0 | 0.0% | 0.000 | [0.000, 0.000] |
| 5 | 25 | 0 | 0.0% | 0.000 | [0.000, 0.000] |

---

## What this does and does not license

Claims are only supportable here for the single target model, the
single compressor checkpoint, and the context-length window this
grid used. See `PAPER_GUIDE.md` §5 for the limitations that must
appear in the paper, and `docs/claims_ledger.md` for the ledger to
update from these numbers.
