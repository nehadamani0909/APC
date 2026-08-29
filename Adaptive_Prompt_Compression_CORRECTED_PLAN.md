# Adaptive Prompt Compression — Corrected Research Plan
### (Supersedes `Adaptive_Prompt_Compression_Research_Plan.md`)

**Status:** This is the fixed, buildable version. Every defect found in audit has been resolved here — not patched, replaced. Read this document alone; it is self-contained. (Deeper backing material — full literature verification, experiment protocols, build prompts, rebuttal kit — lives in the companion `APC_00`–`APC_08` files in this folder, but you do not need them to understand or execute the plan.)

---

## 1. What was wrong, and what changed

Six defects made the original plan unbuildable or rejectable as written. Each is fixed below, and four of the six fixes turned into contributions rather than mere repairs.

| # | Original | Problem | Fixed to |
|---|---|---|---|
| 1 | `b* = min{b : A(x,b) ≥ A(x,1)−ε}` | Assumes quality is monotone in budget. It measurably isn't — LongLLMLingua reports **+17.1%** over the *uncompressed* prompt; other work shows moderate compression sometimes *improves* accuracy. One lucky low budget produces a spurious, unreproducible label. | **Monotone-safe budget**: the smallest `b` such that quality stays safe *at every budget above it too*. Immune to single-point noise, well-defined without assuming monotonicity. |
| 2 | Per-instance quality treated as e.g. "95% accuracy" for one prompt | On exact-match tasks (GSM8K, EM/F1) a single prompt's outcome is binary. The whole ε-constraint quietly conflates instance-level and population-level quality. | **Graded per-instance quality** `ρ(x,b)` from k=5 sampled generations (or task-native graded metrics). Instance-level *prediction*, population-level *guarantee* — kept explicitly separate. |
| 3 | Cost = input tokens only | Published evidence: aggressive compression **increases total cost** — a pre-registered trial found `r≈0.2` raised total cost **+1.8%** despite big input cuts, because under-specified prompts trigger compensatory output verbosity (56× output expansion observed on one benchmark). Output tokens bill 3–5× input. | **True end-to-end cost objective**: `c_in·T_in + c_out·T̂_out + overhead`, with a learned head predicting output-length expansion. Reported in USD and seconds, never raw token counts. |
| 4 | Requested compression rate assumed = realised rate | LLMLingua's actual output length deviates systematically from the requested target, and the deviation is prompt-dependent. Every "we saved 47% of tokens" claim is otherwise unverifiable. | **Rate-adherence model** `r̂ = R(x,b)`, learned and inverted at selection time so the system requests the rate that lands on the intended target. |
| 5 | Separate "PCS" module + a "Compression Gate," hand-tuned threshold | Reviewers' most obvious objection ("PCS is arbitrary") is correct — the threshold has no statistical grounding, and complexity ≠ compressibility (the plan's own §18 admits this). | **One frontier predictor**, no separate gate (folded into `b=1.0`), and the accept/reject threshold is **calibrated by Conformal Risk Control** — a distribution-free, finite-sample guarantee on quality degradation. Nothing hand-tuned. |
| 6 | LLMLingua (2023) as the fixed backend; three cited "prior works," two of which are misidentified | "AdaCoder" doesn't exist in this space; "Adaptive QuerySelect" is a method inside a NeurIPS 2024 paper, not a paper itself; the actual closest prior work (**AdaComp**, 2024) was missing entirely; all 2025–2026 literature was absent. | **LLMLingua-2 as default backend, behind a swappable adapter** (cross-backend transfer becomes a free experiment). Full corrected literature map in §4. |

---

## 2. The corrected research question

**Old framing (too broad, already claimed by others):**
> "Can dynamic/adaptive compression improve on fixed compression?"
— Already answered yes by AdaComp, ACC-RAG, and Nagle et al.'s Adaptive QuerySelect (NeurIPS 2024). Claiming this as novel is a fast path to desk rejection.

**Corrected framing:**
> Every existing prompt compressor — LLMLingua, LLMLingua-2, LongLLMLingua — treats the compression rate as a hyperparameter the *user* supplies. We ask the prior question: for a specific prompt, what rate should have been requested, and can we know cheaply, before paying for inference, **under an explicit statistical guarantee on quality**, and **against the cost the user will actually be billed**?

That reframing is what makes the project defensible. It doesn't compete with prior adaptive methods on "adaptive vs fixed" — it operates one level up, asking what a well-calibrated adaptive method should even be optimizing.

---

## 3. Formal problem statement (corrected)

### 3.1 Objects

- `x` = compressible context (documents, transcript, demonstrations, code). `q` = query/instruction — **never compressed** (compressing instructions is the dominant driver of output blow-up; keeping `q` fixed removes a confound).
- `b ∈ B = {1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2}`, requested compression rate; `b=1.0` means "don't compress" (this replaces the separate Compression Gate — one unified choice set).
- `r(x,b)` = **realised** rate after compression — generally `≠ b`.
- `ρ(x,q,b) ∈ [0,1]` = graded per-instance quality, estimated as the mean of `k=5` sampled task-metric outcomes at `T=0.7` (or a task-native graded metric: F1, ROUGE-L+BERTScore, pass@1).
- `T_in, T_out` = realised input/output token counts. `Cost(x,q,b) = c_in·T_in + c_out·T_out + overhead`.

### 3.2 The optimal budget (fixed)

```
b*(x) = min{ b ∈ B : ρ(x,q,b') ≥ ρ(x,q,1) − ε   for ALL b' ∈ B with b' ≥ b }
```

This is the smallest budget such that safety holds at that budget **and every less-aggressive budget above it** — the infimum of the maximal safe suffix of the curve. It does not require the curve to be monotone; it is simply robust to the single noisy point that breaks the naive `min{b : ρ(x,b) ≥ ρ(x,1)−ε}` definition. Report the naive and corrected labels side by side; their disagreement rate is itself a measurement of how often real compression curves are non-monotone (nobody has measured this at the instance level).

### 3.3 The selection objective (fixed)

```
b̂(x,q) = argmin_{b ∈ B}  Ĉost(x,q,b)     subject to     ŝ_θ(x,q,b) ≥ λ
```

where `ŝ_θ` is the predicted safety score (from the frontier predictor, §5) and `λ` is **calibrated, not hand-set** (§6). Critically: **this is a cost-minimization subject to a safety constraint, not "pick the smallest safe budget."** Because `Ĉost` includes predicted output-token expansion, the two are not the same thing — the cheapest safe budget can be a *less* aggressive one than the smallest technically-safe budget, if aggressive compression is predicted to inflate output tokens enough to raise total cost. This divergence is itself a headline experimental result (§7, E-cost).

---

## 4. Corrected positioning against prior work

The original plan's literature review had three citation errors. Corrected map:

| Work | What it actually is | Delta from ours |
|---|---|---|
| **Nagle et al., NeurIPS 2024** ("Fundamental Limits of Prompt Compression") | Derives the population-level distortion-rate *limit* via a linear program (near-oracle, largely synthetic data); proposes "Adaptive QuerySelect" as a variable-rate method *inside* that paper — it is not a separate citable work | They characterize what's achievable in principle, with oracle access. We build a causal, feature-based estimator usable *before* inference, with a deployable guarantee. We use their LP as an evaluation upper bound, not a competitor. |
| **AdaComp** (arXiv:2409.01579) | Labels the minimum sufficient top-k documents for RAG, trains a rate predictor — **structurally the closest prior work**, and what the original plan's "Version A" essentially reinvents | Document-granularity, RAG-only, point-estimate, no guarantee, input-tokens-only cost. We predict a full token-level curve for arbitrary long prompts, with a calibrated guarantee and a true-cost objective. |
| **ACC-RAG** (Findings EMNLP 2025) | RL-trained adaptive rate selection over *embedding* granularity | Requires model internals — unusable against a black-box API, which is the deployment regime we target. |
| **LLMLingua-2** (Findings ACL 2024) | 3–6× faster than LLMLingua-1, better out-of-domain — the correct 2026 default backend | Adopted as our default compressor, not a competitor. |
| **TAAC preprint series (2026)** | Non-peer-reviewed; documents the "compression paradox" (aggressive compression raising total cost via output expansion) using **truncation as a compression stand-in** and **embedding similarity** as the quality metric, on one model | We credit the observation, then supply the rigorous version: real neural compressors, task-native metrics, multiple models, a learned and calibrated policy. Their own future-work list names "learned compression policies" — that's this project. |

**One sentence for the abstract, doing the positioning work:** *"Prior adaptive compression methods choose a rate; we predict the full per-instance quality-vs-rate frontier, select the cheapest point on it that satisfies a calibrated, distribution-free safety guarantee, and optimize against the total cost the user actually pays — including compression-induced output expansion, which no prior adaptive method models."*

---

## 5. Corrected architecture

```
                         CONTEXT x, QUERY q
                                │
                    ┌───────────────────────┐
                    │  FEATURE EXTRACTOR     │   L0 surface (length, gzip-ratio,
                    │  (tiered, <20% of      │   redundancy, digit/code ratio…)
                    │  compressor latency)   │   L1 small-LM perplexity stats
                    └───────────┬────────────┘   L2 optional encoder embeddings
                                │
                    ┌───────────────────────┐
                    │  FRONTIER PREDICTOR    │   Three heads, one forward pass,
                    │  (single trunk)        │   emits a value for every b∈B:
                    └───────────┬────────────┘
                    │            │            │
              H1: quality   H2: realised  H3: output-length
              curve ŝ(b)    rate r̂(b)     T̂out(b)
              MONOTONE by   (rate         (drives the
              construction  adherence     true-cost
              (cumulative-  correction)   objective)
              logit head)
                    │            │            │
                    └────────────┼────────────┘
                                 ▼
                    ┌───────────────────────┐
                    │  RISK-CALIBRATED       │   λ set by Conformal Risk
                    │  SELECTOR              │   Control on held-out data:
                    │  b̂ = argmin Ĉost(b)    │   finite-sample guarantee
                    │  s.t. ŝ(b) ≥ λ         │   E[quality loss] ≤ ε
                    └───────────┬────────────┘
                                 │
                     feasible set empty? ──yes──▶ abstain, send uncompressed (b=1.0)
                                 │ no
                                 ▼
                    r̂⁻¹ inverts adherence: request the b that
                    lands on the intended realised rate
                                 │
                                 ▼
                    ┌───────────────────────┐
                    │  COMPRESSOR ADAPTER    │   LLMLingua-2 default;
                    │  (swappable, black-box)│   LongLLMLingua / CPC / others
                    └───────────┬────────────┘   pluggable, unmodified
                                 │
                                 ▼
                          TARGET LLM (black-box)
                                 │
                                 ▼
                    log everything to a cost ledger
                    (USD, latency, realised rate, quality)
```

**Inference cost invariant:** exactly one compressor call, one target-LLM call. The predictor never queries the target LLM to pick its own budget — that would erase the entire efficiency benefit, and it's enforced as a unit test in the build.

**Why the monotone head matters:** a free-form regressor on quality curves fits the non-monotone noise described in §3.2 and produces an unusable `b*`. The cumulative-logit head structurally guarantees `ŝ(b)` is non-decreasing in `b`, which (a) makes `b*` well-defined, (b) regularizes against label noise, and (c) is exactly the monotone-in-threshold structure Conformal Risk Control requires — so the fix for defect #1 is also the mechanism that makes the guarantee in defect #5 exact.

---

## 6. The guarantee (replaces the hand-tuned PCS gate)

The selector family `b̂_λ(x,q)` is monotone in `λ` by construction (raising `λ` shrinks the feasible set toward `b=1.0`, driving degradation risk to exactly zero). This is precisely the precondition **Conformal Risk Control** (Angelopoulos et al., ICLR 2024) needs. Calibrating `λ` on a held-out set gives:

```
E[ max(0, ρ(x,1) − ρ(x,b̂_λ(x))) ] ≤ ε
```

— a distribution-free, finite-sample guarantee, for exchangeable future prompts, with no assumption on the predictor, the compressor, or the target LLM. Verified during literature review: **no existing prompt-compression work applies risk control to rate selection.** This is the least-crowded, highest-value claim in the whole project, and it directly resolves the reviewer objection the original plan itself flagged as its biggest risk ("PCS is arbitrary").

**Honest scope, stated up front, not discovered by a reviewer:** the guarantee assumes exchangeability between calibration and deployment prompts, which breaks under task shift. The plan measures this directly (calibrate on task family A, deploy on B, report the violation) rather than asserting the guarantee holds everywhere.

---

## 7. Corrected experimental plan

### 7.1 Gate 1 (Week 3, go/no-go) — do this first, before building anything else

200 prompts × 7 budgets × 1 local model × k=5 samples (~7,000 generations, a few GPU-hours). Compute, per prompt: the monotone-safe `b*`, the noise floor via split-half resampling of the k samples, realised-vs-requested rate, and output-token expansion per task family.

**Pass condition:** variance of `b*(x)` *within* a task family exceeds the sampling-noise floor (bootstrap CI excludes it). This is deliberately the strict test — cross-family heterogeneity ("code is fragile, summaries aren't") is already established in the literature and is captured by a simple per-task-tuned fixed budget, not a learned policy.

- **Pass** → proceed to the full corpus build.
- **Fail** → pivot immediately to a corpus/measurement paper (§8) — pre-committed now, not improvised later.

### 7.2 Full corpus, predictor, calibration (Weeks 4–12)

Tiered grid (~262,000 generations total, ~300–400 GPU-hours + $400–900 API spend) across 6 task families (LongBench's sub-families, GSM8K, MeetingBank, HumanEval/MBPP, conversational), 4 compression backends, 3 target LLMs (chosen so two show opposite output-expansion directions). Train the three-head predictor; calibrate H1's probabilities; calibrate `λ` via Conformal Risk Control on a disjoint split.

### 7.3 Main comparison (Weeks 13–15)

**Primary baseline, corrected from the original plan:** not a single global fixed budget, but the **best fixed budget tuned per task family** — the real bar an adaptive method must clear. Also compared: length-only heuristic, gzip-redundancy heuristic, random budget, an AdaComp-style point predictor (re-implemented at token granularity), an attention-threshold adaptive method with no learned predictor, model routing (the competing cost lever), a prompt-caching sensitivity analysis (the "why compress at all" objection), and the oracle upper bound.

**Metrics, reported in USD and seconds, never raw token counts:** quality-at-matched-cost curves; Compression Performance Gap Recovered (fraction of the fixed-budget→oracle gap closed); Cost to Preserve Threshold; paired bootstrap CIs throughout.

### 7.4 The headline non-obvious result

Measure how often `argmin_b Cost(b)` (cheapest safe) differs from `min{b : safe}` (smallest safe), and the total-cost difference between a policy optimized for input-token reduction versus one optimized for true cost. Expected finding, grounded in the published evidence in §1: an input-token-optimal policy raises total cost on a measurable fraction of prompts; a true-cost-optimal policy does not.

### 7.5 Generalization and failure analysis

Cross-model, cross-task-family, cross-compressor transfer (including a cheap "recalibrate `λ` only, no retraining" condition — the practically important number for deployment). Failure taxonomy: code-compression cliffs, numeric/identifier loss, broken reasoning chains, short prompts where there's nothing to remove, distribution shift where the guarantee is shown to degrade honestly rather than silently.

---

## 8. The fallback (decided now, not improvised under pressure)

If Gate 1 or the feature-predictability gate (Week 9) fails: the project becomes a measurement paper — *"Measuring Instance-Level Compression Frontiers: How Much Can Adaptive Prompt Compression Actually Help?"* — releasing the corpus, the heterogeneity/predictability decomposition, and the oracle-headroom bound. This is a legitimate NeurIPS Datasets & Benchmarks or Findings paper, not a consolation prize, and it means the project cannot produce zero output regardless of how the core hypothesis resolves.

---

## 9. What stays from the original plan, unchanged

These were right the first time and should not be touched:
- Folding the Compression Gate into `b=1.0` (one unified budget set instead of two components).
- The complexity ≠ compressibility distinction (§18 of the original).
- Compression regret as a metric.
- The claims/evidence matrix discipline (never let a claim into the paper without a status column).
- "Measure the curves before building anything" as the guiding instinct — this document elaborates that instinct, it doesn't replace it.

---

## 10. Immediate next steps

1. Freeze the corrected `b*` definition and the graded-quality definition in writing (§3.2, §3.3) — done here.
2. Add output tokens, USD cost, and latency to the logging schema *before* any grid run — retrofitting later means re-running everything.
3. Run the Gate 1 pilot (§7.1). This is the single highest-value three days in the whole project.
4. Only after Gate 1 passes: build the corpus, then the predictor, then calibrate.

**Do not build the predictor, the calibration layer, or any UI before Gate 1 has passed.** That was the original plan's own warning (§42/§43), and it still applies — it's just now backed by a formal go/no-go test instead of a general instinct.
