# APC-06 — Experimental Protocol

Every experiment maps to a claim in `APC_03 §3` and to a table or figure in the paper. Nothing here is optional except where marked.

---

## 1. Variables

| | |
|---|---|
| **Independent** | requested budget `b ∈ {1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2}` · compressor `ψ` · target model `M` · task family · policy |
| **Dependent** | graded quality `ρ` · realised rate `r` · `T_in` · `T_out` · latency · **USD** |
| **Controlled** | temperature & seed · prompt template · query never compressed · pinned model revisions · pinned price table |

---

## 2. E1 — Per-instance frontiers *(Gate 1; claims 1, 2)*

**Question:** do instances within a task family differ in compression tolerance by more than sampling noise?

**Design:** 200 pilot prompts (stratified over 4 families) × 7 budgets × local model × `k=5`.

**The go/no-go test — this is the most important number in the project.**

Per-instance quality is estimated from `k` samples, so `Var[ρ̂] ≈ ρ(1−ρ)/k ≤ 1/(4k) = 0.05` at `k=5`. Propagated to `b*`, this gives a **noise floor** `σ²_noise` on `Var[b*]`, which we estimate directly by **split-half resampling**: split the `k` samples into two halves, compute `b*` from each, and take the disagreement as the noise variance. Then:

```
GATE 1 PASSES  iff   Var_within-family[ b*(x) ]  >  2 · σ²_noise    (bootstrap CI excludes the floor)
```

**Why *within*-family is the right test.** Cross-family heterogeneity is already established (TAAC's code-cliff vs CoT-gradual). A policy that only learns "code is fragile, summaries are not" is a per-family lookup table — i.e. baseline **B2b** — and would not justify the method. The contribution requires signal *inside* a family.

**Also report (all feed later contributions):**
- **E1b** — non-monotonicity: `P(b*_naive ≠ b*)` and the magnitude of the gap *(claim 2, audit D1)*.
- **E1c** — rate adherence: `r` vs `b` scatter per compressor and family *(feeds C4)*.
- **E1d** — output expansion: `T_out(b)/T_out(1)` per family and model *(feeds C2; the MBPP-vs-HumanEval contrast is the target)*.
- **E1e** — curve taxonomy: cluster the `ρ(x,·)` curves (cliff / gradual / flat / inverted-U) and report family composition. Good figure, and it makes the phenomenon legible.

**Figure 1:** spaghetti plot of per-instance frontiers, faceted by family, with the family mean overlaid — this single figure is the paper's motivating image.

---

## 3. E2 — Frontier prediction *(claims 3, 11)*

**Question:** can cheap features predict the curve, without a target-LLM call?

| Metric | Definition |
|---|---|
| Curve MAE | mean \|ρ̂(x,b) − ρ(x,b)\| over budgets |
| `b*` MAE / exact-match | error in the predicted monotone-safe budget |
| **Calibration** | ECE + reliability diagram of `ŝ`, **per budget** |
| Ranking | Spearman ρ between predicted and true `b*` across instances |
| **Compression regret** | `ρ(x,b*) − ρ(x,b̂)` — the plan's §24 metric, kept |
| **Token / cost regret** | `Cost(x,b̂) − Cost(x,b*)` |
| Overhead | predictor wall-time, ms |

- **E2a** feature tiers: L0 / L0+L1 / L0+L1+L2, each with its latency. *If L0 alone wins, report it as the finding.*
- **E2b** query-aware vs query-agnostic (the plan's §28 — keep it, it is a good experiment).
- **E2c** **rate-adherence head** *(claim 11, contribution C4)*: realised-rate error with and without `R_φ` inversion, plus the requested-vs-realised adherence table for every compressor.

---

## 4. E3 — End-to-end policy comparison *(claims 6, 7, 8, 10; the main result)*

**Primary metric — quality at matched cost.** Sweep each policy's operating parameter to trace a quality-vs-USD curve, then compare curves, not points. Comparing single operating points is how adaptive-compression papers accidentally cheat.

**Headline scalars (adapted from RouteLLM's evaluation design, `APC_02 §3.2`):**
- **CPGR** — *Compression Performance Gap Recovered*: fraction of the (**B2b** → oracle) quality gap recovered at matched cost. One number for the abstract.
- **CPT(x%)** — cost in USD to retain `x%` of uncompressed quality, at `x ∈ {95, 98, 99}`.
- **AUC** of the quality-vs-cost curve over the achievable cost range.

All reported with **paired bootstrap over prompts** (10,000 resamples), and in **USD and seconds**, never input tokens alone.

---

## 5. Baselines (all implemented behind the single `Policy` interface)

| ID | Baseline | Why it must exist |
|---|---|---|
| **B0** | No compression | Reference for `ρ(x,1)` |
| **B1** | Fixed budget, each `b ∈ B` | The frontier of the naive approach |
| **B2a** | Best global fixed budget (validation-tuned) | The usual bar in the literature |
| **B2b** | **Best fixed budget tuned per task family** | ⚠ **The real bar. Gate 3 is defined against this** |
| **B2c** | Best fixed budget per (family × model) | Strongest non-learned competitor |
| **B3** | Length-only heuristic | Plan's Risk 4 — "you didn't need ML" |
| **B3b** | gzip-ratio (redundancy) heuristic | Stronger free heuristic; often surprisingly good |
| **B4** | Random budget (rate-matched) | Isolates "does variability alone explain it?" |
| **B5a** | **AdaComp-style** point-estimate rate predictor, re-implemented at token granularity | The nearest prior work (`APC_02 §D2`) |
| **B5b** | **Adaptive QuerySelect**-style query-aware variable-rate | Nagle et al.'s method |
| **B5c** | **AttnComp-style** threshold-adaptive (no learned predictor) | The strongest *cheap* adaptive rival |
| **B6** | Random-token-drop at matched rate | Isolates whether *which* tokens are dropped matters at all |
| **B7** | **Model routing** (route to a cheaper model instead of compressing) | The competing cost lever; TAAC says it wins for reasoning |
| **B8** | **Prompt caching** sensitivity analysis | Neutralises "why compress when caching is free?" |
| **ORACLE** | True `b*` from held-out labels | **Upper bound — label as such in every table** |
| **ORACLE-noisy** | Oracle computed from `k`-sample estimates | Separates irreducible label noise from predictor weakness |
| **OURS** | CRC-calibrated frontier policy | |

**On B7 and B8** (audit E1): these are not padding. They are the two questions a good reviewer asks in the first five minutes, and having pre-computed answers is worth more than an extra ablation.

---

## 6. E4 — Ablations *(claim 4)*

Every row answers one scientific question; drop any row that does not.

| Ablation | Question |
|---|---|
| Full method | Does the system work? |
| − monotone head (free-form) | Does monotone-by-construction help under noisy labels? |
| − curve head (direct `K`-way classifier) | **Is curve prediction better than classification?** (C3's core claim) |
| − CRC (heuristic fixed threshold) | Does calibration deliver the promised risk? *(pairs with E5b)* |
| − output-length head (input-cost only) | **Does the true cost model change decisions?** (C2's core claim) |
| − rate-adherence head | Does adherence correction matter? |
| − query features | Does query-awareness matter? |
| − L1 (surface only) | Is the small LM worth its latency? |
| − abstain action (`b=1.0` removed) | Is the no-compression option useful? (plan's RQ3/H4) |
| Oracle | How much headroom remains? |

---

## 7. E5 — Risk-control validity *(claim 5; contribution C1)*

- **E5a** — for `ε ∈ {0.02, 0.05, 0.10}`, over **≥20 random `D_cal_b`/`D_test` splits**: empirical risk vs target `ε`. Plot with the CRC bound; expect empirical ≤ ε with slight conservatism. This plot **is** the guarantee.
- **E5b** — **uncalibrated heuristic threshold** (e.g. `ŝ ≥ 0.9`) vs CRC: measure how far the heuristic misses `ε`, and in which direction. This is the experiment that proves the guarantee is doing work rather than decorating.
- **E5c** — tail control via Learn-then-Test: `P(L > τ) ≤ δ`.
- **E5d** — cost of the guarantee: quality/cost lost relative to an unconstrained cost-minimising policy. Honest accounting; guarantees are not free and pretending otherwise invites suspicion.

---

## 8. E6 — Generalisation *(claims 12, 13, 15)*

- **E6a — cross-model.** Train on local model, test on API models 1 and 2. Report *(i)* zero-shot transfer, *(ii)* transfer after recalibrating **λ only** (no retraining — cheap, and the practically important number), *(iii)* full retrain. If λ-recalibration alone recovers most of the gap, that is a strong deployability result.
- **E6b — cross-task.** Leave-one-family-out over LongBench's six families.
- **E6c — guarantee under shift.** Calibrate on family A, deploy on family B, **report the risk violation**. Exchangeability fails here by construction; measuring the failure honestly is stronger than pretending it does not exist, and it pre-empts the reviewer who would find it.
- **E6d — recalibration sample efficiency.** How many labelled in-domain prompts restore validity? (Hypothesis: ~200.) A practically useful curve.

---

## 9. E7 — Cross-compressor transfer *(claim 14)*

Train the policy on LLMLingua-2; deploy on LongLLMLingua, CPC, truncate-tail. Report zero-shot and after-`R_φ`-refit.

**Why this matters more than it looks:** it establishes that the contribution is a **policy layer**, not an LLMLingua-2 accessory. That is the difference between "a tweak to one compressor" and "a component the field can reuse", and it is the cheapest available upgrade to the paper's perceived scope.

---

## 10. E8 — Distance to the fundamental limit *(optional, high value)*

On a small subset, compute Nagle et al.'s distortion-rate function (the LP) and place B2b / B5a / ours / oracle against it.

Turns the biggest scoop risk into our most rigorous instrument: *"we recover X% of the achievable gap to the information-theoretic limit."* Costs a few days. Do it if Phase 6 is on schedule.

---

## 11. E9 — Cheapest safe ≠ smallest safe *(claim 9; contribution C2)*

The dedicated experiment for C2.

- Frequency with which `argmin_b Ĉost(x,q,b) ≠ min{b : safe}`.
- Total cost of a policy optimised for **input tokens** vs one optimised for **total cost**, at matched quality.
- Sensitivity sweep over the price ratio `c_out/c_in ∈ {1, 2, 3, 4, 5}` — so the conclusion is stated as a function of pricing, not tied to one vendor's price list on one date.
- Per-family output-expansion curves (expected: code ≫ QA; MBPP ≫ HumanEval).

**Expected headline:** *"An input-token-optimal policy increases total cost on X% of prompts; ours does not."* If the effect is small, report it small — a well-measured null on this is still novel and still worth its subsection.

---

## 12. E10 — Net efficiency *(claim 10)*

Full accounting: predictor + compressor + LLM, in wall-clock seconds and USD, at three deployment points (local GPU / API / batched offline).

**Include the ECIR-2026 operating-window analysis explicitly:** for which `(prompt length × rate × hardware)` regions is the whole pipeline net-positive? Publishing the region where our own method *does not pay off* is a credibility asset, and it is far better to state it than to have a reviewer derive it.

---

## 13. E11 — Failure analysis *(plan §31, kept and sharpened)*

Taxonomy with instance counts and worked examples:
1. **Code cliff** — verify the r≈0.55 threshold with a *real* compressor and *functional* tests (TAAC used truncation + embedding similarity; this is our upgrade).
2. **Numeric/identifier loss** — the "perplexity paradox": syntactically predictable numerals get pruned despite being task-critical.
3. **Reasoning-chain breaks** — a removed intermediate step.
4. **Short prompts** — little to remove; overhead dominates.
5. **Very long prompts** — prediction degrades?
6. **Distribution shift** — the E6c cases.

For each: how often the policy *correctly* abstains (`b=1.0`), which is the direct evidence for the plan's H4.

---

## 14. Statistics *(fixing audit E4)*

- **Unit of analysis:** the prompt. **Paired bootstrap** over prompts, 10,000 resamples, BCa intervals.
- **Power:** with `N_test ≈ 1,200`, paired comparison, per-prompt quality s.d. ≈ 0.3, the minimum detectable paired difference at 80% power is ≈ **0.024** (2.4 quality points). Anything smaller is not claimable — pre-commit to that.
- **Multiplicity:** Holm–Bonferroni across the baseline family for the primary metric.
- **Effect sizes:** report Cohen's *d* or the paired mean difference with CI. Never a bare p-value.
- **Seeds:** ≥3 training seeds for the predictor; report mean ± s.d.
- **Pre-registration:** primary hypothesis = *OURS beats B2b on quality-at-matched-cost*. Everything else is secondary and labelled as such in the paper.
- **Determinism:** `T=0` runs need no repetition for the *generation*, but `k>1` at `T>0` is still required for graded `ρ` (audit D2). Do not conflate these two things — the plan's §30 does.

---

## 15. Results tables to fill

| Table | Content |
|---|---|
| **T1** | Positioning table (`APC_02 §4`) |
| **T2** | Corpus statistics — prompts, families, budgets, models, generations, total $ |
| **T3** | **Main:** policies × (quality, USD, latency, CPGR, CPT95) with CIs |
| **T4** | Ablations (E4) |
| **T5** | Risk control: target ε vs empirical risk, ±CI, vs uncalibrated |
| **T6** | Cross-model / cross-task / cross-compressor transfer |
| **T7** | Rate adherence: requested vs realised, per compressor × family |
| **T8** | Output expansion + `input-optimal vs total-optimal` cost comparison |
| **T9** | Failure taxonomy with counts |
| **F1** | Per-instance frontier spaghetti plot *(the motivating figure)* |
| **F2** | Quality-vs-USD Pareto curves, all policies |
| **F3** | Risk-control validity plot *(the guarantee)* |
| **F4** | Feature-tier latency vs benefit |

---

*Next:* `APC_07_BUILD_PROMPTS.md` — executable prompts to construct all of this.
