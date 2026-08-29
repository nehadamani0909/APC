# APC-00 — Master Index

**Project:** Adaptive Prompt Compression → re-planned as **risk-controlled instance-adaptive compression-rate selection**
**Source document audited:** `Adaptive_Prompt_Compression_Research_Plan.md`
**Date:** 2026-08-28/29 · **Status:** planning complete, no code written

---

## The one-paragraph verdict

The original plan picks a **real problem** and is unusually honest about its own weaknesses — it already kills its own "dynamic compression is novel" claim, which most proposals never do. But as written it would be **rejected**, for two separable reasons. First, **positioning**: three of the four works it cites are misnamed or misattributed, the single closest prior work (**AdaComp**) is missing, its recommended title collides head-on with a NeurIPS 2024 paper, and the entire 2025–2026 literature — including a 2026 preprint series that observes its central empirical claim — is absent. Second, and more seriously, **the formulation is technically ill-posed in four places**: the optimal-budget definition assumes a monotonicity that measurably does not hold; per-instance quality on exact-match tasks is binary, making the per-instance ε-constraint degenerate; the cost model counts input tokens only, when published evidence shows aggressive compression *raises* total cost via output expansion; and it conflates the requested compression rate with the realised one.

All four are fixable *in planning*, before a single GPU-hour is spent — and each fix is worth more than a repair. Fixing the quality definition creates the opening for a **conformal risk guarantee** (verified unclaimed in this setting). Fixing the cost model produces a genuinely non-obvious result: **the cheapest safe budget is often not the smallest safe budget**. Fixing rate adherence yields a small, sharp, entirely unclaimed contribution. That is the difference between a project that reads as "LLMLingua plus a classifier" and one with five defensible contributions.

---

## Read in this order

| # | Document | What it settles | Read if |
|---|---|---|---|
| **00** | *this file* | Verdict, deltas, next actions | Start here |
| **01** | `APC_01_AUDIT.md` | Every defect, ranked, with fixes | You want to know what was wrong |
| **02** | `APC_02_LITERATURE_MAP.md` | Verified related work; the positioning table | You are writing §3 or checking novelty |
| **03** | `APC_03_NOVELTY.md` | The five contributions; claims ledger; forbidden claims | You are writing the abstract |
| **04** | `APC_04_ARCHITECTURE.md` | Formalism, layers, heads, guarantee, repo layout | You are building |
| **05** | `APC_05_BUILD_PLAN.md` | 20-week plan, budget, gates, fallback | You are scheduling |
| **06** | `APC_06_EXPERIMENTS.md` | Every experiment, baseline, metric, statistic | You are running things |
| **07** | `APC_07_BUILD_PROMPTS.md` | Copy-pasteable agent prompts P0–P11 | You are coding today |
| **08** | `APC_08_REBUTTAL_KIT.md` | Objections + answers, paper plan, titles, venues | You are submitting |

---

## What changed, in one table

| | Original plan | This re-plan |
|---|---|---|
| Core question | "Can we predict the optimal compression budget?" | "Can we predict the frontier and select the **cheapest safe** rate **with a guarantee**?" |
| Optimal budget | `min{b : A ≥ A₀−ε}` (assumes monotonicity) | **Monotone-safe** budget over the safe suffix |
| Quality | Binary/ambiguous, instance vs population conflated | Graded `ρ(x,b)`; instance-level *prediction*, population-level *guarantee* |
| Safety threshold | Hand-picked, via a "PCS" module | **Conformal risk control** — finite-sample, distribution-free |
| Objective | Minimise input tokens | Minimise **realised total cost** (in + out + overheads) |
| Rate | Requested = realised (assumed) | **Two-stage**, with a learned adherence model |
| Components | PCS + gate + budget predictor | One trunk, three heads; gate folded into `b=1.0` |
| Backbone | LLMLingua (2023) | LLMLingua-2 default, **swappable adapter** |
| Primary baseline | "best fixed budget" | **Best fixed budget tuned per task family (B2b)** |
| Fallback if it fails | none | **Compression Frontier Corpus** — publishable on its own |

---

## The five contributions (detail in `APC_03`)

| | Contribution | Novelty | Risk |
|---|---|---|---|
| **C1** | **Risk-controlled compression** — first prompt-compression policy with a distribution-free finite-sample guarantee on quality degradation (conformal risk control) | **High** — verified unclaimed | Low |
| **C2** | **True-cost objective** — predict compression-induced *output expansion*; show the cheapest safe rate ≠ the smallest safe rate | **High** — phenomenon known, never acted on | Low |
| **C3** | **Instance-level frontier prediction** with a monotone-by-construction head | Moderate | Medium (AdaComp adjacency) |
| **C4** | **Rate-adherence-corrected control** — requested ≠ realised, modelled and inverted | Moderate, entirely unclaimed | Low |
| **C5** | **Compression Frontier Corpus** + heterogeneity × predictability decomposition | Moderate–High | **Low — this is the insurance policy** |

**C1 and C2 are the paper.** C5 means the project cannot produce zero output.

---

## The three gates

Each has a pre-committed alternative, decided now rather than under pressure later.

| Gate | Week | Test | If it fails |
|---|---|---|---|
| **1** | 3 | Within-family `Var[b*]` exceeds the `k=5` sampling-noise floor (bootstrap CI) | Pivot to the C5 corpus/negative-result paper |
| **2** | 9 | L0+L1 features beat length-only at predicting `b*` | Corpus + decomposition paper |
| **3** | 15 | Beats **B2b** on quality-at-matched-cost | Reframe: *"when does adaptive compression help?"* — C1/C2/C5 survive |

**Gate 1 is within-family on purpose.** Cross-family heterogeneity ("code is fragile, summaries aren't") is already known and is exactly what baseline B2b captures. The contribution requires signal *inside* a family.

---

## Budget

**≈300–400 GPU-hours + $400–900 API** · ~262,000 generations across a tiered grid · 20 weeks at ~1 FTE.
Full breakdown in `APC_05 §4`. Includes a 40% re-run contingency, which is not padding.

---

## What was verified during this audit

Read or metadata-confirmed on 2026-08-28. Full map with verification flags in `APC_02`.

- **Nagle et al., NeurIPS 2024** (arXiv:2407.15504) — "Adaptive QuerySelect" is a *method inside* this paper, not a paper. It owns the phrase *"rate-distortion framework for prompt compression"* — hence the title change.
- **AdaComp** (arXiv:2409.01579) — labels the minimum sufficient context and trains a rate predictor. **The plan's "Version A" is essentially this paper.** Absent from the original plan; the plan's "AdaCoder" is a phantom.
- **ACC-RAG**, Findings EMNLP 2025 (arXiv:2507.22931) — adaptive rate by input complexity, but *embedding*-granularity and RAG-specific.
- **LLMLingua-2**, Findings ACL 2024 (arXiv:2403.12968) — 3–6× faster than LLMLingua, better OOD. The correct backbone.
- **arXiv:2505.00019** (ICLR 2025 WS) — moderate compression sometimes *improves* accuracy; output-length effects are **model-dependent**. → breaks the plan's monotonicity assumption.
- **arXiv:2603.23525 / 2603.23527** (TAAC series, 2026) — code cliff at r≈0.55; the **compression paradox** (aggressive compression *raises* total cost); Ψ instruction-survival. Non-peer-reviewed, uses **truncation as a compression stand-in** and **embedding similarity as the metric**, single model — and lists *"learned compression policies"* as future work. Biggest scoop risk **and** clearest opening.
- **ECIR 2026** (arXiv:2604.02985) — LLMLingua gives end-to-end speedups **only inside a narrow operating window**; outside it the compressor cancels the gains.
- **Verified gap:** no existing work applies conformal prediction / risk control to prompt-compression **rate selection**. → contribution C1.

---

## Do these seven things next

| # | Action | Effort | Blocking |
|---|---|---|---|
| 1 | Read Nagle et al. + AdaComp in full; write the two delta paragraphs | 1 day | **Yes** |
| 2 | Read TAAC ×2 + ECIR 2026; copy their stated limitations into the positioning doc | ½ day | **Yes** |
| 3 | Freeze the label definition (monotone-safe `b*` + graded `ρ`) in writing | ½ day | **Yes** |
| 4 | Add output tokens, USD, and latency to the logging schema **before** any grid runs | ½ day | **Yes** — retrofitting means re-running the grid |
| 5 | Run the Gate-1 pilot: 200 prompts × 7 budgets × 1 model × k=5 | 3 days | **Yes** — the go/no-go |
| 6 | Pre-register the primary hypothesis and analysis plan (OSF) | 2 hours | No — but the competing series is pre-registered |
| 7 | Write down the null-result pivot | 1 hour | No — but it is free insurance |

Then run `APC_07` prompts **P0 → P3** and stop at Gate 1.

---

## The one thing to keep from the original plan

> *"The first actual experiment should be: generate per-prompt compression curves… If the curves show strong prompt-level heterogeneity, we proceed. If they do not, we reconsider the research direction before investing in a complicated architecture."* — §43/§52

That instinct is correct and is the backbone of this entire re-plan. Everything here is an elaboration of it: measure first, gate hard, and make sure the measurement is publishable on its own.
