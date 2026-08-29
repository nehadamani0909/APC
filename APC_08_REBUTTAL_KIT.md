# APC-08 — Reviewer-Proofing: Objections, Answers, Paper Plan

**Premise:** "unrejectable" does not mean a paper nobody can criticise. It means every foreseeable objection already has a *pre-computed experiment* behind it, so the rebuttal cites a table instead of arguing. This document is that list.

---

## 1. The three reviews you will actually get

### Reviewer A — "the adaptive-compression expert" (highest risk)

> *"Adaptive compression rate selection is well-explored. AdaComp trains a rate predictor from minimum-sufficient-context labels; ACC-RAG adapts compression by input complexity; Nagle et al. propose Adaptive QuerySelect, a query-aware variable-rate method. The delta over these is unclear."*

**Answer (rehearse this until it is one breath):**
> We agree these are the closest works and we compare against all three (B5a, B5b, and ACC-RAG's setting). Our contribution is not that compression should adapt — we credit that to them explicitly in the introduction. It is **what is predicted, under what guarantee, against what cost**. AdaComp predicts a *point* rate at *document* granularity in RAG; we predict a *monotone quality curve* at *token* granularity for arbitrary long prompts. ACC-RAG operates on *embeddings*, which is unavailable against black-box APIs — the regime where the token bill actually is. Nagle et al. characterise the *fundamental limit* with oracle access on largely synthetic prompts; we build a *causal, feature-based estimator* usable before inference, and we use their limit as an evaluation instrument (Table E8). Beyond that, **no prior adaptive method offers a statistical guarantee on quality degradation (Table T5), models compression-induced output expansion (Table T8), or models the requested-vs-realised rate gap (Table T7)** — the last four columns of Table 1 are empty for every prior method.

**Pre-computed evidence:** T1, T3, T5, T7, T8, E8.

### Reviewer B — "the empiricist"

> *"How much of the gain comes from adaptation rather than from tuning? A per-task fixed budget may capture most of it. And is the improvement worth the added complexity?"*

**Answer:**
> This is precisely why **B2b — the best fixed budget tuned per task family — is our primary comparison, not a global fixed budget** (which is the usual and much weaker bar). We pre-registered "OURS beats B2b at matched cost" as the primary hypothesis. We further report B2c (tuned per family × model), the oracle upper bound, and ORACLE-noisy so the reader can separate irreducible label noise from predictor error. Complexity is accounted in USD and seconds including all overheads (T3, E10), and we publish the operating regions where our pipeline does **not** pay off.

**Pre-computed evidence:** T3, E3, E10, ORACLE-noisy.

### Reviewer C — "the sceptic"

> *"Why compress at all in 2026? Prefix caching is free and lossless, and routing to a cheaper model is a bigger lever."*

**Answer:**
> Both are real and we address them explicitly rather than ignoring them. Caching helps only on **repeated static prefixes**; our target regime is **unique dynamic long context** — retrieved documents, transcripts, session history — where cache hit rates are near zero. We include a cache-hit-rate sensitivity analysis (B8) showing the break-even hit rate above which compression stops paying. On routing, we run it as baseline **B7** and study **composition**: routing chooses *which model*, compression chooses *how much context*, and they are complementary levers. Where routing dominates — plausibly for reasoning tasks, per prior observations — we report it as such.

**Pre-computed evidence:** B7, B8, E10.

---

## 2. The full objection table

| # | Objection | Pre-computed answer | Evidence |
|---|---|---|---|
| 1 | "Adaptive compression exists" | 4 axes of delta; direct comparison to all three neighbours | T1, T3 |
| 2 | "LLMLingua already has a budget controller" | It *allocates* a user-given budget; it never *chooses* one. Verified across every method in T1 | T1, §2 of paper |
| 3 | "PCS is arbitrary" | **PCS is removed.** The threshold is set by conformal risk control with a finite-sample guarantee, not chosen by hand | T5, E5b |
| 4 | "A length heuristic would do" | B3 and B3b (gzip redundancy); tiered feature ablation with latencies | E2a, T4, F4 |
| 5 | "Overhead eats the savings" | Full USD + seconds accounting; hard latency budget (predictor < 20% of compressor); the operating-window analysis is published | E10, T3 |
| 6 | "It memorised the dataset" | Leave-one-family-out; two held-out target models; held-out compressors; splits disjoint by source document | E6, E7 |
| 7 | "Labels are noisy" | Graded `ρ` from `k=5` samples; noise floor measured by split-half; ORACLE-noisy control; monotone-safe labels resist single-point noise | E1, ORACLE-noisy |
| 8 | "Reimplementation of prior work" | Named in the intro, run as baselines, fidelity documented in `baseline_fidelity.md` | T3 |
| 9 | **"Quality curves aren't monotone, so `b*` is ill-defined"** | Anticipated (audit D1). Monotone-safe definition + monotone head; non-monotonicity is *measured and reported* | E1b, §3.2 |
| 10 | **"Per-instance accuracy is binary, so per-instance ε is meaningless"** | Anticipated (audit D2). Graded `ρ`; instance-level *prediction*, population-level *guarantee* | §3.1.1, T5 |
| 11 | **"You only counted input tokens"** | Anticipated (audit D3). Output-length head; total-cost objective; the cheapest-safe ≠ smallest-safe result | T8, E9 |
| 12 | **"Requested rate ≠ realised rate"** | Anticipated (audit D4). Adherence head + inversion; adherence table for every compressor | T7, E2c |
| 13 | "Only one compressor" | Four backends; policy transfers across them | E7 |
| 14 | "Only one target model" | Three, deliberately including two with *opposite* output-expansion directions | E6a |
| 15 | "Conformal assumes exchangeability" | Conceded, measured under deliberate shift, with the recalibration sample-efficiency curve | E6c, E6d |
| 16 | "MeetingBank is LLMLingua-2's training domain" | Flagged as an in-domain **control**, never a headline; stated in Limitations | T2, Limitations |
| 17 | "No comparison to the theoretical limit" | Distortion-rate LP computed on a subset | E8 |
| 18 | "Improvements are within noise" | Power analysis pre-committed (MDE ≈ 0.024); paired BCa bootstrap; Holm correction | §14 |
| 19 | "Cherry-picked operating point" | Full quality-vs-cost curves, not points; AUC + CPGR + CPT | F2, T3 |
| 20 | "Not reproducible" | Corpus + code + calibration constants + one-command reproduction + cost ledger | Artefacts |

**Objections 9–12 are the ones that would have sunk the original plan.** They are now design features with tables attached. That transformation is the point of this whole re-plan.

---

## 3. Self-inflicted wounds to avoid

| Don't | Do |
|---|---|
| Claim novelty for adaptive compression | Claim the *guarantee*, the *cost model*, and the *curve* |
| Bury AdaComp in Related Work | Name it in the intro, paragraph 3 |
| Report input-token savings as the headline | Report USD |
| Compare against a global fixed budget only | Make B2b primary |
| Hide the code-cliff failures | Give them their own subsection |
| Say "we approach the oracle" | "Oracle (upper bound, post-hoc)" in every table |
| Use "significantly" loosely | Only with a test, an effect size, and a CI |
| Skip the operating window where you lose | Publish it; it is credibility, not weakness |
| Present the CRC guarantee as unconditional | State exchangeability; measure its failure |

---

## 4. The pre-emptive Limitations section (draft it now, verbatim)

> **Limitations.** (i) Our risk guarantee assumes exchangeability between calibration and deployment prompts; §E6c shows it degrades under deliberate task shift, and §E6d quantifies the ~200 in-domain labelled prompts needed to restore validity. (ii) Per-instance quality is estimated from k=5 samples, giving a standard error of ≈0.22; §E1 reports the resulting noise floor and §T3 includes an ORACLE-noisy control isolating irreducible label noise. (iii) MeetingBank is LLMLingua-2's training domain and is included as an in-domain control, not as evidence of generalisation. (iv) §E10 identifies prompt-length and hardware regimes where the full pipeline is not net-positive; we report them rather than restricting evaluation to favourable settings. (v) We do not modify the compressor; gains from jointly optimising compressor and rate policy remain open. (vi) Our cost model uses published API prices as of [date]; §E9 reports sensitivity across `c_out/c_in ∈ [1,5]`.

A Limitations section written *before* the experiments is worth more than one written after — it constrains the claims rather than apologising for them, and reviewers can tell the difference.

---

## 5. Paper structure (8 pages + appendix)

| § | Content | Pages |
|---|---|---|
| 1 | **Intro.** Long prompts cost money → compressors exist → **every one takes the rate as a user hyperparameter** → instances differ → can we predict the frontier and act on it *with a guarantee*, *at true cost*? Contributions C1–C5. **Name Nagle et al. + AdaComp here.** | 1.0 |
| 2 | **Background.** Prompt compression; LLMLingua family; the rate-distortion view (Nagle et al.); conformal risk control | 0.75 |
| 3 | **Related work.** Five clusters + **Table 1** positioning | 0.75 |
| 4 | **Problem formulation.** Graded `ρ`; monotone-safe `b*`; true-cost objective; the risk-control statement | 1.0 |
| 5 | **Method.** Features; three heads (monotone / adherence / output-length); CRC selector; inference algorithm | 1.25 |
| 6 | **Setup.** Corpus (T2), models, compressors, baselines, metrics (CPGR/CPT), stats | 0.75 |
| 7 | **Results.** F1 frontiers → T3 main → F2 Pareto → T5 risk validity → T8 output expansion / E9 | 1.5 |
| 8 | **Ablations + generalisation.** T4, T6, T7, E7 | 0.75 |
| 9 | **Failure analysis + efficiency.** T9, E10 operating window | 0.5 |
| 10 | **Discussion, limitations, conclusion** | 0.5 |
| App. | Full grids, datasheet, claims ledger, baseline fidelity, cost ledger, proofs | — |

**Figure priority if space is short:** F1 (frontiers) > F2 (Pareto) > F3 (risk validity) > F4 (latency/benefit). F1 and F3 are the two that make the paper's argument visually; never cut both.

---

## 6. Venues

| Venue | Fit | Notes |
|---|---|---|
| **ACL / EMNLP main** (via ARR) | **Best fit** | Efficiency + rigorous empirics is core ARR territory. Aim here |
| **EMNLP / ACL Findings** | Strong fallback | Automatic consideration via ARR |
| **NeurIPS D&B** | Good for C5 | If Gates fail, this is the corpus paper's home |
| **ICLR** | Viable | Prefers a stronger methodological/theoretical core; the CRC angle helps |
| **MLSys / EuroSys** | Viable | If the systems/cost story dominates the results |
| ⚠ **Not a theory venue** | Poor fit | The guarantee is a correct *application* of CRC, not a new theorem. Mis-fit is itself a rejection cause |

**Timing:** ARR cycles are monthly; target a submission ~3 weeks after Milestone M7 so the buffer absorbs one round of internal review. Re-run the novelty sweep (`APC_07`, deep-research prompt) in the same week you submit.

---

## 7. Titles

Ranked by how well each signals the *actual* delta rather than the crowded part.

1. **How Much Can This Prompt Be Compressed? Risk-Controlled Instance-Adaptive Prompt Compression** ← recommended: question-first, states the guarantee, avoids the Nagle collision
2. **Compress As Much As You Safely Can: Quality-Guaranteed Adaptive Prompt Compression**
3. **Predicting the Compression Frontier: Risk-Controlled Rate Selection for LLM Prompts**
4. **The Cheapest Safe Prompt: Adaptive Compression Under a Quality Guarantee** ← best if E9 (cheapest ≠ smallest) is the strongest result
5. **Routing Over Compression Rates: Instance-Adaptive Prompt Compression with Distribution-Free Guarantees**

⚠ **Do not use the plan's §51 first choice** — *"Adaptive Prompt Compression via Instance-Level Rate-Distortion Prediction"* — it collides directly with Nagle et al.'s framing and reads as a land-grab on their contribution. This is a one-word-level decision with a real effect on desk outcomes.

---

## 8. The one-paragraph pitch (for supervisors, funding, collaborators)

> Every prompt compressor — LLMLingua, LLMLingua-2, LongLLMLingua — takes the compression rate as a hyperparameter the user picks, then applies it uniformly to inputs that tolerate compression very differently. We ask the prior question: *for this prompt, what rate should we have asked for, and can we know before paying for inference?* We measure per-instance quality-versus-rate frontiers at scale, predict them from features costing milliseconds, and select the **cheapest safe rate** using conformal risk control — giving a distribution-free finite-sample guarantee on quality degradation, which no prior adaptive compression method offers. We also show the field has been optimising the wrong objective: because aggressive compression inflates *output* tokens, which bill at 3–5× input, the cheapest safe rate is often not the smallest safe rate. We release the first corpus of per-instance compression frontiers, so the measurement stands even if the method does not.

---

## 9. Final checklist before submission

- [ ] Every claim in the abstract traces to a ✓ row in the claims ledger
- [ ] Nagle et al. and AdaComp named in the introduction
- [ ] Table 1 positioning table present and accurate
- [ ] Primary comparison is **B2b**, not a global fixed budget
- [ ] All efficiency numbers in USD and seconds
- [ ] Oracle rows labelled "(upper bound)" everywhere
- [ ] Risk-validity plot (F3) present
- [ ] Output-expansion result (T8/E9) present
- [ ] Rate-adherence table (T7) present
- [ ] Limitations section covers all six items from §4
- [ ] Negative results included (E6c shift violation, E10 unfavourable operating regions)
- [ ] Novelty sweep re-run within the last 3 weeks
- [ ] Every ⚠ arXiv ID in the bibliography verified
- [ ] Artefact reproduces T3 from a fresh clone in one command
- [ ] No forbidden phrase from `APC_03 §4` appears anywhere in the text

---

*This is the last document in the set. Start at `APC_00_MASTER_INDEX.md`, execute `APC_05`, and gate hard at Week 3.*
