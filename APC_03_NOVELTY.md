# APC-03 — Novelty: What We Claim, What We Prove, What We Must Not Say

**Purpose:** convert the audit (`APC_01`) and the literature map (`APC_02`) into a defensible contribution set. Every claim here is paired with the experiment that earns it and the reviewer objection it must survive.

**Governing principle:** *a claim without a pre-specified experiment attached is not a contribution, it is a liability.* The plan's own §41 said this; this document enforces it.

---

## 1. The reframe, in one paragraph

> Every prompt compressor — LLMLingua, LLMLingua-2, LongLLMLingua, CPC, TACO-RL — takes the compression rate as **a hyperparameter the user supplies**. The literature has spent three years optimising *what to delete at a given rate*. Nobody has asked the prior question: **what rate should we have asked for, for this particular input, and how would we know before paying for inference?** Answering it requires three things no prior work provides: a per-instance estimate of the quality-versus-rate curve, a calibrated notion of what "safe" means, and a cost model that reflects what the user is actually billed.

That paragraph is the paper. Everything below makes it survive review.

---

## 2. The five contributions

Ordered by defensibility. **C1 and C2 are the paper.** C3 and C4 are the mechanism. C5 is the insurance policy.

---

### C1 — Risk-controlled prompt compression *(headline; highest novelty × lowest risk)*

**Claim.** We give the first prompt-compression policy with a **distribution-free, finite-sample guarantee on quality degradation**. Given a user-chosen tolerance `ε` and confidence `1−δ`, the deployed policy satisfies `E[L(x)] ≤ ε` (or `P(L > τ) ≤ δ`) on exchangeable future prompts, with no assumptions on the predictor, the compressor, the target LLM, or the data distribution.

**Mechanism.** Selector `b̂_λ(x) = min{ b ∈ B : ŝ_θ(x,q,b) ≥ λ }`, with the single scalar `λ` calibrated on a held-out set by **Conformal Risk Control** (Angelopoulos et al., ICLR 2024) or **Learn-then-Test** when the risk is non-monotone.

**Why it is the right tool, not decoration.** CRC requires a risk monotone in the threshold. Ours is monotone *by construction*: `λ → 1` forces `b̂ → 1.0` (no compression), driving degradation to exactly zero; `λ → 0` selects the most aggressive budget. The hypothesis class is one-dimensional, so the guarantee is exact and needs no correction beyond the standard CRC bound. Calibration cost is one held-out sweep — negligible.

**Verified gap ✓.** Conformal/risk-control work exists for LLM outputs (Prompt Risk Control, conformal factuality), for routing/cascades, for *quantised and sparse model weights* (arXiv:2606.01850), and for communication coding (arXiv:2503.08340). **Searches for risk-controlled prompt-compression *rate selection* return nothing.** Independently, no row in `APC_02 §4`'s table has a guarantee column checked.

**What it buys us politically.** It converts the plan's own Risk 3 — *"PCS is arbitrary"* — from the paper's weakest point into a theorem. That inversion is the highest-leverage single change in this whole re-plan.

**Earned by:** `APC_06 §E5` (calibration validity: empirical risk ≤ ε across ≥20 random calibration/test splits, with the coverage plot) + `§E6` (guarantee holds under cross-task and cross-model shift — and, where exchangeability breaks, *we show where it breaks*, which is itself a finding).

**Reviewer objection → answer.**
- *"Conformal prediction is bolted on."* → No: the selector's monotone-in-λ structure is what makes the guarantee exact, and we ablate against an uncalibrated heuristic threshold, which fails to hit `ε` by a measured margin (`§E5b`).
- *"The guarantee assumes exchangeability, which fails under task shift."* → Conceded and measured. We report the risk violation under deliberate cross-task shift and give the practical fix (per-domain recalibration, ~200 labelled prompts). Honest scoping is stronger than an overclaim.

---

### C2 — The true-cost objective: compression that accounts for output expansion *(highest practical impact)*

**Claim.** Adaptive compression must be optimised against **realised end-to-end cost** — input tokens + output tokens + compressor compute + predictor compute — not input-token reduction. We show that policies optimised for input reduction **select budgets that increase the user's total bill**, and that a policy optimised for true cost does not.

**The evidence base is already published and points our way (all ✓ verified):**
- Pre-registered RCT, 358 production runs (arXiv:2603.23525): `r≈0.5` → **−27.9% total cost**; `r≈0.2` → **+1.8% total cost** despite large input reduction. Output tokens bill at **3–5×** input.
- arXiv:2603.23527: **56× output expansion on MBPP vs 5× on HumanEval**, same model, same rate.
- arXiv:2505.00019: direction is **model-dependent** — GPT-family lengthens under compression, Claude-3-Haiku shortens.
- ECIR 2026 (arXiv:2604.02985): end-to-end speedup **only inside a narrow operating window**; outside it the compressor "dominates and cancels out the gains."

**Why this is ours to take.** Every adaptive-compression paper in `APC_02 §4` reports input-token savings. The one line of work that *has* measured output expansion (TAAC) has no learned policy and uses truncation as a compression stand-in. **The intersection — a learned policy optimised against measured total cost — is empty.**

**The concrete novel object:** a second predictor head `T̂_out(x,q,b)` estimating compression-induced output expansion, entering the selection objective directly:
```
b̂ = argmin_b  [ c_in·T̂_in(x,b) + c_out·T̂_out(x,q,b) ]   s.t.   ŝ_θ(x,q,b) ≥ λ
```
Note this changes the *shape* of the decision: the objective is no longer monotone in `b`, so the naive "smallest safe budget" rule is provably suboptimal — an aggressive budget can be both safe *and* more expensive. **That is a genuinely non-obvious, empirically-grounded result**, and it is a full section of the paper.

**Earned by:** `APC_06 §E3` (main results reported in $ and seconds) + `§E9` (the *"cheapest safe budget ≠ smallest safe budget"* analysis, with the frequency and magnitude of the divergence).

**Reviewer objection → answer.**
- *"Output expansion is already known (TAAC)."* → Credited explicitly. Known ≠ *acted upon*: we are the first to **predict** it per-instance and **optimise against it**, and the first to measure it with real neural compressors and task-native metrics rather than truncation and embedding similarity.
- *"Prices change; the result won't hold."* → We report the full `c_out/c_in` sensitivity sweep, so the conclusion is stated as a function of the price ratio rather than a single vendor's price list.

---

### C3 — Instance-level frontier prediction with a monotone head *(the mechanism)*

**Claim.** We predict, from cheap features and **without a target-LLM call**, the full per-instance quality-versus-rate curve `ρ̂(x,q,·)` — not a point estimate of a budget — using a head that is **monotone by construction**, and show curve prediction beats direct budget classification.

**Delta vs. the two nearest works:**
- vs. **Nagle et al.** (NeurIPS 2024): they compute the population **distortion-rate limit** as an LP, with oracle access, largely on synthetic Markov prompts. We build a **causal, feature-based estimator of the per-instance curve**, usable before inference. *Characterising a limit and estimating it are complements, not competitors* — and we use their LP as an upper-bound instrument (`§E8`), which turns our biggest scoop risk into our most rigorous baseline.
- vs. **AdaComp**: point estimate of a rate at **document granularity in RAG**; ours is a **curve at token granularity for arbitrary long prompts**, plus C1's guarantee and C2's cost model.

**Why monotone-by-construction matters (this is the part that is actually hard).** Audit **D1** established that measured curves are non-monotone — moderate compression sometimes *helps*. A free-form regressor fits that noise and produces unusable labels. We therefore predict a **monotone-safe** curve via a cumulative-logit / isotonic head, which (i) makes `b*` well-defined, (ii) regularises against label noise, and (iii) is exactly the structure CRC needs. The non-monotonicity we smooth away is then reported as a **separate measurement** — instance-level curve non-monotonicity has never been quantified, and it is a small independent finding.

**Earned by:** `APC_06 §E2` (curve prediction: calibration, MAE, monotone vs. free-form) + `§E4` ablation (curve head vs. direct budget classifier vs. regression).

**Reviewer objection → answer.**
- *"Just a classifier with extra steps."* → The ablation is pre-specified and quantitative; if curve prediction does **not** beat classification, we report that, and C1/C2 still stand (they only need a score `ŝ`, not a curve). Contribution independence is deliberate.

---

### C4 — Rate-adherence-corrected control *(small, sharp, entirely unclaimed)*

**Claim.** Requested compression rate ≠ realised rate, the gap is systematic and prompt-dependent, and correcting for it with a learned `R_φ(x,b)` measurably improves budget control. We report requested-vs-realised adherence for every compressor we use.

**Why nobody has this.** ECIR 2026 measures rate adherence as a *property of compressors*; no one **models and inverts** it inside a control loop. Yet without it, every "we saved 47% of tokens" claim in the adaptive-compression literature is a claim about a *request*, not a *result*.

**Cost:** one extra regression head and one extra table. **Return:** every token/cost number in our paper becomes verifiable, and we get a clean methodological criticism of prior work that costs us nothing to make.

**Earned by:** `APC_06 §E2c` + the adherence table.

---

### C5 — The Compression Frontier Corpus *(the insurance policy — build this first)*

**Claim.** We release the first large-scale measurement of **per-instance** compression frontiers: `N` prompts × `K` budgets × `M` target LLMs × `C` compressors, with per-instance graded quality, **realised** rate, input **and output** token counts, latency, and cost. Plus an analysis decomposing the achievable gain into **heterogeneity** (do instances differ?) × **predictability** (can features see the difference?).

**Why this exists as a separate contribution — read this twice.** If C1–C4 fail — if compression tolerance turns out to be unpredictable from cheap features — **the corpus and the decomposition are still publishable**, as a resource/analysis paper (NeurIPS D&B, LREC, or an *Findings* empirical paper). The null result *"instance-level compression tolerance is not predictable from lightweight features, and here is the dataset proving it, and here is the oracle headroom showing what would be available if it were"* is a real contribution that saves the field wasted effort.

The original plan had **no fallback**. Committing to this one now, in writing, before any GPU time is spent, is the difference between a project that always produces a paper and one that might produce nothing. Build the corpus first, in Phase 2, so the insurance is in force before the risky work begins.

**Earned by:** `APC_06 §E1` + the release artefacts in `APC_05 §7`.

---

## 3. The claims ledger

Living document — update the status column weekly. Nothing enters the paper at status ✗ or ?.

| # | Claim | Experiment | Status |
|---|---|---|---|
| 1 | Per-instance compression frontiers are heterogeneous beyond label noise | E1 | ? |
| 2 | Curves are frequently non-monotone; naive `b*` is unstable | E1b | ? |
| 3 | Cheap features predict frontier position better than chance/length | E2 | ? |
| 4 | Curve prediction > direct budget classification | E4 | ? |
| 5 | CRC calibration attains target risk in-distribution | E5 | ? |
| 6 | Adaptive policy beats **per-task-tuned** fixed budget (B2b) on quality-at-cost | E3 | ? |
| 7 | Adaptive policy beats AdaComp-style point predictor (B5a) | E3 | ? |
| 8 | Adaptive policy beats threshold-adaptive AttnComp-style (B5c) | E3 | ? |
| 9 | Compression-induced output expansion changes the optimal budget | E9 | ? |
| 10 | Net $ and latency savings are positive after all overheads | E10 | ? |
| 11 | Rate-adherence correction improves realised-rate control | E2c | ? |
| 12 | Policy transfers across target LLMs | E6a | ? |
| 13 | Policy transfers across task families | E6b | ? |
| 14 | Policy transfers across compression backends | E7 | ? |
| 15 | Guarantee survives (or measurably degrades under) distribution shift | E6c | ? |

---

## 4. Forbidden claims

Write these on the wall. Each has a specific prior owner.

| ❌ Never write | Why | ✅ Write instead |
|---|---|---|
| "We introduce adaptive/dynamic prompt compression" | AdaComp, ACC-RAG, Adaptive QuerySelect, AttnComp | "We study *how to choose* the compression rate under an explicit quality guarantee" |
| "We introduce a rate-distortion framework for prompt compression" | Nagle et al., NeurIPS 2024 | "Building on the rate-distortion framework of Nagle et al., we predict the *per-instance* curve" |
| "We are the first to observe heterogeneous compression tolerance" | LLMLingua §, TAAC series, arXiv:2505.00019 | "We provide the first *instance-level, multi-model, real-compressor* measurement of this heterogeneity" |
| "We are the first to note output expansion under compression" | TAAC series | "We are the first to *predict* it per-instance and *optimise against* it" |
| "Our method reduces inference cost by X%" (input-only) | Contradicted by 2603.23525 | "Reduces *total realised* cost by X% (input+output+overheads), measured in $" |
| "Our compression achieves 20×" | That is LLMLingua's number, not ours | Report *our* measured realised rates with adherence error bars |
| "PCS measures prompt complexity" | Plan's own §18 refutes it | Report the derived *safe compression headroom*, `1 − b*(x)` |

---

## 5. Novelty grading — before and after

| Axis | Original plan | This re-plan | What moved it |
|---|---|---|---|
| Conceptual | **Weak** (self-assessed) | **Moderate** | "Routing over compression operating points, with a guarantee" is a new framing; the *primitives* are known |
| Methodological | **Weak/incremental** (self-assessed) | **Strong** | CRC-calibrated selection + monotone frontier head + rate-adherence inversion + output-length head is a genuinely new composite |
| Technical | Moderate | **Moderate–Strong** | The non-monotone-objective result (C2) is non-obvious; the monotone head is principled rather than ad hoc |
| Empirical | Strong (potential) | **Strong** | Corpus + heterogeneity×predictability decomposition; multi-model, multi-compressor, real metrics |
| Theoretical | Moderate | **Moderate–Strong** | A real finite-sample guarantee, correctly derived, correctly scoped — not a new theorem, but a correct and non-trivial application |

**Honest overall:** this is a **strong applied/empirical paper with a rigorous guarantee**, not a foundational theory paper. Target it accordingly (`APC_08 §6`): ACL/EMNLP main or Findings, or NeurIPS D&B for the corpus. Aiming it at a theory venue would be a mis-fit, and mis-fit is itself a rejection cause.

---

## 6. The abstract, drafted now (working draft — write it before the experiments)

> Prompt compression reduces LLM inference cost, but every existing compressor takes the compression rate as a user-specified hyperparameter, applied uniformly across inputs that tolerate compression very differently. We study the prior question: *what rate should be requested for a given prompt, and can we know before paying for inference?* We first measure per-instance quality-versus-rate frontiers across **N** prompts, **K** rates, **M** target LLMs and **C** compressors, and find substantial heterogeneity **[+ headline number]** that fixed-rate policies cannot exploit. We then predict these frontiers from lightweight features using a monotone curve head, and select the cheapest rate satisfying a quality constraint. Unlike prior adaptive methods, our selector is calibrated by **conformal risk control**, yielding a distribution-free finite-sample guarantee that expected quality degradation stays below a user-specified ε. We further show that optimising input-token reduction is the wrong objective: compression-induced **output expansion** means the cheapest safe rate is often *not* the smallest safe rate, and we predict output length jointly to optimise realised end-to-end cost. Against per-task-tuned fixed budgets and recent adaptive baselines, our policy recovers **[X]%** of the oracle quality–cost gap while reducing total cost by **[Y]%** at matched quality, with overheads included. We release the Compression Frontier Corpus and all code.

Filling `[X]`, `[Y]` and the headline number honestly is the entire job of `APC_06`.

---

*Next:* `APC_04_ARCHITECTURE.md` — the formalism and the system that produces those numbers.
