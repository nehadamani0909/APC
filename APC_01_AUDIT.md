# APC-01 — Critical Audit of `Adaptive_Prompt_Compression_Research_Plan.md`

**Audit date:** 2026-08-28
**Auditor role:** adversarial reviewer + research engineer
**Verdict:** The plan is *directionally right and unusually self-aware* (it already kills its own weak novelty claim, which most students never do). But as written it would be **rejected or heavily contested** at ACL/EMNLP/NeurIPS for six specific reasons, four of which are technical holes rather than positioning problems.

This document lists every defect found, ranked. Fixes are specified; the fixes are what make the project publishable, and they are carried into `APC_03` (novelty), `APC_04` (architecture), and `APC_06` (experiments).

---

## 0. Scorecard

| Dimension | Grade | Note |
|---|---|---|
| Problem selection | **A−** | Real, live, economically relevant problem. |
| Self-criticism / honesty | **A** | §33, §40, §41, §48 are genuinely good practice. Keep all of them. |
| Literature coverage | **D+** | 3 of 4 named prior works are misnamed or mis-attributed; the single closest prior work (AdaComp) is absent; all 2025–2026 work is absent. |
| Problem formalisation | **C−** | The central optimisation (§12, §20) is **ill-posed** as written. Two independent defects. See D1, D2. |
| Cost model | **D** | Counts input tokens only. Contradicted by published evidence. See D3. |
| Experimental design | **B−** | Good instincts (oracle, regret, ablations); missing power analysis, missing the two baselines reviewers will demand, one leakage risk. |
| Novelty as stated | **C** | "Instance-level rate-distortion prediction" collides head-on with a NeurIPS 2024 paper title. Needs re-framing, not re-wording. |
| Buildability | **B** | No compute budget, no cost estimate, no repo design, no label-noise handling. |

**Bottom line:** the *research question* survives audit. The *formulation, the cost model, and the positioning* do not, and all three are fixable in planning, before a single GPU-hour is spent.

---

## Part I — Factual and citation defects

### C1. `Adaptive QuerySelect` is not a paper. ❗

The plan (§1) cites *"Adaptive QuerySelect / rate-distortion work (NeurIPS 2024)"* as if it were a work. It is a **method proposed inside** a paper:

> Nagle, Girish, Bondaschi, Gastpar, Makkuva, Kim. **"Fundamental Limits of Prompt Compression: A Rate-Distortion Framework for Black-Box Language Models."** NeurIPS 2024. arXiv:2407.15504. ✓*verified*

Why this matters far more than a citation slip: **that paper already owns the phrase "rate-distortion framework for prompt compression."** The plan's recommended title (§51) is *"Adaptive Prompt Compression via Instance-Level Rate-Distortion Prediction"*. A reviewer who has read Nagle et al. sees an apparent land-grab on their framing. This is the single fastest path to a desk-level "insufficient delta" rejection.

**Fix:** Read Nagle et al. properly and state the delta in one sentence in the abstract. The real delta is sharp and defensible — see `APC_03 §2.1`. Short version: *they compute the population-level distortion-rate **limit** (a linear program, with oracle access, largely on synthetic Markov-chain prompts); we **predict the per-instance curve** from features, before inference, and then act on it under a statistical guarantee.* Limit-characterisation vs. deployable-estimator is a legitimate, well-precedented delta — but only if stated explicitly.

### C2. `AdaCoder` is a phantom; the real nearest neighbour is `AdaComp`, and it is *much* closer than the plan realises. ❗❗

The plan (§1) lists *"AdaCoder, which explores adaptive compressed preprompts."* "AdaCoder" is a code-generation prompting paper, unrelated. The work that actually matters is:

> **AdaComp: Extractive Context Compression with Adaptive Predictor for Retrieval-Augmented Large Language Models.** arXiv:2409.01579. ✓*verified*

AdaComp's method, in the plan's own vocabulary:
1. For each query, **label the minimum top-k documents needed for the RAG system to still answer correctly** → this is *exactly* the plan's `b* = min{b : A(x,b) ≥ A(x,1) − ε}` label.
2. Build `(query, retrieved docs, compression rate)` triplets.
3. **Train a compression-rate predictor** on them.
4. At inference, predict the rate, compress once, call the LLM once.

That is the plan's Phases 3→7, published in 2024. **The plan's §36 "Version A — Minimal Publishable" is not publishable.** It is AdaComp with LLMLingua swapped in for top-k document truncation.

**Fix:** This is survivable and even useful — AdaComp is *document-granularity, RAG-only, single-point-estimate, no guarantee, no cost model*. Four independent axes of delta. But the paper must cite it in the intro, not bury it, and must run it as baseline **B5a**. Hiding it is fatal; confronting it is a strength. See `APC_02 §3`.

### C3. `ACC-RAG` is real and correctly identified — but under-read.

> Guo, Zhang, Ren. **"Enhancing RAG Efficiency with Adaptive Context Compression."** Findings of EMNLP 2025. arXiv:2507.22931. ✓*verified*

It "dynamically adjusts compression rates based on input complexity", using a hierarchical compressor + RL-trained context selector, >4× faster inference. Note what it is: **soft/embedding-granularity, RAG-specific, trained end-to-end.** Our setting is **hard prompts for black-box APIs** — a genuinely different constraint regime (you cannot feed embeddings to a commercial API). State that; it is a clean delta.

### C4. The entire 2025–2026 literature is missing.

Absent from the plan, all directly relevant:

| Work | Why it matters to us |
|---|---|
| **LLMLingua-2** (Findings ACL 2024, arXiv:2403.12968) ✓ | BERT-level token classifier, data-distilled from GPT-4. **3–6× faster than LLMLingua and better out-of-domain.** The plan builds on LLMLingua-1 — that is the wrong backbone in 2026; see D6. |
| **TACO-RL** (Findings ACL 2025, arXiv:2409.13035) | Task-aware RL compression. A reviewer will ask why you didn't just RL the compressor. |
| **LLM-DCP** (arXiv:2504.11004) | Compression as an MDP with a reward balancing rate/distribution/key-info. Direct competitor framing. |
| **AttnComp** (arXiv:2509.17486) | Adaptive top-p over attention mass — an *adaptive rate without any predictor*. Strong cheap baseline. |
| **"An Empirical Study on Prompt Compression for LLMs"** (ICLR 2025 Building Trust WS, arXiv:2505.00019) ✓ | 6 methods × 13 datasets. Finds **moderate compression sometimes *improves* accuracy** on LongBench, and that response-length effects are model-dependent. Both findings break the plan's assumptions — see D1 and D3. |
| **Prompt Compression in the Wild** (ECIR 2026, arXiv:2604.02985) ✓ | 30k queries, 3 GPU classes. LLMLingua gives **up to 18% end-to-end speedup only inside a narrow operating window; outside it, the compression step cancels the gains entirely.** This is the plan's Risk 5, already empirically confirmed by someone else. |
| **TAAC preprint series** (Johnson, arXiv:2603.23525 / 2603.23527) ✓ | Non-peer-reviewed but *on the nose*: task–compression interaction, a **code "cliff effect" below r≈0.55 vs. gradual CoT degradation**, and a self-described *"quality-gated algorithm that dynamically adjusts compression based on predicted quality degradation."* |
| **Prompt Compression for LLMs: A Survey** (NAACL 2025) | The required related-work anchor. |

**The TAAC series is the plan's biggest scoop risk and its biggest gift.** Risk: it claims the plan's Experiment 1 finding (heterogeneous compression tolerance) and gestures at the plan's method. Gift: read its own stated limitations — *"simulated vs. neural compression (truncation-based, **not** LLMLingua-2 token classification)"*, *"single-model evaluation"*, *"embedding similarity as proxy — does not assess functional correctness"* — and its own future-work list, which literally reads **"learned compression policies."** It is a preprint series by one author using *truncation as a stand-in for compression*, measured with *embedding similarity instead of task metrics*, on *one model*. Every one of those is a hole we fill. Cite it, credit the phenomenon, and own the rigorous version.

**Fix:** `APC_02` is a full literature map with a positioning table. Related Work must be written *before* the experiments, not after.

---

## Part II — Technical defects in the formulation

These are the serious ones. A reviewer with a formal background finds D1 and D2 in ten minutes.

### D1. The optimal-budget definition assumes monotonicity that does not hold. ❗❗

Plan §20:
```
b* = min{ b : A(x,b) ≥ A(x,1) − ε }
```
This is only well-posed if `A(x,·)` is non-decreasing in `b`. **It is not.** Published counter-evidence:
- LongLLMLingua reports **+17.1% over the *uncompressed* prompt at 4× fewer tokens** (compression as denoising / lost-in-the-middle mitigation).
- arXiv:2505.00019 finds moderate compression **enhances** LLM performance on LongBench.

So the real curve is frequently non-monotone — often inverted-U. Under `min{...}`, a single lucky low budget makes `b*` tiny, and the label is then unreproducible noise. You would be training a predictor on artefacts.

**Fix — the monotone-safe budget.** Require safety at the chosen budget *and everywhere above it*:

```
b*(x) = min{ b ∈ B :  ρ(x,b') ≥ ρ(x,1) − ε   for all b' ∈ B with b' ≥ b }
```

This is the infimum of the maximal safe *suffix*, is robust to single-point noise, and is exactly what "safe operating point" means in engineering. Report both the naive and the monotone-safe label, and report how often they differ — that discrepancy is itself a publishable measurement (it quantifies curve non-monotonicity, which nobody has done at instance level).

### D2. Instance-level quality on exact-match tasks is binary, which makes the per-instance ε constraint degenerate. ❗❗

Plan §19's illustrative table shows `P1 @ 100% → 95% accuracy`. **A single prompt does not have 95% accuracy.** On GSM8K with greedy decoding, `A(x,b) ∈ {0,1}`. Therefore `A(x,b) ≥ A(x,1) − ε` collapses: for any `ε < 1` it means "was correct before ⟹ must still be correct", and for `ε ≥ 1` it is vacuous. The whole rate-distortion story silently switches between *instance-level* and *population-level* quality without acknowledging it. This is the plan's deepest technical hole, and it propagates into §12, §17, §20, §24, and §39.

**Fix — pick a graded per-instance quality `ρ(x,b) ∈ [0,1]` and be explicit.** Three legitimate options; the plan should use (a) as primary and (c) as a robustness check:

- **(a) Sampled pass-rate.** `ρ(x,b) = (1/k)Σ 1[correct]` over `k` samples at `T>0`. Graded, standard (it is `pass@1` estimated à la HumanEval), and the sampling noise is quantifiable: s.e. `≤ 1/(2√k)`. Costs `k×`; use `k=5` on the cheap local model where the full grid runs, `k=1` at `T=0` on the expensive API models.
- **(b) Native graded metric** where one exists (ROUGE/BERTScore for summarisation, F1 for QA, pass@1 for code).
- **(c) Answer-preservation.** `ρ(x,b) = 1[answer(x,b) = answer(x,1)]` — measures *fidelity to the uncompressed behaviour* rather than absolute correctness. Cheaper, no gold labels needed, and arguably the right target for a compression *system* (you promise the user "same answer, fewer tokens"). Sidesteps the case where the model was wrong anyway.

**Then separate the two levels cleanly:** the *predictor* estimates instance-level `ρ̂(x,q,b)`; the *guarantee* is population-level (`E[loss] ≤ ε`). That separation is not a compromise — it is precisely what Conformal Risk Control is built for, and it turns this defect into the paper's headline contribution (`APC_03 §2.2`).

### D3. The cost model is wrong: it ignores output tokens, and output tokens are where compression bites back. ❗❗

Plan §29: `Net Cost = Prediction + Compression + Target LLM`, with savings implicitly measured in *input tokens* (§19's schema records `compressed_tokens`, never output tokens).

Documented reality:
- **The compression paradox** (arXiv:2603.23525, ✓verified pre-registered RCT, 358 production runs, Claude Sonnet 4.5): moderate compression `r≈0.5` cut total cost **−27.9%**, while aggressive `r≈0.2` **increased total cost +1.8%** despite large input reduction — because under-specified prompts trigger compensatory verbosity. Output tokens are billed **3–5×** input.
- arXiv:2603.23527 (✓verified): **56× output expansion on MBPP vs 5× on HumanEval** under aggressive compression, on the same model.
- arXiv:2505.00019 (✓verified): direction is **model-dependent** — GPT-family responses lengthen under compression; Claude-3-Haiku's shorten.

**Consequence:** a plan that optimises input tokens can select budgets that *increase the user's bill*, and its headline "token reduction" number would be an artefact. This is a rejection-grade flaw — and, inverted, it is the project's best practical contribution.

**Fix — optimise realised end-to-end cost.** Define
```
Cost(x,b) = c_in·T_in(x,b) + c_out·T_out(x,b) + c_pred(x) + c_comp(x,b)
```
predict `T_out` as a second head of the model, and report the whole paper in **$ and seconds**, not just token counts. Add an explicit *Cost Ledger* to the harness (`APC_04 §7`). **No prior adaptive-compression paper does this.** See `APC_03 §2.3`.

### D4. Requested budget ≠ realised rate; the plan conflates them everywhere. ❗

`LLMLingua(0.4)` does not reliably produce a prompt with 40% of the tokens — token-level thresholding, chunking, forced-token retention and the budget controller's inter-component reallocation all perturb the realised rate, and the deviation is *systematically prompt-dependent* (short prompts and code overshoot most). ECIR 2026 (arXiv:2604.02985) treats "rate adherence" as a first-class measurement precisely because of this.

If `b` is the *requested* rate, then the trained curve `ρ̂(x,q,b)` silently absorbs the adherence error, and "we saved 60% of tokens" becomes unverifiable.

**Fix — two-stage control.** Model `r̂ = R_φ(x,b)` (realised rate given request) as a separate head; define the frontier over **realised** rate `r`, and invert `R_φ` at selection time to issue the request that lands on the target. Report *requested vs realised* rate adherence as a table. Cheap to build, nobody does it, and it makes every token-saving number in the paper defensible.

### D5. `PCS` is defined as a scalar but is inherently a function of the budget. ❗

Plan §17: `PCS(x) = P(quality degradation > ε | x, compression)`. "Compression" is not a value — degradation risk depends on *how much* you compress. A scalar PCS is only meaningful at a fixed reference budget.

**Fix — kill the scalar, or define it as a functional of the curve.** Two coherent options:
- **Drop PCS as a component.** The curve `ρ̂(x,q,·)` already contains everything PCS would say. The plan's own §15 correctly argues the gate is redundant; the same argument kills the scalar PCS. Recommended.
- **Keep it as a derived, reportable scalar** for interpretability and figures: `PCS(x) := 1 − b*(x)` (the *safe compression headroom*), or the area under the predicted frontier `∫ρ̂(x,q,b)db`. Defensible because it is *derived from* a measured quantity rather than posited.

Either way: **do not ship a separately-trained PCS module.** It is the component reviewers will call arbitrary (the plan's own Risk 3), and it is unnecessary.

### D6. Wrong compressor backbone for 2026.

The plan pins LLMLingua-1 (EMNLP 2023) as *the* backend. LLMLingua-2 is 3–6× faster, better out-of-domain, and is the field's current default; LongLLMLingua is the standard for the long-context/QA settings the plan wants to evaluate on. Building a whole pipeline on the 2023 model invites "evaluated on an obsolete backbone."

**Fix — make the compressor a swappable adapter, and default to LLMLingua-2.** This costs nothing at design time and buys a whole contribution: **the policy layer is compressor-agnostic**, so demonstrating transfer across `{LLMLingua-2, LongLLMLingua, CPC, truncation}` becomes an ablation that strengthens generality (`APC_06 §E7`). Keep LLMLingua-1 as the historical reference point only.

---

## Part III — Experimental-design defects

### E1. Two baselines that reviewers *will* demand are missing.

- **Prompt caching.** In 2026, provider-side prefix caching gives compression-class savings at literally zero quality risk on repeated prefixes. Reviewer question: *"why compress at all?"* **Answer to bake in:** caching helps only on **repeated static prefixes**; compression targets **unique, dynamic, long context** (retrieved docs, transcripts, session history) where cache hit rate is ~0. Scope the paper to that regime *in the introduction*, and include a cache-hit-rate sensitivity analysis. This is a one-paragraph fix that neutralises a fatal question.
- **Model routing / cascades.** FrugalGPT, RouteLLM. Routing to a cheaper model is the *competing* cost lever, and TAAC's headline finding is literally "compression-first for code, routing-first for reasoning." Reviewer question: *"is compression even the right knob?"* **Answer to bake in:** run routing as baseline **B7**, and — better — run **compression × routing jointly** as an extension. If our policy composes with routing, that is a strong result; if routing dominates, honest reporting of that is *still* a contribution and we say so.

### E2. The "best fixed budget" bar is set too low.

Plan §22 B2 says "choose the best fixed budget using validation data." Many adaptive-compression papers beat only a *globally* fixed budget, which is a straw man. The honest bar, and the one a good reviewer imposes:

- **B2a** best single global fixed budget (validation-tuned)
- **B2b** best fixed budget **tuned per task family** on validation — *this is the real baseline to beat*
- **B2c** best fixed budget **tuned per (task family × target model)** — the strongest non-learned competitor

If the method cannot beat **B2b**, the paper is not there yet. Design for that from day one; discovering it in month four is the standard way these projects die.

### E3. Oracle-budget leakage.

Plan §25's oracle is computed on the same data it is compared on. That is fine **as an upper bound / headroom measurement** — and must be labelled that way in every table (`Oracle (upper bound, not achievable)`). It is *not* a baseline, and any phrasing like "we approach the oracle" needs the caveat that the oracle is post-hoc.

Additionally: report **Oracle-with-noise** — the oracle's own performance when labels are estimated from `k` samples — so the reader can see how much of the oracle–method gap is irreducible label noise rather than predictor weakness. This is a genuinely informative control that almost nobody runs.

### E4. No statistical power analysis, no `N`, no cost estimate.

§30 says "bootstrap confidence intervals" but the plan never states how many prompts, how many samples per prompt, or what effect size it is powered to detect. A 2-point accuracy difference on 200 test prompts is not detectable. **Fix:** `APC_06 §9` specifies `N`, `k`, a minimum detectable effect, paired bootstrap over prompts, and Holm correction across the baseline family. Also: **pre-register** the primary hypothesis and analysis (the competing TAAC series is pre-registered; not doing so is a needless credibility gap, and OSF registration is free and takes an hour).

### E5. No single headline number.

The plan's metrics (§23) are a list, not an argument. Reviewers and abstracts need one scalar.

**Fix — borrow and adapt RouteLLM's evaluation design**, which solved exactly this problem for routing:
- **CPGR — Compression Performance Gap Recovered**: fraction of the (best-fixed-budget → oracle-budget) quality gap that the policy recovers at matched cost.
- **CPT(x%) — Cost to Preserve Threshold**: the cost (in $) required to retain `x%` of uncompressed quality.
- **Quality-vs-cost AUC** over the achievable frontier, with a paired-bootstrap CI.

One table, three numbers, directly comparable across methods. This is a large, cheap credibility upgrade.

### E6. Feature-set risk: no plan for what happens when cheap features are enough — or aren't.

§16 lists 14 candidate features. Two failure modes are unplanned-for:
- **Cheap features suffice** ⟹ "you didn't need ML" (plan's Risk 4). *This is a fine outcome* if framed as a finding — but only if the tiered comparison was designed in advance. See `APC_04 §5` (Tier L0/L1/L2) and `APC_06 §E4`.
- **Nothing predicts anything** ⟹ null result. The plan has no fallback. See `APC_05 §8` for the pre-committed pivot: the *Compression Frontier Corpus* + heterogeneity/predictability decomposition is publishable as a resource/analysis paper **even if the policy fails**. Decide this now, while it costs nothing.

### E7. Label cost is unbudgeted and is the project's main resource risk.

`N` prompts × `K` budgets × `M` models × `k` samples target-LLM calls, plus a compressor forward pass each. The plan never multiplies this out. At `N=6000, K=7, M=3, k=5` that is **630,000 generations** — infeasible for a student project. `APC_05 §5` gives a tiered design that lands at ~**$400–900 + ~200 GPU-hours**, with the expensive API models used only on stratified subsets.

---

## Part IV — Things the plan gets right (do not "fix" these)

Preserve verbatim; they are the plan's real strengths:

1. **§15 — collapsing the gate into the budget set** (`b=1.0` *is* "don't compress"). Correct, and it removes a component reviewers would attack.
2. **§18 — complexity ≠ compressibility.** Sharp, correct, and the right intuition for the whole project.
3. **§24 — compression regret.** Keep it; add *token regret* and *cost regret* as the plan already hints.
4. **§41 — the claims/evidence matrix with a "Current Status" column.** Rare discipline. Keep it and update it as a living file.
5. **§42–§43 — measure the curves before building anything.** This is exactly right and is the single most valuable instruction in the document.
6. **§48 / §49 "Limitations" — refusing to claim novelty before checking.** Keep this posture in the paper's Limitations section; reviewers reward it.

---

## Part V — The six things that change, in one table

| # | Plan as written | Replace with | Doc |
|---|---|---|---|
| 1 | `b* = min{b : A ≥ A₀−ε}` | **Monotone-safe** budget over the safe suffix | `APC_04 §3.2` |
| 2 | Per-instance ε on binary accuracy | Graded `ρ(x,b)` (sampled pass-rate / answer-preservation) + **population-level** risk guarantee | `APC_04 §3.1`, `§6` |
| 3 | Input-token cost | **Realised end-to-end $ and latency**, incl. predicted output expansion | `APC_04 §3.4`, `§7` |
| 4 | Requested budget = realised rate | **Two-stage** rate-adherence model `R_φ` | `APC_04 §5.3` |
| 5 | Separate PCS module + gate | **One monotone frontier head**; PCS derived, not trained | `APC_04 §5` |
| 6 | Heuristic threshold on predicted quality | **Conformal Risk Control** calibration ⟹ finite-sample guarantee | `APC_03 §2.2`, `APC_04 §6` |

Changes 3, 4 and 6 are not just repairs — each is an independent contribution no prior work in this area has. That is how the audit turns into the novelty case.

---

## Part VI — Immediate action list (before any model is trained)

| # | Action | Effort | Blocking? |
|---|---|---|---|
| 1 | Read Nagle et al. (2407.15504) and AdaComp (2409.01579) end-to-end; write the two delta paragraphs | 1 day | **Yes** |
| 2 | Read TAAC series (2603.23525, 2603.23527) + ECIR 2026 (2604.02985); extract their stated limitations verbatim into the positioning doc | half day | **Yes** |
| 3 | Fix the label definition (D1+D2) and freeze it in writing | half day | **Yes** |
| 4 | Add output tokens + $ + latency to the logging schema | half day | **Yes** — retro-fitting means re-running the grid |
| 5 | Pilot: 200 prompts × 7 budgets × 1 local model, `k=5` — measure heterogeneity **and** rate adherence **and** output expansion | 3 days | **Yes** — this is the go/no-go gate |
| 6 | Pre-register primary hypothesis + analysis plan (OSF) | 2 hours | No, but do it |
| 7 | Decide and write down the null-result pivot | 1 hour | No, but do it |

**Do not skip #5.** Everything downstream is conditional on it. `APC_06 §E1` specifies the exact go/no-go criterion, with numbers.

---

*Cross-references:* `APC_02` literature map · `APC_03` novelty · `APC_04` architecture · `APC_05` build plan · `APC_06` experiments · `APC_07` build prompts · `APC_08` rebuttal kit.
