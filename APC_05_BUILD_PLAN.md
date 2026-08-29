# APC-05 — Build Plan: Phases, Budget, Gates

**Horizon:** 20 weeks to a submittable paper (≈1 FTE, or 2 people at ~60%).
**Shape:** front-load measurement, gate hard at Week 3, keep the fallback paper alive throughout.

---

## 1. The critical path in one picture

```
W1-2   Harness + compressor adapters + cost ledger
         │
W3     ▼ PILOT (200 prompts × 7 budgets × 1 model × k=5)
       ┌──────────── GATE 1: is there heterogeneity above the noise floor? ──────────┐
       │ PASS → continue          BORDERLINE → widen tasks, re-gate          FAIL → pivot to C5 corpus paper │
       └────────────────────────────────────────────────────────────────────────────┘
         │
W4-7   ▼ FULL GRID  → Compression Frontier Corpus  ◄── the insurance policy is now in force
         │
W8-9   ▼ Features + labels + heterogeneity/predictability decomposition
       ┌──────────── GATE 2: do features beat length-only? ──────────────────────────┐
       │ PASS → method paper       FAIL → corpus + negative-result paper (still publishable) │
       └────────────────────────────────────────────────────────────────────────────┘
         │
W10-12 ▼ Predictor (3 heads) + calibration + CRC
       ┌──────────── GATE 3: does the policy beat B2b (per-task-tuned fixed)? ───────┐
W13-15 ▼ Baselines + main results + ablations                                        │
W16-17 ▼ Generalisation (cross-task, cross-model, cross-compressor) + failure analysis
W18-20 ▼ Writing, figures, artefact release, internal review
```

**Everything before Gate 1 is unconditional.** Everything after is contingent, and each gate has a pre-committed alternative so no outcome produces zero output.

---

## 2. Phase detail

### Phase 0 — Setup (Week 1)
- Repo skeleton per `APC_04 §7`; hydra configs; pinned environment.
- Read the seven papers in `APC_02 §5.2` (Nagle, AdaComp, LLMLingua-2, ACC-RAG, NAACL survey, TAAC ×2, ECIR).
- Write the two delta paragraphs (vs Nagle, vs AdaComp) **now**, while the reading is fresh. They go verbatim into the intro.
- Pre-register on OSF: primary hypothesis, `b*` definition, primary metric, sample size, analysis plan. **One hour; disproportionate credibility**, particularly given the competing TAAC series is itself pre-registered.
- **Exit:** `pytest` green; one prompt round-trips `harness → compressor → LLM → ledger row`.

### Phase 1 — Harness + backends (Weeks 1–2)
- Task adapters: GSM8K, LongBench (multi-doc QA / single-doc QA / summarisation / few-shot / code / synthetic — six families in one benchmark, which is why it is the workhorse), MeetingBank, HumanEval+MBPP, ShareGPT-style conversations.
- Compressors: LLMLingua-2 (default), LongLLMLingua, truncate-tail, random-drop.
- Target-LLM adapters: 1 local (vLLM: Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct) + 2 API (one GPT-family, one Claude-family — deliberately chosen because arXiv:2505.00019 shows they expand/contract output in **opposite** directions, which is exactly the contrast C2 needs).
- Cost ledger + price table with a version stamp.
- **Exit:** `realised_r` logged for every compression; ledger reconciles against provider-reported usage to <1%.

### Phase 2 — Pilot and **Gate 1** (Week 3)
`200 prompts (stratified across 4 families) × 7 budgets × 1 local model × k=5` ≈ **7,000 generations**, a few GPU-hours.

Measure four things:
1. **Heterogeneity** of `b*(x)` — the go/no-go quantity.
2. **Rate adherence** — `r` vs `b` (feeds C4; if adherence is near-perfect, C4 shrinks to a footnote and that is fine).
3. **Output expansion** — `T_out(b)/T_out(1)` per family (feeds C2).
4. **Curve non-monotonicity** — frequency of `b*_naive ≠ b*`.

**Gate 1 criterion (pre-specified, `APC_06 §E1`):** the variance of `b*(x)` within task families must significantly exceed the variance attributable to sampling noise at `k=5`. *Within* families is the strict version — cross-family heterogeneity is already known (TAAC) and a policy that only learns "code is fragile, summaries aren't" is a lookup table, not a contribution.

- **PASS** → Phase 3.
- **BORDERLINE** (heterogeneity only across families) → add long-context families where redundancy varies most (multi-doc QA, transcripts), raise `k` to 10 on a subset, re-gate. Do **not** proceed on a borderline result; a weak signal here becomes an unbeatable-baseline problem in Week 14.
- **FAIL** → pivot immediately to the C5 corpus/negative-result paper. Weeks 4–12 become a broader, better-measured corpus. **This is a real paper, not a consolation prize.**

### Phase 3 — Full grid → **Compression Frontier Corpus** (Weeks 4–7)
Tiered to control cost (§5). Ship the corpus with a datasheet and a validation script. **From the end of this phase the project cannot produce zero output.**

- **Exit:** corpus validated (schema, no NaNs, cost reconciled, splits disjoint by source document), datasheet drafted, v0 released internally.

### Phase 4 — Labels, features, decomposition + **Gate 2** (Weeks 8–9)
- Compute `b*` (monotone-safe) and `b*_naive` at `ε ∈ {0.02, 0.05, 0.10}`.
- Extract L0 / L1 / L2 features; record extraction latency for each tier.
- **Heterogeneity × predictability decomposition:** how much of the oracle gain is *available* (heterogeneity) vs how much is *visible to features* (mutual information / a simple probe)? This analysis is a paper section on its own and tells you in Week 9 whether Week 14 will work.
- **Gate 2:** L0+L1 features must beat length-only on `b*` prediction with a paired-bootstrap CI excluding zero.
  - **PASS** → Phase 5. **FAIL** → corpus + negative-result paper, with the decomposition as the scientific core.

### Phase 5 — Predictor + calibration (Weeks 10–12)
Three heads (`APC_04 §5.3`), monotone H1, temperature/isotonic calibration on `D_cal_a`, CRC threshold on `D_cal_b`.
- **Exit:** reliability diagrams; empirical risk ≤ ε over ≥20 random calibration/test splits; predictor latency **< 20% of compressor latency**, measured.

### Phase 6 — Baselines + main results + **Gate 3** (Weeks 13–15)
All baselines through the single `Policy` interface (`APC_06 §5`). **Gate 3:** beat **B2b** (per-task-tuned fixed budget) on quality-at-matched-cost with a paired-bootstrap CI excluding zero.
- **PASS** → full paper.
- **FAIL** → the paper becomes *"when does adaptive compression help, and when does a well-tuned fixed budget suffice?"* — which, given how many adaptive papers only beat *global* fixed budgets, is a genuinely useful and citable result. Say so plainly rather than torturing the numbers.

### Phase 7 — Generalisation, ablations, failure analysis (Weeks 16–17)
Cross-model, cross-task, cross-compressor; ablations E4; failure taxonomy (code cliff, numeric loss, reasoning-chain breaks, short prompts, shift).

### Phase 8 — Writing and release (Weeks 18–20)
Paper, artefact, model card, datasheet, reproduction script. **Re-run the novelty sweep in Week 18** (`APC_02 §5.3`) — this field moved six papers in six months.

---

## 3. Milestones

| # | Week | Deliverable | Definition of done |
|---|---|---|---|
| M0 | 1 | Repo + preregistration | tests green, OSF timestamped |
| M1 | 2 | Harness + 4 compressors + 3 models | ledger reconciles <1% |
| M2 | 3 | **Gate 1** pilot report | heterogeneity vs noise floor, with CI |
| M3 | 7 | **Compression Frontier Corpus v1** | validated + datasheet |
| M4 | 9 | **Gate 2** + decomposition | features > length-only, CI excludes 0 |
| M5 | 12 | Calibrated predictor | risk ≤ ε over 20 splits; latency budget met |
| M6 | 15 | **Gate 3** main results | beats B2b, CI excludes 0 |
| M7 | 17 | Generalisation + ablations | all E4–E7 tables filled |
| M8 | 20 | Submission | paper + artefact + claims ledger all green |

---

## 4. Compute and data budget

### 4.1 The tiered grid (this is what makes the project affordable)

A naive full factorial (`6,000 × 7 × 3 models × k=5` = **630,000 generations**) is infeasible. Tier it:

| Tier | Prompts | Budgets | Model | k | Generations | Purpose |
|---|---|---|---|---|---|---|
| **A** | 6,000 | 7 | local 7–8B | 5 | **210,000** | Main corpus; training + all instance-level analysis |
| **B** | 1,200 (stratified) | 7 | API model 1 | 1 | 8,400 | Cross-model transfer |
| **C** | 1,200 (same prompts) | 7 | API model 2 | 1 | 8,400 | Cross-model transfer + opposite output-expansion direction |
| **D** | 800 | 7 | local | 5 | 28,000 | Cross-**compressor** (LongLLMLingua, CPC, truncate) |
| **Pilot** | 200 | 7 | local | 5 | 7,000 | Gate 1 |
| | | | | **Total** | **≈262,000** | |

Tier A dominates and runs on a local GPU, where the marginal cost is time, not money. API spend is confined to Tiers B and C, ~17k short-output calls.

### 4.2 Estimated cost

| Item | Estimate |
|---|---|
| Local generation (Tier A+D, ~238k gens, batched vLLM on 1×A100/H100) | **~150–220 GPU-hours** |
| Compression passes (LLMLingua-2 is BERT-scale; ~4 backends × grid) | **~25–40 GPU-hours** |
| API generation (Tiers B+C, ~17k calls, long inputs / short outputs) | **~$250–600** |
| Feature extraction + training + calibration | **~15–25 GPU-hours** |
| Re-runs and debugging contingency (**assume one full Tier-A re-run**) | **+40%** |
| **Total** | **≈300–400 GPU-hours + $400–900** |

**The 40% contingency is not padding.** Grids get re-run: a metric bug, a tokenizer mismatch, a price-table change. Caching (`APC_04 §7`) is what keeps a re-run cheap; budget for it anyway.

### 4.3 Datasets

| Dataset | Family | Role |
|---|---|---|
| **LongBench** (6 sub-families) | multi-doc QA, single-doc QA, summ, few-shot, code, synthetic | Workhorse: cross-task generalisation *within one benchmark* |
| **GSM8K** | reasoning | CoT; gradual-degradation case (per TAAC) |
| **MeetingBank** | summarisation | ⚠ **LLMLingua-2's training domain** — an in-domain *control*, never the headline. Label it as such |
| **HumanEval + MBPP** | code | The cliff case; also where output expansion diverges most (Ψ≈0.72 vs 0.15) |
| **NaturalQuestions multi-doc** | retrieval QA | LongLLMLingua's setting; comparability |
| **ShareGPT-style** | conversation | Long history; the plan's original motivation |

---

## 5. Risk register

| Risk | P | Impact | Mitigation | Trigger |
|---|---|---|---|---|
| Insufficient within-family heterogeneity | Med | **Fatal to method** | Gate 1 at Week 3; C5 pivot pre-committed | Gate 1 |
| Cheap features suffice / nothing predicts | Med | High | Tiered features designed for this; the finding is publishable either way | Gate 2 |
| Cannot beat per-task-tuned fixed budget | Med | High | Reframe as "when does adaptation help?"; C1/C2/C5 survive independently | Gate 3 |
| Overhead cancels savings (ECIR operating window) | Med | High | Hard latency budget on the predictor; report the operating window explicitly as a finding | Phase 5 |
| Label noise swamps signal at k=5 | Med | High | Noise floor computed at Gate 1; raise `k` on a subset; use answer-preservation as a lower-variance secondary | Gate 1 |
| **Scooped** (field moving fast) | Med | Med | Novelty sweep at W1 and W18; C1 (CRC) and C2 (true cost) are the least-crowded claims | Continuous |
| API model silently changes version | Med | Med | Pin model IDs, stamp every ledger row, re-run a canary set weekly | Continuous |
| Grid job crashes mid-run | High | Low | Idempotent, resumable, content-hash cached | Continuous |
| Cost overrun | Low | Med | API confined to Tiers B/C; hard spend cap; ledger monitored daily | Continuous |

---

## 6. Team split (if 2 people)

- **Person A — measurement:** L0/L1 harness, corpus, cost ledger, all evaluation and figures. Owns C5 and C2.
- **Person B — method:** features, predictor heads, calibration, CRC, baselines. Owns C1, C3, C4.
- **Shared:** literature, positioning, writing. Both read Nagle and AdaComp in Week 1 — the positioning is too load-bearing to delegate.

Interface between them is `GridRow` and `Policy` (`APC_04 §8`), frozen in Week 2. Freeze it in writing; a mid-project schema change costs a re-run.

---

## 7. Release artefacts

1. **Compression Frontier Corpus** — parquet + datasheet + validation script.
2. **Code** — MIT, one-command reproduction of every table.
3. **Trained predictor + calibration constants**, with a model card stating the calibration domain and the measured shift behaviour.
4. **Claims ledger** (`APC_03 §3`) shipped in the appendix, statuses resolved.
5. **Cost ledger summary** — total $ and GPU-hours spent producing the paper. Rare, cheap, and quietly persuasive to reviewers assessing whether the efficiency claims were measured or asserted.

---

## 8. The pre-committed fallback (decide now, in writing)

> **If Gate 1 or Gate 2 fails**, the project becomes: *"Measuring Instance-Level Compression Frontiers: How Much Adaptive Prompt Compression Can Actually Help."* Contribution = the corpus + the heterogeneity×predictability decomposition + the oracle-headroom bound + the output-expansion/rate-adherence measurements. Venue: NeurIPS D&B, LREC, or an *Findings* empirical paper.

Writing this down **before** the data arrives is what stops the far more common failure mode: p-hacking a weak signal for six weeks because there is no other way to get a paper. It also means Phase 3 is unconditionally worth doing, which is why the corpus is built before the predictor.

---

*Next:* `APC_06_EXPERIMENTS.md` — the protocol that fills the tables.
