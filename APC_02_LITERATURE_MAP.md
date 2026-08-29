# APC-02 — Literature Map and Positioning

**Purpose:** the verified related-work base for the paper, organised so that Section 3 (Related Work) can be written directly from it, and so that every "is this novel?" question has a pre-computed answer.

**Verification key:**
- ✓ = abstract/paper read during this audit (2026-08-28); claims below are sourced from it.
- ○ = identified by search, metadata confirmed, full text not read — **read before citing in the paper**.
- ⚠ = arXiv ID recalled, not re-verified this session — **check the ID before it goes in the .bib**.

---

## 1. The five clusters

Prompt compression splits into five clusters. Our work lives in a sixth that sits *on top of* cluster A.

```
                        ┌─────────────────────────────────────┐
                        │  F. POLICY LAYER  ← we are here     │
                        │  which operating point, per input,  │
                        │  under what guarantee, at what      │
                        │  true end-to-end cost               │
                        └──────────────┬──────────────────────┘
                                       │ wraps
   ┌───────────────┬───────────────────┼───────────────────┬──────────────────┐
   ▼               ▼                   ▼                   ▼                  ▼
A. Hard-prompt  B. Soft-prompt   C. Task/query-    D. Adaptive-rate    E. Measurement
   token/       (embeddings,        aware RL          (rate chosen        & limits
   sentence     not usable          compression       per input)          (theory,
   selection    on black-box                                              benchmarks)
                APIs)
```

---

## 2. Cluster-by-cluster

### A — Hard-prompt selective compression (our compression backend)

| Work | Venue | Mechanism | Rate selection |
|---|---|---|---|
| **LLMLingua** ⚠ arXiv:2310.05736 | EMNLP 2023 | Budget controller + iterative token-level compression (ITPC) + distribution alignment; small LM perplexity as the signal; up to ~20× | **User-specified global target**; controller only *allocates* it across instruction / demos / question |
| **LongLLMLingua** ⚠ arXiv:2310.06839 | ACL 2024 | Question-aware coarse-to-fine; contrastive perplexity; document reordering; sub-sequence recovery. Reports **+17.1% over the uncompressed prompt at 4× fewer tokens** | User-specified; "dynamic ratios" = *within-prompt* allocation |
| **LLMLingua-2** ✓ arXiv:2403.12968 | Findings ACL 2024 | Reframes compression as **binary token classification** with a bidirectional BERT-level encoder, trained by **data distillation from GPT-4** on MeetingBank. **3–6× faster than LLMLingua, better OOD** | User-specified |
| **CPC** ○ arXiv:2409.01227 | 2024 | Context-aware *sentence* encoder trained contrastively; select sentences by question similarity | User-specified |
| Perception Compressor ○ 2409.19272 · QGC ○ 2406.02376 · DAC ○ 2507.11942 · Style-Compress ○ 2410.14042 | — | Various training-free / query-guided / style-aware variants | User-specified |

> **The load-bearing observation for the whole project:** across *every* method in cluster A, the compression rate is a **hyperparameter the user supplies**. The "budget controller" allocates a given budget; it does not decide what the budget should be. Prior work optimises *what to delete given a rate*; nobody in cluster A optimises *what rate to ask for, for this input*. That is the gap.

### B — Soft-prompt / embedding compression

500xCompressor ⚠ 2408.03094, ICAE, Gist Tokens, xRAG, ACC-RAG's hierarchical compressor.

**Why we exclude it, and why saying so is an asset:** soft prompts require access to the target model's embedding layer. They are *unusable against a commercial black-box API*, which is where the token bill actually is. Scoping to hard prompts is a deliberate deployment-realism choice, and stating it pre-empts "why not soft prompts?"

### C — Task-aware / RL-optimised compression

| Work | Venue | What it optimises |
|---|---|---|
| **TACO-RL** ○ arXiv:2409.13035 | Findings ACL 2025 | RL-trains the *compressor* with task-specific reward |
| **LLM-DCP** ○ arXiv:2504.11004 | 2025 | Compression as an MDP; DCP-Agent picks tokens step-wise; reward balances rate / output distribution / key-info retention; +9.03% ROUGE-2 over LLMLingua-2 on summarisation |

**Delta:** these make the *compressor* smarter at a given rate. We leave the compressor untouched and make the *rate decision* smarter. **Orthogonal and composable** — and we should say so, then demonstrate it by running our policy on top of two different backends (`APC_06 §E7`). "Orthogonal, and here is the experiment proving composition" is a much stronger position than "different."

### D — Adaptive-rate compression ← **the contested cluster**

This is where acceptance or rejection is decided. Four works, each a genuine neighbour.

#### D1. Nagle et al., *Fundamental Limits of Prompt Compression* ✓ (NeurIPS 2024, arXiv:2407.15504)

- Unifies token-level hard-prompt compression for black-box LLMs into one framework.
- Derives the **distortion-rate function as a linear program**, solved via its dual → the *fundamental limit*.
- Shows a **large gap** between existing methods and the optimum.
- Proposes **Adaptive QuerySelect**: a query-aware, **variable-rate** adaptation of prior work that closes much of the gap.
- Evaluated primarily on a **synthetic Markov-chain prompt dataset**, plus a **small** natural-language set.

**This is the paper we must out-position, and the delta is clean:**

| | Nagle et al. | Ours |
|---|---|---|
| Object | The **limit** `D(R)` — what is achievable in principle | The **per-instance estimate** `ρ̂(x,q,·)` — what is achievable *for this input*, known *before* inference |
| Access | Oracle / post-hoc; the LP needs the answer distribution | Feature-based, **causal at inference time**, no target-LLM call |
| Data | Synthetic Markov chains + small NL set | Real benchmarks, real compressors, multiple production LLMs |
| Output | A benchmark curve to measure methods against | A **deployable policy** with a calibrated risk guarantee |
| Cost | Input-rate only | Realised end-to-end **$ and latency**, incl. output expansion |

One-sentence framing for the intro: *"Nagle et al. characterise how good prompt compression could be; we ask whether, for a given prompt, we can **predict** how good it will be — cheaply enough to act on it before paying for inference."* Characterising a limit and building an estimator of it are standard, well-precedented complements. **Cite them as the foundation we build on, never as a competitor.** That framing also lets us use their distortion-rate function as an *evaluation instrument* (`APC_06 §E8`), which converts the biggest scoop risk into our most rigorous upper-bound baseline.

#### D2. AdaComp ✓ (arXiv:2409.01579) — **the closest prior work**

Labels the **minimum top-k retrieved documents** needed to still answer, builds `(query, docs, rate)` triplets, trains a **compression-rate predictor**, infers once.

Structurally identical to the original plan's Version A. The four axes of separation:

| Axis | AdaComp | Ours |
|---|---|---|
| Granularity | **Documents** (top-k) — a coarse, discrete, RAG-only knob | Token-level rate over any long prompt (transcripts, CoT, few-shot, code) |
| Scope | RAG pipelines with a retriever | Any long prompt; retrieval not assumed |
| Prediction target | **Point estimate** of the rate | **Full monotone quality-vs-rate curve** + realised-rate head + output-length head |
| Guarantee | None | **Distribution-free population risk control** (CRC) |
| Objective | Inference cost ≈ input tokens | Realised end-to-end **$**, incl. compression-induced output expansion |

**Handling instruction:** name AdaComp in the *introduction*, not just related work; run it as baseline **B5a** (re-implemented at token granularity so the comparison is fair). Confronting your nearest neighbour head-on is the single most effective anti-rejection move available.

#### D3. ACC-RAG ✓ (Findings EMNLP 2025, arXiv:2507.22931)

Hierarchical compressor + RL context selector choosing per-query embedding granularity; >4× faster than standard RAG on 5 QA datasets. Explicitly motivated by "existing methods apply fixed compression rates — over-compressing simple queries or under-compressing complex ones."

**Delta:** ACC-RAG is **soft/embedding-granularity** (cluster B constraints — needs model internals) and **RAG-specific**, trained end-to-end with the generator. We are **hard-prompt, black-box, compressor-agnostic**, with a guarantee. Their motivating sentence is *also our motivating sentence* — cite it as independent confirmation that the problem is real, then differentiate on the deployment regime. Shared motivation with a different solution regime is a healthy position, not a threat.

#### D4. AttnComp ○ (arXiv:2509.17486) and adaptive-without-a-predictor methods

Top-p over attention mass: keep the minimum document set whose cumulative attention exceeds a threshold. Rate varies per query with **no learned predictor at all**.

**Why this matters more than its citation count suggests:** it is the strongest *cheap* rival. If a threshold on an attention/entropy statistic matches our learned policy, ML was unnecessary (audit E6 / plan Risk 4). **Must be baseline B5c.** Beating it is a real result; failing to beat it, discovered late, is a dead project. Related: the TAAC series' *entropy-guided adaptive compression*, which allocates budget by local perplexity.

### E — Measurement, benchmarks, and limits

| Work | Verified finding we depend on |
|---|---|
| **Prompt Compression for LLMs: A Survey** ○ NAACL 2025 (2025.naacl-long.368) | Required taxonomy anchor for §3 |
| **An Empirical Study on Prompt Compression** ✓ arXiv:2505.00019 (ICLR 2025 Building Trust WS) | 6 methods × 13 datasets. Compression matters more in **long** contexts; **moderate compression sometimes *improves*** LongBench performance; **response-length effects are model-dependent** (GPT-family lengthens, Claude-3-Haiku shortens) → grounds audit **D1** and **D3** |
| **Prompt Compression in the Wild** ✓ arXiv:2604.02985 (ECIR 2026) | 30k queries, multiple open LLMs, 3 GPU classes. LLMLingua gives **up to 18% end-to-end speedup only when prompt length × rate × hardware are well matched; outside that window the compression step dominates and cancels the gains** → grounds audit **D3/E1** and our net-latency analysis |
| **TAAC series** ✓ (Johnson; arXiv:2603.23525, 2603.23527, + 4 companions) | Task–compression interaction; **code cliff below r≈0.55** vs gradual CoT decay; **compression paradox** (aggressive compression → up to 38× output expansion → total cost *increases*); **Ψ (instruction survival probability)** and **CRI**; benchmark-dependent output dynamics (MBPP Ψ≈0.15 vs HumanEval Ψ≈0.72) |

**On the TAAC series specifically — read this before writing the intro.** It is a single-author, non-peer-reviewed preprint series, and its *own* stated limitations are our opening:

> *"simulated vs. neural compression (**truncation-based, not LLMLingua-2 token classification**)"* · *"single-model evaluation"* · *"embedding similarity as proxy — **does not assess functional correctness**"* · *"deterministic decoding"* · *"corpus representativeness — snapshot of two deployments"*

…and its future-work list names **"learned compression policies."** So: **credit them for observing the phenomenon, then supply the version that survives review** — real neural compressors, task-native metrics, multiple models, public benchmarks, a learned policy, and a guarantee. Being second to an observation but first to the rigorous measurement of it is a normal and respectable contribution; pretending the observation is ours is not.

---

## 3. Cross-domain borrowings (the methodological scaffolding)

These are not prompt-compression papers. They are where the *rigour* comes from, and using them is what separates this from an incremental-combination paper.

### 3.1 Risk control — the source of the headline contribution

| Work | What we take |
|---|---|
| **Conformal Risk Control** ⚠ Angelopoulos, Bates, Fisch, Lei, Schuster — ICLR 2024, arXiv:2208.02814 | Extends conformal prediction from coverage to **expected-loss control for bounded, monotone losses**; a single data-dependent threshold with **finite-sample distribution-free** guarantees |
| **Learn then Test** ⚠ arXiv:2110.01052 | Calibrating an arbitrary predictor to control a risk via multiple-hypothesis testing over a threshold grid; handles non-monotone risks |
| **Prompt Risk Control** ⚠ Zollo et al., ICLR 2024, arXiv:2311.13628 | Precedent for applying distribution-free risk control to *prompting* decisions — closest in spirit, different decision variable |

**Verified gap ✓:** searches for conformal / risk-controlled **prompt-compression rate selection** return nothing. Existing conformal×compression work is about *quantised/sparse model weights* (arXiv:2606.01850) or *communication coding* (arXiv:2503.08340) — different objects entirely. **This is open, and it is our best claim.**

Why it fits so precisely: CRC needs a risk that is **monotone in the threshold**. Our selector `b̂_λ(x) = min{b : ŝ_θ(x,q,b) ≥ λ}` has that by construction — raising `λ` monotonically pushes the chosen budget toward `b=1.0` (no compression), monotonically reducing degradation risk to zero. The hypothesis class is one-dimensional and the guarantee is exact and finite-sample. This is not conformal-prediction-as-decoration; it is the correct tool, and it converts the plan's weakest point ("PCS is arbitrary", Risk 3) into a theorem.

### 3.2 Routing and cascades — the evaluation design

| Work | What we take |
|---|---|
| **RouteLLM** ⚠ Ong et al., ICLR 2025, arXiv:2406.18665 | The **evaluation protocol**: APGR (performance gap recovered) and CPT (call-performance threshold), which we adapt into **CPGR** and **CPT($)** (`APC_06 §7`). Also the framing "a router is a learned decision over a cost–quality frontier" |
| **FrugalGPT** ⚠ Chen, Zaharia, Zou, arXiv:2305.05176 | Cascade precedent; the competing cost lever (baseline **B7**) |
| Cost-aware routing surveys ○ (e.g. arXiv:2603.04445) | Positioning: **compression-rate routing is a routing problem over operating points of one model, rather than over models.** That sentence is the paper's conceptual hook |

**Framing to use in the intro:** *the field has learned to route queries across **models**; nobody has learned to route them across **compression operating points** of a single model, with a guarantee, under a true cost model.* One line, and it places the work on a map every reviewer already has in their head.

---

## 4. The positioning table (paper-ready)

Drop this in as Table 1. It is the fastest way to make a reviewer see the delta.

| Method | Rate is | Granularity | Query-aware | Black-box target | Predicts quality **before** inference | Risk guarantee | Optimises **total** $ (in+out) | Models rate adherence |
|---|---|---|---|---|---|---|---|---|
| LLMLingua / -2 | fixed, user-set | token | ✗ / ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| LongLLMLingua | fixed, user-set | token+doc | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| TACO-RL, LLM-DCP | fixed, user-set | token | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Nagle et al. (Adaptive QuerySelect) | variable | token | ✓ | ✓ | ✗ (oracle/post-hoc limit) | ✗ | ✗ | ✗ |
| AdaComp | **predicted** | **document** | ✓ | ✓ | point estimate only | ✗ | ✗ | ✗ |
| ACC-RAG | **predicted (RL)** | embedding | ✓ | **✗** (needs internals) | ✗ | ✗ | ✗ | ✗ |
| AttnComp | threshold-adaptive | document | ✓ | ✗ (needs attention) | ✗ | ✗ | ✗ | ✗ |
| TAAC series | heuristic-adaptive | truncation | ✗ | ✓ | ✗ | ✗ | **✓** (partial) | ✗ |
| **Ours** | **predicted curve** | **token** | **✓** | **✓** | **✓ full frontier** | **✓ CRC** | **✓** | **✓** |

The last four columns are empty for everyone but us. **That is the paper.** Note that no single column is unprecedented in all of ML — the claim is the *conjunction*, instantiated and validated in this setting, which is the standard bar for an empirical systems paper.

---

## 5. Bibliography hygiene (do this before writing)

1. **Verify every ⚠ arXiv ID** against the ACL Anthology / arXiv listing. One wrong ID in a related-work table costs disproportionate credibility.
2. **Read in full, in this order:** Nagle et al. → AdaComp → LLMLingua-2 → ACC-RAG → NAACL survey → TAAC 2603.23525 + 2603.23527 → ECIR 2604.02985. Roughly two days.
3. **Re-run the novelty sweep ~4 weeks before submission.** This field produced ≥6 relevant papers in the first half of 2026 alone. Standing queries: `adaptive compression rate prediction`, `compression budget selection LLM`, `conformal prompt compression`, `compression routing`, `instance-level compression tolerance`.
4. **Keep a `related_work.csv`** with columns `venue, granularity, rate-selection, guarantee, cost-model, black-box` — the positioning table should be *generated* from it so it never drifts out of date.
5. **Set alerts** on arXiv cs.CL for `prompt compression` and on the LLMLingua GitHub releases.

---

*Next:* `APC_03_NOVELTY.md` — the five defensible contributions and the exact claims we may and may not make.
