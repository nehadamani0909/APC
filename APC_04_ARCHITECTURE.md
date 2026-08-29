# APC-04 — System Architecture and Formalism

**Name:** **FRONTIER** — *Frontier-predicting Risk-cONTrolled Instance-adaptive comprEssion Rate selection.*
(Use it as an internal codename; pick the paper's name after the results are in.)

---

## 1. Design principles

1. **The compressor is a black box behind an adapter.** We never modify LLMLingua. This keeps the contribution orthogonal (`APC_02 §C`) and makes cross-backend transfer a free experiment.
2. **The target LLM is a black box.** No logits, no attention, no embeddings. This is the deployment regime that matters, and it excludes soft-prompt methods from competing with us on our own turf.
3. **Everything is measured in dollars and seconds.** Token counts are intermediate quantities, never headline results (audit **D3**).
4. **Every predicted quantity is calibrated, and calibration is tested.** Uncalibrated scores are not decisions.
5. **The corpus is the product; the policy is a consumer of it.** Build Layer 0–2 to publication quality before Layer 3 exists (`APC_03 C5`).
6. **Offline-heavy, online-trivial.** All target-LLM cost is paid once, at dataset-construction time. Inference-time overhead must be *provably* smaller than the compressor's own cost, or the whole premise collapses (ECIR 2026's operating-window finding).

---

## 2. Layered view

```
┌──────────────────────────────────────────────────────────────────────────┐
│ L5  EVALUATION & ANALYSIS                                                │
│     frontier metrics · CPGR / CPT($) · regret · calibration · ablations   │
└──────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌──────────────────────────────────────────────────────────────────────────┐
│ L4  RISK-CONTROLLED SELECTOR                              [contribution C1│
│     b̂_λ(x) = argmin_b Ĉost(x,q,b)  s.t.  ŝ_θ(x,q,b) ≥ λ    + C2]         │
│     λ calibrated offline by Conformal Risk Control on D_cal              │
└──────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌──────────────────────────────────────────────────────────────────────────┐
│ L3  FRONTIER PREDICTOR  F_θ : (x,q) → three heads       [contribution C3, │
│     H1  ŝ(b)     quality-retention curve   (monotone in b)          C4]  │
│     H2  r̂(b)     realised compression rate (rate adherence)              │
│     H3  T̂out(b)  output-token count        (cost model)                  │
└──────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌──────────────────────────────────────────────────────────────────────────┐
│ L2  FEATURE EXTRACTOR   tiered: L0 surface · L1 small-LM · L2 encoder     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌──────────────────────────────────────────────────────────────────────────┐
│ L1  COMPRESSION BACKEND ADAPTER                                          │
│     LLMLingua-2 (default) · LongLLMLingua · CPC · truncation · random     │
└──────────────────────────────────────────────────────────────────────────┘
                                    ▲
┌──────────────────────────────────────────────────────────────────────────┐
│ L0  HARNESS: task registry · target-LLM adapters · metrics · COST LEDGER  │
└──────────────────────────────────────────────────────────────────────────┘
```

**Offline path (expensive, once):** `L0 → L1 → grid over budgets → Compression Frontier Corpus → labels → train L3 → calibrate L4`.
**Online path (cheap, every request):** `L2 features → L3 forward pass → L4 select b̂ → L1 compress once → target LLM once`.

---

## 3. Formalism

### 3.1 Objects

| Symbol | Meaning |
|---|---|
| `x` | context to be compressed (docs, transcript, demonstrations, code) |
| `q` | query / instruction (never compressed; see §4.3) |
| `b ∈ B` | **requested** compression rate; `B = {1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2}`, with `b=1.0` ≡ no compression |
| `C_ψ(x,b)` | compressor backend ψ → compressed context |
| `r(x,b)` | **realised** rate = `|tok(C_ψ(x,b))| / |tok(x)|`. **`r ≠ b` in general** (audit D4) |
| `y(x,b)` | target-LLM output |
| `ρ(x,q,b) ∈ [0,1]` | **graded per-instance quality** (§3.1.1) |
| `T_in, T_out` | realised input / output token counts |
| `L(x) = max(0, ρ(x,q,1) − ρ(x,q,b̂))` | **degradation loss**, clipped at 0 so accidental *improvement* is not rewarded as negative risk (required for CRC's bounded-loss condition) |

#### 3.1.1 The quality function — resolving audit D2

Per-instance quality must be **graded**, not binary. Primary definition:

```
ρ(x,q,b) = (1/k) Σ_{j=1..k} m( y_j(x,b), gold )      y_j ~ LLM(·| q, C_ψ(x,b)),  T = 0.7
```

- `m` = task-native metric: EM/F1 (QA), pass@1 unit tests (code), ROUGE-L + BERTScore (summarisation), EM on final answer (GSM8K).
- `k = 5` on the cheap local model where the full grid runs; `k = 1` at `T=0` on expensive API models (there, `ρ` is binary and we say so, and only aggregate-level claims are made from it).
- Standard error of `ρ̂` is `≤ 1/(2√k)` = **0.224 at k=5**. This is *not negligible* and it directly bounds how much heterogeneity we can claim to detect — which is exactly why the E1 go/no-go test (`APC_06 §E1`) must compare observed variance against this noise floor rather than against zero.

**Secondary definition — answer preservation** (robustness check, and arguably the better product spec):
```
ρ_ap(x,q,b) = agreement( y(x,b), y(x,1) )
```
Needs no gold labels, measures fidelity to the *uncompressed system's* behaviour, and sidesteps cases where the model was wrong anyway. Report both; if conclusions differ, that difference is a finding worth a subsection.

### 3.2 The monotone-safe optimal budget — resolving audit D1

Naive (**rejected**, but reported for comparison):
```
b*_naive(x) = min{ b ∈ B : ρ(x,q,b) ≥ ρ(x,q,1) − ε }
```
Fails because measured curves are non-monotone (LongLLMLingua's +17.1%; arXiv:2505.00019), so one lucky low budget produces a spuriously tiny label.

**Adopted — safety must hold at the chosen budget and everywhere above it:**
```
b*(x) = min{ b ∈ B : ρ(x,q,b') ≥ ρ(x,q,1) − ε   ∀ b' ∈ B, b' ≥ b }
```
This is the infimum of the maximal safe *suffix*: robust to single-point noise, well-defined without assuming monotonicity, and matching the engineering meaning of "safe operating point".

> **Report `|b*_naive − b*|` as a measurement.** Its distribution quantifies instance-level curve non-monotonicity, which no prior work has measured. Small independent finding, free.

### 3.3 The selection problem — resolving audit D3

The plan's objective (`min_b T(x,b)`) is wrong because it ignores output tokens. Correct objective:

```
b̂(x,q) = argmin_{b ∈ B}  Ĉost(x,q,b)
         s.t.  ŝ_θ(x,q,b) ≥ λ

Ĉost(x,q,b) = c_in · T_in(x)·r̂_φ(x,b)          ← input, via the ADHERENCE model
             + c_out · T̂out(x,q,b)              ← output, via the EXPANSION head
             + c_comp(x,b) + c_pred(x)          ← measured overheads
```

**The consequence is the interesting part.** `Ĉost` is **not monotone in `b`**: aggressive compression can be simultaneously *safe* and *more expensive*, because it inflates `T̂out`. So `argmin cost s.t. safe` ≠ `min safe budget`. The plan's entire §20 rule is therefore provably suboptimal under a true cost model. That divergence is contribution **C2** and gets its own experiment (`§E9`).

### 3.4 The risk guarantee — contribution C1

Selector family indexed by one scalar `λ ∈ [0,1]`:
```
b̂_λ(x,q) = argmin_b Ĉost(x,q,b)   s.t.   ŝ_θ(x,q,b) ≥ λ
           (fallback b = 1.0 if the feasible set is empty)
```
Risk: `R(λ) = E[ L(x) ] = E[ max(0, ρ(x,q,1) − ρ(x,q,b̂_λ(x))) ]`.

**Monotonicity holds by construction.** As `λ ↑ 1`, the feasible set shrinks toward `{1.0}`, so `b̂ → 1.0` and `R(λ) → 0`. `L` is bounded in `[0,1]`. Both CRC preconditions are met.

**Calibration (CRC, Angelopoulos et al. ICLR 2024).** On `n` held-out calibration prompts, choose
```
λ̂ = inf{ λ :  (n/(n+1))·R̂_n(λ) + 1/(n+1)  ≤  ε }
```
giving `E[L(X_{n+1})] ≤ ε` for exchangeable test points, distribution-free and finite-sample.

**Tail variant** — when the user wants "rarely bad" rather than "good on average", control `P(L > τ) ≤ δ` via **Learn-then-Test** (fixed-sequence testing over a `λ` grid). Report both; they answer different product questions and cost nothing extra.

**Scope honestly.** Exchangeability fails across task families. Therefore: (i) calibrate per deployment domain, (ii) **measure and report the risk violation under deliberate cross-task shift** (`§E6c`) — a negative result here is informative, not fatal, and pre-empts the reviewer who would otherwise find it, (iii) note that ~200 labelled in-domain prompts suffice to recalibrate.

---

## 4. L0 — Harness

### 4.1 Task registry
Each task implements:
```python
class Task(Protocol):
    name: str; family: Literal["qa","multidoc_qa","summ","reason","code","conv"]
    def load(self, split, n) -> list[Instance]     # Instance: id, context, query, gold, meta
    def build_prompt(self, ctx, query) -> str
    def metric(self, pred, gold) -> float          # graded, in [0,1]
    def parse(self, raw) -> str
```
Contexts and queries are stored **separately** — `x` is compressible, `q` is not.

### 4.2 Target-LLM adapters
Uniform interface over local (vLLM) and API models. Every call returns `(text, T_in, T_out, latency_s, usd)`. Adapters must expose **provider-reported** token counts, never local re-tokenisation estimates — a systematic mismatch here would corrupt every cost number in the paper.

### 4.3 What never gets compressed
The query/instruction `q` is never compressed. Rationale: TAAC's Ψ (instruction survival probability) finding shows instruction damage is the dominant driver of output explosion (MBPP Ψ≈0.15 → 56× expansion). Fixing `q` removes a confound and matches how compressors are deployed in practice. **Document this as a design decision in the paper**, or a reviewer will read it as a hidden advantage.

### 4.4 Cost ledger (mandatory)
Append-only JSONL, one row per LLM/compressor call:
```
run_id, ts, phase, prompt_id, task, backend, requested_b, realised_r,
model, T_in, T_out, latency_ms, usd_in, usd_out, usd_total,
gpu_seconds, sample_idx, temperature, seed, code_version, price_table_version
```
Not optional bookkeeping — it *is* the evidence for C2 and for the net-efficiency analysis, and it cannot be reconstructed after the fact.

---

## 5. L1–L3 — Compressor, features, predictor

### 5.1 Compression backend adapter
```python
class Compressor(Protocol):
    name: str
    def compress(self, ctx: str, query: str|None, rate: float) -> CompressedResult
    # -> compressed_text, realised_rate, wall_ms, gpu_ms, model_version
```
Implementations: **`llmlingua2`** (default), `longllmlingua` (query-aware, long-context), `cpc` (sentence-level), `truncate_tail` (the TAAC-equivalent control), `random_drop` (the null control that isolates "does *which* tokens you drop matter?").

### 5.2 Feature tiers (designed so audit E6 is answerable)

| Tier | Features | Cost | Purpose |
|---|---|---|---|
| **L0 — surface** | token count, sentence count, type-token ratio, n-gram repetition rate, compression ratio of `gzip(x)` *(a genuinely strong free proxy for redundancy — do not skip it)*, digit ratio, code-token ratio, punctuation/structure ratio, avg sentence length, query length, query-context lexical overlap | <1 ms | The "you didn't need ML" bar. If L0 alone wins, that is the paper's honest finding |
| **L1 — small-LM statistics** | per-token NLL from a 0.5B LM (mean, var, p10/p50/p90, entropy), contrastive perplexity `PPL(x|q) − PPL(x)` (LongLLMLingua's signal), attention-free saliency spread | ~10–40 ms | Where the signal probably is |
| **L2 — encoder** | frozen encoder (e.g. a small BERT/E5) `[CLS]` over `x` and `q`, plus their cosine similarity; optionally 2 fine-tuned layers | ~20–60 ms | Highest ceiling; must justify its latency |

**Hard constraint:** total predictor latency **must be < 20% of the compressor's own latency**, or the ECIR-2026 operating-window objection lands. Measure and report it; if L2 breaks the budget, L2 is out — a fast policy that wins by less is worth more than a slow one that wins by more.

### 5.3 Predictor `F_θ` — three heads

Shared trunk (small MLP over L0+L1, optional encoder concat). Budget `b` is **not** an input feature; each head emits a value for **every** `b ∈ B` at once, which is what makes the curve coherent and monotonicity enforceable.

**H1 — quality-retention curve (monotone by construction).** Cumulative-logit / ordinal head:
```
ŝ(x,q,b_j) = σ( u(x,q) + Σ_{i≤j} softplus(v_i(x,q)) )
```
Non-negative increments ⇒ `ŝ` is non-decreasing in `b` by construction. This is the structural fix for audit **D1**: it regularises against curve noise, makes `b*` well-defined, and supplies exactly the monotone score CRC requires. Loss: binary cross-entropy against `1[ρ(x,q,b) ≥ ρ(x,q,1) − ε]`, plus a ranking term over budgets.
*Ablation:* free-form (non-monotone) head, and a direct `K`-way budget classifier (audit-required, `§E4`).

**H2 — rate adherence `r̂_φ(x,b)`** (contribution C4). Regression on `log(r/b)`; features are L0-only (it is a property of the text and the compressor, not the task). Inverted at selection time to request the `b` that lands on the target realised rate.

**H3 — output length `T̂out(x,q,b)`** (contribution C2). Regression on `log T_out`; needs task-family identity and query features, since expansion is strongly task- and model-dependent (MBPP 56× vs HumanEval 5×).

**Calibration.** H1 is a probability and must be calibrated before CRC: temperature scaling or isotonic regression on `D_cal_a`, with reliability diagrams and ECE reported per budget. **Never calibrate and select the CRC threshold on the same split** — that would silently void the guarantee, and it is the single most likely way to accidentally break C1.

### 5.4 Data splits (leakage discipline)

```
D_train  (~60%)  train F_θ
D_cal_a  (~10%)  probability calibration of H1
D_cal_b  (~10%)  CRC threshold λ̂          ← must be disjoint from D_cal_a
D_test   (~20%)  everything reported
D_shift  (held-out task families & target models, never trained or calibrated on)
```
Split **by prompt id and by source document** — several benchmarks reuse a document across instances, and splitting naively leaks.

---

## 6. Inference-time algorithm

```
Input: context x, query q, tolerance ε (via pre-calibrated λ̂), price table
 1. f  ← Features(x, q)                       # L0 (+L1 if within budget)
 2. ŝ, r̂, T̂out ← F_θ(f)                       # one forward pass, all budgets
 3. S ← { b ∈ B : ŝ(b) ≥ λ̂ }                  # feasible (safe) set
 4. if S = ∅:  b̂ ← 1.0                        # abstain: send uncompressed
 5. else:      b̂ ← argmin_{b∈S} Ĉost(x,q,b)   # cheapest safe, NOT smallest safe
 6. b_req ← r̂⁻¹(b̂)                            # adherence correction
 7. x̃ ← C_ψ(x, b_req)          [skipped entirely if b̂ = 1.0]
 8. y ← LLM(q, x̃)
 9. log everything to the cost ledger
```
**One compressor call, one LLM call.** Never query the target LLM to discover its own budget — the plan's §21 warning is correct and is an invariant of the design, enforced by a unit test.

---

## 7. Repository layout

```
frontier/
├── frontier/
│   ├── harness/      tasks.py  models.py  metrics.py  ledger.py  prices.py
│   ├── compress/     base.py  llmlingua2.py  longllmlingua.py  cpc.py  baselines.py
│   ├── features/     surface.py  smallm.py  encoder.py  registry.py
│   ├── corpus/       build_grid.py  labels.py  schema.py  validate.py
│   ├── predict/      heads.py  train.py  calibrate.py  monotone.py
│   ├── select/       policy.py  crc.py  ltt.py  baselines.py
│   └── eval/         metrics.py  frontier.py  bootstrap.py  figures.py
├── configs/          hydra yaml: tasks, models, budgets, splits, prices
├── scripts/          00_pilot … 09_paper_figures
├── data/             raw/  corpus/  splits/  cache/
├── tests/            incl. test_no_target_llm_at_inference.py, test_split_disjoint.py
└── paper/            main.tex  tables/  figures/  claims_ledger.md
```

**Three non-negotiable engineering rules:**
1. **Cache every compression and every generation** keyed by `sha256(prompt + model + params + seed)`. The grid *will* be re-run after bugs; without caching that is fatal to the budget.
2. **Resumable, idempotent grid jobs.** Crash on prompt 4,000 of 6,000 must not cost the first 4,000.
3. **Pin everything** — model revisions, LLMLingua version, tokenizer version, price table. Record in every ledger row. API model IDs silently change under you.

---

## 8. Interface contracts (freeze these before coding)

```python
# corpus row — the atomic unit of the whole project
@dataclass(frozen=True)
class GridRow:
    prompt_id: str; task: str; family: str
    backend: str; requested_b: float; realised_r: float
    target_model: str; sample_idx: int; temperature: float; seed: int
    quality: float                  # graded, in [0,1]
    T_in: int; T_out: int
    latency_ms: float; compress_ms: float
    usd_in: float; usd_out: float
    raw_output_hash: str            # full text stored separately
    code_version: str; price_table_version: str
```

```python
class Policy(Protocol):
    """Every baseline and our method implement exactly this."""
    def select(self, x: str, q: str, task_family: str) -> Selection
    # Selection: requested_b, predicted_quality|None, predicted_cost|None, overhead_ms
```
Baselines-as-policies is what makes the comparison airtight: fixed budgets, length heuristic, random, AdaComp-style point predictor, AttnComp-style threshold, oracle, and ours all go through **one** evaluation path. Any baseline that needs a special evaluation path is a baseline you will be accused of handicapping.

---

## 9. Where each audit defect is fixed

| Audit | Fix location |
|---|---|
| D1 non-monotonicity | §3.2 monotone-safe label + §5.3 H1 monotone head |
| D2 binary quality | §3.1.1 graded `ρ` + §3.4 population-level guarantee |
| D3 cost model | §3.3 objective + §4.4 ledger + §5.3 H3 |
| D4 rate adherence | §5.3 H2 + §6 step 6 |
| D5 scalar PCS | Removed; derived as `1 − b*(x)` for reporting only |
| D6 backbone | §5.1 adapter, LLMLingua-2 default |
| E3 oracle leakage | §5.4 splits; oracle labelled "upper bound" everywhere |
| E7 label cost | §7 caching + `APC_05 §5` tiered grid |

---

*Next:* `APC_05_BUILD_PLAN.md` — phases, budget, and the go/no-go gates.
