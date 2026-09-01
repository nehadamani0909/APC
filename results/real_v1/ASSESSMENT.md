# Assessment of the real grid

630 cells · 30 instances · 3 families · 0 failures
LLMLingua-2 (`bert-base-multilingual-meetingbank`) · `llama3.2:3B-Q4_K_M` · k=3
2026-08-31

---

## Verdict on the original question

**A regional-IEEE paper is now writable.** It is not the paper the planning
documents describe, and it is not the paper I would have predicted twelve
hours ago. It is a **measurement and methodology paper** with one clean
positive result, one carefully-bounded negative, and one design finding that
changes how the rest of the project should be run.

That is a real paper for INDICON/TENCON/CONECCT. It is not an ACL paper, and
it should not be dressed as one.

---

## 1. What is solid enough to publish

### C4 — rate adherence. The strongest result here.

LLMLingua-2 **systematically over-compresses at every requested budget**,
measured in the target model's tokenizer:

| requested `b` | realised `r` | signed error | 95% CI |
|---|---|---|---|
| 0.20 | 0.173 | −0.027 | [−0.030, −0.025] |
| 0.30 | 0.262 | −0.038 | [−0.041, −0.035] |
| 0.40 | 0.354 | −0.046 | [−0.050, −0.042] |
| 0.50 | 0.444 | −0.056 | [−0.061, −0.052] |
| 0.65 | 0.590 | −0.060 | [−0.067, −0.053] |
| 0.80 | 0.751 | −0.049 | [−0.057, −0.041] |

Every CI excludes zero, n=90 per budget, and the error is not constant — it
peaks near b=0.65 and shrinks at both extremes.

A **separate sweep needing no target-model calls** (`runs/adherence_v1`,
`scripts/24_adherence.py`) replicates this at scale: **1,044 measurements
across 174 instances and three families, zero errors.**

| requested `b` | realised `r` | signed error | 95% CI (n=174) |
|---|---|---|---|
| 0.20 | 0.1751 | −0.0249 | [−0.0265, −0.0233] |
| 0.30 | 0.2656 | −0.0344 | [−0.0365, −0.0323] |
| 0.40 | 0.3581 | −0.0419 | [−0.0448, −0.0391] |
| 0.50 | 0.4492 | −0.0508 | [−0.0538, −0.0479] |
| 0.65 | 0.5973 | −0.0527 | [−0.0568, −0.0487] |
| 0.80 | 0.7596 | −0.0404 | [−0.0453, −0.0355] |

Two structural facts, and both matter for the contribution:

1. **The error is an inverted U in `b`**, peaking mid-range at b=0.65. It is
   not a constant offset, so it cannot be corrected by a single scalar.
2. **It is family-dependent** — at b=0.65 the mean error is −0.074 on
   `2wikimqa`, −0.060 on `multifieldqa_en`, and −0.025 on `gsm8k`, a ~3×
   spread. This is the *prompt-dependence* half of C4's claim, and it is what
   makes a learned `R_φ(x,b)` the right object rather than a lookup table.

Directional, consistent, well-powered, and unreported. It costs one figure
and one paragraph.

The methodological point it licenses: **a token-savings number reported
against a requested rate is a claim about a request, not a result.**

### Non-monotonicity

Naive and monotone-safe `b*` disagree on **20.0%** of instances (mean gap
0.483), and aggregate quality peaks at b=0.8 (0.428) *above* uncompressed
(0.372). This is why a free-form regressor on `b*` would fit noise, and it
justifies the monotone-safe label the architecture already specifies.

---

## 2. The bounded negative — and it is a finding, not a failure

Output expansion is **real and significant**:

- `qa` at b=0.2: **1.68×** (95% CI [1.27, 2.14])
- `reason` at b=0.2: **1.28×** (95% CI [1.09, 1.48])

Both CIs exclude 1.0. This is, as far as the project's literature map shows,
the first measurement of compression-induced output expansion using a **real
neural compressor and task-native metrics** rather than truncation and
embedding similarity.

**And yet the cheapest safe budget equalled the smallest safe budget for all
25 evaluable instances, at every price ratio `c_out/c_in ∈ [1,5]`.**

The reason is visible in the raw magnitudes:

| | T_in (b=1.0 → 0.2) | T_out (b=1.0 → 0.2) |
|---|---|---|
| `qa` | 2405 → 498 (**−1907**) | 30 → 40 (**+10**) |

A 1.68× expansion on a 30-token answer cannot outweigh a 1900-token input
reduction. Even at `c_out/c_in = 5`, that is 1907 against 50.

This does **not** contradict the TAAC compression paradox — it *bounds* it.
TAAC measured code generation, where outputs are long. The honest claim, and
a genuinely useful one:

> Compression-induced output expansion is real and measurable, but it changes
> the cost-optimal budget only when the output-to-input token ratio is high.
> In short-answer QA it does not.

That tells a practitioner exactly when to worry. Publishing the 1.68× without
the 0/25 divergence would be the same error, in the opposite direction, as
the input-tokens-only claims the project set out to correct.

---

## 3. The design finding — this is the one that matters most

Gate 1 does not pass, but **the reason is not that heterogeneity is absent.**
The per-instance curves show it plainly: within `multifieldqa_en`, instance
`d5422b0f` degrades 0.98 → 0.73 while `8e79ff61` collapses 0.53 → 0.11.

The problem is that **k=3 makes the per-instance label too noisy to resolve
it.** Testing on a more robust statistic than `b*` — compression sensitivity
`S(x) = ρ(x,1.0) − ρ(x,0.2)` — the between-instance variance still sits below
the split-half noise floor:

| family | n | mean S | Var[S] | noise floor (k=3) | verdict |
|---|---|---|---|---|---|
| `qa` | 10 | +0.211 | 0.0194 | 0.0654 | fails |
| `multidoc_qa` | 9 | +0.049 | 0.0022 | 0.0052 | fails |
| `reason` | 6 | −0.056 | 0.1519 | 0.1624 | fails |

The floor scales roughly as `1/k`. Solving for the k at which the observed
variance would clear twice the floor:

| family | required k |
|---|---|
| `qa` | **≥ 20** |
| `multidoc_qa` | ≥ 14 |
| `reason` | ≥ 6 |

**`APC_06 §2` plans for k=5. That is roughly 4× too few for the QA family.**
Running the full 262k-generation grid at k=5 would have produced an
underpowered Gate 1 and burned the budget to reach an uninterpretable answer.
Discovering this on 630 cells instead is the single most valuable thing this
run did.

Two levers follow, and they compose:

1. **Raise k**, at least for the gate pilot. k=20 on a small instance set
   costs far less than k=5 on a large one and is the only way to resolve the
   label.
2. **Prefer graded metrics.** Token-F1 (`qa`) has a floor of 0.0654 against
   exact-match (`reason`) at 0.1624 — **2.5× lower for free**. Binary
   exact-match is the most expensive metric choice available in terms of
   samples needed.

---

## 4. What is not supported, and must not be written

- No heterogeneity claim. Gate 1 fails on every family.
- No predictor, no conformal guarantee, no policy comparison. Nothing was
  trained on this grid.
- No cross-model or cross-backend claim. One model, one checkpoint.
- `multidoc_qa` (2wikimqa) is **degenerate**: ρ(x,1) = 0.105, and the target
  model effectively cannot do the task. Report it as an excluded family with
  the reason stated, not as a result.
- `reason` (gsm8k) has only 6/10 instances solvable uncompressed and its
  quality peaks at b=0.4 — noise-dominated.

`multifieldqa_en` is the only family with **no validity problems** (10/10
solvable, ρ(x,1)=0.544). It should carry the paper's main figure.

---

## 5. Recommended paper

**Title direction:** *"Requested Is Not Realised: Measuring Rate Adherence
and Output Expansion in Neural Prompt Compression"*

**Structure:** C4 adherence as the headline · output expansion measured, with
the divergence bound as the honest second result · the k-requirement as a
methodology contribution · `multifieldqa_en` frontiers as the motivating
figure.

This is a coherent 6-page paper in which **every number is measured**, the
negative results are load-bearing rather than embarrassing, and the
limitations section is a strength. Hold C1 (conformal risk control) and C3
(frontier prediction) for the ACL submission — see `PAPER_GUIDE.md` §7 on the
archival constraint.

## 6. Before submitting, run this

The single highest-value follow-up, and it is affordable:

```bash
# k=20 on multifieldqa_en only: 10 instances x 7 budgets x 20 samples = 1400 cells
uv run python scripts/20_real_grid.py --out runs/k20 --device cpu \
  --n-per-family 10 --samples 20 --workers 1 --min-tok 500 --max-tok 3500
```

At the ~7s/cell measured here that is **under 3 hours** and it directly tests
whether the heterogeneity visible in the curves survives a properly-powered
noise floor. If it does, Gate 1 passes and the paper gains its strongest
figure. If it does not, that is a clean, well-powered negative — which for
this project is also publishable, and considerably more informative than an
underpowered one.
