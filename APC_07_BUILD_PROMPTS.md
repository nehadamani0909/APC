# APC-07 — Build Prompts

Copy-pasteable prompts for a coding agent (Claude Code or equivalent), one per module, in dependency order. Each has **context**, **task**, **acceptance criteria**, and **do-not**. Run them sequentially; do not start P*n+1* until P*n*'s acceptance criteria pass.

**Standing preamble — prepend to every prompt below:**

> You are building `frontier`, a research system for risk-controlled instance-adaptive prompt compression. Design docs are in `~/Downloads/APC_04_ARCHITECTURE.md` (architecture and formalism) and `~/Downloads/APC_06_EXPERIMENTS.md` (experiments). Read the relevant sections before writing code. Python 3.11+, typed, `ruff` + `mypy` clean, `pytest` for every module. No notebooks in the repo. Never invent an API you have not verified exists — check the installed package's signature first. If a design doc and my instruction conflict, stop and ask rather than guessing.

---

## P0 — Repository skeleton

> Create the `frontier` repo skeleton exactly as specified in `APC_04 §7`. Use `pyproject.toml` (uv or poetry), hydra for configs, `structlog` for logging. Implement `frontier/harness/ledger.py`: an append-only JSONL cost ledger with the `GridRow` schema from `APC_04 §8`, plus `read_ledger() -> pandas.DataFrame` and a `validate_ledger()` that checks for NaNs, duplicate `(prompt_id, backend, requested_b, target_model, sample_idx)` keys, and cost-field consistency. Implement `frontier/harness/prices.py` with a **versioned** price table (USD per 1M input / output tokens per model) — every ledger row records `price_table_version`.
>
> **Acceptance:** `pytest` green; `python -m frontier.harness.ledger --selftest` writes 100 synthetic rows, reads them back, and validates. `mypy` clean.
> **Do not:** add any model or dataset code yet.

---

## P1 — Task registry and target-LLM adapters

> Implement `frontier/harness/tasks.py` per the `Task` protocol in `APC_04 §4.1`, with adapters for: GSM8K, LongBench (each of its six sub-families as a separate `Task` with a shared `family` label), MeetingBank, HumanEval, MBPP, and a ShareGPT-style conversation set. **Critical:** each `Instance` stores `context` (compressible) and `query` (never compressed) as separate fields — see `APC_04 §4.3`.
>
> Implement `frontier/harness/models.py`: a `TargetLLM` protocol returning `(text, T_in, T_out, latency_ms, usd)`, with a vLLM local backend and an API backend. **Token counts must come from the provider's reported usage**, never from local re-tokenisation. Pin model IDs/revisions and record them in every ledger row.
>
> Implement `frontier/harness/metrics.py`: EM, token-F1, ROUGE-L, BERTScore, and `pass@1` via sandboxed unit-test execution for code. All return floats in [0,1].
>
> **Acceptance:** for each task, `load(split, n=5)` returns well-formed instances; one end-to-end call per task produces a valid ledger row; local ledger cost fields reconcile with provider-reported usage to <1% on a 50-call sample. Code execution is sandboxed with a timeout and cannot touch the filesystem outside a temp dir.
> **Do not:** compress anything yet.

---

## P2 — Compression backend adapters

> Implement `frontier/compress/` per the `Compressor` protocol in `APC_04 §5.1`: `llmlingua2` (default), `longllmlingua`, `cpc` *(only if a usable released implementation exists — otherwise stub it and record that in the docs)*, `truncate_tail`, and `random_drop` (seeded).
>
> **Critical:** `compress()` must return the **realised** rate measured with the *target model's* tokenizer, not the requested rate and not the compressor's own tokenizer — see `APC_04 §5.1` and audit defect D4. Also return `wall_ms` and `gpu_ms`.
>
> Add a content-hash cache: key `sha256(context + query + backend + rate + backend_version)`, stored under `data/cache/`. Cache hits must be logged distinctly so cached calls never contaminate latency measurements.
>
> **Acceptance:** `tests/test_compressors.py` shows, for 20 prompts × 5 rates × all backends, that realised rate is recorded and finite, caching is hit on the second call, and `random_drop` is reproducible under a fixed seed. A `notebooks/`-free script prints the requested-vs-realised adherence table (this is table **T7**).
> **Do not:** modify LLMLingua internals. It is a black box behind the adapter.

---

## P3 — Pilot grid runner (Gate 1)

> Implement `frontier/corpus/build_grid.py`: a **resumable, idempotent** grid runner over `(prompt × requested_b × backend × target_model × sample_idx)` that writes one ledger row per generation. It must survive a mid-run kill and resume without recomputing completed cells (use the content-hash cache plus a completion index). Support `--dry-run` printing the exact cell count and an estimated USD/GPU-hour cost before executing.
>
> Then `scripts/00_pilot.py`: 200 prompts stratified across 4 task families × 7 budgets `{1.0,0.8,0.65,0.5,0.4,0.3,0.2}` × 1 local model × `k=5` at `T=0.7`.
>
> Then `frontier/eval/gate1.py` implementing `APC_06 §2` exactly:
> - graded per-instance `ρ` (mean over k samples);
> - **monotone-safe** `b*` per `APC_04 §3.2` **and** naive `b*_naive`, at `ε ∈ {0.02,0.05,0.10}`;
> - the **noise floor** `σ²_noise` via split-half resampling of the `k` samples;
> - the gate test `Var_within-family[b*] > 2·σ²_noise` with a bootstrap CI;
> - E1b non-monotonicity rate, E1c rate adherence, E1d output expansion `T_out(b)/T_out(1)` per family;
> - **Figure 1**: per-instance frontier spaghetti plot faceted by family.
>
> **Acceptance:** `scripts/00_pilot.py --dry-run` prints the cost estimate; a full run produces `reports/gate1.md` containing every number above with CIs, plus Figure 1.
> **Do not:** train anything. This phase only measures. Do not proceed past this without the gate result.

---

## P4 — Full corpus

> Scale P3 to the tiered design in `APC_05 §4.1` (Tiers A/B/C/D). Add `frontier/corpus/validate.py`: schema conformance, no NaNs, no duplicate keys, splits disjoint **by prompt id and by source document** (`APC_04 §5.4`), and cost reconciliation. Export to parquet with a datasheet template (motivation, composition, collection, preprocessing, uses, limitations).
>
> **Acceptance:** `frontier-validate data/corpus/v1` exits 0; `reports/corpus_stats.md` contains table **T2**; a fresh clone can load the corpus and reproduce T2 in one command.
> **Do not:** let API spend exceed the configured cap — fail loudly instead.

---

## P5 — Features

> Implement `frontier/features/` with the three tiers from `APC_04 §5.2`. `surface.py` (L0) must include the **gzip compression ratio** of the context — a strong free redundancy proxy. `smallm.py` (L1) computes per-token NLL statistics from a ~0.5B LM plus contrastive perplexity `PPL(x|q) − PPL(x)`. `encoder.py` (L2) uses a frozen small encoder for `[CLS]` embeddings of context and query plus their cosine similarity.
>
> Every extractor **must record its own wall-clock latency per instance** — the latency budget in `APC_04 §5.2` (predictor < 20% of compressor latency) is a hard acceptance criterion for the whole project, and it cannot be checked retroactively.
>
> **Acceptance:** `frontier-features --tier L0|L1|L2` writes a features parquet keyed by `prompt_id`; `reports/feature_latency.md` reports per-tier ms; unit tests assert L0 < 5 ms/instance.
> **Do not:** use any feature that requires a target-LLM call. Add `tests/test_no_target_llm_in_features.py` asserting this by monkeypatching the LLM adapter to raise.

---

## P6 — Predictor with monotone head

> Implement `frontier/predict/heads.py` with the three heads from `APC_04 §5.3`:
> - **H1** quality-retention curve using a **cumulative-logit head with softplus increments**, so `ŝ(x,q,b)` is non-decreasing in `b` **by construction**. Also implement a free-form head and a direct `K`-way classifier as ablation variants behind the same interface.
> - **H2** rate adherence `r̂_φ(x,b)`, regressing `log(r/b)` on L0 features only, with an `invert(target_rate) -> requested_b` method.
> - **H3** output length, regressing `log T_out` with task-family and query features.
>
> Budget `b` is **not** an input feature: each head emits a vector over all `b ∈ B` in one forward pass.
>
> `train.py` uses the splits from `APC_04 §5.4`; `calibrate.py` does temperature scaling and isotonic calibration of H1 on `D_cal_a` **only**, with reliability diagrams and per-budget ECE.
>
> **Acceptance:** `tests/test_monotone.py` asserts `ŝ(b_i) ≤ ŝ(b_j)` for `b_i < b_j` on 1,000 random inputs; `tests/test_split_disjoint.py` asserts `D_cal_a ∩ D_cal_b = ∅`; training reproduces across 3 seeds with reported mean ± s.d.; `reports/e2.md` contains all E2 metrics.
> **Do not:** calibrate on `D_cal_b`. That split is reserved for CRC and using it here silently voids the guarantee.

---

## P7 — Risk-controlled selector

> Implement `frontier/select/crc.py` (Conformal Risk Control) and `ltt.py` (Learn-then-Test), and `policy.py` implementing the inference algorithm in `APC_04 §6` exactly:
> feasible set → abstain to `b=1.0` if empty → **argmin predicted total cost over the feasible set** (NOT the smallest feasible budget — see `APC_04 §3.3`) → adherence-corrected request.
>
> CRC: `λ̂ = inf{λ : (n/(n+1))·R̂_n(λ) + 1/(n+1) ≤ ε}` on `D_cal_b`, with loss `L = max(0, ρ(x,1) − ρ(x,b̂_λ))` clipped at 0.
>
> **Acceptance:** `tests/test_crc_monotone.py` asserts empirical risk is non-increasing in λ; `tests/test_no_target_llm_at_inference.py` asserts `select()` never calls the target LLM (monkeypatch to raise); `reports/e5.md` contains the E5a validity plot over ≥20 random calibration/test splits, showing empirical risk ≤ ε for `ε ∈ {0.02,0.05,0.10}`.
> **Do not:** implement "smallest safe budget" as the default rule. The cost-minimising rule is contribution C2 and the difference between them is experiment E9.

---

## P8 — Baselines

> Implement every baseline in `APC_06 §5` behind the **single** `Policy` protocol (`APC_04 §8`): B0, B1, B2a/b/c, B3, B3b, B4, B5a (AdaComp-style point predictor at token granularity), B5b (Adaptive QuerySelect-style), B5c (AttnComp-style threshold-adaptive), B6, B7 (model routing), B8 (caching sensitivity), ORACLE, ORACLE-noisy, OURS.
>
> **Every policy must go through one identical evaluation path.** If a baseline appears to need a special path, stop and tell me — that is a design smell and a fairness risk, not a reason to fork the evaluator.
>
> For B5a and B5b, read the descriptions in `APC_02 §D1/§D2` and reimplement faithfully at token granularity; document every adaptation you make in `docs/baseline_fidelity.md`, because reviewers will ask whether the comparison was fair.
>
> **Acceptance:** `frontier-eval --policies all` produces table **T3** with paired-bootstrap CIs; ORACLE rows are labelled `(upper bound)` in every output; `docs/baseline_fidelity.md` exists and is specific.
> **Do not:** tune OURS on the test split. Tune every baseline at least as carefully as OURS, and record the tuning budget for each.

---

## P9 — Evaluation, statistics, figures

> Implement `frontier/eval/` with: **CPGR**, **CPT(x%)**, quality-vs-USD AUC, compression/token/cost regret, paired BCa bootstrap (10,000 resamples), Holm–Bonferroni across the baseline family, and the power calculation from `APC_06 §14`.
>
> Figures F1–F4 per `APC_06 §15`, publication quality: vector output, colour-blind-safe palette, legible at print size, readable in greyscale.
>
> **Acceptance:** `scripts/09_paper_figures.py` regenerates every table and figure from the corpus in one command; every reported number carries a CI; `reports/power.md` states the minimum detectable effect.
> **Do not:** report a bare p-value without an effect size and CI.

---

## P10 — Generalisation, ablations, failure analysis

> Implement E4 (ablations), E6 (cross-model / cross-task / cross-compressor, including the **λ-recalibration-only** transfer condition and the E6c risk-violation-under-shift measurement), E7, E9 (cheapest-safe ≠ smallest-safe, with the `c_out/c_in` price sweep), E10 (net efficiency incl. the operating-window analysis), E11 (failure taxonomy with instance counts and worked examples).
>
> **Acceptance:** tables T4–T9 filled; `reports/failures.md` contains ≥3 worked examples per failure category, each showing the original context, the compressed context, and the model output.
> **Do not:** omit negative results. E6c is *expected* to show a risk violation under task shift, and E10 is *expected* to show regions where the pipeline does not pay off. Both are findings and both go in the paper.

---

## P11 — Release

> Produce: MIT-licensed code, corpus parquet + datasheet, trained predictor + calibration constants + model card (stating the calibration domain and measured shift behaviour), the resolved claims ledger from `APC_03 §3`, and a `REPRODUCE.md` that regenerates every paper number from scratch with one command. Include a cost-ledger summary reporting total USD and GPU-hours spent producing the paper.
>
> **Acceptance:** a fresh clone on a clean machine reproduces T3 from the released corpus in one command.

---

## Deep-research prompt (run at Week 1 and again at Week 18)

> Search arXiv, ACL Anthology, OpenReview and NeurIPS/ICLR/ICML proceedings for work published or updated since **2025-01-01** on: adaptive or variable-rate prompt compression; predicting compression ratio or compression tolerance per instance; quality-constrained or risk-controlled context compression; conformal prediction or risk control applied to prompt/context compression; compression-induced output-length expansion; and compression-rate routing.
>
> For each hit, extract: venue, granularity (token / sentence / document / embedding), how the rate is selected (fixed / heuristic-adaptive / learned / RL), whether it predicts quality **before** inference, whether it offers any statistical guarantee, whether it models total cost including output tokens, and whether it models rate adherence.
>
> Return a table in the schema of `APC_02 §4` plus, for any work that overlaps two or more of our contributions C1–C5 (see `APC_03 §2`), a paragraph on the precise delta. **Flag anything that would falsify a contribution — I want the bad news first and explicitly, not softened.**

---

## Paper-writing prompt (Week 18)

> Write the paper from `APC_03` (contributions and forbidden claims), `APC_02` (related work and the positioning table), `APC_06 §15` (tables and figures) and the resolved claims ledger. Structure per `APC_08 §5`.
>
> Hard constraints:
> - **Every claim must trace to a ledger row with status ✓.** If a claim has no supporting experiment, delete the claim — do not soften it.
> - Use the forbidden/permitted phrasing table in `APC_03 §4` verbatim.
> - Nagle et al. and AdaComp must be named **in the introduction**, with the delta stated in one sentence each.
> - Report all efficiency numbers in **USD and seconds**, never input tokens alone.
> - The Limitations section must include: exchangeability failure under task shift (E6c), the operating window where the pipeline does not pay off (E10), label noise at `k=5`, and the fact that MeetingBank is LLMLingua-2's training domain.
> - Do not use the words "novel", "we are the first", or "significantly" without a citation-checked or statistically-tested basis.

---

*Next:* `APC_08_REBUTTAL_KIT.md` — the reviews to expect and the answers.
