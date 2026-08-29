# Mission

This repo (APC) is a fully-scaffolded but non-runnable research pipeline for "risk-controlled
instance-adaptive prompt-compression rate selection." All contracts, the cost ledger, the
CRC calibration math, L0 features, and the eval harness exist and pass tests, but every
connection to the real world is an unfilled injection point and the core predictor cannot be
trained. Your job is to make the pipeline execute end-to-end on real benchmarks, real
compressors, and real target LLMs, following the existing design docs — without breaking the
frozen contracts or the offline test suite.

# Read first (do not skip)

- APC_04_ARCHITECTURE.md  — the formalism, the four data splits (D_train/D_cal_a/D_cal_b/
  D_test), the three predictor heads, the selection algorithm, the repo layout. This is
  the spec you implement to.
- APC_05_BUILD_PLAN.md    — the tiered grid design, budget, gates.
- APC_06_EXPERIMENTS.md   — every experiment (E1..E10), baseline (B0..B8), metric (CPGR,
  CPT, quality-cost AUC), and the statistics (BCa bootstrap, Holm, power). Your outputs
  must match what E-sections expect.
- APC_07_BUILD_PROMPTS.md — the P0..P11 phase definitions the current code was built from.
- handover_p11_complete.md and docs/claims_ledger.md — what is deliberately deferred.

Then read the existing code you will extend: frontier/compress/*, frontier/harness/*,
frontier/features/*, frontier/predict/*, frontier/select/*, frontier/corpus/*,
frontier/eval/*, scripts/*.

# Ground rules

1. Do NOT change the frozen dataclass schemas (GridRow, CompressedResult, GenerationResult,
   Instance, Selection, PolicyPrediction) or the Protocol signatures without writing a short
   note in the relevant handover file explaining why and updating every caller + test.
2. `uv run pytest`, `uv run ruff check .`, and `uv run mypy frontier scripts tests` must stay
   green at every commit. Current suite is ~37 tests; it must only grow.
3. Keep the offline determinism guarantee: no test may download data or hit a network/API.
   Real integrations go behind `@pytest.mark.integration` (skipped by default) and behind a
   config flag / injected client. The existing fixture fallbacks in tasks.py must keep working.
4. Secrets: read API keys only from env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.). Never
   write keys, datasets, or model weights into the repo. Add large artifacts to .gitignore.
5. Every run must record: model id + revision, compressor backend + version, tokenizer
   version, price-table version, seeds, temperature, and (for API) a spend cap. The cost
   ledger already reconciles USD to provider usage within 1% — preserve that.
6. Work in small, reviewable commits, one phase per branch/PR. At the top of each PR
   description, state which APC_06 experiment(s) the phase unblocks.

# Work plan (phases; land them in this order)

## Phase A — Benchmark ETL
Create `frontier/data/` with one loader per task family that downloads the real dataset
(HuggingFace `datasets` where possible), normalizes each record to the Instance JSONL schema
(`id, context, query, gold, split, meta`), keeps context and query strictly separate (the
query is never compressed), and records source document ids so split-leakage can be checked.
Targets, matching tasks.py: GSM8K (reason), LongBench subsets (multidoc_qa/single_doc_qa/
summarisation/few_shot/code/synthetic), MeetingBank (summ), HumanEval + MBPP (code),
ShareGPT (conv). Deterministic train/cal_a/cal_b/test partition BY PROMPT ID using
`frontier.predict.train.split_prompt_ids` (60/10/10/20). Add `scripts/03_fetch_data.py` to
materialize normalized JSONL under `data/raw/` and `data/splits/`. Wire `default_tasks()` (or
a config) so every Task points at a real `source=` instead of falling back to `_fixture`.
Add integration tests that assert schema, non-empty splits, and zero id overlap across splits.

## Phase B — Compressor integration
Implement real backends behind the existing adapters:
- LLMLingua2Compressor / LongLLMLinguaCompressor: wire the actual `llmlingua` package
  (`PromptCompressor`), passing the target-model tokenizer through so `realised_rate` is
  measured on the target tokenizer, not the compressor's. Populate `backend_version` and
  `model_version` from the installed package + model revision.
- Keep the `compressor` callable injection path for tests.
- CPCCompressor: leave raising, but add a `TruncateHeadCompressor`/`RandomDropCompressor` if
  APC_06 baselines (B6, backend ablation E7) need more deterministic backends than what's in
  baselines.py.
Add `scripts/compress_smoke.py` that compresses 20 real prompts at all 7 budgets and prints
requested-vs-realised rate, so the C4 rate-adherence claim has data.
Integration test: realised rate is monotone-ish in requested rate and within sane bounds.

## Phase C — Target-LLM clients
Implement `ProviderClient` classes in `frontier/harness/providers.py`:
- OpenAIClient, AnthropicClient (chat completions; return text + exact input/output token
  usage from the API response, NOT locally re-tokenized).
- A local `HFClient` or `VLLMClient` for the cheap model where the full k=5 grid runs.
Add ret/backoff, concurrency limiting, a hard USD spend cap that aborts the run, and a
disk request cache keyed by (prompt, model, revision, temperature, seed). Wire them into
`APIBackend` / `VLLMBackend`. Config selects provider + model + price row.
Integration test (skipped by default): one real call reconciles USD to usage within 1%.

## Phase D — Code-execution metric
Implement a sandboxed runner for `pass_at_1` on HumanEval/MBPP (subprocess with timeout,
resource limits, no network; or a container hook). Replace any placeholder in
frontier/harness/metrics.py. Unit-test with known-passing and known-failing solutions.

## Phase E — Grid runner at scale
Extend `frontier/corpus/build_grid.py` / `GridRunner`: parallel workers, resumable via the
existing completed-cells ledger, per-cell try/except that records failures instead of
aborting, progress + ETA, and honoring the spend cap. Implement the TIERED design from
APC_05 §5 (cheap local model gets the full N×K×k grid; expensive API models run only on a
stratified subset). `scripts/04_corpus.py` must produce a real `data/corpus/v1/corpus.parquet`
conforming to `frontier/corpus/schema.py` with input AND output tokens, realised rate,
latency, USD, and graded per-instance quality ρ(x,b). Update the datasheet
(docs/corpus_datasheet.md) from the real run.

## Phase F — Predictor training  (this is the critical gap)
In `frontier/predict/`:
- Give `CumulativeLogitHead` a real `fit(features, quality_labels)` — cumulative-logit /
  ordinal NLL with the softplus-parameterized non-negative increments preserving
  monotonicity; L2 reg; deterministic. Add `fit` to `FreeFormHead` and `BudgetClassifier`
  for the E4 ablation. Wire `OutputLengthHead.fit` (Poisson/log-normal regression on T_out).
  `RateAdherenceHead.fit` already exists — keep it.
- Labels: build the graded quality target ρ(x,b) and the monotone-safe b*(x) from the corpus
  per APC_04 §3.1–3.2 (sampled pass-rate primary; answer-preservation as robustness).
- Training script `scripts/06_train.py`: extract features (L0 always; L1 via injected
  small-LM; L2 via injected encoder), fit on D_train, temperature/isotonic-calibrate H1 on
  D_cal_a ONLY (calibrate.py functions exist), persist a real `artifacts/predictor.json`
  (replace the scaffold) with feature list, weights, calibration params, and provenance.
- Report E2 (curve calibration, MAE, monotone vs free-form) and E4 (curve vs classifier)
  with real numbers, replacing the hardcoded smoke report in train.py.

## Phase G — End-to-end predictor + policy
Implement a concrete `FrontierPredictor` (satisfies the Protocol in select/policy.py) that:
takes (x, q, task_family) → extracts features → runs the 3 heads → assembles
`PolicyPrediction(quality, cost, realised_rate)` where `cost` is the TRUE-cost objective
from APC_03 C2: `c_in·T̂_in + c_out·T̂_out + c_pred + c_comp`, using the price table.
Confirm `RiskControlledPolicy` selects the cheapest safe budget and inverts adherence.
`scripts/07_crc_report.py`: calibrate λ̂ via `calibrate_crc` on D_cal_b (disjoint from
D_cal_a), produce the E5 coverage plot (empirical risk ≤ ε across ≥20 random splits) and the
E5b uncalibrated-threshold ablation. Fix `calibrate_ltt` to a proper multi-hypothesis
Learn-then-Test (Bonferroni-Holm over the λ grid with a valid per-hypothesis p-value) if the
risk is non-monotone; document the choice.

## Phase H — Evaluation, baselines, figures
Make `frontier/eval/` run on the real corpus + trained predictor:
- Implement the real versions of baselines currently stubbed in baselines.py / policies.py:
  B2a/B2b/B2c (validation-tuned fixed budget: global / per-family / per-family×model),
  B5a (AdaComp-style trained point predictor), B5c (AttnComp-style threshold), B7 (model
  routing), B8 (cache-hit sensitivity), ORACLE and ORACLE-noisy (labelled upper bound).
- Compute CPGR, CPT(x%), quality-vs-cost AUC with paired BCa bootstrap over prompts and
  Holm correction across the baseline family; run the power calc from APC_06 §9.
- `scripts/09_paper_figures.py` and `scripts/10_generalisation.py`: emit real vector figures
  and tables into `paper/figures/` and `paper/tables/` (E3 main result, E6 transfer across
  models/families, E7 backend transfer, E9 "cheapest safe ≠ smallest safe", E10 net $/latency).

## Phase I — Config, reproduction, docs
- Hydra configs under `configs/` for: pilot, tiered full grid, train, calibrate, eval — each
  pinning models, price rows, seeds, spend caps.
- `scripts/reproduce.py` must run the real pipeline from raw data → corpus → features →
  train → calibrate → eval → figures, with a `--fast` mode that uses a small sample.
- Update REPRODUCE.md, docs/model_card.md, docs/corpus_datasheet.md, docs/compressors.md,
  docs/cost_ledger_summary.md from real runs. Do NOT edit docs/claims_ledger.md status
  columns — leave claim validation to the human researcher.

# Acceptance criteria

- `uv run pytest` green; new integration tests exist but are skipped without creds/data.
- `uv run ruff check .` and `uv run mypy frontier scripts tests` clean.
- `uv run python scripts/reproduce.py --fast` runs the full real pipeline on a small sample
  and produces a non-fixture corpus, a trained+calibrated predictor with provenance, and
  populated paper/figures + paper/tables.
- A short `RUNBOOK.md` explaining how to do the full run: env vars, `datasets` downloads,
  estimated GPU-hours and USD, how to resume, how to raise the spend cap.
- No secrets, datasets, or weights committed; .gitignore updated.

# If you hit an ambiguity

Prefer the choice written in APC_04/APC_06. If the docs are silent, pick the simplest option
that keeps contracts stable, leave a `# DECISION:` comment explaining it, and list all such
decisions in the PR description. Do not invent new scientific claims or fill in
docs/claims_ledger.md.
