# Handover: P3 to P4 audit

Date: 2026-08-29  
Repository: `APC`, branch `main`  
Last pushed commit: `9761d23` (`Implement P3 pilot grid and Gate 1 analysis`)

## Executive decision

The P0–P3 engineering path is reproducible and currently clean, but the
project is **not scientifically ready for P4**. The checked-in P3 pilot uses a
deterministic offline target and fixture-style prompts because no vLLM or
benchmark dataset package is installed. Its generated Gate 1 report says
`PASS`, but that result is a harness smoke result only and must not be used as
the research go/no-go decision.

Per the corrected plan, P4 must remain blocked until the same pilot is rerun
with a pinned real local model and real pilot prompts, and the Gate 1 result is
explicitly accepted.

## Audit scope

Audited against:

- `Adaptive_Prompt_Compression_CORRECTED_PLAN.md`, especially §§3, 7, 8, and 10
- `APC_04_ARCHITECTURE.md`, especially §§4, 5.1, 7, and 8
- `APC_05_BUILD_PLAN.md`, especially §§2–4 and §8
- `APC_06_EXPERIMENTS.md`, especially §2
- `APC_07_BUILD_PROMPTS.md`, prompts P0–P3

## Verification performed

All commands were run in `.venv` using Python 3.12.13:

```text
17 passed
mypy frontier: Success: no issues found
ruff check .: All checks passed
python -m frontier.harness.ledger --selftest: 100 rows written and validated
scripts/00_pilot.py --dry-run: 7,000 cells; estimated 0.972222 GPU-hours
pilot ledger validation: 7,000 rows; 7,000 unique grid keys
second pilot run: executed 0 cells (resume/idempotence)
no notebooks found
```

The pilot report and SVG were regenerated at `reports/gate1.md` and
`reports/figure1.svg`; generated pilot/report artifacts are ignored by Git.

Additional regression tests now cover:

- monotone-safe `b*` rejecting an unsafe middle budget;
- split-half noise-floor calculation;
- resuming from the ledger when the completion index is absent;
- one end-to-end row for every registered P1 task;
- cache hits and zero cache-hit timing;
- 20 prompts × 5 rates across every runnable P2 backend;
- duplicate/cost/NaN ledger validation and sandboxed code checks.

## Prompt-by-prompt status

### P0 — repository skeleton and ledger

Status: **engineering acceptance met**.

Implemented the architecture layout, `pyproject.toml`, Hydra/structlog
dependencies, frozen `GridRow`, append-only JSONL writer, pandas reader,
duplicate/NaN/cost validation, and versioned prices. `run_id`, `phase`, and
`ts` are ledger metadata added at write time because APC-04 §4.4 requires them
but the frozen §8 dataclass omits them.

Known limitation: the price table only knows the three current IDs in
`frontier/harness/prices.py`; a real local model ID must be added as a pinned,
versioned price entry before a real run.

### P1 — tasks, models, and metrics

Status: **harness acceptance only; benchmark integration incomplete**.

Implemented:

- separate `Instance.context` and `Instance.query`;
- GSM8K, six LongBench category adapters, MeetingBank, HumanEval, MBPP, and
  ShareGPT task objects;
- normalized JSONL ingestion;
- provider-client API adapter and injected vLLM adapter;
- provider-usage-based token/cost accounting;
- EM, token-F1, ROUGE-L, injected BERTScore hook, and sandboxed code checks.

Gaps before scientific use:

- loaders default to five deterministic fixtures and do not yet load the real
  GSM8K/LongBench/MeetingBank/HumanEval/MBPP/ShareGPT sources;
- the vLLM adapter requires an injected engine and sampling-parameter factory;
  it does not instantiate and pin the planned Qwen/Llama deployment;
- no real provider SDK adapter has been verified in this environment;
- BERTScore is an injection point, not an installed/verified BERTScore model;
- the code sandbox is a conservative subprocess/unit-test sandbox, not a
  hardened security boundary. It rejects imports and selected filesystem
  primitives, but should not execute hostile arbitrary code.

### P2 — compressor adapters

Status: **adapter acceptance met for runnable/injected backends; real backend
integration incomplete**.

Implemented deterministic `truncate_tail` and seeded `random_drop`, black-box
LLMLingua-2 and LongLLMLingua adapters, target-tokenizer realised-rate
measurement, wall/GPU timing fields, content-hash JSON cache, and an explicit
CPC unavailable stub documented in `docs/compressors.md`.

Gaps before scientific use:

- no LLMLingua-2, LongLLMLingua, or CPC package is installed or API-verified;
- the target tokenizer must be injected; the offline pilot uses whitespace
  tokenization and therefore is not a target-model rate measurement;
- injected neural adapters currently report `gpu_ms=0.0`; real compressor
  timing must be wired before E1c/E10 claims.

### P3 — pilot grid and Gate 1

Status: **engineering path met; scientific Gate 1 not yet established**.

Implemented:

- generic resumable/idempotent `GridRunner` over the required cell key;
- completion index plus ledger fallback, with canonicalized budget keys;
- dry-run count and cost/GPU-hour estimate;
- 200 synthetic prompts across four family labels, seven budgets, one target,
  and five samples;
- monotone-safe and naive budget labels for all requested epsilons;
- repeated split-half noise-floor estimation;
- family-wise bootstrap variance checks;
- E1b budget disagreement/gap, E1c adherence, E1d output expansion, and SVG
  Figure 1 report output.

Critical scientific caveat: the pilot target is `PilotTarget`, not a local
LLM, and the prompts are generated fixtures. The resulting zero noise floor
and Gate 1 PASS are expected consequences of that deterministic simulation;
they do not demonstrate within-family heterogeneity above sampling noise in
the intended population.

The Gate 1 aggregation has been made conservative: the overall result is PASS
only when every family passes the family-wise criterion for every tested
epsilon. The report is still not a valid research result until real data are
used.

## Plan alignment

### On track

- The project has not implemented predictors, calibration, selectors, or UI.
- No P4 work has been started.
- Query text remains separate from and is not passed through the compressor.
- Ledger rows contain realised rate, output tokens, USD fields, latency, model
  revision metadata, and price-table version.
- Grid execution is resumable and idempotent.
- The fallback corpus/measurement paper remains available if Gate 1 fails.

### Not yet on track / must resolve

1. Install and verify the pinned local model/vLLM stack and its target
   tokenizer.
2. Replace fixture loaders with real, pinned benchmark data for the 200-prompt
   pilot, recording source/document IDs and licenses.
3. Add a real versioned local model price/compute accounting policy; local USD
   may be zero, but GPU seconds must be measured.
4. Verify the real LLMLingua-2 adapter signature and backend revision before
   collecting scientific compression results.
5. Decide and document the verified BERTScore implementation and hardening
   requirements for code execution.
6. Rerun the pilot, validate the real ledger, regenerate `reports/gate1.md`,
   inspect CIs, and make the pre-registered PASS/BORDERLINE/FAIL decision.

## Explicit handoff gate

P4 may start only after all of the following are true:

- real local model and tokenizer are installed, pinned, and API-verified;
- real pilot data replace offline fixtures;
- 7,000 real generations complete with reconciled ledger costs and measured
  compressor/GPU latency;
- Gate 1 report contains the required E1 quantities and CIs;
- the result is reviewed as PASS or BORDERLINE/FAIL according to
  `APC_05 §2`; and
- the project owner gives explicit go-ahead to cross Gate 1.

No predictor, calibration layer, selector, or P4 corpus scaling should be
started before that handoff.
