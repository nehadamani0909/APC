# Handover: P4 to P5 audit

Date: 2026-08-29  
Repository: `APC`, branch `main`  
Last pushed commit before P4: `9761d23` (`Implement P3 pilot grid and Gate 1 analysis`)

## Executive status

P4’s corpus engineering layer is implemented and tested, but the repository
does **not** yet contain the planned full scientific corpus. The available
`data/corpus/v1` artifact is a 7,000-row internal corpus materialized from the
deterministic P3 pilot. It is useful for validating parquet/schema/tooling
paths, not for training features or making scientific claims.

P5 must not begin feature extraction on this fixture corpus. Replace it with
real, licensed, pinned corpus data and real-model ledger rows first.

## Implemented P4 components

- `frontier/corpus/schema.py`
  - `CorpusRow` adds `source_document_id`, `split`, and `tier` to a frozen
    `GridRow`.
- `frontier/corpus/validate.py`
  - parquet/JSONL loading;
  - schema conformance;
  - empty corpus and NaN checks;
  - duplicate grid-key checks;
  - prompt-ID and source-document split leakage checks;
  - delegated ledger cost reconciliation;
  - configurable API spend-cap failure;
  - parquet export and reproducible T2 statistics/report.
- `frontier/corpus/build_full.py`
  - P4 tier specifications A/B/C/D;
  - planned generation accounting;
  - metadata attachment and ledger-to-parquet export.
- `scripts/04_corpus.py`
  - materializes, validates, and reports a corpus from an existing ledger.
- `docs/corpus_datasheet.md`
  - datasheet template with explicit pending real-data fields and limitations.
- `frontier-validate` console entry point in `pyproject.toml`.

## Tier accounting

The implemented tier specs are:

| Tier | Prompts | Budgets | Models | k | Generations |
|---|---:|---:|---:|---:|---:|
| A | 6,000 | 7 | 1 | 5 | 210,000 |
| B | 1,200 | 7 | 1 | 1 | 8,400 |
| C | 1,200 | 7 | 1 | 1 | 8,400 |
| D | 800 | 7 | 1 | 5 | 28,000 |
| **Total** | | | | | **254,800** |

The source build plan describes this total as approximately 262,000; the
implemented exact factorial sum is 254,800. Resolve that documentation
arithmetic discrepancy before budgeting or preregistering generation counts.

## Validation performed

Using the project `.venv` with Python 3.12.13:

```text
20 passed
mypy frontier: Success: no issues found
ruff check .: All checks passed
ledger self-test: 100 rows written and validated
parquet export: data/corpus/v1/corpus.parquet
frontier-validate data/corpus/v1: valid corpus: 7000 rows
T2 report: reports/corpus_stats.md
```

The parquet round trip was exercised through pandas’ installed
`DataFrame.to_parquet`/`read_parquet` path with `pyarrow==19.0.1` declared and
installed. Nonfatal Arrow hardware-query warnings appeared in this restricted
environment; export and validation completed successfully.

Regression tests cover:

- tier generation accounting;
- parquet round trip;
- corpus schema and ledger-cost validation;
- duplicate/split-leakage rejection;
- API spend-cap rejection;
- T2 report generation.

## P4 acceptance status

### Met for the engineering fixture path

- `frontier-validate data/corpus/v1` exits 0;
- schema, NaN, duplicate-key, split-leakage, and cost checks exist;
- parquet export exists and round-trips;
- T2 generation exists;
- datasheet template exists;
- API spend-cap checks fail loudly;
- no feature/predictor/selector code was added.

### Not met for the scientific corpus

1. The 6,000/1,200/1,200/800 real prompts have not been collected.
2. The corpus currently derives from the 200 deterministic fixture-pilot
   prompts and contains only one backend/model.
3. `source_document_id` defaults to `prompt_id` and `split` defaults to
   `unspecified` when metadata is not supplied. This validates mechanics but
   does not establish source-document-disjoint train/calibration/test splits.
4. No actual benchmark licenses, source-document mapping, or datasheet
   composition numbers have been filled in.
5. No real LLMLingua-2/LongLLMLingua/CPC integration or real target-model
   tokenizer has supplied the measurements.
6. The current corpus has zero USD local cost and synthetic timing; it cannot
   support cost or efficiency claims.
7. The generated `data/corpus/v1` and `reports/` artifacts are local working
   artifacts and are not part of a release or pushed commit yet.

## P0–P3 carry-forward audit

- P0: ledger and price-table mechanics pass tests; real local model IDs still
  need versioned pricing/compute accounting.
- P1: task adapters are normalized-JSONL plus offline fixtures, not verified
  benchmark downloads; BERTScore remains an injection point; vLLM is injected
  rather than instantiated.
- P2: deterministic backends run; neural compressor APIs and target tokenizer
  integration remain unverified; CPC is explicitly unavailable.
- P3: resumability and Gate 1 reporting work; the Gate 1 result remains a
  deterministic offline smoke result, not evidence about the research
  population.
- P4: corpus tooling works, but the full data collection and leakage-safe
  metadata are still outstanding.

## Required work before P5

1. Install and pin the actual local target model, vLLM revision, tokenizer, and
   compressor versions; verify every callable signature.
2. Collect the real tiered datasets with licenses and stable source-document
   IDs.
3. Define disjoint `train`, `cal_a`, `cal_b`, `test`, and shift splits by both
   prompt ID and source document.
4. Run the full tiered grid under the configured API spend cap, with measured
   GPU seconds and provider-reported token usage.
5. Export corpus v1 parquet, validate it from a clean environment, complete
   the datasheet, and reproduce T2 from one command.
6. Reconcile the exact tier total (254,800 vs the plan’s approximate 262,000)
   in the research records.
7. Use only that validated real corpus as P5 input. Do not train or extract
   features from the fixture corpus.
