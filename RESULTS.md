# Results index

Everything produced by the overnight run of 2026-08-31. Nothing in this file
is copied from the planning documents; every artefact below was generated
from a real measurement.

## Headline results

Full interpretation in `results/real_v1/ASSESSMENT.md`. Three findings.

**1. Rate adherence (publishable).** LLMLingua-2 systematically
*over*-compresses at every requested budget, measured in the target's
tokenizer. Replicated at scale in a sweep needing no target-model calls:
**1,044 measurements, 174 instances, 0 errors**, signed error −0.025 (b=0.2)
to −0.053 (b=0.65), every 95% CI excluding zero. The error is an **inverted U
in b** (not a constant offset, so no scalar fixes it) and is
**family-dependent** — −0.074 on `2wikimqa` vs −0.025 on `gsm8k` at b=0.65.
A token-savings figure reported against a requested rate is a claim about a
request, not a result.

**2. Output expansion is real but does not flip the cost optimum here.**
`qa` expands **1.68×** at b=0.2 (CI [1.27, 2.14]) and `reason` **1.28×**
(CI [1.09, 1.48]) — both excluding 1.0. Yet the cheapest safe budget equalled
the smallest safe budget for **all 25** evaluable instances at every price
ratio in [1,5], because input reduction dominates ~40:1 (T_in 2405→498 against
T_out 30→40). This *bounds* the TAAC compression paradox rather than
contradicting it: the paradox needs a high output-to-input token ratio.

**3. k=3 is too few — and so is the plan's k=5.** Gate 1 fails on every
family, but heterogeneity is visible in the raw curves (one instance degrades
0.98→0.73, another collapses 0.53→0.11). The per-instance label is simply
noise-dominated. Solving for the k at which between-instance variance clears
twice the split-half floor gives **k ≥ 20** for `qa`, ≥14 for `multidoc_qa`,
≥6 for `reason`. `APC_06 §2` plans k=5. Graded metrics also help: token-F1's
noise floor is 0.065 against exact-match's 0.162, **2.5× lower for free**.

Not supported, and must not be written: any heterogeneity claim, any
predictor/guarantee/policy result, any cross-model or cross-backend claim.
`multidoc_qa` is degenerate (ρ(x,1)=0.105) and `reason` has 6/10 instances
solvable; `multifieldqa_en` is the only family with no validity problems.

## Where things are

| Path | What it is |
|---|---|
| `runs/real_v1/ledger.jsonl` | The grid. One row per (instance, budget, sample) with quality, realised rate, `T_in`, `T_out`, latency, compression time |
| `runs/real_v1/outputs` | Raw generations, keyed by hash, so answer preservation stays computable |
| `runs/real_v1/manifest.json` | Exact run configuration: model, `num_ctx`, compressor checkpoint, device, token window, tokenizer fallback flag |
| `runs/real_v1/failures.jsonl` | Cells that raised, with the error. Absent or empty means none did |
| `runs/adherence_v1/adherence.csv` | Contribution C4 at scale: requested vs realised rate, no target-LLM calls |
| `results/real_v1/` | Analysis output: `results.json` plus CSV tables |
| `results/real_v1/FINDINGS.md` | The human-readable report to write the paper from |
| `results/real_v1/figures/` | Vector (PDF) figures |
| `PAPER_GUIDE.md` | How to turn all of this into the regional-IEEE submission |

## How to regenerate

```bash
uv sync --extra real --extra dev
export HF_HOME="/Volumes/Adith HDD/apc-work/hf-cache"

# the grid (needs ollama serve with OLLAMA_NUM_PARALLEL=4 and llama3.2 pulled)
uv run python scripts/20_real_grid.py --out runs/real_v1 --device mps \
  --n-per-family 17 --samples 3 --workers 2 --min-tok 500 --max-tok 3500

# rate adherence, no LLM required
uv run python scripts/24_adherence.py --out runs/adherence_v1 --device cpu --n 60

# analysis, report, figures
uv run python scripts/21_analyse.py --ledger runs/real_v1/ledger.jsonl --out results/real_v1
uv run python scripts/23_report.py  --results results/real_v1/results.json \
                                    --out results/real_v1/FINDINGS.md
uv run python scripts/22_figures.py --ledger runs/real_v1/ledger.jsonl \
                                    --out results/real_v1/figures
```

## Code changes made during this run

Four defects were found and fixed. Two of them would have silently corrupted
results rather than failing visibly, which is the dangerous kind.

1. **`frontier/harness/metrics.py` — code sandbox dead on macOS.**
   `RLIMIT_AS` capped *address space* at 512MB; CPython on arm64 Darwin
   reserves far more than that before running any bytecode, so `preexec_fn`
   raised, `sandboxed_code_check` caught it, and **every** candidate scored
   0.0 — correct ones included. HumanEval/MBPP would have read as a flat
   zero that looks like a real measurement. The limit is now skipped on
   Darwin and kept elsewhere, and `verify_sandbox()` refuses to start a run
   whose sandbox rejects a known-correct canary.

2. **`frontier/harness/tasks.py` — GSM8K answers unparseable.**
   `parse` cut the response at the first blank line. An instruction-tuned
   model opens with a preamble sentence, so the cut discarded the entire
   derivation and left a fragment with no digits; `parse` then returned the
   raw text, which never exact-matches a numeral. **The whole `reason`
   family scored 0.0 at every compressed budget** — indistinguishable from a
   compression cliff. Now truncates only at genuine next-question markers and
   prefers GSM8K's own `####` delimiter. Two regression tests added.

3. **`frontier/harness/ollama_client.py` (new) — silent context truncation.**
   Ollama drops prompt overflow without erroring. At `b = 1.0` that would
   mean the *uncompressed* reference quality was measured on a truncated
   prompt, contaminating every frontier in the corpus against a bad baseline.
   The client now raises `ContextOverflow` when `prompt_eval_count` reaches
   the usable window, so the b=1.0 cell can be trusted.

4. **`frontier/compress/llmlingua_base.py` — engine load race.**
   Grid workers each raced to build their own `PromptCompressor`: 90s and a
   full model copy each on MPS, for an object they immediately share.
   Double-checked locking added.

Test suite: **134 passed, 5 skipped** (was 126 passed, 8 failed).

## Measurements taken while tuning the run

Recorded because they determine what scale is feasible on this hardware, and
because two of them are counterintuitive.

| Finding | Number |
|---|---|
| LLMLingua-2 on MPS vs CPU | **17× faster** (0.54s vs 9.42s mean per compression) |
| Ollama parallelism, *shared* context | 4 workers ≈ 1.3× faster than 1 |
| Ollama parallelism, *distinct* contexts | 4 workers **1.4× SLOWER** (16.8s vs 12.0s per cell) — concurrent prefills contend for one GPU |
| Generation throughput, llama3.2:3B Q4_K_M | ~9–12 tok/s; prefill ~140–220 tok/s |

The parallelism result is why the grid runs with `--workers 2`, not 8. On a
single-GPU machine, prompt compression research is prefill-bound, and prefill
does not parallelise.
