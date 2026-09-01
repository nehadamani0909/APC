# Writing the regional-IEEE paper from this repository

**Scope.** This is the guide for turning `runs/real_v1` into a 6-page
double-column IEEE conference paper. It is deliberately *not* the ACL/EMNLP
plan in `APC_03`/`APC_08` — that plan describes five contributions and needs
a grid roughly a hundred times this one. Trying to compress it into six pages
produces a survey of your own intentions. Pick one contribution, land it, and
keep the rest for the longer paper.

---

## 1. Which venue, and what it actually requires

Candidates: **INDICON, TENCON, CONECCT, INDISCON, UPCON, ANTS, SILCON,
ASIANCON, Bombay Section Symposium**.

What these reviewers reward:

| They want | They do not require |
|---|---|
| A clearly stated problem a practitioner recognises | State of the art |
| A working implementation | Novel theory |
| **Real** experiments — not synthetic, not simulated | Large scale |
| At least one honest baseline | Multiple models or datasets |
| Reproducibility (code, seeds, configs) | Human evaluation |
| Clean figures and correct statistics | A new benchmark |

The single most common desk-reject at these venues is *"the evaluation is on
toy or synthetic data."* That is precisely the failure mode the pre-existing
`data/corpus/v1` fixture would have triggered, and it is why the run in
`runs/real_v1` — real compressor, real benchmark, real target model — is the
whole difference between submittable and not.

**Page budget.** 6 pages including references, IEEE conference template
(`IEEEtran`, `\documentclass[conference]{IEEEtran}`). Do not use the journal
class. Anonymise only if the venue says double-blind; most of these are not.

---

## 2. The one contribution to lead with

**Lead with C2 — the true-cost / output-expansion result.** Ordered reasons:

1. **It is legible to a non-NLP reviewer.** "Compressing your prompt harder
   can increase your bill" is counterintuitive, immediately practical, and
   needs no background in conformal prediction to review.
2. **It does not depend on the predictor working.** It is a property of the
   measured grid, so it survives even if Gate 1 fails.
3. **The evidence is already in the ledger.** `T_in`, `T_out`, `usd_*` are
   recorded per row; `scripts/21_analyse.py` computes the divergence
   frequency directly.

**Support it with C4 — rate adherence.** Requested rate ≠ realised rate is a
two-paragraph section plus one scatter plot, it needs no target-model calls
at all, and it lets you make a fair methodological point: token-savings
numbers reported against a *requested* rate are claims about a request, not a
result.

**Hold back C1 (conformal risk control) and C3 (frontier prediction).** C1 is
the highest-novelty asset in the project and deserves the ACL/EMNLP paper.
C3 needs far more data than one night produces. Mention both in Future Work
in one sentence each — enough to establish precedence of idea, not enough to
burn the novelty.

---

## 3. Structure, with page budget

| § | Content | Pages |
|---|---|---|
| I | **Introduction** — compressors take rate as a user-set hyperparameter; nobody asks what rate to request; cost is billed on input *and* output | 0.75 |
| II | **Related work** — LLMLingua/LLMLingua-2, AdaComp, Nagle et al., ACC-RAG. Use the verified table in `APC_02`; do not cite from memory | 0.75 |
| III | **Method / measurement protocol** — budgets, families, target model, graded quality, the monotone-safe label, the cost model | 1.25 |
| IV | **Experimental setup** — datasets, splits, k, seeds, hardware, the exact limitations in §5 below | 0.75 |
| V | **Results** — Fig. 1 frontiers, Fig. 2 adherence, Fig. 3 cost-vs-budget, Table I divergence frequency | 1.75 |
| VI | **Limitations and future work** | 0.4 |
| VII | **Conclusion** | 0.2 |
| — | References | 0.4 |

Cut §II before you cut §V. Reviewers at these venues forgive a thin related-
work section; they do not forgive thin results.

---

## 4. Figures and tables — generate all of these from `runs/real_v1`

Everything below comes from `scripts/21_analyse.py`; nothing is hand-made.

- **Fig. 1 — per-instance frontiers.** Spaghetti plot of `ρ(x,b)` vs `b`,
  faceted by family, family mean overlaid. This is the motivating image;
  it is the one figure a skimming reviewer will actually look at.
- **Fig. 2 — rate adherence.** Requested `b` on x, realised `r` on y, with
  the `y = x` identity line. The gap between the cloud and the diagonal *is*
  contribution C4. One panel per backend.
- **Fig. 3 — cost vs budget.** `T_in + ρ·T_out` against `b`, one line per
  price ratio `c_out/c_in ∈ {1..5}`. If any line is non-monotone, that
  single panel carries the paper.
- **Table I — cheapest ≠ smallest.** From `cheapest_vs_smallest.csv`:
  price ratio, % of instances where the cheapest safe budget is not the
  smallest safe budget, and the mean saving when they diverge.
- **Table II — output expansion.** `T_out(b)/T_out(1)` per family, with CIs.

Plot rules: vector (PDF) not raster; font size ≥ 8pt *after* scaling to
column width; do not encode meaning in colour alone (these proceedings are
often printed greyscale); every error bar must say in the caption what it is
(here: BCa bootstrap, 10,000 resamples, paired over prompts).

---

## 5. Limitations you must state — this is not optional

Stating these costs you nothing at a regional venue and protects you if the
work is later extended. Omitting them is how a paper gets retracted or a
reviewer gets annoyed.

1. **One target model** (Llama-3.2-3B-Instruct, Q4_K_M via Ollama). Output
   expansion is known to be model-dependent, so report the *direction* you
   measured and do not generalise it to other models.
2. **One compressor family** (LLMLingua-2, BERT-base multilingual
   checkpoint). Cross-backend transfer is not tested.
3. **MeetingBank is the compressor's training domain.** None of the three
   families used here is MeetingBank, which is good — say so explicitly,
   because it pre-empts the obvious objection.
4. **Tokenizer proxy.** Realised rate is measured with the GPT-2 BPE, not
   Llama-3.2's tokenizer, which is gated. This shifts the absolute level of
   the adherence gap but not its shape across budgets. The run manifest
   records `tokenizer_fallback` so this is auditable.
5. **k = 3 samples per cell** at temperature 0.7. The noise floor is
   measured by split-half resampling and reported; do not claim any
   heterogeneity result that does not clear it.
6. **Context-length window.** Only instances between 500 and 4500 tokens are
   used. `hotpotqa` (median 14k tokens) and `qmsum` (13.5k) were excluded
   rather than truncated, because truncating them would drop the supporting
   document and force `ρ(x,1)=0`. Say this — it is a methodological
   *strength*, and a reviewer who spots the exclusion without an explanation
   will assume the worst.
7. **Local inference is zero-priced.** USD in the ledger is 0 by
   construction. All monetary conclusions are stated as a function of the
   price ratio `c_out/c_in`, never as one vendor's bill.

---

## 6. Claims discipline

`docs/claims_ledger.md` is the contract. **A claim goes in the paper only
when its ledger row is supported by `runs/real_v1`.** Update the ledger from
the analysis output, not from memory, and not from the planning documents —
those describe an experiment that has not been run at this scale.

Carry over the forbidden-claims table from `APC_03 §4` verbatim. The three
that matter most here:

- ❌ "We introduce adaptive prompt compression" → AdaComp, ACC-RAG own this.
- ❌ "Our method reduces inference cost by X%" using input tokens only →
  this is the exact error the paper is arguing against; making it in your
  own abstract is fatal.
- ❌ "We are the first to observe output expansion" → the TAAC series
  observed it. You are the first to *measure it with a real neural
  compressor and task-native metrics* rather than truncation and embedding
  similarity. That is still a real and defensible delta — say that instead.

---

## 7. The archival warning — decide this before you submit

These proceedings go to **IEEE Xplore and are archival**. Publishing C2 there
constrains the later ACL/EMNLP paper: you will need substantial new material
(conventionally ~30%+) and must cite your own prior paper. Standard practice
permits the extension; doing it accidentally does not.

So decide now, deliberately:

- Keep **C1 and C3 entirely out** of the IEEE paper. One future-work
  sentence each is fine; a method section is not.
- The IEEE paper is the *measurement*: frontiers, adherence, output
  expansion, cost divergence.
- The ACL paper is the *method*: the predictor, the conformal guarantee, the
  end-to-end policy comparison against B2b.

That split is clean, defensible, and leaves the stronger paper intact.

---

## 8. Pre-submission checklist

- [ ] Every number in the paper traces to a file under `runs/real_v1`
- [ ] `uv run pytest` passes; commit hash recorded in the paper
- [ ] `paper/` contains no output carrying the `SCAFFOLD` banner
      (`tests/test_no_fabricated_results.py` enforces this — keep it green)
- [ ] Claims ledger updated from the real run
- [ ] All seven limitations from §5 stated in the paper
- [ ] Figures legible at column width in greyscale
- [ ] IEEE conference template, page limit respected, references complete
- [ ] Every prior work cited has been *read*, not just cited — `APC_02`
      flags which ones were verified
- [ ] Code and configs released, or a statement saying when they will be
