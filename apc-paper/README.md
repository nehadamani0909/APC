# Requested Is Not Realised (IEEE conference paper)

A 6-page IEEE conference paper reporting the measurements from the `real_v1`
grid: LLMLingua-2 rate adherence, compression-induced output expansion and its
effect on the cost-optimal budget, and a power analysis of per-instance
frontier estimation.

Final PDF: **`main.pdf`** (6 pages).

## Compiling

The IEEE class and BibTeX style are vendored here, so no TeX package
installation is needed beyond a base LaTeX distribution:

```bash
cd apc-paper
pdflatex main.tex
bibtex   main
pdflatex main.tex
pdflatex main.tex
```

`latexmk -pdf main.tex` also works if available.

Two notes on this machine's toolchain. The local TeX Live is BasicTeX, which
ships the Times metrics the IEEE template needs but not URW Courier or
Helvetica, and `tlmgr` refuses to fetch them across a release boundary
(local 2025 against remote 2026). `main.tex` therefore maps `\ttdefault` to
Computer Modern Typewriter. Times remains the body face, so the paper matches
the IEEE format; only the few verbatim strings differ from a full
installation. Remove those two `\renewcommand` lines if you compile somewhere
with the full font set.

`IEEEtran.cls` (v1.8b) and `IEEEtran.bst` (v1.14) were downloaded from CTAN
and are included for reproducibility.

## Regenerating the figures

The three figures and the sensitivity statistics are computed from the
repository ledgers, not stored by hand:

```bash
# from the repository root
uv run python apc-paper/make_figures.py
```

This writes `figures/fig_adherence.pdf`, `figures/fig_frontiers.pdf`,
`figures/fig_expansion.pdf`, `sensitivity.json` (Table V) and
`correction.json` (Table IV, the held-out adherence-correction experiment).

## Source of truth

Every number in the paper traces to one of these, and all were re-checked
against the repository after the final compile:

| Source | Used for |
|---|---|
| `runs/real_v1/ledger.jsonl` | 630-cell grid: quality, realised rate, token counts, timings |
| `runs/real_v1/manifest.json` | run configuration reported in Table I |
| `runs/adherence_v1/adherence.csv` | 1,044-point adherence sweep (Table IV, Fig. 1) |
| `results/real_v1/results.json` | expansion, cost divergence, Gate-1 statistics |
| `results/real_v1/ASSESSMENT.md` | interpretation and scope boundaries |
| `PAPER_GUIDE.md` | limitations that had to appear, and venue scoping |
| `docs/claims_ledger.md` | which claims the evidence supports |
| `apc-paper/sensitivity.json` | Var[S], noise floor, required k (Table V) |
| `apc-paper/correction.json` | held-out adherence correction, 200 splits (Table IV) |

## Scope of the claims

All claims are restricted to what the completed experiments establish: one
target model (Llama-3.2-3B-Instruct, Q4_K_M), one compressor checkpoint
(LLMLingua-2 `bert-base-multilingual-cased-meetingbank`), three task families,
and contexts of 500 to 3500 tokens.

The paper reports its negative and bounded results rather than only the
favourable ones. Specifically: output expansion is significant in the mean but
never moved the cost optimum in the measured data (0 of 25 instances at every
price ratio tested), the expansion magnitude rests on a heavy-tailed
distribution over ten instances and is labelled exploratory, and per-instance
heterogeneity could not be established at three samples per cell. The
`required k` column of Table V is an extrapolation, not a measurement, and is
labelled as such.

Following `PAPER_GUIDE.md`, the conformal risk-control and frontier-predictor
contributions are deliberately **not** claimed here. They are cited as context
only. These proceedings are archival, so publishing them now would constrain a
later submission built on them.

## Citations

All thirteen references were checked against the arXiv API on 2026-08-31 by
querying each identifier and comparing the returned title, first author and
date. This corrected one entry carried over from the project's earlier
bibliography: arXiv:2507.22931 is "Enhancing RAG Efficiency with Adaptive
Context Compression" (Guo, Cheng and Zhang), not the "ACC-RAG" title
previously recorded. Two references (arXiv:2603.23525, arXiv:2603.23527) are
preprints that are not peer reviewed; `refs.bib` says so, and Section II
states the methodological difference (they use truncation as the compression
operator) that the paper's positioning depends on.

## Files

```
main.tex           paper source
refs.bib           bibliography (all entries arXiv-verified)
main.pdf           compiled output, 6 pages
make_figures.py    regenerates figures and sensitivity.json from the ledgers
sensitivity.json   Var[S], noise floor and required k per family
correction.json    held-out adherence-correction results, 200 splits
figures/           three vector figures used by the paper
IEEEtran.cls       IEEE conference class v1.8b (CTAN)
IEEEtran.bst       IEEE BibTeX style v1.14 (CTAN)
```
