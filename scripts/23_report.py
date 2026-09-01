"""Turn analysis output into a findings report a paper can be written from.

Deliberately conservative: where a precondition for interpreting a number is
not met, this says so instead of printing the number bare. A plausible figure
with no valid measurement behind it is the most expensive thing this pipeline
could emit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return "n/a" if x != x else f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    r = json.loads(args.results.read_text())

    L: list[str] = [
        "# Findings — real grid",
        "",
        f"Ledger: `{r['ledger']}` · rows **{r['rows']}** · "
        f"instances **{r['instances']}**",
        f"Families: {', '.join(r['families'])} · "
        f"backends: {', '.join(r['backends'])} · "
        f"targets: {', '.join(r['target_models'])}",
        "",
        "> Local inference is zero-priced, so USD is 0 by construction. Every",
        "> monetary statement below is in **token-cost units** "
        "`T_in + (c_out/c_in)·T_out`",
        "> and is therefore a claim about the price *ratio*, not about one",
        "> vendor's bill.",
        "",
        "## 1. Quality by budget",
        "",
        "| backend | b | mean quality | n |",
        "|---|---|---|---|",
    ]
    for row in r.get("quality_by_budget", []):
        L.append(f"| {row['backend']} | {row['requested_b']:.2f} | "
                 f"{_fmt(row['mean'])} | {row['count']} |")

    L += ["", "## 2. Gate 1 — within-family frontier heterogeneity", ""]
    for backend, entry in r.get("gate1", {}).items():
        L.append(f"### backend: `{backend}`")
        problems = entry.get("validity_problems") or []
        if problems:
            L += ["", "**⚠ Validity preconditions NOT met — the verdict below is "
                  "not interpretable:**", ""]
            L += [f"- {p}" for p in problems]
            L += ["", "Do not report this as a Gate 1 FAIL; it is an "
                  "uninterpretable measurement, which is a different thing and "
                  "has a different fix: a stronger target model, or instances "
                  "the model can actually solve uncompressed.", ""]
        else:
            L += ["", "Validity preconditions met.", ""]
        L += ["| ε | Var[b*] | 95% CI | noise floor | 2×floor | passes |",
              "|---|---|---|---|---|---|"]
        for eps in (0.02, 0.05, 0.10):
            e = entry.get(f"eps_{eps}")
            if not e:
                continue
            ci = e.get("var_ci") or [None, None]
            L.append(
                f"| {eps} | {_fmt(e['var_b_star'],4)} | "
                f"[{_fmt(ci[0],4)}, {_fmt(ci[1],4)}] | "
                f"{_fmt(e['noise_floor'],4)} | {_fmt(e['threshold_2x_floor'],4)} | "
                f"**{e['passes']}** |")
        nm = entry.get("non_monotonicity", {})
        L += ["", f"Non-monotonicity: naive and monotone-safe `b*` disagree on "
              f"**{_fmt(100*nm.get('p_disagree',float('nan')),1)}%** of "
              f"{nm.get('n','?')} instances "
              f"(mean gap {_fmt(nm.get('mean_gap_when_disagree'))}).", ""]

    L += ["## 3. Rate adherence (C4) — requested vs realised", "",
          "| backend | requested b | realised (mean) | sd | signed error | "
          "95% CI | n |", "|---|---|---|---|---|---|---|"]
    for row in r.get("rate_adherence", []):
        L.append(f"| {row['backend']} | {row['requested_b']:.2f} | "
                 f"{_fmt(row['realised_mean'])} | {_fmt(row['realised_sd'])} | "
                 f"{_fmt(row['signed_error'])} | "
                 f"[{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}] | {row['n']} |")

    L += ["", "## 4. Output expansion (C2 mechanism)", "",
          "| family | b | T_out(b)/T_out(1) | 95% CI | n |",
          "|---|---|---|---|---|"]
    for row in r.get("output_expansion", []):
        L.append(f"| {row['family']} | {row['requested_b']:.2f} | "
                 f"{_fmt(row['expansion_mean'])} | "
                 f"[{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}] | {row['n']} |")

    L += ["", "## 5. Cheapest safe budget ≠ smallest safe budget (C2 headline)",
          "", "| c_out/c_in | instances | divergent | % | mean saving when "
          "divergent | 95% CI |", "|---|---|---|---|---|---|"]
    for row in r.get("cheapest_vs_smallest", []):
        L.append(f"| {row['c_out_over_c_in']} | {row['n_instances']} | "
                 f"{row['n_divergent']} | {_fmt(row['pct_divergent'],1)}% | "
                 f"{_fmt(row['mean_saving_when_divergent'])} | "
                 f"[{_fmt(row['ci_lo'])}, {_fmt(row['ci_hi'])}] |")

    L += ["", "---", "",
          "## What this does and does not license",
          "",
          "Claims are only supportable here for the single target model, the",
          "single compressor checkpoint, and the context-length window this",
          "grid used. See `PAPER_GUIDE.md` §5 for the limitations that must",
          "appear in the paper, and `docs/claims_ledger.md` for the ledger to",
          "update from these numbers.", ""]

    args.out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
