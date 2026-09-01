"""Analyse a real grid ledger: Gate 1, rate adherence (C4), output expansion (C2).

Every number this emits is computed from the ledger passed in. Nothing is
hardcoded, and where a quantity cannot be estimated the report says so rather
than substituting a default -- the failure mode that matters here is a
plausible-looking number with no measurement behind it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frontier.eval.gate1 import (
    BUDGETS,
    _bootstrap_variance,
    _labels,
    _noise_floor,
    check_validity,
)


def _boot_ci(values, stat=np.mean, n=10000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        point = float(stat(values)) if len(values) else float("nan")
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n, len(values)))
    samples = stat(values[draws], axis=1)
    return (
        float(stat(values)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def rate_adherence(frame: pd.DataFrame) -> pd.DataFrame:
    """C4: requested vs realised rate, per backend x requested budget."""
    rows = []
    for (backend, b), grp in frame.groupby(["backend", "requested_b"]):
        realised = grp["realised_r"].astype(float)
        err = realised - float(b)
        mean, lo, hi = _boot_ci(err.values)
        rows.append({
            "backend": backend, "requested_b": float(b),
            "realised_mean": float(realised.mean()),
            "realised_sd": float(realised.std(ddof=1)) if len(realised) > 1 else 0.0,
            "signed_error": mean, "ci_lo": lo, "ci_hi": hi,
            "abs_error": float(err.abs().mean()), "n": int(len(grp)),
        })
    return pd.DataFrame(rows).sort_values(["backend", "requested_b"])


def output_expansion(frame: pd.DataFrame) -> pd.DataFrame:
    """C2: T_out(b) / T_out(1.0), per family. The compression paradox test."""
    rows = []
    base = (frame[frame["requested_b"] == 1.0]
            .groupby("prompt_id")["T_out"].mean().rename("T_out_base"))
    merged = frame.merge(base, on="prompt_id", how="inner")
    merged = merged[merged["T_out_base"] > 0]
    for (family, b), grp in merged.groupby(["family", "requested_b"]):
        ratio = (grp["T_out"] / grp["T_out_base"]).values
        mean, lo, hi = _boot_ci(ratio)
        rows.append({
            "family": family, "requested_b": float(b),
            "expansion_mean": mean, "ci_lo": lo, "ci_hi": hi,
            "median": float(np.median(ratio)), "n": int(len(grp)),
        })
    return pd.DataFrame(rows).sort_values(["family", "requested_b"])


def true_cost_curve(frame: pd.DataFrame, price_ratios=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """C2's decision-relevant object: total cost vs budget as c_out/c_in varies.

    Local inference is zero-priced, so USD in the ledger is 0 by construction.
    Cost is therefore reconstructed from measured token counts under an
    explicit price ratio -- which is what makes the conclusion a statement
    about the ratio rather than about one vendor's price list.
    """
    rows = []
    for ratio in price_ratios:
        for (family, b), grp in frame.groupby(["family", "requested_b"]):
            cost = grp["T_in"].astype(float) + ratio * grp["T_out"].astype(float)
            mean, lo, hi = _boot_ci(cost.values)
            rows.append({
                "c_out_over_c_in": ratio, "family": family, "requested_b": float(b),
                "cost_units_mean": mean, "ci_lo": lo, "ci_hi": hi, "n": int(len(grp)),
            })
    return pd.DataFrame(rows)


def cheapest_vs_smallest(
    frame: pd.DataFrame, epsilon=0.05, price_ratios=(1, 2, 3, 4, 5)
):
    """How often is the cheapest safe budget NOT the smallest safe budget?

    This is C2's headline claim, stated as a frequency. It is only meaningful
    on instances that are solvable uncompressed, so those are the denominator.
    """
    q = frame.groupby(["prompt_id", "requested_b"])["quality"].mean().unstack()
    tin = frame.groupby(["prompt_id", "requested_b"])["T_in"].mean().unstack()
    tout = frame.groupby(["prompt_id", "requested_b"])["T_out"].mean().unstack()
    budgets = [b for b in BUDGETS if b in q.columns]
    out = []
    for ratio in price_ratios:
        divergent = total = 0
        savings = []
        for pid in q.index:
            vals = {b: q.loc[pid, b] for b in budgets}
            if not np.isfinite(vals.get(1.0, np.nan)) or vals[1.0] <= 0:
                continue  # unsolvable uncompressed: no tolerance to measure
            safe = [
                b
                for b in budgets
                if all(vals[x] >= vals[1.0] - epsilon for x in budgets if x >= b)
            ]
            if not safe:
                continue
            total += 1
            smallest = min(safe)
            costs = {b: tin.loc[pid, b] + ratio * tout.loc[pid, b] for b in safe}
            cheapest = min(costs, key=costs.get)
            if cheapest != smallest:
                divergent += 1
                if costs[smallest] > 0:
                    savings.append(1 - costs[cheapest] / costs[smallest])
        mean, lo, hi = _boot_ci(savings) if savings else (0.0, 0.0, 0.0)
        out.append({
            "c_out_over_c_in": ratio, "n_instances": total,
            "n_divergent": divergent,
            "pct_divergent": 100.0 * divergent / total if total else float("nan"),
            "mean_saving_when_divergent": mean, "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(out)


def non_monotonicity(frame: pd.DataFrame, epsilon=0.05) -> dict:
    """E1b: how often does the naive smallest-safe budget disagree with the
    monotone-safe one? Non-monotone curves are what make naive b* unstable."""
    lab = _labels(frame, epsilon)
    disagree = float((lab["b_star"] != lab["b_star_naive"]).mean())
    gap = (lab["b_star"] - lab["b_star_naive"]).abs()
    return {
        "p_disagree": disagree,
        "mean_gap_when_disagree": (
            float(gap[gap > 0].mean()) if (gap > 0).any() else 0.0
        ),
        "n": int(len(lab)),
    }


def per_family_gate(frame: pd.DataFrame, epsilon: float = 0.05) -> dict:
    """Gate 1 per family, restricted to instances the model can actually do.

    The global gate mixes families, and one family the target model cannot
    solve at all drags the whole verdict into "not interpretable". That is
    the correct global answer but it hides the useful one: within the family
    where a frontier *is* measurable, is there heterogeneity beyond noise?

    Instances with rho(x,1) = 0 are dropped, not because they are
    inconvenient but because they carry no compression tolerance to measure:
    every budget trivially satisfies rho(b) >= 0 - epsilon, so their b* is an
    artefact of the model failing, not of compression.
    """
    out: dict[str, object] = {}
    for family, grp in frame.groupby("family"):
        base = grp[grp["requested_b"] == 1.0].groupby("prompt_id")["quality"].mean()
        solvable = set(base[base > 0].index)
        entry: dict[str, object] = {
            "n_instances": int(grp["prompt_id"].nunique()),
            "n_solvable": len(solvable),
            "mean_quality_uncompressed": float(base.mean()) if len(base) else None,
        }
        sub = grp[grp["prompt_id"].isin(solvable)]
        if len(solvable) < 3:
            entry["verdict"] = (
                "not measurable: fewer than 3 instances are solvable "
                "uncompressed, so this family carries no frontier to measure"
            )
            out[str(family)] = entry
            continue
        entry["validity_problems"] = check_validity(sub)
        lab = _labels(sub, epsilon)
        floor = _noise_floor(sub, epsilon)
        var, lo, hi = _bootstrap_variance(lab["b_star"].tolist())
        entry.update({
            "var_b_star": var, "var_ci": [lo, hi], "noise_floor": floor,
            "passes": None if floor is None else bool(lo > 2 * floor),
            "b_star_hist": {str(k): int(v) for k, v in
                            lab["b_star"].value_counts().sort_index().items()},
        })
        out[str(family)] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epsilon", type=float, default=0.05)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_json(args.ledger, lines=True)
    frame["requested_b"] = frame["requested_b"].round(4)
    results: dict[str, object] = {
        "ledger": str(args.ledger), "rows": int(len(frame)),
        "instances": int(frame["prompt_id"].nunique()),
        "families": sorted(frame["family"].unique().tolist()),
        "backends": sorted(frame["backend"].unique().tolist()),
        "target_models": sorted(frame["target_model"].unique().tolist()),
    }

    # --- Gate 1, per backend, with the validity preconditions kept in front
    gate: dict[str, object] = {}
    for backend, grp in frame.groupby("backend"):
        entry: dict[str, object] = {"validity_problems": check_validity(grp)}
        for eps in (0.02, 0.05, 0.10):
            lab = _labels(grp, eps)
            floor = _noise_floor(grp, eps)
            var, lo, hi = _bootstrap_variance(lab["b_star"].tolist())
            entry[f"eps_{eps}"] = {
                "var_b_star": var, "var_ci": [lo, hi], "noise_floor": floor,
                "threshold_2x_floor": None if floor is None else 2 * floor,
                "passes": None if floor is None else bool(lo > 2 * floor),
                "b_star_hist": {str(k): int(v) for k, v in
                                lab["b_star"].value_counts().sort_index().items()},
            }
            # within-family, which is the test that actually matters
            per_family = {}
            for fam, fgrp in lab.groupby("family"):
                if len(fgrp) > 1:
                    var_f, lo_f, hi_f = _bootstrap_variance(fgrp["b_star"].tolist())
                    per_family[fam] = {
                        "var": var_f,
                        "ci": [lo_f, hi_f],
                        "n": int(len(fgrp)),
                    }
            entry[f"eps_{eps}"]["within_family"] = per_family
        entry["non_monotonicity"] = non_monotonicity(grp, args.epsilon)
        gate[str(backend)] = entry
    results["gate1"] = gate
    results["gate1_per_family_solvable"] = per_family_gate(frame, args.epsilon)

    tables = {
        "rate_adherence": rate_adherence(frame),
        "output_expansion": output_expansion(frame),
        "true_cost_curve": true_cost_curve(frame),
        "cheapest_vs_smallest": cheapest_vs_smallest(frame, args.epsilon),
    }
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        results[name] = json.loads(table.to_json(orient="records"))

    # mean quality by budget: the first thing a reader checks
    qual = frame.groupby(["backend", "requested_b"])["quality"].agg(["mean", "count"])
    qual.to_csv(args.out / "quality_by_budget.csv")
    results["quality_by_budget"] = json.loads(
        qual.reset_index().to_json(orient="records")
    )

    (args.out / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    headline = {
        k: v
        for k, v in results.items()
        if k in ("rows", "instances", "families", "backends")
    }
    print(json.dumps(headline, indent=2))
    print(f"\nwrote {args.out}/results.json and {len(tables)+1} CSV tables")


if __name__ == "__main__":
    main()
