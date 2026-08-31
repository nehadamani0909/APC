"""Generate the paper's figures and the sensitivity table from repository data.

Run from the repository root:
    uv run python apc-paper/make_figures.py

Every value written here is computed from runs/real_v1/ledger.jsonl and
runs/adherence_v1/adherence.csv. Nothing is hardcoded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# IEEEtran single-column text width is 3.5in. Sizing at 3.4in avoids any
# scaling in \includegraphics, so the 8pt figure text matches the 8pt the
# caption is set in rather than being shrunk to illegibility.
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
})

MARK = ["o", "s", "^", "D"]
LINE = ["-", "--", "-.", ":"]
LABEL = {
    "multifieldqa_en": "MultiFieldQA",
    "2wikimqa": "2WikiMQA",
    "gsm8k": "GSM8K",
    "qa": "MultiFieldQA",
    "multidoc_qa": "2WikiMQA",
    "reason": "GSM8K",
}

ledger = pd.read_json(ROOT / "runs/real_v1/ledger.jsonl", lines=True)
ledger["requested_b"] = ledger["requested_b"].round(4)
adh = pd.read_csv(ROOT / "runs/adherence_v1/adherence.csv")


def boot_ci(values, n=10000, seed=0):
    v = np.asarray(values, float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(v), (n, len(v)))
    s = v[draws].mean(axis=1)
    return float(v.mean()), float(np.quantile(s, .025)), float(np.quantile(s, .975))


# ---------------------------------------------------------------- figure 1
def fig_adherence() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    for i, fam in enumerate(["2wikimqa", "multifieldqa_en", "gsm8k"]):
        g = adh[adh["family"] == fam].groupby("requested_b")["signed_error"]
        ax.plot(g.mean().index, g.mean().values, marker=MARK[i], ls=LINE[i],
                ms=3.2, lw=1.0, color="0.15" if i == 0 else ("0.45" if i == 1 else "0.0"),
                label=LABEL[fam])
    pooled = adh.groupby("requested_b")["signed_error"].mean()
    ax.plot(pooled.index, pooled.values, color="k", lw=2.0, alpha=0.28,
            zorder=0, label="pooled")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("requested compression budget $b$")
    ax.set_ylabel("signed rate error $r-b$")
    ax.legend(frameon=False, loc="lower left", handlelength=2.2)
    fig.savefig(OUT / "fig_adherence.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_frontiers() -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    sub = ledger[ledger["family"] == "qa"]
    piv = sub.groupby(["prompt_id", "requested_b"])["quality"].mean().unstack()
    for pid in piv.index:
        ax.plot(piv.columns, piv.loc[pid], color="0.6", lw=0.5, alpha=0.85)
    mean = piv.mean(axis=0)
    ax.plot(mean.index, mean.values, "k-o", lw=1.5, ms=3.2, zorder=3,
            label="family mean")
    ax.set_xlabel("requested compression budget $b$")
    ax.set_ylabel(r"token-F1 $\rho(x,b)$")
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(OUT / "fig_frontiers.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_expansion() -> None:
    """Per-instance expansion, not a mean. The mean hides a heavy tail."""
    base = ledger[ledger["requested_b"] == 1.0].groupby("prompt_id")["T_out"].mean()
    m = ledger.merge(base.rename("base"), on="prompt_id")
    m = m[m["base"] > 0]
    m["ratio"] = m["T_out"] / m["base"]
    fams = ["qa", "reason", "multidoc_qa"]
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    rng = np.random.default_rng(0)
    for i, fam in enumerate(fams):
        v = m[(m["family"] == fam) & (m["requested_b"] == 0.2)]
        v = v.groupby("prompt_id")["ratio"].mean().values
        x = np.full(len(v), i) + rng.uniform(-0.11, 0.11, len(v))
        ax.scatter(x, v, s=13, facecolors="none", edgecolors="k", linewidths=0.7)
        ax.plot([i - 0.26, i + 0.26], [np.median(v)] * 2, color="k", lw=1.6)
        ax.plot([i - 0.20, i + 0.20], [v.mean()] * 2, color="0.45", lw=1.1, ls="--")
    ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels([LABEL[f] for f in fams])
    ax.set_ylabel(r"$T_{\mathrm{out}}(b{=}0.2)\,/\,T_{\mathrm{out}}(1)$")
    ax.plot([], [], color="k", lw=1.6, label="median")
    ax.plot([], [], color="0.45", lw=1.1, ls="--", label="mean")
    ax.legend(frameon=False, loc="upper right")
    fig.savefig(OUT / "fig_expansion.pdf")
    plt.close(fig)


# ------------------------------------------------- sensitivity / required k
def sensitivity_table() -> dict:
    """Var[S] against a split-half noise floor, and the k that would clear it.

    S(x) = rho(x,1.0) - rho(x,0.2). Reported instead of Var[b*] because b* is
    a threshold crossing of a noisy curve and is the noisier statistic of the
    two. Instances with rho(x,1)=0 are excluded: they carry no compression
    tolerance to measure.
    """
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for fam, g in ledger.groupby("family"):
        base = g[g["requested_b"] == 1.0].groupby("prompt_id")["quality"].mean()
        solvable = base[base > 0].index
        g = g[g["prompt_id"].isin(solvable)]
        if len(solvable) < 3:
            continue
        hi = g[g["requested_b"] == 1.0].groupby("prompt_id")["quality"].mean()
        lo = g[g["requested_b"] == 0.2].groupby("prompt_id")["quality"].mean()
        S = (hi - lo).dropna()
        floors = []
        for _ in range(200):
            h1, h2 = [], []
            for _pid, pg in g.groupby("prompt_id"):
                idx = sorted(pg["sample_idx"].unique())
                if len(idx) < 2:
                    continue
                a = set(rng.choice(idx, len(idx) // 2, replace=False))
                for store, sel in ((h1, a), (h2, set(idx) - a)):
                    p = pg[pg["sample_idx"].isin(sel)]
                    u = p[p["requested_b"] == 1.0]["quality"].mean()
                    w = p[p["requested_b"] == 0.2]["quality"].mean()
                    store.append(u - w)
            h1, h2 = np.array(h1), np.array(h2)
            ok = ~(np.isnan(h1) | np.isnan(h2))
            if ok.sum() > 1:
                floors.append(np.nanmean((h1[ok] - h2[ok]) ** 2) / 2)
        floor = float(np.mean(floors))
        var = float(S.var(ddof=1))
        out[str(fam)] = {
            "n": int(len(S)), "mean_S": float(S.mean()), "var_S": var,
            "floor_k3": floor, "required_k": 3 * (2 * floor) / var,
        }
    return out


if __name__ == "__main__":
    fig_adherence()
    fig_frontiers()
    fig_expansion()
    stats = sensitivity_table()
    (Path(__file__).resolve().parent / "sensitivity.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    for fam, s in stats.items():
        print(f"{fam:12s} n={s['n']:2d} mean_S={s['mean_S']:+.3f} "
              f"Var[S]={s['var_S']:.4f} floor={s['floor_k3']:.4f} "
              f"k>={s['required_k']:.0f}")
    print(f"\nwrote 3 figures to {OUT}")
