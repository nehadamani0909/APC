"""Generate the paper's figures from a real ledger. Vector output only.

Greyscale-safe: every series is distinguished by marker and linestyle as well
as colour, because these proceedings are frequently printed in greyscale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
MARKERS = ["o", "s", "^", "D", "v", "P"]
STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def fig1_frontiers(frame: pd.DataFrame, out: Path) -> None:
    """Per-instance quality-vs-rate frontiers, faceted by family."""
    fams = sorted(frame["family"].unique())
    fig, axes = plt.subplots(
        1, len(fams), figsize=(2.4 * len(fams), 2.2), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, fams, strict=False):
        sub = frame[frame["family"] == fam]
        piv = sub.groupby(["prompt_id", "requested_b"])["quality"].mean().unstack()
        for pid in piv.index:
            ax.plot(
                piv.columns, piv.loc[pid], color="0.75", lw=0.4, alpha=0.6, zorder=1
            )
        mean = piv.mean(axis=0)
        ax.plot(
            mean.index, mean.values, "k-o", lw=1.4, ms=3, zorder=3,
            label="family mean",
        )
        ax.set_title(f"{fam} (n={piv.shape[0]})")
        ax.set_xlabel("requested budget $b$")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel(r"quality $\rho(x,b)$")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig1_frontiers.pdf")
    plt.close(fig)


def fig2_adherence(frame: pd.DataFrame, out: Path) -> None:
    """Requested vs realised rate. The gap from the diagonal is C4."""
    backends = sorted(frame["backend"].unique())
    fig, axes = plt.subplots(1, len(backends), figsize=(2.6 * len(backends), 2.4),
                             sharey=True, squeeze=False)
    for ax, backend in zip(axes[0], backends, strict=False):
        sub = frame[frame["backend"] == backend]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="requested = realised")
        for i, fam in enumerate(sorted(sub["family"].unique())):
            f = sub[sub["family"] == fam]
            g = f.groupby("requested_b")["realised_r"]
            m, sd = g.mean(), g.std(ddof=1).fillna(0.0)
            ax.errorbar(m.index, m.values, yerr=sd.values, marker=MARKERS[i % 6],
                        ls=STYLES[i % 6], ms=3.5, lw=1.0, capsize=2, label=fam)
        ax.set_xlabel("requested budget $b$")
        ax.set_title(backend)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0][0].set_ylabel("realised rate $r$")
    axes[0][0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig2_adherence.pdf")
    plt.close(fig)


def fig3_cost(frame: pd.DataFrame, out: Path, ratios=(1, 2, 3, 4, 5)) -> None:
    """Total cost vs budget as the output/input price ratio varies.

    A non-monotone line here is the paper: it means an aggressive budget can
    be both safe and more expensive than a milder one.
    """
    fams = sorted(frame["family"].unique())
    fig, axes = plt.subplots(
        1, len(fams), figsize=(2.4 * len(fams), 2.2), squeeze=False
    )
    for ax, fam in zip(axes[0], fams, strict=False):
        sub = frame[frame["family"] == fam]
        for i, ratio in enumerate(ratios):
            cost = sub.assign(c=sub["T_in"] + ratio * sub["T_out"])
            g = cost.groupby("requested_b")["c"].mean()
            g = g / g.loc[1.0] if 1.0 in g.index else g / g.max()
            ax.plot(g.index, g.values, marker=MARKERS[i % 6], ls=STYLES[i % 6],
                    ms=3, lw=1.0, label=rf"$c_{{out}}/c_{{in}}={ratio}$")
        ax.axhline(1.0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("requested budget $b$")
        ax.set_title(fam)
        ax.grid(alpha=0.3, lw=0.4)
    axes[0][0].set_ylabel("total cost\n(relative to $b=1$)")
    axes[0][-1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig3_cost.pdf")
    plt.close(fig)


def fig4_expansion(frame: pd.DataFrame, out: Path) -> None:
    """Output expansion T_out(b)/T_out(1) per family -- C2's mechanism."""
    base = (frame[frame["requested_b"] == 1.0]
            .groupby("prompt_id")["T_out"].mean().rename("base"))
    m = frame.merge(base, on="prompt_id")
    m = m[m["base"] > 0]
    m["ratio"] = m["T_out"] / m["base"]
    fig, ax = plt.subplots(figsize=(3.2, 2.3))
    for i, fam in enumerate(sorted(m["family"].unique())):
        g = m[m["family"] == fam].groupby("requested_b")["ratio"]
        mean, sem = g.mean(), g.sem()
        ax.errorbar(mean.index, mean.values, yerr=1.96 * sem.values,
                    marker=MARKERS[i % 6], ls=STYLES[i % 6], ms=3.5, lw=1.0,
                    capsize=2, label=fam)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("requested budget $b$")
    ax.set_ylabel(r"output expansion $T_{out}(b)/T_{out}(1)$")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig4_expansion.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frame = pd.read_json(args.ledger, lines=True)
    frame["requested_b"] = frame["requested_b"].round(4)
    fig1_frontiers(frame, args.out)
    fig2_adherence(frame, args.out)
    fig3_cost(frame, args.out)
    fig4_expansion(frame, args.out)
    print(f"wrote 4 vector figures to {args.out}")


if __name__ == "__main__":
    main()
