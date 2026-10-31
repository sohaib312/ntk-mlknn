"""Summarise results/<dataset>/ into tables, a markdown summary and two figures.

* summary_<criterion>.csv  test metrics of each family's CV-selected configuration
* ntk_effect.csv           paired-bootstrap differences against ML-kNN + PCA (test split)
* optimism.csv             test score of the CV pick vs the best test score in the grid
* summary.md               the tables above as markdown, for the README
* figures/alpha_sensitivity.png, figures/ntk_effect.png

Usage: python -m ntk_mlknn.report
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import metrics
from .data import DATASETS
from .experiment import CRITERIA, FAMILIES, RESULTS

BASE = "ML-kNN + PCA"
COMPARED = ["ML-kNN", "NTK (raw)", "NTK (distance)", "Hybrid (value fusion)", "Hybrid (kernel distance)"]
SHOWN = ["hamming_loss", "subset_accuracy", "micro_f1", "macro_f1", "ranking_loss", "avg_precision"]
N_BOOT, SEED = 2000, 0

# Chart tokens (light surface): text inks, recessive grid, categorical slots 1-2.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = {"Hybrid (value fusion)": "#2a78d6", "Hybrid (kernel distance)": "#eb6834"}


def _datasets():
    return [d for d in DATASETS if (RESULTS / d / "selected.csv").exists()]


def _per_instance(Y, P):
    """Per-test-instance Hamming loss, and average precision on rankable rows."""
    ok = metrics.rankable(Y)
    return (Y != (P >= 0.5)).mean(1), metrics.ranking_per_instance(Y[ok], P[ok])["avg_precision"]


def _boot_ci(diff: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    return tuple(np.percentile(diff[idx].mean(1), [2.5, 97.5]))


def ntk_effect(ds: str, criterion: str) -> list[dict]:
    z = np.load(RESULTS / ds / "selected_scores.npz")
    sel = pd.read_csv(RESULTS / ds / "selected.csv").set_index(["family", "criterion"])
    Y = z["Y_test"].astype(bool)
    hl0, ap0 = _per_instance(Y, z[f"{BASE}|{criterion}"])
    rng = np.random.default_rng(SEED)
    rows = []
    for fam in COMPARED:
        hl, ap = _per_instance(Y, z[f"{fam}|{criterion}"])
        r = sel.loc[(fam, criterion)]
        row = {"dataset": ds, "criterion": criterion, "family": fam, "alpha": r.alpha, "kind": r.kind}
        for name, a, b in [("hamming_loss", hl, hl0), ("avg_precision", ap, ap0)]:
            lo, hi = _boot_ci(a - b, rng)
            row.update({f"base_{name}": b.mean(), f"d_{name}": (a - b).mean(), f"d_{name}_lo": lo,
                        f"d_{name}_hi": hi, f"d_{name}_rel": (a - b).mean() / b.mean()})
        rows.append(row)
    return rows


def optimism(ds: str) -> list[dict]:
    grid = pd.read_csv(RESULTS / ds / "grid.csv")
    sel = pd.read_csv(RESULTS / ds / "selected.csv").set_index(["family", "criterion"])
    rows = []
    for fam in [BASE, "Hybrid (value fusion)", "Hybrid (kernel distance)"]:
        sub = grid[FAMILIES[fam](grid)]
        oracle = sub.loc[sub.test_hamming_loss.idxmin()]
        rows.append({
            "dataset": ds, "family": fam, "n_configs": len(sub),
            "test_hl_cv_selected": sel.loc[(fam, "hamming_loss")].test_hamming_loss,
            "test_hl_best_in_grid": oracle.test_hamming_loss, "alpha_of_best_in_grid": oracle.alpha,
        })
    return rows


def alpha_curve(ds: str) -> pd.DataFrame:
    """Best mean-CV Hamming loss at each NTK weight alpha, for both fusion styles."""
    grid = pd.read_csv(RESULTS / ds / "grid.csv")
    out = []
    for fam in SERIES:
        sub = grid[FAMILIES[fam](grid)]
        best = sub.groupby("alpha").cv_hamming_loss.min()
        out.append(pd.DataFrame({"family": fam, "alpha": best.index, "cv_hamming_loss": best.values,
                                 "rel_change": best.values / best.loc[0.0] - 1}))
    return pd.concat(out)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def plot_alpha(curves: dict[str, pd.DataFrame], path):
    n = len(curves)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.6 * rows), squeeze=False, facecolor=SURFACE)
    for ax, (ds, cur) in zip(axes.flat, curves.items()):
        _style(ax)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        for fam, color in SERIES.items():
            c = cur[cur.family == fam]
            ax.plot(c.alpha, 100 * c.rel_change, color=color, linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=fam)
        ax.set_title(ds, color=INK, fontsize=10, loc="left")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("CV Hamming loss vs. alpha=0 (%)", color=INK_2, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("NTK weight alpha", color=INK_2, fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=9, labelcolor=INK)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_effect(eff: pd.DataFrame, path):
    """Change vs. ML-kNN + PCA as a % of its test score, so datasets share one scale."""
    eff = eff[(eff.criterion == "hamming_loss") & eff.family.isin(SERIES)]
    dsets = list(dict.fromkeys(eff.dataset))
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.55 * len(dsets) + 1.4), facecolor=SURFACE, sharey=True)
    for ax, (m, title) in zip(axes, [("hamming_loss", "Hamming loss (lower is better)"),
                                     ("avg_precision", "Average precision (higher is better)")]):
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.axvline(0, color=INK_2, linewidth=0.8)
        for j, (fam, color) in enumerate(SERIES.items()):
            e = eff[eff.family == fam].set_index("dataset").loc[dsets]
            mid, lo, hi = (100 * e[f"d_{m}{s}"] / e[f"base_{m}"] for s in ["", "_lo", "_hi"])
            y = np.arange(len(dsets)) + (j - 0.5) * 0.28
            ax.errorbar(mid, y, xerr=[mid - lo, hi - mid], fmt="o", color=color, markersize=5, elinewidth=2,
                        capsize=0, markeredgecolor=SURFACE, label=fam)
        ax.set_yticks(np.arange(len(dsets)), dsets)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("% change vs. ML-kNN + PCA on the test split (95% bootstrap CI)", color=INK_2, fontsize=8)
    axes[0].invert_yaxis()  # shared y: invert once, first dataset on top
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _md(df: pd.DataFrame, floatfmt="{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    dsets = _datasets()
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    md = ["# Results", ""]

    for crit in ["hamming_loss", "avg_precision"]:
        rows = []
        for ds in dsets:
            sel = pd.read_csv(RESULTS / ds / "selected.csv")
            sel = sel[sel.criterion == crit]
            for _, r in sel.iterrows():
                rows.append({"dataset": ds, "method": r.family,
                             **{m: r[f"test_{m}"] for m in SHOWN},
                             "config": f"{r.kind}, pca={int(r.n_comp) or '-'}, alpha={r.alpha:g}, "
                                       f"gamma={r.gamma:g}, k={int(r.k)}, s={r.s:g}"})
        tab = pd.DataFrame(rows)
        tab.to_csv(RESULTS / f"summary_{crit}.csv", index=False)
        md += [f"## Test results, configurations selected by 5-fold CV {crit.replace('_', ' ')}", "", _md(tab), ""]

    eff = pd.DataFrame([r for ds in dsets for c in ["hamming_loss", "avg_precision"] for r in ntk_effect(ds, c)])
    eff.to_csv(RESULTS / "ntk_effect.csv", index=False)
    show = eff[eff.criterion == "hamming_loss"][["dataset", "family", "alpha", "d_hamming_loss", "d_hamming_loss_lo",
                                                 "d_hamming_loss_hi", "d_avg_precision", "d_avg_precision_lo",
                                                 "d_avg_precision_hi"]]
    md += [f"## Differences vs. {BASE} (test split, paired bootstrap, {N_BOOT} resamples; selection by CV Hamming loss)",
           "", _md(show), ""]

    # Better / no clear difference / worse, by whether the 95% CI excludes 0.
    def tally(g, m):
        better = (g[f"d_{m}_hi"] < 0) if m == "hamming_loss" else (g[f"d_{m}_lo"] > 0)
        worse = (g[f"d_{m}_lo"] > 0) if m == "hamming_loss" else (g[f"d_{m}_hi"] < 0)
        return f"{int(better.sum())} / {int((~better & ~worse).sum())} / {int(worse.sum())}"

    verdict = pd.DataFrame([
        {"selected by CV": crit.replace("_", " "), "family": fam,
         **{f"{m.replace('_', ' ')}: better / same / worse": tally(eff[(eff.criterion == crit) & (eff.family == fam)], m)
            for m in ["hamming_loss", "avg_precision"]}}
        for crit in CRITERIA for fam in COMPARED
    ])
    verdict.to_csv(RESULTS / "verdict.csv", index=False)
    md += [f"## How often each method beats {BASE} (number of datasets; 95% CI excludes 0)", "", _md(verdict), ""]

    opt = pd.DataFrame([r for ds in dsets for r in optimism(ds)])
    opt.to_csv(RESULTS / "optimism.csv", index=False)
    md += ["## Choosing by test score vs. by cross-validation", "", _md(opt), ""]

    curves = {ds: alpha_curve(ds) for ds in dsets}
    pd.concat([c.assign(dataset=ds) for ds, c in curves.items()]).to_csv(RESULTS / "alpha_curve.csv", index=False)
    plot_alpha(curves, RESULTS / "figures" / "alpha_sensitivity.png")
    plot_effect(eff, RESULTS / "figures" / "ntk_effect.png")

    meta = pd.DataFrame([json.loads((RESULTS / ds / "meta.json").read_text()) for ds in dsets])
    # Reference points: predicting no labels at all for every test example.
    Ys = {ds: np.load(RESULTS / ds / "selected_scores.npz")["Y_test"] for ds in dsets}
    meta["all_negative_test_hl"] = [float(Ys[ds].mean()) for ds in dsets]
    meta["all_negative_test_subset_acc"] = [float((Ys[ds].sum(1) == 0).mean()) for ds in dsets]
    md += ["## Datasets, run times, and the all-negative predictor", "", _md(meta, "{:.4f}"), ""]
    (RESULTS / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
