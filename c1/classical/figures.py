"""Report figures for the C1 classical method.

    python -m c1.classical.figures                  # local rows
    python -m c1.classical.figures --hw colab-t4    # rows from the Colab T4 re-run

Writes report/figures/c1_classical_*.{pdf,png}. Config parameters are read
from configs.STAGES / configs.FINAL, never parsed from the config-id strings.
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from c1.classical import configs  # noqa: E402
from c1.classical.run import DEFAULT_HW, PERSON, SEED_SETS, hw_of, own_rows  # noqa: E402
from c1.classical.select import select, summarize_configs  # noqa: E402
from common import plots  # noqa: E402

TUNE_LABEL = "tuning seeds 0–4"
FEATURE_LABELS = {"scat": "Scattering", "hog": "HOG", "raw": "Raw pixels",
                  "rawpca": "Raw + PCA(50)", "hsv": "HSV histograms", "lbp": "LBP"}
CLF_LABELS = {"lr": "LR", "linsvm": "Linear SVM", "rbfsvm": "RBF SVM", "rf": "Random forest",
              "hgb": "Hist. GBoost", "xgb": "XGBoost", "knn": "kNN"}


def _stage_table(summary, stage):
    """summary rows for one stage, joined with each config's kwargs."""
    s = summary[summary.index.isin(configs.STAGES[stage])].copy()
    s["kw"] = [configs.STAGES[stage][cid] for cid in s.index]
    return s


def _feat_params(kw):
    (name, params), = kw["features"]
    return name, params


def fig_s1_heatmap(summary):
    s = _stage_table(summary, "s1")
    feats = list(configs.S1_FEATURES)
    clfs = configs.S1_CLASSIFIERS
    grid = np.array([[s.loc[f"s1_{f}_{c}", "acc_mean"] * 100 for c in clfs] for f in feats])
    order = np.argsort(-grid.max(axis=1))
    grid, feats = grid[order], [feats[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    im = ax.imshow(grid, cmap=plots.SEQUENTIAL_CMAP, vmin=60, vmax=90, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(clfs)), [CLF_LABELS[c] for c in clfs], rotation=30, ha="right")
    ax.set_yticks(range(len(feats)), [FEATURE_LABELS[f] for f in feats])
    for i in range(len(feats)):
        for j in range(len(clfs)):
            v = grid[i, j]
            bold = v == grid[i].max()
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 80 else plots.TEXT,
                    fontweight="bold" if bold else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Validation accuracy (%)")
    ax.set_title(f"S1: features × classifiers, default settings ({TUNE_LABEL}; bold = row best)")
    return fig


def fig_sweeps(summary):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    # (a) LR regularization C per feature (S2)
    ax = axes[0]
    s2 = _stage_table(summary, "s2")
    for i, feat in enumerate(["scat", "hog", "raw"]):
        rows = [(kw["clf_params"]["C"], r.acc_mean, r.acc_std) for cid, r in s2.iterrows()
                for kw in [r.kw] if kw["clf"] == "lr" and cid.startswith(f"s2_{feat}_")]
        C, m, sd = map(np.array, zip(*sorted(rows)))
        plots.errorbar_sweep(ax, C, m * 100, sd * 100, label=FEATURE_LABELS[feat],
                             color=plots.SERIES[i])
    ax.set_xscale("log")
    ax.set_xlabel("LR inverse regularization C (log scale)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("(a) Logistic regression: C")
    ax.legend(loc="lower left")

    # (b) scattering scale J and orientations L (S3, RGB, LR-CV)
    ax = axes[1]
    s3 = _stage_table(summary, "s3")
    for i, L in enumerate([4, 6, 8]):
        rows = []
        for _, r in s3.iterrows():
            name, p = _feat_params(r.kw)
            if name == "scattering" and p["L"] == L and p["color"] and p.get("pool", 1) == 1:
                rows.append((p["J"], r.acc_mean, r.acc_std))
        J, m, sd = map(np.array, zip(*sorted(rows)))
        plots.errorbar_sweep(ax, J + (i - 1) * 0.06, m * 100, sd * 100, label=f"L = {L}",
                             color=plots.SERIES[i])
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Scattering scale J (output map 32/2^J px)")
    ax.set_title("(b) Scattering: J and orientations L")
    ax.legend(loc="lower left")

    # (c) HOG cell and block size (S3, gray, 9 orientations, RBF-SVM)
    ax = axes[2]
    for i, ppc in enumerate([4, 8]):
        rows = []
        for _, r in s3.iterrows():
            name, p = _feat_params(r.kw)
            if (name == "hog" and p["pixels_per_cell"] == ppc and not p["color"]
                    and p["orientations"] == 9):
                rows.append((p["cells_per_block"], r.acc_mean, r.acc_std))
        b, m, sd = map(np.array, zip(*sorted(rows)))
        plots.errorbar_sweep(ax, b + (i - 0.5) * 0.06, m * 100, sd * 100,
                             label=f"{ppc}×{ppc} px cells", color=plots.SERIES[i])
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("HOG cells per block (side)")
    ax.set_title("(c) HOG: cell and block size")
    ax.legend(loc="lower right")

    for ax in axes:
        ax.set_ylim(56, 95)
    fig.suptitle(f"Hyperparameter sweeps, mean ± std over {TUNE_LABEL}", y=1.02)
    fig.tight_layout()
    return fig


def fig_acc_vs_time(summary):
    s = summary[summary.index.str.match(r"s[1-4]_")]
    best, fast = select(s)
    t = s["time_mean"].to_numpy()
    a = s["acc_mean"].to_numpy() * 100
    front = plots.pareto_front(t, a)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(t[~front], a[~front], s=14, color=plots.NEUTRAL, alpha=0.6,
               label=f"All tuning configs (n = {len(s)})", linewidths=0)
    order = np.argsort(t[front])
    ax.plot(t[front][order], a[front][order], "-o", color=plots.SERIES[0], markersize=5,
            label="Pareto front")
    # Short labels in the empty band above the Pareto front; config ids go in the caption.
    for cid, name in ((best, "best"), (fast, "fast")):
        r = s.loc[cid]
        ax.scatter([r.time_mean], [r.acc_mean * 100], s=90, facecolor="none",
                   edgecolor=plots.TEXT, linewidths=1.5, zorder=5)
        ax.annotate(f"{name}: {r.acc_mean * 100:.1f}%, {r.time_mean:.2f} s",
                    (r.time_mean, r.acc_mean * 100), textcoords="offset points",
                    xytext=(-8, 10), ha="right", fontsize=8, color=plots.TEXT, zorder=6)
    ax.set_ylim(top=93.5)
    ax.set_xscale("log")
    ax.set_xlabel("Train + inference time per problem instance (s, log scale)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"C1 classical: accuracy vs time, all stages ({TUNE_LABEL})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    return fig


def fig_report_pairs(df, hw):
    seeds = SEED_SETS["report"]
    df = own_rows(df)
    d = df[df["seed"].isin(seeds) & (df["notes"].map(hw_of) == hw)
           & df["config_id"].isin(["best", "fast"])]
    wide = d.pivot_table(index=["seed", "class_pair"], columns="config_id", values="val_acc")
    wide = wide.reset_index().sort_values("best")
    y = np.arange(len(wide))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    has_fast = "fast" in wide
    if has_fast:
        ax.hlines(y, wide["fast"] * 100, wide["best"] * 100, color=plots.GRID, linewidth=2)
        ax.plot(wide["fast"] * 100, y, "s", color=plots.SERIES[1], label="fast", markersize=7)
    ax.plot(wide["best"] * 100, y, "o", color=plots.SERIES[0], label="best", markersize=7)
    ax.set_yticks(y, [f"{p} (seed {s})" for s, p in zip(wide["seed"], wide["class_pair"])])
    ax.set_xlabel("Validation accuracy (%)")
    title = f"Report seeds 10–19: best {wide['best'].mean() * 100:.1f} ± {wide['best'].std(ddof=0) * 100:.1f}%"
    if has_fast:
        diff = (wide["best"] - wide["fast"]) * 100
        title += (f", fast {wide['fast'].mean() * 100:.1f} ± {wide['fast'].std(ddof=0) * 100:.1f}%"
                  f"\npaired best − fast: {diff.mean():+.1f} pp, best wins {(diff > 0).sum()}/"
                  f"{len(diff)}, ties {(diff == 0).sum()}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right")
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Make the C1 classical report figures.")
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--results", default=f"results/{PERSON}.csv")
    p.add_argument("--outdir", default="report/figures")
    a = p.parse_args(argv)

    plots.apply_style()
    df = pd.read_csv(a.results)
    summary = summarize_configs(df, SEED_SETS["tune"], a.hw)
    for name, fig in [("c1_classical_s1_heatmap", fig_s1_heatmap(summary)),
                      ("c1_classical_sweeps", fig_sweeps(summary)),
                      ("c1_classical_acc_vs_time", fig_acc_vs_time(summary)),
                      ("c1_classical_report_pairs", fig_report_pairs(df, a.hw))]:
        for path in plots.save(fig, name, a.outdir):
            print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()
