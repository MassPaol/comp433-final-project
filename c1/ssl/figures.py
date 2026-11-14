"""Report figures for the C1 self-supervised methods (Coates k-means pipeline, SimSiam).

    python -m c1.ssl.figures                  # local rows
    python -m c1.ssl.figures --hw colab-t4    # rows from the Colab T4 run

Writes report/figures/c1_ssl_*.{pdf,png}. Config parameters are read from configs.STAGES and
configs.FINAL_SOURCE, never parsed from config-id strings.
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from c1.ssl import configs  # noqa: E402
from c1.ssl.run import DEFAULT_HW, PERSON, SEED_SETS, hw_of, own_rows  # noqa: E402
from c1.ssl.select import pick_all, summarize_configs, tuning_rows  # noqa: E402
from common import plots  # noqa: E402

TUNE_LABEL = "tuning seeds 0–4"
PATCH_SIZES = [4, 6, 8]


def _stage_table(summary, stage):
    s = summary[summary.index.isin(configs.STAGES[stage])].copy()
    s["kw"] = [configs.STAGES[stage][cid] for cid in s.index]
    return s


def fig_kmeans_sweep(summary):
    """Accuracy against dictionary size K, one line per patch size (stage k1)."""
    s = _stage_table(summary, "k1")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, p in enumerate(PATCH_SIZES):
        rows = sorted((r.kw["n_clusters"], r.acc_mean, r.acc_std)
                      for _, r in s.iterrows() if r.kw["patch_size"] == p)
        if not rows:
            continue
        k, m, sd = map(np.array, zip(*rows))
        plots.errorbar_sweep(ax, k, m * 100, sd * 100, label=f"patch {p}", color=plots.SERIES[i])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Dictionary size K (log scale)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Coates et al.: K × patch size ({TUNE_LABEL})", fontsize=10)
    ax.legend(loc="lower right", title="patch size")
    fig.tight_layout()
    return fig


def fig_kmeans_preprocessing(summary):
    """Preprocessing sensitivity on the chosen K and patch size (stage k2), with error bars."""
    s = _stage_table(summary, "k2")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(s))
    ax.errorbar(x, s.acc_mean * 100, yerr=s.acc_std * 100, fmt="o", color=plots.SERIES[0],
                capsize=3, markersize=5)
    ax.set_xticks(x, [c.replace("k2_", "") for c in s.index], rotation=30, ha="right")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Coates et al.: contrast and whitening constants ({TUNE_LABEL})", fontsize=10)
    fig.tight_layout()
    return fig


def fig_simsiam_epochs(summary):
    """Accuracy against SimSiam pretraining epochs; epochs=0 is the random-init baseline (ss1)."""
    s = _stage_table(summary, "ss1")
    s = s.assign(epochs=[r.kw["epochs"] for _, r in s.iterrows()]).sort_values("epochs")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(s))
    plots.errorbar_sweep(ax, x, s.acc_mean.to_numpy() * 100, s.acc_std.to_numpy() * 100,
                         label="SimSiam + linear SVM", color=plots.SERIES[1])
    ax.set_xticks(x, [("0\n(random init)" if e == 0 else str(e)) for e in s.epochs])
    ax.set_xlabel("SimSiam pretraining epochs")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"SimSiam: pretraining length ({TUNE_LABEL})", fontsize=10)
    return fig


def fig_simsiam_arch(summary):
    """Heatmap of backbone width x projection dimension (stage ss2)."""
    s = _stage_table(summary, "ss2")
    widths = sorted({r.kw["width"] for _, r in s.iterrows()})
    projs = sorted({r.kw["proj_dim"] for _, r in s.iterrows()})
    grid = np.full((len(widths), len(projs)), np.nan)
    for _, r in s.iterrows():
        grid[widths.index(r.kw["width"]), projs.index(r.kw["proj_dim"])] = r.acc_mean * 100
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    im = ax.imshow(grid, cmap="viridis")
    for i in range(len(widths)):
        for j in range(len(projs)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        color="white", fontsize=7)
    ax.set_xticks(range(len(projs)), projs)
    ax.set_yticks(range(len(widths)), widths)
    ax.set_xlabel("Projection dimension")
    ax.set_ylabel("Backbone width")
    ax.set_title(f"SimSiam: architecture ({TUNE_LABEL})", fontsize=10)
    fig.colorbar(im, ax=ax, label="Validation accuracy (%)")
    fig.tight_layout()
    return fig


def fig_acc_vs_time(summary):
    """Accuracy against train + inference time for every tuning config, with the Pareto front."""
    s = tuning_rows(summary)
    picks = pick_all(s)
    t, a = s["time_mean"].to_numpy(), s["acc_mean"].to_numpy() * 100
    method = np.array([configs.method_of(c) for c in s.index])
    front = plots.pareto_front(t, a)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, m in enumerate(["kmeans", "simsiam"]):
        sel = method == m
        ax.scatter(t[sel], a[sel], s=16, color=plots.SERIES[i], alpha=0.6, linewidths=0,
                   label=f"{m} configs (n = {sel.sum()})")
    order = np.argsort(t[front])
    ax.plot(t[front][order], a[front][order], "-", color=plots.TEXT, linewidth=1,
            label="Pareto front")
    for name, cid in picks.items():
        r = s.loc[cid]
        ax.scatter([r.time_mean], [r.acc_mean * 100], s=90, facecolor="none",
                   edgecolor=plots.TEXT, linewidths=1.5, zorder=5)
        ax.annotate(name.replace("_", " "), (r.time_mean, r.acc_mean * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=7, color=plots.TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("Train + inference time per problem instance (s, log scale)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"C1 self-supervised: accuracy vs time ({TUNE_LABEL})", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    return fig


def fig_report(df, hw):
    """Frozen best and fast configs on the report seeds: one dot per seed, diamond = mean ± std."""
    seeds = SEED_SETS["report"]
    df = own_rows(df)
    names = list(configs.FINAL_SOURCE)
    d = df[df["seed"].isin(seeds) & (df["notes"].map(hw_of) == hw) & df["config_id"].isin(names)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rng = np.random.RandomState(0)
    for i, name in enumerate(names):
        v = d[d["config_id"] == name]["val_acc"].to_numpy() * 100
        if len(v) == 0:
            continue
        color = plots.SERIES[0] if name.startswith("kmeans") else plots.SERIES[1]
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", color=color, alpha=0.6, markersize=5)
        ax.errorbar([i], [v.mean()], yerr=[v.std(ddof=0)], fmt="D", color=plots.TEXT,
                    capsize=4, markersize=6)
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names])
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Report seeds 10–19 (dots = seeds, diamond = mean ± std)", fontsize=10)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Make the C1 self-supervised report figures.")
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--results", default=f"results/{PERSON}.csv")
    p.add_argument("--outdir", default="report/figures")
    a = p.parse_args(argv)

    plots.apply_style()
    df = pd.read_csv(a.results)
    summary = summarize_configs(df, SEED_SETS["tune"], a.hw)
    figs = [("c1_ssl_kmeans_sweep", fig_kmeans_sweep(summary)),
            ("c1_ssl_kmeans_preprocessing", fig_kmeans_preprocessing(summary)),
            ("c1_ssl_simsiam_epochs", fig_simsiam_epochs(summary)),
            ("c1_ssl_simsiam_arch", fig_simsiam_arch(summary)),
            ("c1_ssl_acc_vs_time", fig_acc_vs_time(summary)),
            ("c1_ssl_report", fig_report(df, a.hw))]
    for name, fig in figs:
        for path in plots.save(fig, name, a.outdir):
            print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()