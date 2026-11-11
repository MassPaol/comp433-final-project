"""Report figures for the C1 self-supervised methods (k-means, SimSiam).

    python -m c1.ssl.figures                  # local rows
    python -m c1.ssl.figures --hw colab-t4    # rows from the Colab T4 run

Writes report/figures/c1_ssl_*.{pdf,png}. Config parameters are read from
configs.STAGES / configs.FINAL_SOURCE, never parsed from the config-id strings.
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


def _stage_table(summary, stage):
    s = summary[summary.index.isin(configs.STAGES[stage])].copy()
    s["kw"] = [configs.STAGES[stage][cid] for cid in s.index]
    return s


def fig_kmeans_sweep(summary):
    s = _stage_table(summary, "k1")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, mode, title in zip(axes, ["vote", "dist"],
                               ["(a) cluster = majority label", "(b) centroid distances → LR"]):
        for i, (fname, feats) in enumerate(configs.K_FEATURES.items()):
            rows = [(r.kw["k"], r.acc_mean, r.acc_std) for _, r in s.iterrows()
                    if r.kw["mode"] == mode and r.kw["features"] == feats]
            if not rows:
                continue
            k, m, sd = map(np.array, zip(*sorted(rows)))
            plots.errorbar_sweep(ax, k, m * 100, sd * 100, label=fname, color=plots.SERIES[i])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of clusters k (log scale)")
        ax.set_title(title)
    axes[0].set_ylabel("Validation accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle(f"k-means: features × k, mean ± std over {TUNE_LABEL}", y=1.02)
    fig.tight_layout()
    return fig


def fig_simsiam_epochs(summary):
    s = _stage_table(summary, "ss1")
    epochs = sorted({r.kw["epochs"] for _, r in s.iterrows()})
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, head in enumerate(configs.HEADS):
        rows = {r.kw["epochs"]: (r.acc_mean, r.acc_std) for _, r in s.iterrows()
                if r.kw["head"] == head}
        xs = [j for j, e in enumerate(epochs) if e in rows]
        if not xs:
            continue
        m = np.array([rows[epochs[j]][0] for j in xs]) * 100
        sd = np.array([rows[epochs[j]][1] for j in xs]) * 100
        plots.errorbar_sweep(ax, np.array(xs) + (i - 1) * 0.05, m, sd, label=head,
                             color=plots.SERIES[i])
    ax.set_xticks(range(len(epochs)), [("0 (random init)" if e == 0 else str(e)) for e in epochs])
    ax.set_xlabel("SimSiam pretraining epochs")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"SimSiam: pretraining length × head ({TUNE_LABEL})")
    ax.legend(loc="lower right", title="head")
    return fig


def fig_acc_vs_time(summary):
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
    ax.plot(t[front][order], a[front][order], "-", color=plots.TEXT, linewidth=1, label="Pareto front")
    for name, cid in picks.items():
        r = s.loc[cid]
        ax.scatter([r.time_mean], [r.acc_mean * 100], s=90, facecolor="none",
                   edgecolor=plots.TEXT, linewidths=1.5, zorder=5)
        ax.annotate(name.replace("_", " "), (r.time_mean, r.acc_mean * 100),
                    textcoords="offset points", xytext=(6, 6), fontsize=7, color=plots.TEXT)
    ax.set_xscale("log")
    ax.set_xlabel("Train + inference time per problem instance (s, log scale)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"C1 self-supervised: accuracy vs time ({TUNE_LABEL})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8)
    return fig


def fig_report(df, hw):
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
    for name, fig in [("c1_ssl_kmeans_sweep", fig_kmeans_sweep(summary)),
                      ("c1_ssl_simsiam_epochs", fig_simsiam_epochs(summary)),
                      ("c1_ssl_acc_vs_time", fig_acc_vs_time(summary)),
                      ("c1_ssl_report", fig_report(df, a.hw))]:
        for path in plots.save(fig, name, a.outdir):
            print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()
