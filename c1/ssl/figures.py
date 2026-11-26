"""Report figures for the C1 self-supervised methods (Coates k-means, SimSiam).

    python -m c1.ssl.figures --hw colab-t4 --results <csv>

Writes report/figures/c1_ssl_*.{pdf,png}. Config parameters are read from configs.STAGES and
configs.FINAL_SOURCE, never parsed from config-id strings. Each figure is drawn only when the
rows it needs exist, so this can be re-run after every stage.
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from c1.ssl import configs  # noqa: E402
from c1.ssl.run import (DEFAULT_HW, PERSON, SEED_SETS, default_results_dir, hw_of,  # noqa: E402
                        own_rows)
from c1.ssl.select import paired, pick_all, summarize_configs, tuning_rows  # noqa: E402
from common import plots  # noqa: E402

TUNE_LABEL = "tuning seeds 0–4"
PATCH_SIZES = [4, 6, 8]
ALVIN = plots.METHOD_COLORS["c1_ssl"]
SS_ABLATION_STAGES = ("ss2", "ss3", "ss4", "ss5", "ss6")


def _stage_table(summary, stage):
    s = summary[summary.index.isin(configs.STAGES[stage])].copy()
    s["kw"] = [configs.STAGES[stage][cid] for cid in s.index]
    return s


def fig_kmeans_sweep(summary):
    """Accuracy against dictionary size K, one line per patch size (stage k1)."""
    s = _stage_table(summary, "k1")
    if s.empty:
        return None
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
    """Preprocessing sensitivity on the chosen K and patch size (stage k2)."""
    s = _stage_table(summary, "k2")
    if s.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(s))
    ax.errorbar(x, s.acc_mean * 100, yerr=s.acc_std * 100, fmt="o", color=ALVIN,
                capsize=3, markersize=5)
    ax.set_xticks(x, [c.replace("k2_", "") for c in s.index], rotation=30, ha="right")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Coates et al.: contrast and whitening constants ({TUNE_LABEL})", fontsize=10)
    fig.tight_layout()
    return fig


def fig_simsiam_epochs(summary):
    """Accuracy against SimSiam pretraining epochs; epochs=0 is the random-init baseline (ss1)."""
    s = _stage_table(summary, "ss1")
    if s.empty:
        return None
    s = s.assign(epochs=[r.kw["epochs"] for _, r in s.iterrows()]).sort_values("epochs")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = np.arange(len(s))
    plots.errorbar_sweep(ax, x, s.acc_mean.to_numpy() * 100, s.acc_std.to_numpy() * 100,
                         label="SimSiam + linear SVM", color=ALVIN)
    ax.set_xticks(x, [("0\n(random init)" if e == 0 else str(e)) for e in s.epochs])
    ax.set_xlabel("SimSiam pretraining epochs")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"SimSiam: pretraining length ({TUNE_LABEL})", fontsize=10)
    fig.tight_layout()
    return fig


def fig_simsiam_ablation(df, summary, seeds, hw):
    """Paired change in accuracy of every SimSiam config vs the SS1 base config (stages ss2-ss6)."""
    ref = configs.SS_REF
    if ref not in summary.index:
        return None
    rows, bounds = [], []
    for st in SS_ABLATION_STAGES:
        res = [(c, paired(df, c, ref, seeds, hw)) for c in configs.STAGES[st] if c in summary.index]
        res = sorted([(c, r) for c, r in res if r["n"] == len(seeds)],
                     key=lambda t: t[1]["mean_diff_pp"])
        if res:
            bounds.append((st, len(rows)))
            rows += res
    if not rows:
        return None
    d = np.array([r["mean_diff_pp"] for _, r in rows])
    err = np.array([np.std(list(r["per_seed"].values())) for _, r in rows])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.6, 0.26 * len(rows) + 1.6))
    ax.barh(y, d, xerr=err, color=[ALVIN if v > 0 else plots.NEUTRAL for v in d],
            capsize=2, error_kw={"elinewidth": 1})
    ax.set_yticks(y, [f"{c}  ({r['wins']}/{r['n']})" for c, r in rows], fontsize=7)
    ax.axvline(0, color=plots.TEXT_SECONDARY, linewidth=1)
    for st, start in bounds:
        ax.axhline(start - 0.5, color=plots.GRID, linewidth=1)
        ax.text(1.01, start, st.upper(), transform=ax.get_yaxis_transform(), fontsize=8,
                color=plots.TEXT_SECONDARY, va="center")
    ax.set_xlabel(f"paired Δ accuracy vs {ref} (pp)")
    ax.set_title(f"SimSiam: one change at a time ({TUNE_LABEL}); bars mean ± std, wins/seeds in brackets",
                 fontsize=9)
    fig.tight_layout()
    return fig


def fig_acc_vs_time(summary):
    """Accuracy against train + inference time for every tuning config, with the Pareto front."""
    s = tuning_rows(summary)
    if len(s) < 2:
        return None
    picks = pick_all(s)
    t, a = s["time_mean"].to_numpy(), s["acc_mean"].to_numpy() * 100
    method = np.array([configs.method_of(c) for c in s.index])
    front = plots.pareto_front(t, a)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m, marker in [("kmeans", "o"), ("simsiam", "s")]:
        sel = method == m
        ax.scatter(t[sel], a[sel], s=16, color=ALVIN, marker=marker, alpha=0.6,
                   linewidths=0, label=f"{m} configs (n = {sel.sum()})")
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
    fig.tight_layout()
    return fig


def fig_report(df, hw):
    """Frozen best and fast configs on the report seeds: one dot per seed, diamond = mean ± std."""
    seeds = SEED_SETS["report"]
    names = list(configs.FINAL_SOURCE)
    d = own_rows(df)
    d = d[d["seed"].isin(seeds) & (d["notes"].map(hw_of) == hw) & d["config_id"].isin(names)]
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rng = np.random.RandomState(0)
    for i, name in enumerate(names):
        v = d[d["config_id"] == name]["val_acc"].to_numpy() * 100
        if len(v) == 0:
            continue
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", color=ALVIN, alpha=0.6, markersize=5)
        ax.errorbar([i], [v.mean()], yerr=[v.std(ddof=0)], fmt="D", color=plots.TEXT,
                    capsize=4, markersize=6)
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names])
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Report seeds 10–19 (dots = seeds, diamond = mean ± std)", fontsize=10)
    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Make the C1 self-supervised report figures.")
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--results", default=None, help="default: the results file for --hw")
    p.add_argument("--outdir", default="report/figures")
    a = p.parse_args(argv)

    plots.apply_style()
    df = pd.read_csv(a.results or f"{default_results_dir(a.hw)}/{PERSON}.csv")
    seeds = SEED_SETS["tune"]
    summary = summarize_configs(df, seeds, a.hw)
    figs = [("c1_ssl_kmeans_sweep", fig_kmeans_sweep(summary)),
            ("c1_ssl_kmeans_preprocessing", fig_kmeans_preprocessing(summary)),
            ("c1_ssl_simsiam_epochs", fig_simsiam_epochs(summary)),
            ("c1_ssl_simsiam_ablation", fig_simsiam_ablation(df, summary, seeds, a.hw)),
            ("c1_ssl_acc_vs_time", fig_acc_vs_time(summary)),
            ("c1_ssl_report", fig_report(df, a.hw))]
    for name, fig in figs:
        if fig is None:
            print("skipped", name, "(rows not there yet)")
            continue
        for path in plots.save(fig, name, a.outdir):
            print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()