"""Report figures for the C1 ViT-from-scratch method.

    python -m c1.vit_scratch.figures                  # rows tagged hw=local-cpu
    python -m c1.vit_scratch.figures --hw colab-t4    # rows from a Colab T4 run

Writes report/figures/c1_vit_*.{pdf,png}.
"""
import argparse
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from c1.vit_scratch import configs  # noqa: E402
from common import data, plots  # noqa: E402

COLOR = plots.METHOD_COLORS["c1_vit_scratch"]
STAGE_OF = {cid: st for st, cfgs in configs.STAGES.items() for cid in cfgs}
STAGE_LABEL = {"s0": "S0 baseline", "s1": "S1 one knob", "s2": "S2 small-data ViT"}
STAGE_COLOR = {"s0": plots.NEUTRAL, "s1": plots.SERIES[0], "s2": COLOR}


def load(hw, path="results/ammar.csv"):
    df = pd.read_csv(path)
    df = df[(df.challenge == "c1") & (df.method == "vit_scratch")]
    df = df[df.notes.fillna("").map(lambda n: bool(re.search(rf"hw={re.escape(hw)}\b", n)))]
    return df[df.seed.isin(data.TUNE_SEEDS)]


def fig_paired_deltas(df):
    """Paired per-seed difference vs baseline, one row per config, sorted by mean."""
    acc = df.pivot_table(index="seed", columns="config_id", values="val_acc") * 100
    delta = acc.sub(acc["baseline"], axis=0).drop(columns="baseline")
    order = delta.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(order) + 1.2))
    for i, cid in enumerate(order):
        st = STAGE_OF.get(cid, "s1")
        d = delta[cid].dropna()
        ax.scatter(d, np.full(len(d), i), s=12, color=plots.NEUTRAL, zorder=2)
        ax.plot([d.mean()], [i], marker="D", color=STAGE_COLOR[st], markersize=7, zorder=3,
                linestyle="none")
    ax.axvline(0, color=plots.TEXT_SECONDARY, linewidth=1)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Accuracy difference vs testbed baseline (pp), paired per seed")
    handles = [plt.Line2D([], [], marker="D", color=STAGE_COLOR[s], linestyle="none",
                          label=STAGE_LABEL[s]) for s in ("s0", "s1", "s2")]
    handles.append(plt.Line2D([], [], marker="o", color=plots.NEUTRAL, linestyle="none",
                              markersize=4, label="single seed"))
    ax.legend(handles=handles, loc="lower right")
    ax.set_title("C1 ViT from scratch: change vs baseline (tuning seeds 0–4; diamond = mean)")
    return fig


def fig_acc_vs_time(df):
    s = df.groupby("config_id").agg(acc=("val_acc", "mean"), sd=("val_acc", "std"),
                                    t=("train_time_s", "mean"))
    s[["acc", "sd"]] *= 100
    front = plots.pareto_front(s.t, s.acc)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for st in ("s0", "s1", "s2"):
        m = s.index.map(lambda c: STAGE_OF.get(c) == st)
        ax.scatter(s.t[m], s.acc[m], color=STAGE_COLOR[st], label=STAGE_LABEL[st], zorder=3)
    for cid, r in s[front].iterrows():
        ax.annotate(cid, (r.t, r.acc), textcoords="offset points", xytext=(5, 4), fontsize=8)
    pf = s[front].sort_values("t")
    ax.plot(pf.t, pf.acc, color=plots.TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1,
            label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlabel("Training time per instance (s, log scale)")
    ax.set_ylabel("Mean validation accuracy (%)")
    ax.set_title("C1 ViT from scratch: accuracy vs training time (tuning seeds 0–4)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(loc="upper left")
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hw", default="local-cpu")
    p.add_argument("--results", default="results/ammar.csv")
    a = p.parse_args(argv)
    plots.apply_style()
    df = load(a.hw, a.results)
    for name, fig in [("c1_vit_paired_deltas", fig_paired_deltas(df)),
                      ("c1_vit_acc_vs_time", fig_acc_vs_time(df))]:
        print(plots.save(fig, name))
        plt.close(fig)


if __name__ == "__main__":
    main()
