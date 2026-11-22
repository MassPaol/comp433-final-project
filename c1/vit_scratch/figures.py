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
STAGE_LABEL = {"s0": "S0 baseline", "s1": "S1 one knob", "s2": "S2 small-data ViT",
               "s3": "S3 lr / wd / epochs"}
STAGE_COLOR = {"s0": plots.NEUTRAL, "s1": plots.SERIES[0], "s2": COLOR, "s3": plots.SERIES[3]}


def load(hw, path="results/ammar.csv", seeds=data.TUNE_SEEDS):
    df = pd.read_csv(path)
    df = df[(df.challenge == "c1") & (df.method == "vit_scratch")]
    df = df[df.notes.fillna("").map(lambda n: bool(re.search(rf"hw={re.escape(hw)}\b", n)))]
    return df[df.seed.isin(seeds)]


def report_configs():
    """(config_id, label) for the report-seed figure: two references, then fast and best."""
    out = [("baseline", "Testbed baseline"), ("s1_flip_crop", "+ flip + crop")]
    final = configs.FINAL_SOURCE
    if final["fast"] != final["best"]:
        out.append((final["fast"], f"{final['fast']} (fast)"))
    out.append((final["best"], f"{final['best']} (best)"))
    return out


def fig_report_pairs(df):
    """Report seeds: accuracy per class pair for baseline -> + augmentation -> best."""
    rc = report_configs()
    acc = df.pivot_table(index=["seed", "class_pair"], columns="config_id", values="val_acc") * 100
    acc = acc[[c for c, _ in rc]].sort_values(configs.FINAL_SOURCE["best"])
    labels = [f"{p} ({s})" for s, p in acc.index]
    colors = [plots.NEUTRAL, plots.SERIES[0], plots.SERIES[3], COLOR][-len(rc):]
    offsets = np.linspace(-0.24, 0.24, len(rc))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(acc))
    ax.hlines(y, acc.min(axis=1), acc.max(axis=1), color=plots.GRID, linewidth=2, zorder=1)
    # small vertical offsets + distinct markers so tied points stay visible
    for (cid, name), col, dy, mk in zip(rc, colors, offsets, "os^D"):
        ax.scatter(acc[cid], y + dy, color=col, marker=mk, zorder=3, s=34,
                   label=f"{name}: {acc[cid].mean():.1f} ± {acc[cid].std(ddof=0):.1f}%")
    ax.axvline(50, color=plots.TEXT_SECONDARY, linewidth=1, linestyle=":")
    ax.text(50.5, len(acc) - 0.6, "chance", fontsize=8, color=plots.TEXT_SECONDARY)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Validation accuracy (%)")
    ax.set_title("C1 ViT from scratch: report seeds 10–19 (Colab T4)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=1)
    return fig


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
                          label=STAGE_LABEL[s]) for s in STAGE_LABEL]
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
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for st in STAGE_LABEL:
        m = s.index.map(lambda c: STAGE_OF.get(c) == st)
        ax.scatter(s.t[m], s.acc[m], color=STAGE_COLOR[st], label=STAGE_LABEL[st], zorder=3)
    # label only the configs the text talks about (labelling the whole front overlaps)
    final = configs.FINAL_SOURCE
    labels = {"baseline": ("testbed baseline", (6, -10)),
              "s1_flip_crop": ("+ flip + crop", (-72, 2)),
              final["fast"]: (f"fast: {final['fast']}", (-78, 4)),
              final["best"]: (f"best: {final['best']}", (-20, 12)),
              s.acc.idxmax(): (f"top: {s.acc.idxmax()}", (-40, 10))}
    for cid, (text, off) in labels.items():
        if cid in s.index:
            ax.annotate(text, (s.t[cid], s.acc[cid]), textcoords="offset points", xytext=off,
                        fontsize=8, arrowprops={"arrowstyle": "-", "color": plots.TEXT_SECONDARY,
                                                "linewidth": 0.6})
    pf = s[front].sort_values("t")
    ax.plot(pf.t, pf.acc, color=plots.TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=1,
            label="Pareto front")
    ax.set_xscale("log")
    ticks = [t for t in (2.5, 3, 4, 5, 6, 8, 10, 12) if s.t.min() * 0.9 <= t <= s.t.max() * 1.1]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Training time per instance (s, log scale)")
    ax.set_ylabel("Mean validation accuracy (%)")
    ax.set_title("C1 ViT from scratch: accuracy vs training time (tuning seeds 0–4)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hw", default="local-cpu")
    p.add_argument("--results", default="results/ammar.csv")
    a = p.parse_args(argv)
    plots.apply_style()
    df = load(a.hw, a.results)
    figs = [("c1_vit_paired_deltas", fig_paired_deltas(df)),
            ("c1_vit_acc_vs_time", fig_acc_vs_time(df))]
    rep = load(a.hw, a.results, seeds=data.REPORT_SEEDS)
    if set(c for c, _ in report_configs()) <= set(rep.config_id):
        figs.append(("c1_vit_report_pairs", fig_report_pairs(rep)))
    for name, fig in figs:
        print(plots.save(fig, name))
        plt.close(fig)


if __name__ == "__main__":
    main()
