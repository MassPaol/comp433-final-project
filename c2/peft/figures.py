"""Report figures for the C2 parameter-efficient fine-tuning method.

    python -m c2.peft.figures                  # rows tagged hw=local-cpu
    python -m c2.peft.figures --hw colab-t4    # rows from a Colab T4 run

Writes report/figures/c2_peft_*.{pdf,png}.
"""
import argparse
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from c2.peft import configs  # noqa: E402
from common import data, plots  # noqa: E402

MODE_LABEL = {"head": "Head only (control)", "bitfit": "BitFit", "lora": "LoRA", "vpt": "VPT"}
MODE_COLOR = {"head": plots.NEUTRAL, "bitfit": plots.SERIES[0], "lora": plots.SERIES[1],
              "vpt": plots.SERIES[2]}


def load(hw, path="results/ammar.csv", seeds=data.TUNE_SEEDS):
    df = pd.read_csv(path)
    df = df[(df.challenge == "c2") & (df.method == "peft")]
    df = df[df.notes.fillna("").map(lambda n: bool(re.search(rf"hw={re.escape(hw)}\b", n)))]
    df = df[df.seed.isin(seeds)].copy()
    cfgs = configs.all_configs()
    df["mode"] = df.config_id.map(lambda c: cfgs[c]["mode"])
    return df


def _summary(df):
    # Median time, not mean: one local row (bitfit_lr1e-3_flip, seed 1) includes ~13 h of laptop
    # sleep, and the median is robust to that. T4 runs are unattended, so it changes nothing there.
    s = df.groupby(["config_id", "mode"]).agg(
        acc=("val_acc", "mean"), sd=("val_acc", "std"), t=("train_time_s", "median"),
        params=("n_trainable_params", "first")).reset_index()
    s[["acc", "sd"]] *= 100
    return s


def _scatter(ax, s, x, log=True):
    for mode in MODE_LABEL:
        m = s[s["mode"] == mode]
        if len(m):
            ax.errorbar(m[x], m.acc, yerr=m.sd, fmt="o", color=MODE_COLOR[mode],
                        label=MODE_LABEL[mode], capsize=3, elinewidth=1, markersize=6)
    best = s.loc[s.acc.idxmax()]
    ax.annotate(best.config_id, (best[x], best.acc), textcoords="offset points",
                xytext=(6, 6), fontsize=8)
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("Mean validation accuracy (%)")


OFF_SCALE = 95.0   # configs below this mean accuracy are listed in the title, not plotted


def fig_acc_vs_params_time(df):
    s = _summary(df)
    off = s[s.acc < OFF_SCALE]
    s = s[s.acc >= OFF_SCALE]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    _scatter(axes[0], s, "params")
    axes[0].set_xlabel("Trainable parameters (log scale)")
    axes[0].set_title("(a) Accuracy vs trainable parameters")
    _scatter(axes[1], s, "t", log=False)
    axes[1].set_xlabel("Median training time per instance (s)")
    axes[1].set_title("(b) Accuracy vs training time")
    axes[1].set_ylabel("")
    axes[0].set_ylim(max(s.acc.min() - s.sd.max() - 0.3, 90), 100.3)
    axes[0].legend(loc="lower right")
    title = "C2 PEFT on ViT-Tiny (tuning seeds 0–4; bars = std across seeds)"
    if len(off):
        title += "\nnot shown (off scale): " + ", ".join(
            f"{r.config_id} {r.acc:.1f}%" for r in off.itertuples())
    fig.suptitle(title, y=1.05)
    return fig


REPORT_CONFIGS = [("head", "Head only (fast)", 386), ("lora_r2", "LoRA r2 (best)", 18818),
                  ("vpt_shallow10_lr3e-3", "VPT shallow, 10 prompts", 2306)]


def fig_report_pairs(df):
    """Report seeds: accuracy per class pair for head, LoRA and VPT."""
    acc = df.pivot_table(index=["seed", "class_pair"], columns="config_id", values="val_acc") * 100
    acc = acc[[c for c, _, _ in REPORT_CONFIGS]].sort_values("lora_r2")
    labels = [f"{p} ({s})" for s, p in acc.index]
    colors = [plots.NEUTRAL, plots.SERIES[1], plots.SERIES[2]]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    y = np.arange(len(acc))
    ax.hlines(y, acc.min(axis=1), acc.max(axis=1), color=plots.GRID, linewidth=2, zorder=1)
    # small vertical offsets + distinct markers so tied points stay visible
    for (cid, name, n), col, dy, mk in zip(REPORT_CONFIGS, colors, (-0.18, 0, 0.18), "osD"):
        ax.scatter(acc[cid], y + dy, color=col, marker=mk, zorder=3, s=34,
                   label=f"{name}, {n:,} params: {acc[cid].mean():.2f} ± {acc[cid].std(ddof=0):.2f}%")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Validation accuracy (%)")
    ax.set_title("C2 PEFT on ViT-Tiny: report seeds 10–19 (Colab T4)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=1)
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hw", default="local-cpu")
    p.add_argument("--results", default="results/ammar.csv")
    a = p.parse_args(argv)
    plots.apply_style()
    figs = [("c2_peft_acc_vs_params_time", fig_acc_vs_params_time(load(a.hw, a.results)))]
    rep = load(a.hw, a.results, seeds=data.REPORT_SEEDS)
    if set(c for c, _, _ in REPORT_CONFIGS) <= set(rep.config_id):
        figs.append(("c2_peft_report_pairs", fig_report_pairs(rep)))
    for name, fig in figs:
        print(plots.save(fig, name))
        plt.close(fig)


if __name__ == "__main__":
    main()
