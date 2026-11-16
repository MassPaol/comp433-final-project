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
import pandas as pd  # noqa: E402

from c2.peft import configs  # noqa: E402
from common import data, plots  # noqa: E402

MODE_LABEL = {"head": "Head only (control)", "bitfit": "BitFit", "lora": "LoRA", "vpt": "VPT"}
MODE_COLOR = {"head": plots.NEUTRAL, "bitfit": plots.SERIES[0], "lora": plots.SERIES[1],
              "vpt": plots.SERIES[2]}


def load(hw, path="results/ammar.csv"):
    df = pd.read_csv(path)
    df = df[(df.challenge == "c2") & (df.method == "peft")]
    df = df[df.notes.fillna("").map(lambda n: bool(re.search(rf"hw={re.escape(hw)}\b", n)))]
    df = df[df.seed.isin(data.TUNE_SEEDS)].copy()
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--hw", default="local-cpu")
    p.add_argument("--results", default="results/ammar.csv")
    a = p.parse_args(argv)
    plots.apply_style()
    fig = fig_acc_vs_params_time(load(a.hw, a.results))
    print(plots.save(fig, "c2_peft_acc_vs_params_time"))
    plt.close(fig)


if __name__ == "__main__":
    main()
