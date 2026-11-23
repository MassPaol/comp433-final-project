"""Report figures for the C2 frozen-features method.

    python -m c2.frozen_features.figures                  # local rows
    python -m c2.frozen_features.figures --hw colab-t4    # rows from the Colab T4 re-run

Writes report/figures/c2_frozen_*.{pdf,png}. Config parameters are read from
configs.STAGES / configs.FINAL, never parsed from the config-id strings.
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from c2.frozen_features import backbones as B  # noqa: E402
from c2.frozen_features import configs  # noqa: E402
from c2.frozen_features.run import DEFAULT_HW, PERSON, SEED_SETS, own_rows  # noqa: E402
from c2.frozen_features.select import select, summarize_configs  # noqa: E402
from common import plots  # noqa: E402
from common.sweep import hw_of  # noqa: E402

TUNE_LABEL = "tuning seeds 0–4"
ACC_LABEL = "Validation accuracy (%)"
N_VAL = 700
BACKBONE_LABELS = {"vitti": "ViT-Ti/16", "vits": "ViT-S/16", "vits1k": "ViT-S/16 (IN-1k)",
                   "vitb": "ViT-B/16", "deits": "DeiT-S/16", "rn50": "ResNet-50",
                   "cnxt": "ConvNeXt-T"}
HEAD_LABELS = {"adamw": "AdamW head\n(testbed)", "lr": "LR", "lr_cv": "LR-CV",
               "linsvm": "Linear SVM", "rbfsvm": "RBF SVM", "ncm": "NCM", "knn": "kNN"}


def _log_x(ax, subs=(1, 2, 5)):
    """Log x axis with plain tick labels at subs x 10^k (use subs=(1,) for wide ranges)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(subs=subs))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _tuning_rows(df, hw):
    d = own_rows(df)
    return d[d["seed"].isin(SEED_SETS["tune"]) & (d["notes"].map(hw_of) == hw)]


def _stage_table(summary, stage):
    """summary rows for one stage, joined with each config's kwargs."""
    s = summary[summary.index.isin(configs.STAGES[stage])].copy()
    s["kw"] = [configs.STAGES[stage][cid] for cid in s.index]
    return s


def fig_s1_heatmap(summary):
    bbs, heads = configs.S1_BACKBONES, configs.S1_HEADS
    grid = np.array([[summary.loc[f"s1_{b}_{h}", "acc_mean"] * 100 for h in heads] for b in bbs])
    order = np.argsort(-grid.max(axis=1))
    grid, bbs = grid[order], [bbs[i] for i in order]
    vmin = np.floor(grid.min())

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    im = ax.imshow(grid, cmap=plots.SEQUENTIAL_CMAP, vmin=vmin, vmax=100, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(heads)), [HEAD_LABELS[h] for h in heads], rotation=30, ha="right")
    ax.set_yticks(range(len(bbs)), [BACKBONE_LABELS[b] for b in bbs])
    mid = (vmin + 100) / 2
    for i in range(len(bbs)):
        for j in range(len(heads)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if v > mid else plots.TEXT,
                    fontweight="bold" if v == grid[i].max() else "normal")
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(ACC_LABEL)
    ax.set_title(f"S1: backbones × heads, default settings ({TUNE_LABEL}; bold = row best)")
    return fig


# Label placement (offset in points, horizontal alignment) so the 1.5-2.3 ms cluster stays legible.
# Offsets are relative to the marker, so the layout survives a uniform time shift (e.g. on a T4).
# The two crowded points get leader lines.
_LABEL_POS = {"vitti": ((9, -3), "left"), "vits": ((-9, 7), "right"), "vits1k": ((-30, -48), "center"),
              "deits": ((28, -38), "left"), "cnxt": ((9, 5), "left"), "rn50": ((9, -3), "left"),
              "vitb": ((-9, 7), "right")}
_LEADER = {"vits1k", "deits"}


def fig_backbones(summary, df, hw):
    """Best S1 head per backbone: accuracy vs inference time per image."""
    rows = _tuning_rows(df, hw)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for b in configs.S1_BACKBONES:
        ids = [f"s1_{b}_{h}" for h in configs.S1_HEADS]
        cid = summary.loc[ids, "acc_mean"].idxmax()
        r = summary.loc[cid]
        ms = rows.loc[rows["config_id"] == cid, "infer_time_s"].mean() / N_VAL * 1000
        in1k = B.PRETRAINING[b][0] == B.IN1K
        ax.errorbar([ms], [r.acc_mean * 100], yerr=[r.acc_std * 100], fmt="o", markersize=8,
                    color=plots.SERIES[0], markerfacecolor="white" if in1k else plots.SERIES[0],
                    ecolor=plots.NEUTRAL, capsize=3, elinewidth=1, zorder=3)
        head = configs.STAGES["s1"][cid]["head"]
        offset, ha = _LABEL_POS.get(b, ((9, -3), "left"))
        arrow = {"arrowstyle": "-", "color": plots.TEXT_SECONDARY, "lw": 0.7} if b in _LEADER else None
        ax.annotate(f"{BACKBONE_LABELS[b]} + {HEAD_LABELS[head].splitlines()[0].replace(' head', '')}",
                    (ms, r.acc_mean * 100), textcoords="offset points", xytext=offset, ha=ha,
                    fontsize=7.5, color=plots.TEXT, zorder=4, arrowprops=arrow)
    ax.plot([], [], "o", color=plots.SERIES[0], label="pretrained on IN-21k/22k (then IN-1k)")
    ax.plot([], [], "o", color=plots.SERIES[0], markerfacecolor="white", label="pretrained on IN-1k only")
    _log_x(ax)
    # start the axis at the 1-2-5 value just below the fastest backbone, so it gets a tick label
    lo = ax.get_xlim()[0]
    nice = [m * 10.0 ** k for k in range(-3, 3) for m in (1, 2, 5)]
    ax.set_xlim(left=max(v for v in nice if v <= lo))
    ax.set_xlabel("Inference time per image (ms, log scale; incl. feature extraction)")
    ax.set_ylabel(ACC_LABEL)
    ax.set_title(f"Backbones with their best S1 head (mean ± std, {TUNE_LABEL})")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def _resolved(kw):
    """A config's kwargs with every default filled in, so S1 defaults join S2/S3 curves."""
    from c2.frozen_features.heads import DEFAULT_NORM
    hp = dict(kw.get("head_params") or {})
    if kw["head"] in ("lr", "linsvm", "rbfsvm"):
        hp.setdefault("C", 1.0)
    if kw["head"] == "adamw":
        hp.setdefault("epochs", 10)
        hp.setdefault("lr", 1e-3)
    return {"backbone": kw["backbone"], "head": kw["head"], "hp": hp,
            "norm": kw.get("norm") or DEFAULT_NORM[kw["head"]], "pool": kw.get("pool") or "cls",
            "layers": kw.get("layers", 1), "crop_pct": kw.get("crop_pct"),
            "flip_train": kw.get("flip_train", False), "flip_tta": kw.get("flip_tta", False),
            "precision": kw.get("precision", "fp32")}


def _plain(r):
    """No S3/S4 feature or flip/precision change."""
    return (r["pool"] == "cls" and r["layers"] == 1 and r["crop_pct"] is None
            and not r["flip_train"] and not r["flip_tta"] and r["precision"] == "fp32")


def _curve(summary, pred, xkey):
    """(x, mean %, std %) of every stage config whose resolved kwargs satisfy `pred`, sorted by x."""
    pts = []
    for stage in configs.STAGES.values():
        for cid, kw in stage.items():
            r = _resolved(kw)
            if cid in summary.index and pred(r):
                pts.append((xkey(r), summary.loc[cid, "acc_mean"] * 100, summary.loc[cid, "acc_std"] * 100))
    return tuple(map(np.array, zip(*sorted(pts)))) if pts else (np.array([]),) * 3


def fig_sweeps(summary):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    s2_bbs = configs.S2_BACKBONES
    jitter = np.linspace(-0.04, 0.04, len(s2_bbs))

    # (a) LR inverse regularization C, default (z-scored) features
    ax = axes[0, 0]
    for i, b in enumerate(s2_bbs):
        x, m, sd = _curve(summary, lambda r, b=b: (r["backbone"] == b and r["head"] == "lr"
                                                   and r["norm"] == "standard" and _plain(r)),
                          lambda r: r["hp"]["C"])
        plots.errorbar_sweep(ax, x * 10 ** jitter[i], m, sd, label=BACKBONE_LABELS[b], color=plots.SERIES[i])
    _log_x(ax, subs=(1,))
    ax.set_xlabel("LR inverse regularization C (log scale)")
    ax.set_ylabel(ACC_LABEL)
    ax.set_title("(a) Logistic regression: C")
    ax.legend(loc="lower left", fontsize=8)

    # (b) testbed AdamW head: epochs x learning rate
    ax = axes[0, 1]
    for i, b in enumerate(s2_bbs):
        for lr, style in ((1e-3, "-"), (1e-2, "--")):
            x, m, sd = _curve(summary, lambda r, b=b, lr=lr: (r["backbone"] == b and r["head"] == "adamw"
                                                              and r["hp"]["lr"] == lr and _plain(r)),
                              lambda r: r["hp"]["epochs"])
            line = plots.errorbar_sweep(ax, x * 10 ** jitter[i], m, sd, color=plots.SERIES[i])
            line[0].set_linestyle(style)
    for i, b in enumerate(s2_bbs):
        ax.plot([], [], "-", color=plots.SERIES[i], label=BACKBONE_LABELS[b])
    ax.plot([], [], "-", color=plots.TEXT_SECONDARY, label="lr 1e-3 (testbed)")
    ax.plot([], [], "--", color=plots.TEXT_SECONDARY, label="lr 1e-2")
    _log_x(ax)
    ax.set_xlabel("Training epochs (log scale; testbed = 10)")
    ax.set_title("(b) Testbed AdamW head: epochs and learning rate")
    ax.legend(loc="lower right", fontsize=7.5, ncol=2)

    # (c, d) S3 feature source: pool x last-k blocks, per backbone, with that backbone's S3 head
    for ax, (b, base) in zip(axes[1], configs.S3_BASES.items()):
        base_r = _resolved(base)
        same_head = lambda r: (r["backbone"] == b and r["head"] == base_r["head"]  # noqa: E731
                               and r["hp"] == base_r["hp"] and r["norm"] == base_r["norm"]
                               and r["crop_pct"] is None and not r["flip_train"] and not r["flip_tta"]
                               and r["precision"] == "fp32")
        for i, pool in enumerate(["cls", "mean", "cls+mean"]):
            x, m, sd = _curve(summary, lambda r, pool=pool: same_head(r) and r["pool"] == pool,
                              lambda r: r["layers"])
            plots.errorbar_sweep(ax, x + (i - 1) * 0.06, m, sd, label=f"pool = {pool}", color=plots.SERIES[i])
        crop = f"s3_{b}_crop1"
        if crop in summary.index:
            r = summary.loc[crop]
            ax.errorbar([1 - 0.18], [r.acc_mean * 100], yerr=[r.acc_std * 100], fmt="*", markersize=10,
                        color=plots.SERIES[3], capsize=3, elinewidth=1, label="CLS, whole image (no crop)")
        ax.set_xticks([1, 2, 4])
        ax.set_xlabel("Last blocks concatenated (each after the final norm)")
        ax.set_ylabel(ACC_LABEL)
        head = HEAD_LABELS[base_r["head"]].splitlines()[0]
        ax.set_title(f"({'c' if b == 'vitb' else 'd'}) {BACKBONE_LABELS[b]} + {head}: feature source")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Hyperparameter sweeps, mean ± std over {TUNE_LABEL}", y=1.0)
    fig.tight_layout()
    return fig


def fig_acc_vs_time(summary):
    s = summary[summary.index.str.match(r"s\d_")]
    best, fast = select(s)
    base = configs.FINAL_SOURCE["baseline"]
    t = s["time_mean"].to_numpy()
    a = s["acc_mean"].to_numpy() * 100
    front = plots.pareto_front(t, a)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.scatter(t[~front], a[~front], s=14, color=plots.NEUTRAL, alpha=0.6,
               label=f"All tuning configs (n = {len(s)})", linewidths=0)
    order = np.argsort(t[front])
    ax.plot(t[front][order], a[front][order], "-o", color=plots.SERIES[0], markersize=5,
            label="Pareto front")
    for cid, name, dy in ((best, "best", 10), (fast, "fast", -16), (base, "baseline (testbed)", -16)):
        r = s.loc[cid]
        ax.scatter([r.time_mean], [r.acc_mean * 100], s=90, facecolor="none",
                   edgecolor=plots.TEXT, linewidths=1.5, zorder=5)
        ax.annotate(f"{name}: {r.acc_mean * 100:.2f}%, {r.time_mean:.2f} s",
                    (r.time_mean, r.acc_mean * 100), textcoords="offset points",
                    xytext=(8, dy), fontsize=8, color=plots.TEXT, zorder=6)
    _log_x(ax)
    ax.set_xlabel("Train + inference time per problem instance (s, log scale)")
    ax.set_ylabel(ACC_LABEL)
    ax.set_title(f"C2 frozen features: accuracy vs time, all stages ({TUNE_LABEL})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    return fig


def fig_report_pairs(df, hw):
    d = own_rows(df)
    d = d[d["seed"].isin(SEED_SETS["report"]) & (d["notes"].map(hw_of) == hw)
          & d["config_id"].isin(["best", "fast", "baseline"])]
    wide = d.pivot_table(index=["seed", "class_pair"], columns="config_id", values="val_acc")
    wide = wide.reset_index().sort_values("best")
    y = np.arange(len(wide))

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    lo = wide[[c for c in ("best", "fast", "baseline") if c in wide]].min(axis=1)
    hi = wide[[c for c in ("best", "fast", "baseline") if c in wide]].max(axis=1)
    ax.hlines(y, lo * 100, hi * 100, color=plots.GRID, linewidth=2)
    styles = {"baseline": ("D", plots.SERIES[2]), "fast": ("s", plots.SERIES[1]),
              "best": ("o", plots.SERIES[0])}
    for name, (marker, color) in styles.items():
        if name in wide:
            ax.plot(wide[name] * 100, y, marker, color=color, markersize=7, linestyle="none",
                    label=name if name != "baseline" else "baseline (testbed)")
    ax.set_yticks(y, [f"{p} (seed {s})" for s, p in zip(wide["seed"], wide["class_pair"])])
    ax.set_xlabel(ACC_LABEL)
    parts = [f"{n} {wide[n].mean() * 100:.1f} ± {wide[n].std(ddof=0) * 100:.1f}%"
             for n in ("best", "fast", "baseline") if n in wide]
    title = "Report seeds 10–19: " + ", ".join(parts)
    for other in ("fast", "baseline"):
        if other in wide:
            diff = (wide["best"] - wide[other]) * 100
            title += (f"\nbest − {other}: {diff.mean():+.2f} pp, best wins {(diff > 0).sum()}/"
                      f"{len(diff)}, ties {(diff == 0).sum()}")
    ax.set_title(title, fontsize=9.5)
    ax.legend(loc="center left")  # the hard pairs (bottom) spread left; the rest sit near 100 %
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Make the C2 frozen-features report figures.")
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--results", default=f"results/{PERSON}.csv")
    p.add_argument("--outdir", default="report/figures")
    a = p.parse_args(argv)

    plots.apply_style()
    df = pd.read_csv(a.results)
    summary = summarize_configs(df, SEED_SETS["tune"], a.hw)
    figs = [("c2_frozen_s1_heatmap", fig_s1_heatmap(summary)),
            ("c2_frozen_backbones", fig_backbones(summary, df, a.hw)),
            ("c2_frozen_sweeps", fig_sweeps(summary)),
            ("c2_frozen_acc_vs_time", fig_acc_vs_time(summary)),
            ("c2_frozen_report_pairs", fig_report_pairs(df, a.hw))]
    for name, fig in figs:
        for path in plots.save(fig, name, a.outdir):
            print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()
