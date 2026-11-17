"""C2 CLIP report figures. Reads results/alvin.csv; uses common/plots.py for style and colors.

    python -m c2.clip.figures --hw colab-t4

Tuning figures use seeds 0-4 (every config in a stage). The report figure uses seeds 10-19 and
only the frozen FINAL configs. Writes report/figures/c2_clip_*.{pdf,png}.
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from c2.clip.configs import FINAL, STAGES  # noqa: E402
from c2.clip.run import PERSON, summarize  # noqa: E402
from common import data, plots  # noqa: E402

COLOR = plots.METHOD_COLORS["c2_clip"]
TUNE_LABEL = "tuning seeds 0–4"


def _kw(stage, cid):
    return STAGES[stage][cid]


def _rows(summary, stage):
    return summary[summary.index.isin(STAGES[stage])]


def fig_probe(summary):
    """Probe: accuracy against learning rate, one line per training steps value (stage probe)."""
    s = _rows(summary, "probe")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, steps in enumerate(sorted({_kw("probe", c)["steps"] for c in s.index})):
        sub = sorted((_kw("probe", c)["lr"], r.acc * 100, r["std"] * 100)
                     for c, r in s.iterrows() if _kw("probe", c)["steps"] == steps)
        lr, m, sd = map(np.array, zip(*sub))
        plots.errorbar_sweep(ax, lr, m, sd, label=f"{steps} steps", color=plots.SERIES[i])
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log scale)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Linear probe: learning rate × steps ({TUNE_LABEL})", fontsize=10)
    ax.legend(title="training steps")
    fig.tight_layout()
    return fig


def fig_wiseft(summary, zs_acc):
    """WiSE-FT: accuracy against mix weight (0 = zero-shot head, 1 = trained probe head)."""
    s = _rows(summary, "wiseft")
    mix = np.array([_kw("wiseft", c)["mix"] for c in s.index])
    order = np.argsort(mix)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    plots.errorbar_sweep(ax, np.r_[0.0, mix[order]],
                         np.r_[zs_acc, s.acc.to_numpy()[order] * 100],
                         np.r_[0.0, s["std"].to_numpy()[order] * 100],
                         label="interpolated head", color=COLOR)
    ax.axhline(zs_acc, color=plots.NEUTRAL, linestyle="--", linewidth=1, label="zero-shot")
    ax.set_xlabel("Mix weight toward probe head")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"WiSE-FT-style interpolation ({TUNE_LABEL})", fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def fig_tip_heatmap(summary):
    """Tip-Adapter: accuracy over alpha x beta (stage tip)."""
    s = _rows(summary, "tip")
    alphas = sorted({_kw("tip", c)["alpha"] for c in s.index})
    betas = sorted({_kw("tip", c)["beta"] for c in s.index})
    grid = np.full((len(alphas), len(betas)), np.nan)
    for c, r in s.iterrows():
        kw = _kw("tip", c)
        grid[alphas.index(kw["alpha"]), betas.index(kw["beta"])] = r.acc * 100
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    im = ax.imshow(grid, cmap=plots.SEQUENTIAL_CMAP)
    for i in range(len(alphas)):
        for j in range(len(betas)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] > np.nanmean(grid) else plots.TEXT)
    ax.set_xticks(range(len(betas)), betas)
    ax.set_yticks(range(len(alphas)), alphas)
    ax.set_xlabel("beta (sharpness)")
    ax.set_ylabel("alpha (cache weight)")
    ax.set_title(f"Tip-Adapter: alpha × beta ({TUNE_LABEL})", fontsize=10)
    fig.colorbar(im, ax=ax, label="Validation accuracy (%)")
    fig.tight_layout()
    return fig


def fig_tipf(summary):
    """Tip-Adapter-F: accuracy against key learning rate, one line per steps value (stage tipf)."""
    s = _rows(summary, "tipf")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, steps in enumerate(sorted({_kw("tipf", c)["steps"] for c in s.index})):
        sub = sorted((_kw("tipf", c)["lr"], r.acc * 100, r["std"] * 100)
                     for c, r in s.iterrows() if _kw("tipf", c)["steps"] == steps)
        lr, m, sd = map(np.array, zip(*sub))
        plots.errorbar_sweep(ax, lr, m, sd, label=f"{steps} steps", color=plots.SERIES[i])
    ax.set_xscale("log")
    ax.set_xlabel("Key learning rate (log scale)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"Tip-Adapter-F: key learning rate × steps ({TUNE_LABEL})", fontsize=10)
    ax.legend(title="training steps")
    fig.tight_layout()
    return fig


def fig_time_vs_acc(summary):
    """All tuning configs: accuracy against train + inference time, with the Pareto front."""
    ids = [c for c in summary.index if any(c in STAGES[s] for s in STAGES)]
    s = summary.loc[ids]
    t, a = s.time.to_numpy(), s.acc.to_numpy() * 100
    front = plots.pareto_front(t, a)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    kw = plots.method_marker_kw("c2_clip")
    ax.scatter(t, a, s=22, alpha=0.6, linewidths=0, color=COLOR,
               marker=kw["marker"], label=f"tuning configs (n = {len(s)})")
    order = np.argsort(t[front])
    ax.plot(t[front][order], a[front][order], "-", color=plots.TEXT, linewidth=1,
            label="Pareto front")
    for cid in FINAL.values():
        if cid in s.index:
            r = s.loc[cid]
            ax.scatter([r.time], [r.acc * 100], s=90, facecolor="none",
                       edgecolor=plots.TEXT, linewidths=1.5, zorder=5)
            ax.annotate(cid.replace("c2_", ""), (r.time, r.acc * 100),
                        textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Train + inference time per problem instance (s, log scale)")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"CLIP methods: accuracy vs time ({TUNE_LABEL})", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def fig_report(df, hw):
    """Frozen configs on reporting seeds 10-19: one dot per seed, diamond = mean ± std."""
    d = df[(df.challenge == "c2") & df.seed.isin(data.REPORT_SEEDS)
           & df.notes.fillna("").str.contains(f"hw={hw}")
           & df.config_id.isin(list(FINAL.values()))]
    names = list(FINAL.values())
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    rng = np.random.RandomState(0)
    for i, cid in enumerate(names):
        v = d[d.config_id == cid].val_acc.to_numpy() * 100
        if len(v) == 0:
            continue
        ax.plot(i + rng.uniform(-0.12, 0.12, len(v)), v, "o", color=COLOR, alpha=0.6, markersize=5)
        ax.errorbar([i], [v.mean()], yerr=[v.std(ddof=0)], fmt="D", color=plots.TEXT,
                    capsize=4, markersize=6)
    ax.set_xticks(range(len(names)), [n.replace("c2_", "").replace("_", "\n") for n in names],
                  fontsize=8)
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Report seeds 10–19 (dots = seeds, diamond = mean ± std)", fontsize=10)
    fig.tight_layout()
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description="Make the C2 CLIP report figures.")
    p.add_argument("--hw", default="colab-t4")
    p.add_argument("--results", default=f"results/{PERSON}.csv")
    p.add_argument("--outdir", default="report/figures")
    a = p.parse_args(argv)

    plots.apply_style()
    df = pd.read_csv(a.results)
    tune = summarize(a.results, data.TUNE_SEEDS, a.hw)
    zs = tune.acc.get("c2_zeroshot", np.nan) * 100

    figs = []
    if any(c in tune.index for c in STAGES["probe"]):
        figs.append(("c2_clip_probe_sweep", fig_probe(tune)))
    if any(c in tune.index for c in STAGES["wiseft"]):
        figs.append(("c2_clip_wiseft_mix", fig_wiseft(tune, zs)))
    if any(c in tune.index for c in STAGES["tip"]):
        figs.append(("c2_clip_tip_heatmap", fig_tip_heatmap(tune)))
    if any(c in tune.index for c in STAGES["tipf"]):
        figs.append(("c2_clip_tipf_sweep", fig_tipf(tune)))
    figs.append(("c2_clip_time_vs_acc", fig_time_vs_acc(tune)))
    figs.append(("c2_clip_report", fig_report(df, a.hw)))

    for name, fig in figs:
        for path in plots.save(fig, name, a.outdir):
            print("wrote", path)
        plt.close(fig)


if __name__ == "__main__":
    main()
