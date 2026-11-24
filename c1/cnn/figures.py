"""Report figures for the C1 CNN method.

    python -m c1.cnn.figures --hw colab-t4
    python -m c1.cnn.figures --hw local-cpu --outdir <scratch dir>   # preview from laptop rows

--results / --curves take one or more CSVs; by default the committed results/ files
and the git-ignored results/local/ ones, whichever exist (rows are filtered by --hw).

Writes report/figures/c1_cnn_*.{pdf,png}. Each figure is drawn only when the
rows it needs exist, so this can be re-run after every stage:
  c1_cnn_dynamics      training loss + validation accuracy vs epoch (curves.py rows)
  c1_cnn_stages        per-stage accuracy of every config (mean +/- std over tuning seeds)
  c1_cnn_ablation      paired accuracy change vs a reference config (S2a / S2b)
  c1_cnn_acc_vs_time   accuracy vs train time for every config, Pareto front, best/fast marked
  c1_cnn_report_pairs  best / fast per class pair on the report seeds, next to the classical best
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from c1.cnn import configs  # noqa: E402
from c1.cnn.run import DEFAULT_HW, LOCAL_RESULTS_DIR, PERSON, RESULTS_DIR, SEED_SETS  # noqa: E402
from c1.cnn.select import (DEFAULT_RESULTS, load_results, paired, select,  # noqa: E402
                           summarize_configs)

DEFAULT_CURVES = [f"{RESULTS_DIR}/{PERSON}_curves.csv", f"{LOCAL_RESULTS_DIR}/{PERSON}_curves.csv"]
from common import plots  # noqa: E402

COLOR = plots.METHOD_COLORS["c1_cnn"]
TUNE_LABEL = "tuning seeds 0–4"

# Reference config for each ablation stage: the paired difference is (config - reference).
ABLATION_REF = {"s2a": "s2a_ref", "s2b": "s2b_full"}
# Training curves drawn in the dynamics figure: same backbone (ConvNet-3 w16) and 300 epochs,
# no augmentation -> flip + crop -> the full S2b recipe.
DYNAMICS_CONFIGS = ("s2a_noaug", "s2a_ref", "s2b_full")
# Stage rows of the every-config figure (one panel row each, so the labels stay readable).
STAGE_ROWS = (("s0", "s1", "s2a"), ("s2b", "s3", "s4"))
# Reference method drawn next to ours on the report seeds (its `best` config).
REFERENCE = ("results/massimo.csv", "classical")


def fig_dynamics(curves, hw, outdir, config_ids=DYNAMICS_CONFIGS):
    c = curves[(curves["hw"] == hw) & curves["config_id"].isin(config_ids)]
    if c.empty:
        return None
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, cid in enumerate(config_ids):
        d = c[c["config_id"] == cid]
        if d.empty:
            continue
        color = plots.SERIES[i]
        # per-seed loss smoothed over 5 epochs, then median and inter-quartile range over seeds
        loss = (d.pivot_table(index="epoch", columns="seed", values="train_loss")
                .rolling(5, min_periods=1).mean())
        q = loss.quantile([0.25, 0.5, 0.75], axis=1).T
        ax_l.plot(q.index, q[0.5], color=color, label=cid)
        ax_l.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.2, linewidth=0)
        acc = d.dropna(subset=["val_acc"]).groupby("epoch")["val_acc"].agg(["mean", "std"]) * 100
        ax_a.plot(acc.index, acc["mean"], color=color, marker="o", markersize=4, label=cid)
        ax_a.fill_between(acc.index, acc["mean"] - acc["std"], acc["mean"] + acc["std"],
                          color=color, alpha=0.12, linewidth=0)
    ax_l.set(xlabel="epoch", ylabel="training loss (cross-entropy, log scale)", yscale="log",
             title="Training loss (5-epoch mean; median, IQR over seeds)")
    ax_a.set(xlabel="epoch", ylabel="validation accuracy (%)",
             title="Validation accuracy (mean ± std over seeds)")
    ax_a.legend(loc="lower right")
    fig.suptitle(f"C1 CNN training dynamics ({TUNE_LABEL})", fontsize=11)
    fig.tight_layout()
    return plots.save(fig, "c1_cnn_dynamics", outdir)


def fig_stages(summary, outdir):
    rows = [[(st, cid) for st in group if st in configs.STAGES
             for cid in configs.STAGES[st] if cid in summary.index] for group in STAGE_ROWS]
    rows = [r for r in rows if r]
    if not rows:
        return None
    n = max(len(r) for r in rows)
    fig, axes = plt.subplots(len(rows), 1, figsize=(max(6, 0.3 * n + 1.5), 3.6 * len(rows)),
                             sharey=True, squeeze=False)
    for ax, row in zip(axes[:, 0], rows):
        x = np.arange(len(row))
        s = summary.loc[[cid for _, cid in row]]
        stages = [st for st, _ in row]
        colors = [plots.SERIES[[g for grp in STAGE_ROWS for g in grp].index(st) % len(plots.SERIES)]
                  for st in stages]
        ax.errorbar(x, s["acc_mean"] * 100, yerr=s["acc_std"] * 100, fmt="none",
                    ecolor=plots.NEUTRAL, capsize=2, elinewidth=1)
        ax.scatter(x, s["acc_mean"] * 100, c=colors, edgecolors=plots.TEXT, s=28, zorder=3)
        ax.set_xticks(x, [cid for _, cid in row], rotation=60, ha="right", fontsize=8)
        ax.set_xlim(-0.7, n - 0.3)
        ax.set_ylabel("validation accuracy (%)")
        for i, st in enumerate(stages):  # stage label above its first config
            if i == 0 or stages[i - 1] != st:
                ax.axvline(i - 0.5, color=plots.GRID, linewidth=1)
                ax.text(i - 0.4, 1.02, st.upper(), transform=ax.get_xaxis_transform(),
                        fontsize=9, color=plots.TEXT_SECONDARY)
    fig.suptitle(f"C1 CNN: every tuning config, mean ± std over seeds ({TUNE_LABEL})",
                 fontsize=11)
    fig.tight_layout()
    return plots.save(fig, "c1_cnn_stages", outdir)


def fig_ablation(df, seeds, hw, outdir):
    have = set(summarize_configs(df, seeds, hw).index)
    panels = [(st, ref) for st, ref in ABLATION_REF.items()
              if ref and st in configs.STAGES and ref in have]
    if not panels:
        return None  # no ablation stage has its reference config complete yet
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 3.8), squeeze=False)
    for ax, (stage, ref) in zip(axes[0], panels):
        ids = [c for c in configs.STAGES[stage] if c != ref]
        res = [(c, paired(df, c, ref, seeds, hw)) for c in ids]
        res = [(c, r) for c, r in res if r["n"] == len(seeds)]
        if not res:
            continue
        res.sort(key=lambda t: t[1]["mean_diff_pp"])
        d = np.array([r["mean_diff_pp"] for _, r in res])
        err = np.array([np.std(list(r["per_seed"].values())) for _, r in res])
        y = np.arange(len(res))
        ax.barh(y, d, xerr=err, color=[COLOR if v > 0 else plots.NEUTRAL for v in d],
                capsize=3, error_kw={"elinewidth": 1})
        ax.set_yticks(y, [f"{c}  ({r['wins']}/{r['n']})" for c, r in res])
        ax.axvline(0, color=plots.TEXT_SECONDARY, linewidth=1)
        ax.set(xlabel=f"paired Δ accuracy vs {ref} (pp)", title=stage.upper())
    fig.suptitle(f"C1 CNN augmentation / regularization ablation ({TUNE_LABEL})\n"
                 "bars: mean ± std of per-seed differences; wins/seeds in brackets", fontsize=10)
    fig.tight_layout()
    return plots.save(fig, "c1_cnn_ablation", outdir)


def fig_acc_vs_time(summary, outdir):
    if len(summary) < 2:
        return None
    t = summary["time_mean"].to_numpy()
    a = summary["acc_mean"].to_numpy() * 100
    front = plots.pareto_front(t, a)
    best, fast = select(summary)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.scatter(t[~front], a[~front], color=plots.NEUTRAL, s=18, label="config")
    order = np.argsort(t[front])
    ax.plot(t[front][order], a[front][order], **plots.method_marker_kw("c1_cnn"),
            label="Pareto front")
    for name, cid in (("best", best), ("fast", fast)):
        r = summary.loc[cid]
        ax.annotate(f"{name}: {cid}", (r.time_mean, r.acc_mean * 100), xytext=(6, -12),
                    textcoords="offset points", fontsize=8, color=plots.TEXT)
    ax.set(xscale="log", xlabel="train + inference time per problem instance (s, log scale)",
           ylabel="validation accuracy (%)", title=f"C1 CNN: accuracy vs time ({TUNE_LABEL})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return plots.save(fig, "c1_cnn_acc_vs_time", outdir)


def fig_report_pairs(df, hw, outdir, reference=REFERENCE):
    seeds = SEED_SETS.get("report")
    if not seeds:
        return None
    d = df[df["seed"].isin(seeds) & df["config_id"].isin(["best", "fast"])
           & df["notes"].str.contains(f"hw={hw}", regex=False)]
    wide = d.pivot_table(index=["seed", "class_pair"], columns="config_id", values="val_acc")
    if "best" not in wide or wide["best"].count() < len(seeds):
        return None
    ref_path, ref_name = reference
    ref = None
    if ref_path and os.path.exists(ref_path):
        r = pd.read_csv(ref_path)
        r = r[(r["challenge"] == "c1") & r["seed"].isin(seeds) & (r["config_id"] == "best")
              & r["notes"].astype(str).str.contains(f"hw={hw}", regex=False)]
        if r["seed"].nunique() == len(seeds):
            ref = r.groupby("seed")["val_acc"].mean()
    wide = wide.reset_index().sort_values("best")
    y = np.arange(len(wide))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    has_fast = "fast" in wide and wide["fast"].count() == len(seeds)
    if has_fast:
        ax.hlines(y, wide["fast"] * 100, wide["best"] * 100, color=plots.GRID, linewidth=2)
        ax.plot(wide["fast"] * 100, y, "s", color=plots.SERIES[1], markersize=7, label="CNN fast")
    ax.plot(wide["best"] * 100, y, "o", color=plots.SERIES[0], markersize=7, label="CNN best")
    if ref is not None:
        ax.plot(wide["seed"].map(ref) * 100, y, "D", markerfacecolor="none",
                markeredgecolor=plots.TEXT_SECONDARY, markersize=7, label=f"{ref_name} best")
    ax.set_yticks(y, [f"{p} (seed {s})" for s, p in zip(wide["seed"], wide["class_pair"])])
    ax.set_xlabel("validation accuracy (%)")

    def ms(v):
        return f"{v.mean() * 100:.1f} ± {v.std(ddof=0) * 100:.1f}%"

    title = f"C1 CNN, report seeds 10–19: best {ms(wide['best'])}"
    if has_fast:
        title += f", fast {ms(wide['fast'])}"
    if ref is not None:
        diff = (wide["best"] - wide["seed"].map(ref).to_numpy()) * 100
        title += (f"\nCNN best − {ref_name} best: {diff.mean():+.1f} pp, "
                  f"CNN wins {(diff > 0).sum()}/{len(diff)}")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return plots.save(fig, "c1_cnn_report_pairs", outdir)


def main(argv=None):
    p = argparse.ArgumentParser(description="Draw the C1 CNN report figures.")
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--results", nargs="+", default=DEFAULT_RESULTS)
    p.add_argument("--curves", nargs="+", default=DEFAULT_CURVES)
    p.add_argument("--outdir", default="report/figures")
    a = p.parse_args(argv)

    plots.apply_style()
    df = load_results(a.results)
    seeds = SEED_SETS["tune"]
    summary = summarize_configs(df, seeds, a.hw)
    summary = summary[summary.index.isin([c for s in configs.STAGES.values() for c in s])]
    made = []
    curves = [c for c in a.curves if os.path.exists(c)]
    if curves:
        made.append(fig_dynamics(pd.concat([pd.read_csv(c) for c in curves]), a.hw, a.outdir))
    made += [fig_stages(summary, a.outdir), fig_ablation(df, seeds, a.hw, a.outdir),
             fig_acc_vs_time(summary, a.outdir), fig_report_pairs(df, a.hw, a.outdir)]
    for paths in made:
        if paths:
            print("wrote", paths[1])


if __name__ == "__main__":
    main()
