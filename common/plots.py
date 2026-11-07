"""Shared plotting so every figure in the report looks the same.

OWNER: Massimo.

Colors:
  * METHOD_COLORS: one hue per team member, so each person's C1 and C2 methods
    share a color. The four hues were checked with a colorblind-safety validator
    on all pairs (worst CVD Delta E 13.0), so the four methods of one challenge
    stay distinguishable even in scatter plots. In a figure that mixes C1 and C2,
    tell them apart by marker fill: C1 filled, C2 hollow (see `method_marker_kw`).
  * SERIES: generic categorical colors for series *inside* one method's figure
    (e.g. feature families). Use them in this fixed order and keep scatter plots
    to <= 3 series (the first three validate on all pairs).
  * SEQUENTIAL_CMAP: single-hue blue ramp for heatmaps.
Yellow and magenta are below 3:1 contrast on white: always pair them with a
legend/direct label (never color alone) and keep a table of the numbers.
"""
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d4"
NEUTRAL = "#9c9b96"  # de-emphasized marks (e.g. every config in a sweep cloud)

_PERSON_COLOR = {
    "massimo": "#2a78d6",  # blue
    "ryan": "#eda100",     # yellow
    "ammar": "#008300",    # green
    "alvin": "#e87ba4",    # magenta
}
_METHOD_OWNER = {
    "c1_classical": "massimo", "c2_frozen_features": "massimo",
    "c1_cnn": "ryan", "c2_finetune": "ryan",
    "c1_vit_scratch": "ammar", "c2_peft": "ammar",
    "c1_ssl": "alvin", "c2_clip": "alvin",
}
METHOD_COLORS = {m: _PERSON_COLOR[p] for m, p in _METHOD_OWNER.items()}
_PERSON_MARKER = {"massimo": "o", "ryan": "s", "ammar": "^", "alvin": "D"}
METHOD_MARKERS = {m: _PERSON_MARKER[p] for m, p in _METHOD_OWNER.items()}

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def method_marker_kw(method):
    """Marker style for a method: color + shape, filled for C1, hollow for C2."""
    color = METHOD_COLORS[method]
    hollow = method.startswith("c2")
    return {"color": color, "marker": METHOD_MARKERS[method],
            "markerfacecolor": "white" if hollow else color, "markeredgecolor": color}


def apply_style():
    """Matplotlib defaults for report figures: readable fonts, recessive grid/axes."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID,
        "grid.alpha": 0.3, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.markersize": 6,
        "legend.frameon": False,
    })


def summarize(df, by, value="val_acc"):
    """Mean, std (ddof=0: spread across problem instances) and count of `value` per group."""
    g = df.groupby(by)[value]
    out = g.agg(mean="mean", std=lambda v: float(np.std(v, ddof=0)), n="count")
    return out.reset_index()


def errorbar_sweep(ax, x, mean, std, label=None, color=None, marker="o"):
    """One hyperparameter-sweep curve: mean with +/- std error bars."""
    return ax.errorbar(x, mean, yerr=std, label=label, color=color, marker=marker,
                       markersize=6, linewidth=2, capsize=3, elinewidth=1)


def pareto_front(times, accs):
    """Boolean mask of points no other point beats on both time (lower) and accuracy (higher)."""
    t = np.asarray(times, dtype=float)
    a = np.asarray(accs, dtype=float)
    mask = np.ones(len(t), dtype=bool)
    for i in range(len(t)):
        dominated = (t <= t[i]) & (a >= a[i]) & ((t < t[i]) | (a > a[i]))
        mask[i] = not dominated.any()
    return mask


def save(fig, name, outdir="report/figures"):
    """Write `name`.pdf (for LaTeX) and `name`.png (for quick viewing). Returns the paths."""
    os.makedirs(outdir, exist_ok=True)
    paths = [os.path.join(outdir, f"{name}.{ext}") for ext in ("pdf", "png")]
    for p in paths:
        fig.savefig(p)
    return paths
