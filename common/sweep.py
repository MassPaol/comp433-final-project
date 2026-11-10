"""Helpers for running a hyperparameter sweep through the harness and choosing configs.

Shared by every method that logs to results/<person>.csv. A person's file holds
all their methods, so everything here is scoped to one (challenge, method).
Rows carry the hardware in their notes ("hw=colab-t4 ..."), so runs on
different machines never mix.
"""
import csv
import os
import re

import pandas as pd

EPS = 1e-9


def hw_of(notes):
    """The hw tag in a notes string ("hw=colab-t4 ..." -> "colab-t4"), or None.

    Empty notes cells come back from pandas as NaN, so anything that isn't a
    string has no tag.
    """
    if not isinstance(notes, str):
        return None
    m = re.search(r"hw=([\w.-]+)", notes)
    return m.group(1) if m else None


def note_value(notes, key):
    """A numeric `key=value` from a notes string as a float, or None."""
    if not isinstance(notes, str):
        return None
    m = re.search(rf"(?:^|\s){re.escape(key)}=([-+\d.eE]+)", notes)
    return float(m.group(1)) if m else None


def own_rows(df, challenge, method):
    """Only one method's rows of a per-person results table."""
    return df[(df["challenge"] == challenge) & (df["method"] == method)]


def missing_seeds(path, config_id, seeds, hw, *, challenge, method):
    """Seeds with no row yet for (challenge, method, config_id, hw) in the results CSV at `path`."""
    if not os.path.exists(path):
        return list(seeds)
    with open(path, newline="") as f:
        done = {int(r["seed"]) for r in csv.DictReader(f)
                if r["challenge"] == challenge and r["method"] == method
                and r["config_id"] == config_id and hw_of(r["notes"]) == hw}
    return [s for s in seeds if s not in done]


def merge_results(src, dst):
    """Append the rows of results CSV `src` that `dst` doesn't have yet. Returns how many.

    Used by the Colab notebooks: they log to a copy on Google Drive (survives a
    runtime reset) and merge that copy back into the repo's results file.
    """
    with open(dst, newline="") as f:
        have = {tuple(r) for r in csv.reader(f)}
    with open(src, newline="") as f:
        new = [r for r in list(csv.reader(f))[1:] if tuple(r) not in have]
    with open(dst, "a", newline="") as f:
        csv.writer(f).writerows(new)
    return len(new)


def summarize_configs(df, seeds, hw, *, challenge, method):
    """Per config: mean/std accuracy and mean train+infer time over `seeds` on `hw`.

    Only configs that have a row for every seed are kept.
    """
    df = own_rows(df, challenge, method)
    d = df[df["seed"].astype(int).isin(seeds) & (df["notes"].map(hw_of) == hw)].copy()
    d["time"] = d["train_time_s"].astype(float) + d["infer_time_s"].astype(float)
    g = d.groupby("config_id")
    s = pd.DataFrame({
        "acc_mean": g["val_acc"].mean(),
        "acc_std": g["val_acc"].std(ddof=0),
        "time_mean": g["time"].mean(),
        "n": g["seed"].nunique(),
    })
    return s[s["n"] == len(seeds)]


def select(summary, tie_pp=0.5, fast_pp=2.0):
    """(best_id, fast_id) from a summarize_configs table.

    best = highest mean accuracy; a config within `tie_pp` pp of it wins instead
           if it is faster (lowest train + inference time among those).
    fast = lowest train + inference time among configs with mean accuracy
           >= best - `fast_pp` pp.
    """
    top = summary["acc_mean"].max()
    near_top = summary[summary["acc_mean"] >= top - tie_pp / 100 - EPS]
    best = near_top["time_mean"].idxmin()
    good = summary[summary["acc_mean"] >= summary.loc[best, "acc_mean"] - fast_pp / 100 - EPS]
    fast = good["time_mean"].idxmin()
    return best, fast
