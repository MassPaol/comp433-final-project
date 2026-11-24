"""Rank C1 CNN configs on the tuning seeds, apply the selection rule, compare pairs.

    python -m c1.cnn.select --hw colab-t4                  # all stages
    python -m c1.cnn.select --hw colab-t4 --stage s2       # one stage
    python -m c1.cnn.select --hw colab-t4 --pair s0_noaug s0_flipcrop
    python -m c1.cnn.select --hw colab-t4 --stage s1 --compare-hw local-cpu   # T4 vs laptop

--results takes one or more CSVs (default: results/ryan.csv and the git-ignored
results/local/ryan.csv, whichever exist), so T4 and laptop rows can be compared
without the laptop rows ever being committed.

Accuracy per (config, seed) is first averaged over reps (training noise), then
mean/std are taken over seeds (std = spread across class pairs).

Selection rule: the team's rule from c1/classical/select.py, fixed before any
CNN results were seen:
  best = highest mean tuning accuracy; a config within 0.5 pp of it wins instead
         if it is faster (lowest train + inference time among those).
  fast = lowest train + inference time among configs with mean accuracy
         >= best - 2 pp.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from c1.classical.select import select  # noqa: F401  (same rule as the rest of the team)
from c1.cnn import configs
from c1.cnn.run import (DEFAULT_HW, LOCAL_RESULTS_DIR, PERSON, RESULTS_DIR, SEED_SETS, hw_of,
                        own_rows, rep_of)

DEFAULT_RESULTS = [f"{RESULTS_DIR}/{PERSON}.csv", f"{LOCAL_RESULTS_DIR}/{PERSON}.csv"]


def load_results(paths):
    """Concatenate the results CSVs in `paths` that exist (rows are told apart by their hw tag)."""
    found = [p for p in paths if os.path.exists(p)]
    if not found:
        raise SystemExit(f"none of these results files exist: {paths}")
    return pd.concat([pd.read_csv(p) for p in found], ignore_index=True)


def per_seed(df, seeds, hw):
    """One row per (config_id, seed): accuracy and time averaged over reps, plus the rep count."""
    d = own_rows(df)
    d = d[d["seed"].astype(int).isin(seeds) & (d["notes"].map(hw_of) == hw)].copy()
    d["rep"] = d["notes"].map(rep_of)
    d["time"] = d["train_time_s"].astype(float) + d["infer_time_s"].astype(float)
    d["val_acc"] = d["val_acc"].astype(float)
    g = d.groupby(["config_id", "seed"])
    return pd.DataFrame({"val_acc": g["val_acc"].mean(), "time": g["time"].mean(),
                         "train_time_s": g["train_time_s"].mean(),
                         "reps": g["rep"].nunique()}).reset_index()


def summarize_configs(df, seeds, hw):
    """Per config: mean/std accuracy over seeds, mean time, min reps. Only configs with every seed."""
    p = per_seed(df, seeds, hw)
    g = p.groupby("config_id")
    s = pd.DataFrame({
        "acc_mean": g["val_acc"].mean(),
        "acc_std": g["val_acc"].std(ddof=0),
        "time_mean": g["time"].mean(),
        "n": g["seed"].nunique(),
        "reps": g["reps"].min(),
    })
    return s[s["n"] == len(seeds)]


def paired(df, a, b, seeds, hw):
    """Paired comparison a - b over the seeds both have (rep-averaged), in percentage points."""
    p = per_seed(df, seeds, hw).pivot(index="seed", columns="config_id", values="val_acc")
    if a not in p.columns or b not in p.columns:  # one of them has no rows yet
        return {"mean_diff_pp": float("nan"), "wins": 0, "losses": 0, "n": 0,
                "wilcoxon_p": float("nan"), "per_seed": {}}
    d = ((p[a] - p[b]) * 100).dropna()
    nonzero = d[d != 0]
    pval = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else float("nan")
    return {"mean_diff_pp": float(d.mean()), "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()), "n": int(len(d)), "wilcoxon_p": pval,
            "per_seed": d.round(2).to_dict()}


def compare_hw(df, seeds, hw, other, config_ids=None):
    """Side-by-side accuracy and rank of each config on two hardware tags (e.g. T4 vs laptop).

    Used to check whether decisions made from laptop rows hold on the T4 rows the report uses.
    """
    a = summarize_configs(df, seeds, hw)
    b = summarize_configs(df, seeds, other)
    if config_ids is not None:
        a, b = a[a.index.isin(config_ids)], b[b.index.isin(config_ids)]
    t = pd.DataFrame({f"acc_{hw}": a["acc_mean"] * 100, f"acc_{other}": b["acc_mean"] * 100})
    t = t.dropna()
    t["diff_pp"] = t[f"acc_{hw}"] - t[f"acc_{other}"]
    t[f"rank_{hw}"] = t[f"acc_{hw}"].rank(ascending=False).astype(int)
    t[f"rank_{other}"] = t[f"acc_{other}"].rank(ascending=False).astype(int)
    return t.sort_values(f"rank_{hw}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Rank configs, pick best/fast, or compare two configs.")
    p.add_argument("--stage", choices=sorted(configs.STAGES))
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--seeds", choices=sorted(SEED_SETS), default="tune")
    p.add_argument("--results", nargs="+", default=DEFAULT_RESULTS,
                   help="one or more results CSVs (missing ones are skipped)")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--pair", nargs=2, metavar=("A", "B"))
    p.add_argument("--compare-hw", metavar="OTHER_HW",
                   help="show each config's accuracy and rank on --hw next to OTHER_HW")
    a = p.parse_args(argv)

    df = load_results(a.results)
    seeds = SEED_SETS[a.seeds]
    if a.compare_hw:
        ids = list(configs.STAGES[a.stage]) if a.stage else None
        t = compare_hw(df, seeds, a.hw, a.compare_hw, ids)
        if t.empty:
            raise SystemExit(f"no configs complete on both {a.hw} and {a.compare_hw}")
        pd.set_option("display.width", 140)
        print(t.round(2).to_string())
        print(f"top config: {a.hw} -> {t.index[0]}, "
              f"{a.compare_hw} -> {t[f'acc_{a.compare_hw}'].idxmax()}")
        return
    if a.pair:
        r = paired(df, *a.pair, seeds, a.hw)
        print(f"{a.pair[0]} - {a.pair[1]}: {r['mean_diff_pp']:+.2f} pp, wins {r['wins']}/{r['n']} "
              f"(losses {r['losses']}), Wilcoxon p = {r['wilcoxon_p']:.3f}")
        print("per seed (pp):", r["per_seed"])
        return
    s = summarize_configs(df, seeds, a.hw)
    if a.stage:
        s = s[s.index.isin(configs.STAGES[a.stage])]
    if s.empty:
        raise SystemExit("no complete configs found")
    pd.set_option("display.width", 120)
    print(s.sort_values("acc_mean", ascending=False).head(a.top).round(4).to_string())
    best, fast = select(s)
    for name, cid in (("best", best), ("fast", fast)):
        r = s.loc[cid]
        print(f"{name}: {cid}  acc {r.acc_mean:.4f} +/- {r.acc_std:.4f}  time {r.time_mean:.2f}s")


if __name__ == "__main__":
    main()
