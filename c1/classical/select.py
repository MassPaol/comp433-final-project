"""Rank C1 classical configs on the tuning seeds and apply the selection rule.

    python -m c1.classical.select              # all stages
    python -m c1.classical.select --stage s2   # one stage

Rule (fixed before any results were seen):
  best = highest mean tuning accuracy; a config within 0.5 pp of it wins instead
         if it is faster (lowest train + inference time among those).
  fast = lowest train + inference time among configs with mean accuracy
         >= best - 2 pp.
"""
import argparse

import pandas as pd

from c1.classical import configs
from c1.classical.run import DEFAULT_HW, PERSON, SEED_SETS, hw_of, own_rows

EPS = 1e-9


def summarize_configs(df, seeds, hw):
    """Per config: mean/std accuracy and mean train+infer time over `seeds` on `hw`.

    Only configs that have a row for every seed are kept.
    """
    df = own_rows(df)
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
    """(best_id, fast_id) from a summarize_configs table."""
    top = summary["acc_mean"].max()
    near_top = summary[summary["acc_mean"] >= top - tie_pp / 100 - EPS]
    best = near_top["time_mean"].idxmin()
    good = summary[summary["acc_mean"] >= summary.loc[best, "acc_mean"] - fast_pp / 100 - EPS]
    fast = good["time_mean"].idxmin()
    return best, fast


def main(argv=None):
    p = argparse.ArgumentParser(description="Rank configs and pick best/fast.")
    p.add_argument("--stage", choices=sorted(configs.STAGES))
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--seeds", choices=sorted(SEED_SETS), default="tune")
    p.add_argument("--results", default=f"results/{PERSON}.csv")
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args(argv)

    s = summarize_configs(pd.read_csv(a.results), SEED_SETS[a.seeds], a.hw)
    if a.stage:
        s = s[s.index.isin(configs.STAGES[a.stage])]
    if s.empty:
        raise SystemExit("no complete configs found")
    pd.set_option("display.width", 120)
    print(s.sort_values("acc_mean", ascending=False).head(a.top).round(4).to_string())
    best, fast = select(s)
    for name, cid in (("best", best), ("fast", fast)):
        r = s.loc[cid]
        print(f"{name}: {cid}  acc {r.acc_mean:.4f} +/- {r.acc_std:.4f}  time {r.time_mean:.3f}s")


if __name__ == "__main__":
    main()
