"""Rank C1 self-supervised configs on the tuning seeds and pick best/fast for each method.

    python -m c1.ssl.select              # all stages
    python -m c1.ssl.select --stage ss2  # one stage

Rule (same as c1.classical; fixed before any results are seen), applied separately to
k-means configs and to SimSiam configs:
  best = highest mean tuning accuracy; a config within 0.5 pp of it wins instead
         if it is faster (lowest train + inference time among those).
  fast = lowest train + inference time among configs with mean accuracy >= best - 2 pp.
"""
import argparse

import numpy as np
import pandas as pd

from c1.classical.select import select as select_rule
from c1.ssl import configs
from c1.ssl.run import DEFAULT_HW, PERSON, SEED_SETS, hw_of, own_rows


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


def tuning_rows(summary, method=None):
    """Only configs from the tuning stages (optionally one method)."""
    keep = [configs.method_of(c) is not None and (method is None or configs.method_of(c) == method)
            for c in summary.index]
    return summary[np.array(keep, dtype=bool)]


def pick_all(summary):
    """{'kmeans_best': id, 'kmeans_fast': id, 'simsiam_best': id, 'simsiam_fast': id}."""
    out = {}
    for m in ("kmeans", "simsiam"):
        sub = tuning_rows(summary, m)
        if sub.empty:
            continue
        out[f"{m}_best"], out[f"{m}_fast"] = select_rule(sub)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Rank configs and pick best/fast per method.")
    p.add_argument("--stage", choices=sorted(configs.STAGES))
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--seeds", choices=sorted(SEED_SETS), default="tune")
    p.add_argument("--results", default=f"results/{PERSON}.csv")
    p.add_argument("--top", type=int, default=10)
    a = p.parse_args(argv)

    s = summarize_configs(pd.read_csv(a.results), SEED_SETS[a.seeds], a.hw)
    if a.stage:
        s = s[s.index.isin(configs.STAGES[a.stage])]
    if s.empty:
        raise SystemExit("no complete configs found")
    pd.set_option("display.width", 120)
    for m in ("kmeans", "simsiam"):
        sub = tuning_rows(s, m)
        if sub.empty:
            continue
        print(f"\n== {m} ==")
        print(sub.sort_values("acc_mean", ascending=False).head(a.top).round(4).to_string())
    print()
    for name, cid in pick_all(s).items():
        r = s.loc[cid]
        print(f"{name}: {cid}  acc {r.acc_mean:.4f} +/- {r.acc_std:.4f}  time {r.time_mean:.3f}s")


if __name__ == "__main__":
    main()
