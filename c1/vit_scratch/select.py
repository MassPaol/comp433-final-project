"""Rank C1 ViT-from-scratch configs on the tuning seeds and apply the team selection rule.

    python -m c1.vit_scratch.select --hw colab-t4              # all stages
    python -m c1.vit_scratch.select --hw colab-t4 --stage s3   # one stage

Rule (common/sweep.py, the same for every method):
  best = highest mean tuning accuracy; a config within 0.5 pp of it wins instead
         if it is faster (lowest train + inference time among those).
  fast = lowest train + inference time among configs with mean accuracy
         >= best - 2 pp.
"""
import argparse

import pandas as pd

from c1.vit_scratch import configs
from common import data, sweep

CHALLENGE, METHOD = "c1", "vit_scratch"


def summarize(df, hw, stage=None):
    """Per-config summary on the tuning seeds; stage configs only (not best/fast rows)."""
    s = sweep.summarize_configs(df, data.TUNE_SEEDS, hw, challenge=CHALLENGE, method=METHOD)
    keep = configs.STAGES[stage] if stage else configs.all_configs()
    return s[s.index.isin(keep)]


def pick(df, hw, stage=None):
    """{"best": id, "fast": id} under the team rule."""
    best, fast = sweep.select(summarize(df, hw, stage))
    return {"best": best, "fast": fast}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stage", choices=sorted(configs.STAGES))
    p.add_argument("--hw", default="colab-t4")
    p.add_argument("--results", default="results/ammar.csv")
    p.add_argument("--top", type=int, default=12)
    a = p.parse_args(argv)
    df = pd.read_csv(a.results)
    s = summarize(df, a.hw, a.stage)
    if s.empty:
        raise SystemExit("no complete configs found")
    print(s.sort_values("acc_mean", ascending=False).head(a.top).round(4).to_string())
    picks = pick(df, a.hw, a.stage)
    for name, cid in picks.items():
        r = s.loc[cid]
        print(f"{name}: {cid}  acc {r.acc_mean:.4f} +/- {r.acc_std:.4f}  time {r.time_mean:.3f}s")
    if not a.stage and picks != configs.FINAL_SOURCE:
        print(f"NOTE: configs.FINAL_SOURCE is {configs.FINAL_SOURCE}: update it to match.")


if __name__ == "__main__":
    main()
