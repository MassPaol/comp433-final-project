"""Rank C2 frozen-features configs on the tuning seeds and apply the selection rule.

    python -m c2.frozen_features.select              # all stages
    python -m c2.frozen_features.select --stage s2   # one stage

Rule (the same as C1's, fixed before any results were seen):
  best = highest mean tuning accuracy; a config within 0.5 pp of it wins instead
         if it is faster (lowest train + inference time among those).
  fast = lowest train + inference time among configs with mean accuracy
         >= best - 2 pp.
"""
import argparse

import pandas as pd

from c2.frozen_features import configs
from c2.frozen_features.run import CHALLENGE, DEFAULT_HW, METHOD, PERSON, SEED_SETS
from common import sweep
from common.sweep import select  # noqa: F401  (re-exported: same rule for every method)


def summarize_configs(df, seeds, hw):
    """Per config: mean/std accuracy and mean train+infer time over `seeds` on `hw`."""
    return sweep.summarize_configs(df, seeds, hw, challenge=CHALLENGE, method=METHOD)


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
    else:
        s = s[s.index.str.match(r"s\d_")]  # stage configs only, not best/fast/baseline
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
