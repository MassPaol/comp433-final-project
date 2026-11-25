"""Rank C1 self-supervised configs on the tuning seeds and pick best/fast for each method.

    python -m c1.ssl.select --hw colab-t4                 # all stages
    python -m c1.ssl.select --hw colab-t4 --stage ss2     # one stage
    python -m c1.ssl.select --hw colab-t4 --pair ss6_joint_ra ss1_e200

The selection is the team rule, `select` in common/sweep.py (fixed before any results), applied
separately to the k-means configs and to the SimSiam configs:
  best = highest mean tuning accuracy; a config within 0.5 pp of it wins instead if it is
         faster (lowest train + inference time among those).
  fast = lowest train + inference time among configs with mean accuracy >= best - 2 pp.
"""
import argparse

import pandas as pd
from scipy.stats import wilcoxon

from c1.ssl import configs
from c1.ssl.run import (DEFAULT_HW, METHODS, PERSON, SEED_SETS, default_results_dir, hw_of,
                        own_rows)
from common import sweep
from common.sweep import select as select_rule


def summarize_configs(df, seeds, hw):
    """Per config: mean/std accuracy and mean train+infer time over `seeds` on `hw`.

    Only configs with a row for every seed are kept (common.sweep.summarize_configs, per method).
    """
    parts = [sweep.summarize_configs(df, seeds, hw, challenge="c1", method=m) for m in METHODS]
    return pd.concat(parts)


def tuning_rows(summary, method=None):
    """Only configs from the tuning stages (optionally one method)."""
    keep = [c in {k for st in configs.STAGES.values() for k in st}
            and (method is None or configs.method_of(c) == method) for c in summary.index]
    return summary[pd.Series(keep, index=summary.index, dtype=bool)]


def pick_all(summary):
    """{'kmeans_best': id, 'kmeans_fast': id, 'simsiam_best': id, 'simsiam_fast': id}."""
    out = {}
    for m in ("kmeans", "simsiam"):
        sub = tuning_rows(summary, m)
        if sub.empty:
            continue
        out[f"{m}_best"], out[f"{m}_fast"] = select_rule(sub)
    return out


def paired(df, a, b, seeds, hw):
    """Paired comparison a - b over the seeds both have, in percentage points."""
    d = own_rows(df)
    d = d[d["seed"].astype(int).isin(seeds) & (d["notes"].map(hw_of) == hw)
          & d["config_id"].isin([a, b])]
    p = d.pivot_table(index="seed", columns="config_id", values="val_acc")
    if a not in p.columns or b not in p.columns:
        return {"mean_diff_pp": float("nan"), "wins": 0, "losses": 0, "n": 0,
                "wilcoxon_p": float("nan"), "per_seed": {}}
    diff = ((p[a] - p[b]) * 100).dropna()
    nonzero = diff[diff != 0]
    pval = float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else float("nan")
    return {"mean_diff_pp": float(diff.mean()), "wins": int((diff > 0).sum()),
            "losses": int((diff < 0).sum()), "n": int(len(diff)), "wilcoxon_p": pval,
            "per_seed": diff.round(2).to_dict()}


def main(argv=None):
    p = argparse.ArgumentParser(description="Rank configs and pick best/fast per method.")
    p.add_argument("--stage", choices=sorted(configs.STAGES))
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--seeds", choices=sorted(SEED_SETS), default="tune")
    p.add_argument("--results", default=None, help="default: the results file for --hw")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--pair", nargs=2, metavar=("A", "B"))
    a = p.parse_args(argv)

    path = a.results or f"{default_results_dir(a.hw)}/{PERSON}.csv"
    df = pd.read_csv(path)
    seeds = SEED_SETS[a.seeds]
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