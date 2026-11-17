"""C2 CLIP methods through the shared harness.

    python -m c2.clip.run --stage probe     # tuning seeds 0-4
    python -m c2.clip.run --select          # best and fast per method
    python -m c2.clip.run --final           # frozen FINAL configs, reporting seeds 10-19
"""
import argparse
import os

import pandas as pd

from c2.clip.configs import FINAL, STAGES
from c2.clip.linearProbe import LinearProbe
from c2.clip.tipadapter import TipAdapter, TipAdapterF
from c2.clip.wiseft import WiSEFT
from c2.clip.zeroshot import ZeroShotMethod
from common import data, harness

PERSON, CHALLENGE = "alvin", "c2"
METHOD = {"zeroshot": ZeroShotMethod, "probe": LinearProbe, "wiseft": WiSEFT,
          "tip": TipAdapter, "tipf": TipAdapterF}


def stage_of(config_id):
    for stage, configs in STAGES.items():
        if config_id in configs:
            return stage, configs[config_id]
    raise KeyError(config_id)


def factory(stage, kwargs, seed):
    _, _, _, _, classes = data.load_instance(seed)
    names = [data.CLASS_NAMES[c] for c in classes]
    return METHOD[stage](class_names=names, seed=seed, **kwargs)


def done_pairs(path, hw):
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path)
    df = df[(df.challenge == CHALLENGE) & df.notes.fillna("").str.contains(f"hw={hw}")]
    return set(zip(df.config_id, df.seed))


def run(config_ids, seeds, hw, results_dir="results"):
    done = done_pairs(os.path.join(results_dir, f"{PERSON}.csv"), hw)
    for cid in config_ids:
        stage, kwargs = stage_of(cid)
        for seed in seeds:
            if (cid, seed) in done:
                continue
            harness.run(lambda s=seed, st=stage, kw=kwargs: factory(st, kw, s), [seed],
                        challenge=CHALLENGE, person=PERSON, method_name=METHOD[stage].name,
                        config_id=cid, notes=f"hw={hw}", results_dir=results_dir)


def summarize(path, seeds, hw):
    df = pd.read_csv(path)
    df = df[(df.challenge == CHALLENGE) & df.seed.isin(seeds)
            & df.notes.fillna("").str.contains(f"hw={hw}")].copy()
    df["time"] = df.train_time_s + df.infer_time_s
    g = df.groupby("config_id")
    s = pd.DataFrame({"acc": g.val_acc.mean(), "std": g.val_acc.std(ddof=0),
                      "time": g.time.mean(), "n": g.seed.nunique()})
    return s[s.n == len(seeds)]


def pick(s):
    out = {}
    for stage, configs in STAGES.items():
        ids = [c for c in configs if c in s.index]
        if not ids:
            continue
        sub = s.loc[ids]
        best = sub.acc.idxmax()
        out[f"{stage}_best"] = best
        out[f"{stage}_fast"] = sub[sub.acc >= sub.acc[best] - 0.01].time.idxmin()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", choices=list(STAGES))
    g.add_argument("--select", action="store_true")
    g.add_argument("--final", action="store_true")
    p.add_argument("--hw", default="colab-t4")
    p.add_argument("--results-dir", default="results")
    a = p.parse_args()
    path = os.path.join(a.results_dir, f"{PERSON}.csv")
    if a.stage:
        run(list(STAGES[a.stage]), data.TUNE_SEEDS, a.hw, a.results_dir)
    elif a.final:
        run(list(FINAL.values()), data.REPORT_SEEDS, a.hw, a.results_dir)
    else:
        s = summarize(path, data.TUNE_SEEDS, a.hw)
        print(s.sort_values("acc", ascending=False).round(4).to_string())
        print(pick(s))


if __name__ == "__main__":
    main()
