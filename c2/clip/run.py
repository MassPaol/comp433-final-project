"""C2 CLIP methods through the shared harness.

    python -m c2.clip.run --stage probe     # tuning seeds 0-4
    python -m c2.clip.run --select          # best and fast per stage (shared rule: common/sweep.py)
    python -m c2.clip.run --final           # frozen FINAL configs, reporting seeds 10-19

Writes two files per person:
    results/alvin.csv         harness rows (schema unchanged)
    results/alvin_timing.csv  predict split per seed: CPU preprocessing and CLIP encode
"""
import argparse
import csv
import os

import pandas as pd

from c2.clip.configs import FINAL, STAGES
from c2.clip.linearProbe import LinearProbe
from c2.clip.tipadapter import TipAdapter, TipAdapterF
from c2.clip.wiseft import WiSEFT
from c2.clip.zeroshot import ZeroShotMethod
from common import data, harness
from common.sweep import missing_seeds, select, summarize_configs

PERSON, CHALLENGE = "alvin", "c2"
METHOD = {"zeroshot": ZeroShotMethod, "probe": LinearProbe, "wiseft": WiSEFT,
          "tip": TipAdapter, "tipf": TipAdapterF}
TIMING_COLUMNS = ["challenge", "hw", "config_id", "seed", "infer_preprocess_s", "infer_encode_s"]

# Filled by Timed.predict; written to the timing CSV after each harness.run call.
_breakdown = []


def results_path(results_dir="results"):
    return os.path.join(results_dir, f"{PERSON}.csv")


def timing_path(results_dir="results"):
    return os.path.join(results_dir, f"{PERSON}_timing.csv")


def stage_of(config_id):
    for stage, configs in STAGES.items():
        if config_id in configs:
            return stage, configs[config_id]
    raise KeyError(config_id)


class Timed:
    """Wraps a method for harness.run. Same fit/predict/n_trainable_params interface.
    predict also records the preprocess/encode split from the method's `timing_` dict,
    so common/harness.py does not need to change."""

    def __init__(self, inner, config_id, seed):
        self.inner, self.config_id, self.seed = inner, config_id, seed
        self.name = inner.name

    def fit(self, X, y):
        self.inner.fit(X, y)
        return self

    def predict(self, X):
        self.inner.timing_ = {}
        out = self.inner.predict(X)
        t = self.inner.timing_ or {}
        _breakdown.append({"config_id": self.config_id, "seed": self.seed,
                           "infer_preprocess_s": t.get("preprocess_s", ""),
                           "infer_encode_s": t.get("encode_s", "")})
        return out

    def n_trainable_params(self):
        return self.inner.n_trainable_params()


def factory(cid, stage, kwargs, seed):
    _, _, _, _, classes = data.load_instance(seed)
    names = [data.CLASS_NAMES[c] for c in classes]
    return Timed(METHOD[stage](class_names=names, seed=seed, **kwargs), cid, seed)


def _append_timing(results_dir, hw, records):
    os.makedirs(results_dir, exist_ok=True)
    path = timing_path(results_dir)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_COLUMNS)
        if write_header:
            writer.writeheader()
        for r in records:
            writer.writerow({"challenge": CHALLENGE, "hw": hw, **r})


def run(config_ids, seeds, hw, results_dir="results"):
    """Run every (config, seed) that has no row yet, so re-running only fills gaps."""
    path = results_path(results_dir)
    for cid in config_ids:
        stage, kwargs = stage_of(cid)
        todo = missing_seeds(path, cid, seeds, hw, challenge=CHALLENGE,
                             method=METHOD[stage].name)
        for seed in todo:
            _breakdown.clear()
            harness.run(lambda c=cid, st=stage, kw=kwargs, s=seed: factory(c, st, kw, s),
                        [seed], challenge=CHALLENGE, person=PERSON,
                        method_name=METHOD[stage].name, config_id=cid,
                        notes=f"hw={hw}", results_dir=results_dir)
            _append_timing(results_dir, hw, list(_breakdown))


def tuning_summary(df, hw, seeds=data.TUNE_SEEDS):
    """Accuracy and time per config on tuning seeds, for every stage. Columns:
    acc_mean, acc_std (fractions), time_mean (s), n."""
    parts = []
    for stage in STAGES:
        s = summarize_configs(df, seeds, hw, challenge=CHALLENGE, method=METHOD[stage].name)
        parts.append(s[s.index.isin(list(STAGES[stage]))])
    return pd.concat(parts)


def pick(df, hw, seeds=data.TUNE_SEEDS):
    """best and fast per stage, using the shared rule in common/sweep.py."""
    out = {}
    for stage in STAGES:
        s = summarize_configs(df, seeds, hw, challenge=CHALLENGE, method=METHOD[stage].name)
        s = s[s.index.isin(list(STAGES[stage]))]
        missing = set(STAGES[stage]) - set(s.index)
        if missing:
            print(f"warning: {stage}: no complete tuning runs for {sorted(missing)}")
        if not s.empty:
            out[f"{stage}_best"], out[f"{stage}_fast"] = select(s)
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

    if a.stage:
        run(list(STAGES[a.stage]), data.TUNE_SEEDS, a.hw, a.results_dir)
    elif a.final:
        run(list(FINAL.values()), data.REPORT_SEEDS, a.hw, a.results_dir)
    else:
        df = pd.read_csv(results_path(a.results_dir))
        df["notes"] = df["notes"].fillna("")
        print(tuning_summary(df, a.hw).sort_values("acc_mean", ascending=False)
              .round(4).to_string())
        print(pick(df, a.hw))


if __name__ == "__main__":
    main()