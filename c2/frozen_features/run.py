"""Run C2 frozen-features configs through the shared harness.

    python -m c2.frozen_features.run --stage s1 --seeds tune
    python -m c2.frozen_features.run --config best fast baseline --seeds report --hw colab-t4

Rows go to results/massimo.csv with notes "hw=<tag> load_s=... extract_s=...".
Only (config, seed) pairs without a row for that hw tag are run, so an
interrupted sweep can be restarted with the same command without duplicating rows.
Before each config, an untimed warm-up downloads the weights (first time only)
and runs one fit/predict on dummy images, so CUDA start-up and downloads never
land inside a timed row.
"""
import argparse
import os

import numpy as np

from c2.frozen_features import backbones as B
from c2.frozen_features import configs
from c2.frozen_features.method import FrozenFeatureMethod
from common import data, harness, sweep

PERSON = "massimo"
CHALLENGE = "c2"
METHOD = "frozen_features"
DEFAULT_HW = "local-3070"
SEED_SETS = {"tune": data.TUNE_SEEDS, "report": data.REPORT_SEEDS}


def own_rows(df):
    """Only this method's rows: results/<person>.csv also holds the person's other methods."""
    return sweep.own_rows(df, CHALLENGE, METHOD)


def missing_seeds(path, config_id, seeds, hw):
    """Seeds that have no C2 frozen-features row yet for (config_id, hw) in the CSV at `path`."""
    return sweep.missing_seeds(path, config_id, seeds, hw, challenge=CHALLENGE, method=METHOD)


def warm_up(kw):
    """Untimed: download weights if needed and run the whole pipeline once on dummy images."""
    rng = np.random.RandomState(0)
    X = rng.randint(0, 256, (20, 32, 32, 3)).astype(np.uint8)
    y = np.array([0, 1] * 10)
    FrozenFeatureMethod(**kw).fit(X, y).predict(X)
    B.clear_cache()


def run_configs(config_ids, seeds, hw, results_dir="results"):
    path = os.path.join(results_dir, f"{PERSON}.csv")
    for i, cid in enumerate(config_ids, 1):
        kw = configs.get(cid)
        todo = missing_seeds(path, cid, seeds, hw)
        if not todo:
            print(f"[{i}/{len(config_ids)}] {cid}: already done, skipped", flush=True)
            continue
        warm_up(kw)
        rows = harness.run(lambda kw=kw: FrozenFeatureMethod(**kw), todo,
                           challenge=CHALLENGE, person=PERSON, method_name=METHOD,
                           config_id=cid, notes=f"hw={hw}", results_dir=results_dir)
        B.clear_cache()
        acc = np.array([r["val_acc"] for r in rows])
        fit = np.mean([r["train_time_s"] for r in rows])
        infer = np.mean([r["infer_time_s"] for r in rows])
        print(f"[{i}/{len(config_ids)}] {cid}: acc {acc.mean():.4f} +/- {acc.std():.4f} "
              f"| fit {fit:.2f}s infer {infer:.2f}s | seeds {todo}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--stage", choices=sorted(configs.STAGES))
    which.add_argument("--config", nargs="+")
    p.add_argument("--seeds", choices=sorted(SEED_SETS), required=True)
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--results-dir", default="results")
    a = p.parse_args(argv)
    ids = list(configs.STAGES[a.stage]) if a.stage else a.config
    run_configs(ids, SEED_SETS[a.seeds], a.hw, a.results_dir)


if __name__ == "__main__":
    main()
