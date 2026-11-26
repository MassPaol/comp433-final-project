"""Run C1 self-supervised configs (Coates k-means, SimSiam) through the shared harness.

    python -m c1.ssl.run --stage k1 --seeds tune --hw colab-t4
    python -m c1.ssl.run --config kmeans_best simsiam_best --seeds report --hw colab-t4

Rows go to results/alvin.csv with notes "hw=<tag> ..." when --hw is colab-t4 (the rows the report
uses, committed), and to results/local/alvin.csv for any other hardware tag (a first look only,
git-ignored). Only (config, seed) pairs without a row for that hw tag are run, so an interrupted
sweep restarts with the same command. Resume logic and CSV merging come from common/sweep.py.
"""
import argparse
import os

import numpy as np

from c1.ssl import configs
from c1.ssl.method_kmeans import KMeansMethod
from c1.ssl.method_simsiam import SimSiamMethod
from common import data, harness, sweep
from common.sweep import hw_of, merge_results  # noqa: F401  (re-exported for figures / the notebook)

PERSON = "alvin"
CHALLENGE = "c1"
METHODS = {"kmeans": KMeansMethod, "simsiam": SimSiamMethod}
DEFAULT_HW = "local-3070"
REPORT_HW = "colab-t4"
RESULTS_DIR = "results"              # report rows (hw=colab-t4), committed
LOCAL_RESULTS_DIR = "results/local"  # every other hw tag, git-ignored
SEED_SETS = {"tune": data.TUNE_SEEDS, "report": data.REPORT_SEEDS}


def default_results_dir(hw):
    """Where rows for hardware tag `hw` go when --results-dir isn't given."""
    return RESULTS_DIR if hw == REPORT_HW else LOCAL_RESULTS_DIR


def own_rows(df):
    """Only this part's rows: results/<person>.csv may also hold the person's C2 methods."""
    return df[(df["challenge"] == CHALLENGE) & (df["method"].isin(METHODS))]


def _smoke(method, kw, epochs):
    """Shrink a config for a pipeline smoke test (never for real runs)."""
    if method != "simsiam":
        return {**kw, "num_patches": 2000, "n_init": 1}
    kw = {**kw, "epochs": min(kw.get("epochs", 200), epochs)}
    if kw.get("ft_epochs", 0) > 0:
        kw["ft_epochs"] = min(kw["ft_epochs"], epochs)
    return kw


def run_configs(config_ids, seeds, hw, results_dir="results", smoke_epochs=None):
    path = os.path.join(results_dir, f"{PERSON}.csv")
    for i, cid in enumerate(config_ids, 1):
        method, kw = configs.get(cid)
        if smoke_epochs:
            kw = _smoke(method, kw, smoke_epochs)
        todo = sweep.missing_seeds(path, cid, seeds, hw, challenge=CHALLENGE, method=method)
        if not todo:
            print(f"[{i}/{len(config_ids)}] {cid}: already done, skipped", flush=True)
            continue
        cls = METHODS[method]
        rows = harness.run(lambda cls=cls, kw=kw: cls(**kw), todo,
                           challenge=CHALLENGE, person=PERSON, method_name=method,
                           config_id=cid, notes=f"hw={hw}", results_dir=results_dir)
        acc = np.array([r["val_acc"] for r in rows])
        fit = np.mean([r["train_time_s"] for r in rows])
        print(f"[{i}/{len(config_ids)}] {cid} ({method}): acc {acc.mean():.4f} +/- {acc.std():.4f} "
              f"| fit {fit:.2f}s | seeds {todo}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--stage", choices=sorted(configs.STAGES))
    which.add_argument("--config", nargs="+")
    p.add_argument("--seeds", choices=sorted(SEED_SETS), required=True)
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--results-dir", default=None,
                   help=f"default: {RESULTS_DIR} for --hw {REPORT_HW}, else {LOCAL_RESULTS_DIR}")
    p.add_argument("--smoke-epochs", type=int, default=None,
                   help="shrink configs (smoke tests only; use a separate --hw and --results-dir)")
    a = p.parse_args(argv)
    ids = list(configs.STAGES[a.stage]) if a.stage else a.config
    run_configs(ids, SEED_SETS[a.seeds], a.hw, a.results_dir or default_results_dir(a.hw),
                a.smoke_epochs)


if __name__ == "__main__":
    main()