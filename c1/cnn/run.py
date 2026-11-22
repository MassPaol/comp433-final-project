"""Run C1 CNN configs through the shared harness.

    python -m c1.cnn.run --stage s0 --seeds tune --reps 2 --hw colab-t4
    python -m c1.cnn.run --config best fast --seeds report --hw colab-t4

Rows go to results/ryan.csv with notes "hw=<tag> rep=<k>". A rep is an
independent training run (method seed k) on the same problem instance: neural
nets add training noise on top of the class-pair variance, so close comparisons
average a few reps. Only (config, seed, hw, rep) combinations without a row yet
are run, so an interrupted sweep restarts with the same command.
"""
import argparse
import csv
import os
import re

import numpy as np

from c1.classical.run import hw_of, merge_results  # noqa: F401  (merge_results re-exported for the notebook)
from c1.cnn import configs
from c1.cnn.method import CNNMethod
from common import data, harness

PERSON = "ryan"
CHALLENGE = "c1"
METHOD = "cnn"
DEFAULT_HW = "local-cpu"
SEED_SETS = {"tune": data.TUNE_SEEDS, "report": data.REPORT_SEEDS}


def rep_of(notes):
    """The rep number in a notes string ("hw=colab-t4 rep=1" -> 1); rows without one are rep 0."""
    if not isinstance(notes, str):
        return 0
    m = re.search(r"rep=(\d+)", notes)
    return int(m.group(1)) if m else 0


def own_rows(df):
    """Only this method's rows: results/ryan.csv also holds Ryan's C2 method."""
    return df[(df["challenge"] == CHALLENGE) & (df["method"] == METHOD)]


def missing_seeds(path, config_id, seeds, hw, rep):
    """Seeds with no row yet for (config_id, hw, rep) in the results CSV at `path`."""
    if not os.path.exists(path):
        return list(seeds)
    with open(path, newline="") as f:
        done = {int(r["seed"]) for r in csv.DictReader(f)
                if r["challenge"] == CHALLENGE and r["method"] == METHOD
                and r["config_id"] == config_id and hw_of(r["notes"]) == hw
                and rep_of(r["notes"]) == rep}
    return [s for s in seeds if s not in done]


def run_configs(config_ids, seeds, hw, reps=1, results_dir="results", overrides=None):
    """Run every (config, rep) on `seeds`. `overrides` (e.g. {"epochs": 2}) is for smoke tests only."""
    path = os.path.join(results_dir, f"{PERSON}.csv")
    for i, cid in enumerate(config_ids, 1):
        kw = {**configs.get(cid), **(overrides or {})}
        for rep in range(reps):
            todo = missing_seeds(path, cid, seeds, hw, rep)
            tag = f"[{i}/{len(config_ids)}] {cid} rep {rep}"
            if not todo:
                print(f"{tag}: already done, skipped", flush=True)
                continue
            rows = harness.run(lambda kw=kw, rep=rep: CNNMethod(seed=rep, **kw), todo,
                               challenge=CHALLENGE, person=PERSON, method_name=METHOD,
                               config_id=cid, notes=f"hw={hw} rep={rep}", results_dir=results_dir)
            acc = np.array([r["val_acc"] for r in rows])
            fit = np.mean([r["train_time_s"] for r in rows])
            print(f"{tag}: acc {acc.mean():.4f} +/- {acc.std():.4f} | fit {fit:.1f}s | seeds {todo}",
                  flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--stage", choices=sorted(configs.STAGES))
    which.add_argument("--config", nargs="+")
    p.add_argument("--seeds", choices=sorted(SEED_SETS), required=True)
    p.add_argument("--reps", type=int, default=None,
                   help="default: configs.STAGE_REPS for --stage, 1 for --config")
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--results-dir", default="results")
    p.add_argument("--smoke-epochs", type=int, default=None,
                   help="override epochs (smoke tests only; use a separate --hw and --results-dir)")
    a = p.parse_args(argv)
    ids = list(configs.STAGES[a.stage]) if a.stage else a.config
    reps = a.reps or (configs.STAGE_REPS.get(a.stage, 1) if a.stage else 1)
    overrides = {"epochs": a.smoke_epochs} if a.smoke_epochs else None
    run_configs(ids, SEED_SETS[a.seeds], a.hw, reps, a.results_dir, overrides)


if __name__ == "__main__":
    main()
