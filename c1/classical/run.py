"""Run C1 classical configs through the shared harness.

    python -m c1.classical.run --stage s1 --seeds tune
    python -m c1.classical.run --config best fast --seeds report --hw colab-t4

Rows go to results/massimo.csv with notes "hw=<tag>". Only (config, seed)
pairs without a row for that hw tag are run, so an interrupted sweep can be
restarted with the same command without duplicating rows.
"""
import argparse
import csv
import os
import re

import numpy as np

from c1.classical import configs
from c1.classical.method import ClassicalMethod
from common import data, harness

PERSON = "massimo"
CHALLENGE = "c1"
METHOD = "classical"
DEFAULT_HW = "local-3070"
SEED_SETS = {"tune": data.TUNE_SEEDS, "report": data.REPORT_SEEDS}


def hw_of(notes):
    """The hw tag in a notes string ("hw=colab-t4 ..." -> "colab-t4"), or None.

    Empty notes cells come back from pandas as NaN, so anything that isn't a
    string has no tag.
    """
    if not isinstance(notes, str):
        return None
    m = re.search(r"hw=([\w.-]+)", notes)
    return m.group(1) if m else None


def own_rows(df):
    """Only this method's rows: results/<person>.csv also holds the person's other methods."""
    return df[(df["challenge"] == CHALLENGE) & (df["method"] == METHOD)]


def missing_seeds(path, config_id, seeds, hw):
    """Seeds that have no C1-classical row yet for (config_id, hw) in the results CSV at `path`."""
    if not os.path.exists(path):
        return list(seeds)
    with open(path, newline="") as f:
        done = {int(r["seed"]) for r in csv.DictReader(f)
                if r["challenge"] == CHALLENGE and r["method"] == METHOD
                and r["config_id"] == config_id and hw_of(r["notes"]) == hw}
    return [s for s in seeds if s not in done]


def merge_results(src, dst):
    """Append the rows of results CSV `src` that `dst` doesn't have yet. Returns how many.

    Used by the Colab notebook: it logs to a copy on Google Drive (survives a
    runtime reset) and merges that copy back into the repo's results file.
    """
    with open(dst, newline="") as f:
        have = {tuple(r) for r in csv.reader(f)}
    with open(src, newline="") as f:
        new = [r for r in list(csv.reader(f))[1:] if tuple(r) not in have]
    with open(dst, "a", newline="") as f:
        csv.writer(f).writerows(new)
    return len(new)


def run_configs(config_ids, seeds, hw, results_dir="results"):
    path = os.path.join(results_dir, f"{PERSON}.csv")
    for i, cid in enumerate(config_ids, 1):
        kw = configs.get(cid)
        todo = missing_seeds(path, cid, seeds, hw)
        if not todo:
            print(f"[{i}/{len(config_ids)}] {cid}: already done, skipped", flush=True)
            continue
        rows = harness.run(lambda kw=kw: ClassicalMethod(**kw), todo,
                           challenge=CHALLENGE, person=PERSON, method_name=METHOD,
                           config_id=cid, notes=f"hw={hw}", results_dir=results_dir)
        acc = np.array([r["val_acc"] for r in rows])
        fit = np.mean([r["train_time_s"] for r in rows])
        print(f"[{i}/{len(config_ids)}] {cid}: acc {acc.mean():.4f} +/- {acc.std():.4f} "
              f"| fit {fit:.2f}s | seeds {todo}", flush=True)


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
