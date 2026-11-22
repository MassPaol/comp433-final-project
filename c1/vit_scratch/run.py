"""Run C1 ViT-from-scratch configs through the shared harness.

    python -m c1.vit_scratch.run --stage s0 --seeds tune --hw colab-t4
    python -m c1.vit_scratch.run --config baseline --seeds tune

Rows are appended to results/ammar.csv with notes "hw=<tag>". Resumable: (config,
seed) pairs that already have a row for that hw tag are skipped.
"""
import argparse
import os

import numpy as np

from c1.vit_scratch import configs
from c1.vit_scratch.method import ViTScratch
from common import data, harness
from common.sweep import missing_seeds

SEED_SETS = {"tune": data.TUNE_SEEDS, "report": data.REPORT_SEEDS}


def run_configs(config_ids, seeds, hw, results_dir="results"):
    path = os.path.join(results_dir, "ammar.csv")
    for i, cid in enumerate(config_ids, 1):
        kw = configs.get(cid)
        todo = missing_seeds(path, cid, seeds, hw, challenge="c1", method="vit_scratch")
        if not todo:
            print(f"[{i}/{len(config_ids)}] {cid}: already done, skipped", flush=True)
            continue
        seeds_run = todo
        rows = harness.run(lambda kw=kw: ViTScratch(**kw), seeds_run,
                           challenge="c1", person="ammar", method_name="vit_scratch",
                           config_id=cid, notes=f"hw={hw}", results_dir=results_dir)
        acc = np.array([r["val_acc"] for r in rows])
        fit = np.mean([r["train_time_s"] for r in rows])
        print(f"[{i}/{len(config_ids)}] {cid}: acc {acc.mean():.4f} +/- {acc.std():.4f} "
              f"| fit {fit:.1f}s | seeds {seeds_run}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--stage", choices=sorted(configs.STAGES))
    which.add_argument("--config", nargs="+")
    p.add_argument("--seeds", choices=sorted(SEED_SETS), required=True)
    p.add_argument("--hw", default="local")
    p.add_argument("--results-dir", default="results")
    a = p.parse_args(argv)
    ids = list(configs.STAGES[a.stage]) if a.stage else a.config
    run_configs(ids, SEED_SETS[a.seeds], a.hw, a.results_dir)


if __name__ == "__main__":
    main()
