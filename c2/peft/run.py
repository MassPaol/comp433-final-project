"""Run C2 PEFT configs through the shared harness.

    python -m c2.peft.run --stage p0 --seeds tune --hw colab-t4
    python -m c2.peft.run --config bitfit_lr1e-3 --seeds tune

Rows are appended to results/ammar.csv with notes "hw=<tag>". Resumable: (config,
seed) pairs that already have a row for that hw tag are skipped.
"""
import argparse
import os

import numpy as np

from c2.peft import configs
from c2.peft.method import PEFTMethod
from common import data, harness
from c1.vit_scratch.resume import missing_seeds

SEED_SETS = {"tune": data.TUNE_SEEDS, "report": data.REPORT_SEEDS}


def run_configs(config_ids, seeds, hw, results_dir="results"):
    path = os.path.join(results_dir, "ammar.csv")
    for i, cid in enumerate(config_ids, 1):
        kw = configs.get(cid)
        todo = missing_seeds(path, "c2", "peft", cid, seeds, hw)
        if not todo:
            print(f"[{i}/{len(config_ids)}] {cid}: already done, skipped", flush=True)
            continue
        seeds_run = todo
        rows = harness.run(lambda kw=kw: PEFTMethod(**kw), seeds_run,
                           challenge="c2", person="ammar", method_name="peft",
                           config_id=cid, notes=f"hw={hw}", results_dir=results_dir)
        acc = np.array([r["val_acc"] for r in rows])
        fit = np.mean([r["train_time_s"] for r in rows])
        print(f"[{i}/{len(config_ids)}] {cid}: acc {acc.mean():.4f} +/- {acc.std():.4f} "
              f"| fit {fit:.1f}s | trainable {rows[0]['n_trainable_params']} "
              f"| seeds {seeds_run}", flush=True)


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
