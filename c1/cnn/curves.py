"""Diagnostic training curves on the TUNING seeds only.

    python -m c1.cnn.curves --config s0_noaug s0_flipcrop --every 10 --hw colab-t4

For each (config, tuning seed), trains once and records the training loss every
epoch and the validation accuracy every `--every` epochs. Used to choose epoch
grids (a fixed epoch count chosen on tuning seeds is ordinary hyperparameter
selection) and for the training-dynamics figure. It is never used to stop a run
early: `fit` itself never sees validation data, and report seeds are refused.

Rows go to results/ryan_curves.csv for --hw colab-t4, else results/local/ryan_curves.csv:
    config_id, seed, class_pair, epoch, train_loss, val_acc, hw
(val_acc is empty on epochs where it was not measured).
"""
import argparse
import csv
import os

import numpy as np
import torch

from c1.cnn import configs
from c1.cnn.method import CNNMethod
from c1.cnn.run import DEFAULT_HW, PERSON, default_results_dir
from common import data

COLUMNS = ["config_id", "seed", "class_pair", "epoch", "train_loss", "val_acc", "hw"]


def curve(kw, seed, every=10, rep=0):
    """[(epoch, train_loss, val_acc or None)] for one training run on tuning seed `seed`."""
    if seed not in data.TUNE_SEEDS:
        raise ValueError(f"curves are for tuning seeds {data.TUNE_SEEDS} only, got {seed}")
    X, y, Xv, yv, classes = data.load_instance(seed)
    kw = {**kw, "ensemble": 1}
    method = CNNMethod(seed=rep, **kw)
    xv = None
    accs = {}

    def on_epoch(member, epoch, model):
        nonlocal xv
        if (epoch + 1) % every and epoch + 1 != method.epochs:
            return
        if xv is None:
            xv = method._to_tensor(Xv)
        with torch.no_grad():
            pred = method._probs(model, xv).argmax(1).cpu().numpy()
        accs[epoch + 1] = float((pred == yv).mean())

    method.fit(X, y, callback=on_epoch)
    pair = f"{data.CLASS_NAMES[classes[0]]}-{data.CLASS_NAMES[classes[1]]}"
    return pair, [(e + 1, loss, accs.get(e + 1)) for e, loss in enumerate(method.history_[0])]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", nargs="+", default=configs.CURVE_CONFIGS,
                   help="default: configs.CURVE_CONFIGS")
    p.add_argument("--seeds", nargs="+", type=int, default=data.TUNE_SEEDS)
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--hw", default=DEFAULT_HW)
    p.add_argument("--out", default=None,
                   help="default: <results dir for --hw>/ryan_curves.csv (see run.default_results_dir)")
    a = p.parse_args(argv)
    a.out = a.out or os.path.join(default_results_dir(a.hw), f"{PERSON}_curves.csv")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
    done = set()
    if not new:  # resumable: skip (config, seed, hw) curves already in the file
        with open(a.out, newline="") as f:
            done = {(r["config_id"], int(r["seed"]), r["hw"]) for r in csv.DictReader(f)}
    with open(a.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        for cid in a.config:
            kw = configs.get(cid)
            for seed in a.seeds:
                if (cid, seed, a.hw) in done:
                    print(f"{cid} seed {seed}: already done, skipped", flush=True)
                    continue
                pair, rows = curve(kw, seed, a.every)
                for epoch, loss, acc in rows:
                    w.writerow([cid, seed, pair, epoch, f"{loss:.5f}",
                                "" if acc is None else f"{acc:.5f}", a.hw])
                f.flush()
                final = [r[2] for r in rows if r[2] is not None]
                print(f"{cid} seed {seed} ({pair}): final val acc {final[-1]:.4f}, "
                      f"best-epoch acc {max(final):.4f}", flush=True)


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
