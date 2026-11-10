from common import data
from common import timing

import numpy as np
import csv
import os

RESULTS_COLUMNS = [
    "challenge", "person", "method", "config_id", "seed", "class_pair",
    "val_acc", "train_time_s", "infer_time_s", "n_trainable_params",
    "notes",
]


def _row_notes(notes, model):
    """`notes` plus the method's own per-run values (Method.extra_notes) as key=value."""
    extra = model.extra_notes() if hasattr(model, "extra_notes") else {}
    parts = [notes] + [f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                       for k, v in extra.items()]
    return " ".join(p for p in parts if p)


def run(method_factory, seeds, *, challenge, person, method_name, config_id,
        notes="", results_dir="results"):
    """
    Centralizes the experiment running for the entire team. Also ensures 
    consistency across runs, since every method is timed and scored the 
    same way on the same splits.

    Args:
        method_factory: Function with no arguments that returns a new, untrained
            Method, e.g. `lambda: HogSVM(C=1.0)`. Called once per seed.
        seeds: List of seeds to run, usually `data.TUNE_SEEDS` or
            `data.REPORT_SEEDS`.
        challenge: "c1" or "c2".
        person: "massimo" | "ryan" | "ammar" | "alvin". Also sets the CSV
            file name.
        method_name: Short method name for the results table, e.g. "hog_svm".
        config_id: Short label for this hyperparameter setting, e.g. "C1_rbf",
            so different configs of the same method can be told apart.
        notes: Optional free text saved in every row.
        results_dir: Folder the CSV is written to. Defaults to "results".

    Returns:
        List of row dicts, one per seed, with keys from RESULTS_COLUMNS.
    """
    path = os.path.join(results_dir, f"{person}.csv")
    os.makedirs(results_dir, exist_ok=True)

    rows = []

    for seed in seeds:
        x_train, y_train, x_val, y_val, classes = data.load_instance(seed)

        model = method_factory()

        with timing.Timer() as t:
            model.fit(x_train, y_train)
        train_time_s = t.seconds

        with timing.Timer() as t:
            output = np.asarray(model.predict(x_val))
        infer_time_s = t.seconds

        # Make sure output is a flat list
        if output.shape != y_val.shape:
            raise ValueError(
                f"{method_name}.predict returned shape {output.shape}, "
                f"expected {y_val.shape}. Return a flat array of 0/1 labels."
            )
        
        val_acc = np.float64((output == y_val).sum() / len(output))
        n_trainable_params = model.n_trainable_params()
        class_pair = f"{data.CLASS_NAMES[classes[0]]}-{data.CLASS_NAMES[classes[1]]}"

        row = {
            "challenge": challenge, "person": person, "method": method_name,
            "config_id": config_id, "seed": seed, "class_pair": class_pair,
            "val_acc": val_acc, "train_time_s": train_time_s,
            "infer_time_s": infer_time_s, "n_trainable_params": n_trainable_params,
            "notes": _row_notes(notes, model)}

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
            file_is_new = not os.path.exists(path) or os.path.getsize(path) == 0
            if file_is_new:
                writer.writeheader()
            writer.writerow(row)

        rows.append(row)

    return rows




