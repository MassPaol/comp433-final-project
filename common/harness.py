"""Runs a method over seeds and logs one results row per seed.

OWNER: Massimo. Harness v1 due Mon Nov 9.
The signature and CSV columns below are agreed by the team; keep them.
"""

RESULTS_COLUMNS = [
    "challenge", "person", "method", "config_id", "seed", "class_pair",
    "val_acc", "train_time_s", "infer_time_s", "n_trainable_params",
    "gpu", "git_commit", "notes",
]


def run(method_factory, seeds, *, challenge, person, method_name, config_id,
        notes="", results_dir="results"):
    """For each seed:
        1. X_tr, y_tr, X_val, y_val, classes = common.data.load_instance(seed)
        2. method = method_factory()            # fresh model for every seed
        3. time method.fit(X_tr, y_tr)          # common.timing.Timer
        4. time method.predict(X_val)
        5. append one row to f"{results_dir}/{person}.csv" right away
           (so a Colab disconnect doesn't lose finished seeds)

    challenge: "c1" or "c2"; person: "massimo" | "ryan" | "ammar" | "alvin".
    Returns the list of row dicts.
    """
    raise NotImplementedError("Harness v1: Massimo, due Mon Nov 9")
