import csv

import numpy as np
import pandas as pd

from common import harness, sweep
from common.harness import RESULTS_COLUMNS
from common.method import Method


def _write_rows(path, rows):
    """rows: (challenge, method, config_id, seed, notes)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        for challenge, method, cid, seed, notes in rows:
            w.writerow({"challenge": challenge, "person": "massimo", "method": method,
                        "config_id": cid, "seed": seed, "notes": notes})


def test_note_value():
    notes = "hw=x load_s=0.5 extract_s=1.25"
    assert sweep.note_value(notes, "extract_s") == 1.25
    assert sweep.note_value(notes, "load_s") == 0.5
    assert sweep.note_value(notes, "missing") is None
    assert sweep.note_value(float("nan"), "extract_s") is None


def test_hw_of_with_extra_notes():
    assert sweep.hw_of("hw=local-3070 load_s=0.5 extract_s=1.2") == "local-3070"


def test_missing_seeds_by_challenge_method(tmp_path):
    p = tmp_path / "massimo.csv"
    _write_rows(p, [("c1", "classical", "a", 0, "hw=x"),
                    ("c2", "frozen_features", "a", 1, "hw=x load_s=0.1"),
                    ("c2", "frozen_features", "a", 2, "hw=y")])
    got = sweep.missing_seeds(p, "a", [0, 1, 2], "x", challenge="c2", method="frozen_features")
    assert got == [0, 2]
    assert sweep.missing_seeds(tmp_path / "nope.csv", "a", [0, 1], "x",
                               challenge="c2", method="frozen_features") == [0, 1]


def test_summarize_and_select_generic():
    rows = []
    for cid, acc, t in [("a", 0.90, 10.0), ("b", 0.896, 2.0)]:
        for s in [0, 1]:
            rows.append({"challenge": "c2", "method": "frozen_features", "config_id": cid,
                         "seed": s, "val_acc": acc, "train_time_s": t / 2, "infer_time_s": t / 2,
                         "notes": "hw=x extract_s=0.3"})
    # a C1 row with the same id must not leak into the C2 summary
    rows.append({"challenge": "c1", "method": "classical", "config_id": "a", "seed": 0,
                 "val_acc": 0.1, "train_time_s": 0.0, "infer_time_s": 0.0, "notes": "hw=x"})
    s = sweep.summarize_configs(pd.DataFrame(rows), [0, 1], "x",
                                challenge="c2", method="frozen_features")
    assert s.loc["a", "acc_mean"] == 0.90 and s.loc["a", "n"] == 2
    assert sweep.select(s) == ("b", "b")


class _Dummy(Method):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.int64)


class _DummyWithNotes(_Dummy):
    def extra_notes(self):
        return {"load_s": 0.5, "extract_s": 1.234567}


def _fake_instance(seed):
    X = np.zeros((4, 32, 32, 3), np.uint8)
    y = np.array([0, 0, 1, 1])
    return X, y, X, y, np.array([0, 1])


def test_harness_appends_extra_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(harness.data, "load_instance", _fake_instance)
    harness.run(_DummyWithNotes, [0], challenge="c2", person="p", method_name="m",
                config_id="c", notes="hw=t", results_dir=tmp_path)
    harness.run(_Dummy, [1], challenge="c2", person="p", method_name="m",
                config_id="c", notes="hw=t", results_dir=tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "p.csv", newline="")))
    assert rows[0]["notes"] == "hw=t load_s=0.5 extract_s=1.235"
    assert rows[1]["notes"] == "hw=t"


def test_method_extra_notes_default_empty():
    assert Method().extra_notes() == {}
