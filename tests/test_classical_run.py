import csv

import pandas as pd
import pytest

from c1.classical import configs
from c1.classical.method import ClassicalMethod
from c1.classical.run import hw_of, merge_results, missing_seeds, own_rows
from common.harness import RESULTS_COLUMNS

S1_FEATS = ["raw", "rawpca", "hsv", "hog", "lbp", "scat"]
S1_CLFS = ["lr", "linsvm", "rbfsvm", "rf", "hgb", "xgb", "knn"]


def test_s1_grid():
    s1 = configs.STAGES["s1"]
    assert len(s1) == 42
    assert set(s1) == {f"s1_{f}_{c}" for f in S1_FEATS for c in S1_CLFS}
    assert s1["s1_rawpca_lr"]["pca"] == 50
    for kw in s1.values():
        ClassicalMethod(**kw)


def test_ids_unique_across_stages():
    ids = [cid for stage in configs.STAGES.values() for cid in stage]
    assert len(ids) == len(set(ids))
    assert not set(ids) & set(configs.FINAL)


def test_final_points_at_stage_configs():
    assert set(configs.FINAL) == {"best", "fast"} == set(configs.FINAL_SOURCE)
    for name, cid in configs.FINAL_SOURCE.items():
        assert configs.FINAL[name] == configs.get(cid)


def test_get_finds_stage_config_and_rejects_unknown():
    assert configs.get("s1_hog_lr") == configs.STAGES["s1"]["s1_hog_lr"]
    with pytest.raises(KeyError):
        configs.get("does_not_exist")


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        for row in rows:
            cid, seed, notes = row[:3]
            challenge, method = row[3:] if len(row) == 5 else ("c1", "classical")
            w.writerow({"config_id": cid, "seed": seed, "notes": notes,
                        "challenge": challenge, "person": "massimo", "method": method})


def test_missing_seeds(tmp_path):
    path = tmp_path / "massimo.csv"
    _write_rows(path, [("a", 0, "hw=x"), ("a", 1, "hw=x"), ("a", 2, "hw=y"), ("b", 3, "hw=x")])
    assert missing_seeds(path, "a", [0, 1, 2, 3], "x") == [2, 3]
    assert missing_seeds(tmp_path / "nope.csv", "a", [0, 1], "x") == [0, 1]


def test_missing_seeds_ignores_other_methods(tmp_path):
    # results/massimo.csv is per person: Massimo's C2 method logs there too
    path = tmp_path / "massimo.csv"
    _write_rows(path, [("best", 10, "hw=x", "c2", "frozen_features"), ("best", 11, "hw=x")])
    assert missing_seeds(path, "best", [10, 11], "x") == [10]


@pytest.mark.parametrize("notes", [float("nan"), None, "", "no tag here"])
def test_hw_of_without_tag(notes):
    assert hw_of(notes) is None


def test_hw_of_reads_tag():
    assert hw_of("hw=colab-t4") == "colab-t4"


def test_own_rows_keeps_only_c1_classical():
    df = pd.DataFrame({"challenge": ["c1", "c2", "c1"],
                       "method": ["classical", "frozen_features", "cnn"], "x": [1, 2, 3]})
    assert own_rows(df)["x"].tolist() == [1]


def test_merge_results_appends_only_new_rows(tmp_path):
    dst, src = tmp_path / "repo.csv", tmp_path / "drive.csv"
    _write_rows(dst, [("a", 0, "hw=x"), ("c2cfg", 0, "hw=x", "c2", "frozen_features")])
    _write_rows(src, [("a", 0, "hw=x"), ("a", 1, "hw=t4"), ("a", 2, "hw=t4")])
    assert merge_results(src, dst) == 2
    rows = list(csv.DictReader(open(dst, newline="")))
    assert [(r["config_id"], r["seed"]) for r in rows] == [
        ("a", "0"), ("c2cfg", "0"), ("a", "1"), ("a", "2")]
    assert merge_results(src, dst) == 0
