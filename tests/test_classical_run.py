import csv

import pytest

from c1.classical import configs
from c1.classical.method import ClassicalMethod
from c1.classical.run import missing_seeds
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
        for cid, seed, notes in rows:
            w.writerow({"config_id": cid, "seed": seed, "notes": notes,
                        "challenge": "c1", "person": "massimo", "method": "classical"})


def test_missing_seeds(tmp_path):
    path = tmp_path / "massimo.csv"
    _write_rows(path, [("a", 0, "hw=x"), ("a", 1, "hw=x"), ("a", 2, "hw=y"), ("b", 3, "hw=x")])
    assert missing_seeds(path, "a", [0, 1, 2, 3], "x") == [2, 3]
    assert missing_seeds(tmp_path / "nope.csv", "a", [0, 1], "x") == [0, 1]
