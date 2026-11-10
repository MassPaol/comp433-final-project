import csv

import pandas as pd
import pytest

from c2.frozen_features import configs, run
from c2.frozen_features import select as sel
from c2.frozen_features.method import FrozenFeatureMethod
from common.harness import RESULTS_COLUMNS

S1_BB = ["vitti", "vits", "vits1k", "vitb", "deits", "rn50", "cnxt"]
S1_HEADS = ["adamw", "lr", "linsvm", "rbfsvm", "ncm", "knn"]


def test_s1_grid():
    s1 = configs.STAGES["s1"]
    assert configs.S1_BACKBONES == S1_BB and configs.S1_HEADS == S1_HEADS
    assert set(s1) == {f"s1_{b}_{h}" for b in S1_BB for h in S1_HEADS}
    assert len(s1) == 42
    for kw in s1.values():
        FrozenFeatureMethod(**kw)


def test_ids_unique_across_stages():
    ids = [cid for stage in configs.STAGES.values() for cid in stage]
    assert len(ids) == len(set(ids))
    assert not set(ids) & set(configs.FINAL)


def test_final_points_at_stage_configs():
    assert set(configs.FINAL) == set(configs.FINAL_SOURCE)
    for name, cid in configs.FINAL_SOURCE.items():
        assert configs.FINAL[name] == configs.get(cid)


def test_get_rejects_unknown():
    with pytest.raises(KeyError):
        configs.get("does_not_exist")


def test_baseline_is_testbed_recipe():
    assert configs.FINAL["baseline"] == {"backbone": "vitti", "head": "adamw"}
    m = FrozenFeatureMethod(**configs.FINAL["baseline"])
    assert m.pool is None and m.layers == 1 and m.norm_kind == "none"
    assert m.head_params == {} and not m.flip_train and not m.flip_tta


def _write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULTS_COLUMNS)
        w.writeheader()
        for challenge, method, cid, seed, notes in rows:
            w.writerow({"challenge": challenge, "person": "massimo", "method": method,
                        "config_id": cid, "seed": seed, "notes": notes})


def test_missing_seeds_c2_only(tmp_path):
    p = tmp_path / "massimo.csv"
    _write_rows(p, [("c1", "classical", "a", 2, "hw=x"),
                    ("c2", "frozen_features", "a", 0, "hw=x load_s=0.2 extract_s=0.5"),
                    ("c2", "frozen_features", "a", 1, "hw=x load_s=0.2 extract_s=0.5"),
                    ("c2", "frozen_features", "a", 2, "hw=y")])
    assert run.missing_seeds(p, "a", [0, 1, 2, 3], "x") == [2, 3]


def test_own_rows_and_summary():
    df = pd.DataFrame([
        {"challenge": "c2", "method": "frozen_features", "config_id": "a", "seed": 0,
         "val_acc": 0.9, "train_time_s": 1.0, "infer_time_s": 1.0, "notes": "hw=x extract_s=0.4"},
        {"challenge": "c1", "method": "classical", "config_id": "a", "seed": 0,
         "val_acc": 0.1, "train_time_s": 1.0, "infer_time_s": 1.0, "notes": "hw=x"},
    ])
    assert len(run.own_rows(df)) == 1
    s = sel.summarize_configs(df, [0], "x")
    assert s.loc["a", "acc_mean"] == 0.9
    assert sel.select(s) == ("a", "a")
