import pandas as pd
import pytest

from c1.cnn import configs
from c1.cnn.method import CNNMethod
from c1.cnn.run import missing_seeds, rep_of, run_configs
from c1.cnn.select import paired, summarize_configs


def test_every_stage_config_builds():
    for stage in configs.STAGES.values():
        for kw in stage.values():
            CNNMethod(device="cpu", **kw)


def test_ids_unique_across_stages_and_prefixed_by_stage():
    ids = [cid for stage in configs.STAGES.values() for cid in stage]
    assert len(ids) == len(set(ids))
    assert not set(ids) & {"best", "fast"}
    for name, stage in configs.STAGES.items():
        assert all(cid.startswith(name + "_") for cid in stage)


def test_final_points_at_stage_configs():
    assert set(configs.FINAL) == set(configs.FINAL_SOURCE) <= {"best", "fast"}
    for name, cid in configs.FINAL_SOURCE.items():
        assert configs.FINAL[name] == configs.get(cid)


def test_cfg_merges_one_level_and_copies():
    base = {"aug": {"crop": 4}, "epochs": 10}
    out = configs.cfg(base, aug={"flip": True}, epochs=20)
    assert out == {"aug": {"crop": 4, "flip": True}, "epochs": 20}
    out["aug"]["crop"] = 0
    assert base["aug"]["crop"] == 4
    assert configs.cfg(base, aug={})["aug"] == {}  # empty dict resets


def test_every_stage_has_reps_and_curve_configs_exist():
    assert set(configs.STAGE_REPS) == set(configs.STAGES)
    assert all(r >= 1 for r in configs.STAGE_REPS.values())
    for cid in configs.CURVE_CONFIGS:
        configs.get(cid)


def test_compare_hw_ranks_both_tags():
    from c1.cnn.select import compare_hw
    rows = []
    for hw, accs in (("a", {"x": 0.9, "y": 0.8}), ("b", {"x": 0.7, "y": 0.85})):
        for cid, acc in accs.items():
            for seed in (0, 1):
                rows.append({"challenge": "c1", "method": "cnn", "config_id": cid, "seed": seed,
                             "val_acc": acc, "train_time_s": 1.0, "infer_time_s": 0.1,
                             "notes": f"hw={hw} rep=0"})
    t = compare_hw(pd.DataFrame(rows), [0, 1], "a", "b")
    assert list(t.index) == ["x", "y"]
    assert t.loc["x", "rank_a"] == 1 and t.loc["x", "rank_b"] == 2
    assert t.loc["x", "diff_pp"] == pytest.approx(20.0)


def test_get_rejects_unknown():
    with pytest.raises(KeyError):
        configs.get("does_not_exist")


def test_rep_of():
    assert rep_of("hw=colab-t4 rep=2") == 2
    assert rep_of("hw=colab-t4") == 0
    assert rep_of(float("nan")) == 0


def test_run_is_resumable_and_select_reads_it(tmp_path, monkeypatch):
    """Two short reps of s0_flipcrop on tuning seeds 0-1 with real data, CPU, 1 epoch."""
    import c1.cnn.run as run
    monkeypatch.setattr(run, "SEED_SETS", {"tune": [0, 1]})
    run_configs(["s0_noaug", "s0_flipcrop"], [0, 1], "test", reps=2, results_dir=tmp_path,
                overrides={"epochs": 1, "device": "cpu"})
    path = tmp_path / "ryan.csv"
    df = pd.read_csv(path)
    assert len(df) == 2 * 2 * 2
    assert sorted(df["notes"].unique()) == ["hw=test rep=0", "hw=test rep=1"]
    assert missing_seeds(path, "s0_flipcrop", [0, 1], "test", 1) == []
    assert missing_seeds(path, "s0_flipcrop", [0, 1], "test", 2) == [0, 1]
    run_configs(["s0_flipcrop"], [0, 1], "test", reps=2, results_dir=tmp_path,
                overrides={"epochs": 1, "device": "cpu"})
    assert len(pd.read_csv(path)) == 8  # nothing re-run

    s = summarize_configs(df, [0, 1], "test")
    assert set(s.index) == {"s0_noaug", "s0_flipcrop"}
    assert (s["reps"] == 2).all() and (s["n"] == 2).all()
    r = paired(df, "s0_flipcrop", "s0_noaug", [0, 1], "test")
    assert r["n"] == 2 and r["wins"] + r["losses"] <= 2

    from c1.cnn import figures
    monkeypatch.setattr(figures, "SEED_SETS", {"tune": [0, 1]})
    figures.main(["--hw", "test", "--results", str(path), "--curves", str(tmp_path / "none.csv"),
                  "--outdir", str(tmp_path / "fig")])
    assert (tmp_path / "fig" / "c1_cnn_stages.png").exists()
    assert (tmp_path / "fig" / "c1_cnn_acc_vs_time.png").exists()
