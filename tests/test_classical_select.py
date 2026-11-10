import pandas as pd
import pytest

from c1.classical.select import select, summarize_configs

SEEDS = [0, 1]


def _df(spec, hw="local-3070", seeds=SEEDS, challenge="c1", method="classical"):
    """spec: {config_id: (acc, total_time)} -> results rows on every seed."""
    rows = []
    for cid, (acc, t) in spec.items():
        for s in seeds:
            rows.append({"challenge": challenge, "method": method, "config_id": cid, "seed": s,
                         "val_acc": acc, "train_time_s": t / 2, "infer_time_s": t / 2,
                         "notes": f"hw={hw}"})
    return pd.DataFrame(rows)


def _pick(spec):
    return select(summarize_configs(_df(spec), SEEDS, "local-3070"))


def test_best_is_top_acc():
    assert _pick({"a": (0.90, 10), "b": (0.80, 1)})[0] == "a"


def test_tie_goes_to_faster():
    assert _pick({"a": (0.900, 10), "b": (0.896, 2)})[0] == "b"


def test_fast_within_2pp():
    assert _pick({"a": (0.90, 10), "b": (0.885, 1), "c": (0.85, 0.1)})[1] == "b"


def test_summary_columns():
    s = summarize_configs(_df({"a": (0.9, 4)}), SEEDS, "local-3070")
    assert list(s.columns) == ["acc_mean", "acc_std", "time_mean", "n"]
    assert s.loc["a", "time_mean"] == pytest.approx(4)
    assert s.loc["a", "n"] == 2


def test_filters_hw_and_incomplete():
    df = pd.concat([
        _df({"a": (0.9, 1)}),
        _df({"other_hw": (0.99, 1)}, hw="colab-t4"),
        _df({"partial": (0.99, 1)}, seeds=[0]),
        _df({"not_tuning": (0.99, 1)}, seeds=[10, 11]),
    ])
    s = summarize_configs(df, SEEDS, "local-3070")
    assert list(s.index) == ["a"]


def test_ignores_other_methods_and_empty_notes():
    df = pd.concat([
        _df({"a": (0.9, 1)}),
        _df({"c2_best": (0.99, 0.1)}, challenge="c2", method="frozen_features"),
        _df({"a": (0.1, 1)}, challenge="c2", method="frozen_features"),
    ])
    blank = _df({"untagged": (0.99, 1)})
    blank["notes"] = float("nan")  # what pandas reads for an empty notes cell
    s = summarize_configs(pd.concat([df, blank]), SEEDS, "local-3070")
    assert list(s.index) == ["a"]
    assert s.loc["a", "acc_mean"] == pytest.approx(0.9)
