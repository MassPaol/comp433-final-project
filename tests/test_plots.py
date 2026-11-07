import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from common import plots  # noqa: E402

ALL_METHODS = ["c1_classical", "c1_cnn", "c1_vit_scratch", "c1_ssl",
               "c2_frozen_features", "c2_finetune", "c2_peft", "c2_clip"]


def test_colors_cover_all_methods():
    assert set(ALL_METHODS) <= set(plots.METHOD_COLORS)
    assert set(ALL_METHODS) <= set(plots.METHOD_MARKERS)


@pytest.mark.parametrize("challenge", ["c1", "c2"])
def test_colors_distinct_within_challenge(challenge):
    methods = [m for m in ALL_METHODS if m.startswith(challenge)]
    assert len({plots.METHOD_COLORS[m] for m in methods}) == 4


def test_same_person_same_color_across_challenges():
    # one hue per person; C1 vs C2 is told apart by filled vs hollow markers
    pairs = [("c1_classical", "c2_frozen_features"), ("c1_cnn", "c2_finetune"),
             ("c1_vit_scratch", "c2_peft"), ("c1_ssl", "c2_clip")]
    for a, b in pairs:
        assert plots.METHOD_COLORS[a] == plots.METHOD_COLORS[b]


def test_summarize():
    df = pd.DataFrame({"cfg": ["a"] * 3 + ["b"] * 3,
                       "val_acc": [0.7, 0.8, 0.9, 0.5, 0.5, 0.5]})
    s = plots.summarize(df, ["cfg"]).set_index("cfg")
    assert list(plots.summarize(df, ["cfg"]).columns) == ["cfg", "mean", "std", "n"]
    assert s.loc["a", "mean"] == pytest.approx(0.8)
    assert s.loc["a", "std"] == pytest.approx(np.std([0.7, 0.8, 0.9]))  # ddof=0
    assert s.loc["b", "std"] == 0
    assert s.loc["a", "n"] == 3


def test_pareto_front():
    mask = plots.pareto_front([1, 2, 3, 0.5], [0.8, 0.9, 0.85, 0.7])
    assert mask.tolist() == [True, True, False, True]


def test_save_writes_both(tmp_path):
    plots.apply_style()
    fig, ax = plt.subplots()
    plots.errorbar_sweep(ax, [1, 2], [0.5, 0.6], [0.1, 0.1], label="x",
                         color=plots.METHOD_COLORS["c1_classical"])
    paths = plots.save(fig, "demo", outdir=tmp_path)
    assert sorted(os.path.basename(p) for p in paths) == ["demo.pdf", "demo.png"]
    assert all(os.path.getsize(p) > 0 for p in paths)
