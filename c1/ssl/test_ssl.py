"""Smoke tests for c1.ssl. Synthetic data, runs in about a minute on CPU."""
import numpy as np
import pytest

from c1.ssl import configs
from c1.ssl.method_kmeans import KMeansMethod
from c1.ssl.method_simsiam import SimSiamMethod


@pytest.fixture(scope="module")
def toy():
    rng = np.random.RandomState(0)
    y = np.repeat([0, 1], 20)
    X = rng.randint(0, 256, size=(40, 32, 32, 3)).astype(np.uint8)
    X[y == 1, :, :16] = 255  # class 1 has a bright left half: easy to separate
    return X, y


def test_kmeans_extract_patches_shape(toy):
    X, _ = toy
    m = KMeansMethod(patch_size=6, n_clusters=8, num_patches=500)
    assert m.extract_patches(X[:5]).shape == (5 * 27 * 27, 6 * 6 * 3)


def test_kmeans_fit_predict(toy):
    X, y = toy
    m = KMeansMethod(patch_size=6, n_clusters=16, num_patches=2000, n_init=2,
                     C_grid=(0.1, 1.0)).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == y.shape and set(pred) <= {0, 1}
    assert (pred == y).mean() > 0.7
    assert m.encode(X[:3]).shape == (3, 4 * 16)
    assert m.n_trainable_params() > 0


@pytest.mark.parametrize("epochs", [0, 2])
def test_simsiam_fit_predict(toy, epochs):
    X, y = toy
    m = SimSiamMethod(epochs=epochs, width=8, proj_dim=16, proj_hidden=32, pred_hidden=8,
                      batch_size=20, C_grid=(0.1, 1.0)).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == y.shape and set(pred) <= {0, 1}
    assert np.isfinite(m.collapse_std_)
    assert m.n_trainable_params() > 0


def test_every_config_constructs():
    cls = {"kmeans": KMeansMethod, "simsiam": SimSiamMethod}
    for stage in configs.STAGES.values():
        for cid in stage:
            method, kw = configs.get(cid)
            cls[method](**kw)  # constructor only: must accept every kwarg
    for name in configs.FINAL_SOURCE:
        assert configs.get(name)[0] in cls


def test_final_names_exist_in_stages():
    for name, cid in configs.FINAL_SOURCE.items():
        assert any(cid in stage for stage in configs.STAGES.values()), (name, cid)


def test_unknown_config_raises():
    with pytest.raises(KeyError):
        configs.get("nope")