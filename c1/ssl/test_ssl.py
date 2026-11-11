"""Smoke tests for c1.ssl (put in tests/). Synthetic data, runs in seconds on CPU."""
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


@pytest.mark.parametrize("mode", ["vote", "dist"])
def test_kmeans_fit_predict(toy, mode):
    X, y = toy
    m = KMeansMethod(features=[("raw", {})], k=4, mode=mode, flip_train=True, flip_tta=True).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == y.shape and set(pred) <= {0, 1}
    assert (pred == y).mean() > 0.8
    assert m.n_trainable_params() > 0


@pytest.mark.parametrize("head", ["lr", "knn", "kmeans"])
def test_simsiam_fit_predict(toy, head):
    X, y = toy
    params = {"k": 4} if head == "kmeans" else {}
    m = SimSiamMethod(epochs=2, width=8, proj_dim=16, batch_size=20, head=head,
                      head_params=params).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == y.shape and set(pred) <= {0, 1}
    assert np.isfinite(m.collapse_std_)


def test_simsiam_random_init_baseline(toy):
    X, y = toy
    m = SimSiamMethod(epochs=0, width=8, proj_dim=16, head="lr", flip_head=True).fit(X, y)
    assert m.predict(X).shape == y.shape


def test_every_config_resolves_and_constructs():
    cls = {"kmeans": KMeansMethod, "simsiam": SimSiamMethod}
    for stage in configs.STAGES.values():
        for cid in stage:
            method, kw = configs.get(cid)
            cls[method](**kw)  # constructor only: must accept every kwarg
    for name in configs.FINAL_SOURCE:
        assert configs.get(name)[0] in cls


def test_unknown_config_raises():
    with pytest.raises(KeyError):
        configs.get("nope")
