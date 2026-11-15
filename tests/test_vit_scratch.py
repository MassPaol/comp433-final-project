import numpy as np

from c1.vit_scratch.method import ViTScratch
from common.method import Method


def _toy_data():
    rng = np.random.RandomState(0)
    dark = rng.randint(0, 100, (20, 32, 32, 3), dtype=np.uint8)
    bright = rng.randint(150, 256, (20, 32, 32, 3), dtype=np.uint8)
    return np.concatenate([dark, bright]), np.array([0] * 20 + [1] * 20)


def test_is_a_method():
    assert issubclass(ViTScratch, Method)


def test_fit_predict_shapes_and_learns():
    X, y = _toy_data()
    m = ViTScratch(epochs=10, flip=True)
    assert m.fit(X, y) is m
    pred = m.predict(X)
    assert pred.shape == (40,)
    assert np.issubdtype(pred.dtype, np.integer)
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == y).mean() >= 0.9          # dark vs bright is easy
    assert m.n_trainable_params() > 0
