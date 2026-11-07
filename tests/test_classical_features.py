import numpy as np
import pytest

from c1.classical import features
from c1.classical.features import FEATURES, extract, flip, hsv_hist, lbp, raw, scattering


@pytest.fixture
def X():
    return np.random.RandomState(0).randint(0, 256, (6, 32, 32, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", sorted(FEATURES))
def test_shapes_and_dtype(X, name):
    Z = FEATURES[name](X)
    assert Z.dtype == np.float32
    assert Z.ndim == 2
    assert Z.shape[0] == 6
    assert np.isfinite(Z).all()


@pytest.mark.parametrize("name", sorted(FEATURES))
def test_deterministic(X, name):
    np.testing.assert_array_equal(FEATURES[name](X), FEATURES[name](X))


def test_known_dims(X):
    assert raw(X).shape[1] == 3072
    assert raw(X, gray=True).shape[1] == 1024
    assert hsv_hist(X, bins=16).shape[1] == 48
    assert lbp(X, P=8, R=1, grid=2).shape[1] == (8 + 2) * 4
    assert scattering(X, J=2, L=8, color=True).shape[1] == 3 * 81 * 8 * 8
    assert scattering(X, J=2, L=8, color=False, pool=2).shape[1] == 81 * 4 * 4


def test_extract_concatenates(X):
    Z = extract(X, [("raw", {}), ("hsv_hist", {"bins": 8})])
    assert Z.shape == (6, 3072 + 24)
    assert Z.dtype == np.float32


def test_flip(X):
    np.testing.assert_array_equal(flip(flip(X)), X)
    np.testing.assert_array_equal(flip(X)[:, :, 0], X[:, :, -1])


@pytest.mark.parametrize("value", [0, 255])
@pytest.mark.parametrize("name", sorted(FEATURES))
def test_features_finite_on_constant_images(name, value):
    X = np.full((3, 32, 32, 3), value, dtype=np.uint8)
    assert np.isfinite(FEATURES[name](X)).all()


def test_clear_cache_empties_scattering_ops(X):
    scattering(X[:1], J=1, L=4)
    assert features._SCATTERING_OPS
    features.clear_cache()
    assert not features._SCATTERING_OPS
