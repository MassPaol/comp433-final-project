import numpy as np
import pytest

from c1.classical.method import CLASSIFIERS, ClassicalMethod
from common.method import Method


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    dark = rng.randint(0, 100, (20, 32, 32, 3), dtype=np.uint8)
    bright = rng.randint(150, 256, (20, 32, 32, 3), dtype=np.uint8)
    X = np.concatenate([dark, bright])
    y = np.array([0] * 20 + [1] * 20)
    return X, y


def test_is_a_method():
    assert issubclass(ClassicalMethod, Method)


@pytest.mark.parametrize("clf", CLASSIFIERS)
def test_each_classifier_fits_and_predicts(data, clf):
    X, y = data
    m = ClassicalMethod(features=[("hsv_hist", {})], clf=clf)
    assert m.fit(X, y) is m
    pred = m.predict(X)
    assert np.issubdtype(pred.dtype, np.integer)
    assert pred.shape == (40,)
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == y).mean() >= 0.9


def test_predict_single_image(data):
    X, y = data
    m = ClassicalMethod(features=[("hsv_hist", {})], clf="lr").fit(X, y)
    assert m.predict(X[:1]).shape == (1,)


def test_flip_train_doubles(data):
    X, y = data
    m = ClassicalMethod(features=[("hsv_hist", {})], clf="knn",
                        clf_params={"n_neighbors": 1}, flip_train=True).fit(X, y)
    assert m.clf_.n_samples_fit_ == 80


def test_flip_tta_shape(data):
    X, y = data
    m = ClassicalMethod(features=[("hog", {})], clf="lr", flip_tta=True).fit(X, y)
    assert m.predict(X).shape == (40,)


def test_pca_clamped(data):
    X, y = data
    m = ClassicalMethod(features=[("raw", {})], clf="lr", pca=500).fit(X, y)
    assert m.pca_.n_components_ <= 40
    assert m.predict(X).shape == (40,)


def test_lr_cv_groups_flips(data):
    _, y = data
    y2 = np.concatenate([y, y])
    splits = list(ClassicalMethod._cv_splits(40, y2, flip_train=True))
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert not ({i % 40 for i in test_idx} & {i % 40 for i in train_idx})


@pytest.mark.parametrize("clf", CLASSIFIERS)
def test_n_trainable_params(data, clf):
    X, y = data
    m = ClassicalMethod(features=[("hsv_hist", {})], clf=clf).fit(X, y)
    n = m.n_trainable_params()
    assert isinstance(n, int) and n >= 0


def test_n_trainable_params_lr_pca(data):
    X, y = data
    m = ClassicalMethod(features=[("raw", {})], clf="lr", pca=10).fit(X, y)
    assert m.n_trainable_params() == 11


def test_rbf_gamma_mult(data):
    X, y = data
    kw = dict(features=[("hsv_hist", {})], clf="rbfsvm")
    m1 = ClassicalMethod(**kw, clf_params={"gamma_mult": 1.0}).fit(X, y)
    m2 = ClassicalMethod(**kw, clf_params={"gamma_mult": 2.0}).fit(X, y)
    assert m2.clf_.gamma == pytest.approx(2 * m1.clf_.gamma)
    assert m2.gamma_ == m2.clf_.gamma


def test_lr_cv_quiet_and_log_loss(data):
    import warnings
    X, y = data
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = ClassicalMethod(features=[("hsv_hist", {})], clf="lr_cv").fit(X, y)
    assert not [w for w in caught if issubclass(w.category, FutureWarning)]
    assert m.clf_.scoring == "neg_log_loss"


def test_unknown_classifier_raises(data):
    X, y = data
    with pytest.raises(ValueError):
        ClassicalMethod(features=[("hsv_hist", {})], clf="nope").fit(X, y)
