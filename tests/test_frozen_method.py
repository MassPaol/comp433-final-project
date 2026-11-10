import numpy as np
import pytest

from c2.frozen_features import heads
from c2.frozen_features.method import FrozenFeatureMethod
from common.method import Method

VIT = "test_vit.r160_in1k"
CNN = "test_resnet.r160_in1k"


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = np.concatenate([rng.randint(0, 80, (20, 32, 32, 3)),
                        rng.randint(170, 256, (20, 32, 32, 3))]).astype(np.uint8)
    y = np.array([0] * 20 + [1] * 20)
    return X, y


def _m(**kw):
    return FrozenFeatureMethod(**{"backbone": VIT, "head": "lr", "pretrained": False, **kw})


def test_is_a_method():
    assert isinstance(_m(), Method)


def test_fit_predict_shape(data):
    X, y = data
    pred = _m().fit(X, y).predict(X)
    assert pred.dtype == np.int64 and pred.shape == (40,)
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == y).mean() >= 0.9


def test_predict_single_image(data):
    X, y = data
    m = _m().fit(X, y)
    pred = m.predict(X[:1])
    assert pred.shape == (1,)
    assert pred.dtype == np.int64


def test_flip_train_doubles(data):
    X, y = data
    m = _m(flip_train=True).fit(X, y)
    assert m.train_features_.shape[0] == 80
    np.testing.assert_array_equal(m.groups_, np.arange(80) % 40)


def test_flip_tta_shape(data):
    X, y = data
    assert _m(flip_tta=True).fit(X, y).predict(X).shape == (40,)


def test_lr_cv_groups_flips(data):
    X, y = data
    m = _m(head="lr_cv", flip_train=True).fit(X, y)
    np.testing.assert_array_equal(m.groups_, np.arange(80) % 40)
    folds = m.head_.clf_.cv  # the folds the fitted LR-CV actually used
    assert len(folds) == 5
    for tr, te in folds:
        assert not {i % 40 for i in tr} & {i % 40 for i in te}


def test_backbone_list_concatenates(data):
    X, y = data
    m = _m(backbone=[VIT, CNN]).fit(X, y)
    assert m.train_features_.shape[1] == 64 + 96
    assert m.predict(X[:3]).shape == (3,)


def test_extra_notes(data):
    X, y = data
    notes = _m().fit(X, y).extra_notes()
    assert set(notes) == {"load_s", "extract_s"}
    assert all(isinstance(v, float) and v >= 0 for v in notes.values())


def test_n_trainable_params(data):
    X, y = data
    assert _m().fit(X, y).n_trainable_params() == 65


def test_default_norm_follows_head():
    assert _m(head="adamw").norm_kind == "none"
    assert _m(head="ncm").norm_kind == "l2"
    assert _m(head="lr", norm="cl2n").norm_kind == "cl2n"


def test_head_params_reach_head(data):
    X, y = data
    m = _m(head="adamw", head_params={"epochs": 3}).fit(X, y)
    assert m.head_.epochs == 3
