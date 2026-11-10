import numpy as np
import pytest

from c2.frozen_features.heads import DEFAULT_NORM, HEADS, NORMS, FeatureNorm, cv_splits, make_head


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    Z = rng.randn(40, 16).astype(np.float32)
    # classes on opposite sides of the origin, so cosine heads separate them too
    Z[:20] -= 1.5
    Z[20:] += 1.5
    y = np.array([0] * 20 + [1] * 20)
    return Z, y


@pytest.mark.parametrize("name", HEADS)
def test_each_head_separates(data, name):
    Z, y = data
    norm = FeatureNorm(DEFAULT_NORM[name]).fit(Z)
    Zn = norm.transform(Z)
    head = make_head(name).fit(Zn, y)
    s = head.score(Zn)
    assert s.shape == (40,)
    assert ((s > 0).astype(int) == y).mean() >= 0.95


def test_default_norms():
    assert DEFAULT_NORM["adamw"] == "none"
    assert DEFAULT_NORM["ncm"] == "l2" and DEFAULT_NORM["knn"] == "l2"
    assert all(DEFAULT_NORM[h] == "standard" for h in ("lr", "lr_cv", "linsvm", "rbfsvm"))
    assert set(DEFAULT_NORM) == set(HEADS)


def test_norms(data):
    Z, _ = data
    assert NORMS == ("none", "l2", "standard", "cl2n")
    np.testing.assert_array_equal(FeatureNorm("none").fit(Z).transform(Z), Z)
    l2 = FeatureNorm("l2").fit(Z).transform(Z)
    np.testing.assert_allclose(np.linalg.norm(l2, axis=1), 1, atol=1e-5)
    std = FeatureNorm("standard").fit(Z).transform(Z)
    np.testing.assert_allclose(std.mean(0), 0, atol=1e-5)
    cl2n = FeatureNorm("cl2n").fit(Z).transform(Z)
    np.testing.assert_allclose(np.linalg.norm(cl2n, axis=1), 1, atol=1e-5)
    expected = (Z - Z.mean(0)) / np.linalg.norm(Z - Z.mean(0), axis=1, keepdims=True)
    np.testing.assert_allclose(cl2n, expected, atol=1e-5)


def test_cl2n_uses_training_mean(data):
    Z, _ = data
    norm = FeatureNorm("cl2n").fit(Z)
    other = Z[:5] + 10.0
    expected = other - Z.mean(0)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(norm.transform(other), expected, atol=1e-5)


def test_adamw_deterministic(data):
    Z, y = data
    a = make_head("adamw", seed=0).fit(Z, y).score(Z)
    b = make_head("adamw", seed=0).fit(Z, y).score(Z)
    c = make_head("adamw", seed=1).fit(Z, y).score(Z)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_adamw_defaults():
    h = make_head("adamw")
    assert (h.lr, h.weight_decay, h.epochs, h.batch_size) == (1e-3, 0.01, 10, 32)
    h = make_head("adamw", {"epochs": 30})
    assert (h.lr, h.weight_decay, h.epochs, h.batch_size) == (1e-3, 0.01, 30, 32)


@pytest.mark.parametrize("name,expected", [("adamw", 34), ("lr", 17), ("lr_cv", 17),
                                           ("linsvm", 17), ("ncm", 32), ("knn", 0)])
def test_n_params(data, name, expected):
    Z, y = data
    assert make_head(name).fit(Z, y).n_params() == expected


def test_rbfsvm_n_params(data):
    Z, y = data
    h = make_head("rbfsvm").fit(Z, y)
    assert h.n_params() == h.clf_.dual_coef_.size + 1


def test_rbf_gamma_mult(data):
    Z, y = data
    g1 = make_head("rbfsvm").fit(Z, y).clf_.gamma
    g2 = make_head("rbfsvm", {"gamma_mult": 2.0}).fit(Z, y).clf_.gamma
    assert g2 == pytest.approx(2 * g1)
    assert g1 == pytest.approx(1 / (Z.shape[1] * Z.var()))


def test_ncm_sign(data):
    Z, y = data
    h = make_head("ncm").fit(Z, y)
    mu0, mu1 = Z[y == 0].mean(0), Z[y == 1].mean(0)
    assert h.score(mu1[None])[0] > 0
    assert h.score(mu0[None])[0] < 0


def test_cv_splits_keep_groups_together():
    y = np.array([0] * 20 + [1] * 20 + [0] * 20 + [1] * 20)
    groups = np.arange(80) % 40
    folds = cv_splits(y, groups)
    assert len(folds) == 5
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te])


def test_lr_cv_uses_groups(data):
    Z, y = data
    h = make_head("lr_cv").fit(np.concatenate([Z, Z]), np.concatenate([y, y]),
                               groups=np.arange(80) % 40)
    for tr, te in h.clf_.cv:
        assert not {i % 40 for i in tr} & {i % 40 for i in te}


def test_unknown_head_and_norm_raise():
    with pytest.raises(ValueError):
        make_head("tree")
    with pytest.raises(ValueError):
        FeatureNorm("zca")
