import numpy as np
import pytest
import torch

from c1.cnn.method import CNNMethod, balanced_batches, lr_factor
from c1.cnn.models import build_model, count_params


def toy_data(n=40, seed=0):
    """Two classes that differ in brightness, so a CNN can learn them in a few epochs."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    X = rng.integers(0, 120, size=(n, 32, 32, 3)).astype(np.uint8)
    X[y == 1] += 120
    return X, y


SMALL = {"type": "convnet", "width": 8, "convs": 1}


@pytest.mark.parametrize("arch", [
    {"type": "convnet", "width": 16, "convs": 1}, {"type": "convnet", "width": 16, "convs": 2},
    {"type": "resnet", "width": 16, "n": 1}, {"type": "resnet", "width": 16, "n": 3, "norm": "gn"},
])
def test_models_output_two_logits(arch):
    model = build_model(arch)
    assert model(torch.zeros(3, 3, 32, 32)).shape == (3, 2)


def test_resnet20_has_the_usual_parameter_count():
    # CIFAR ResNet-20 at width 16 has ~0.27M parameters (He et al. 2016, Table 6)
    n = count_params(build_model({"type": "resnet", "width": 16, "n": 3}))
    assert 0.26e6 < n < 0.28e6


def test_bad_configs_fail_at_construction():
    with pytest.raises(ValueError):
        CNNMethod(arch={"type": "mlp"})
    with pytest.raises(ValueError):
        CNNMethod(aug={"flips": True})
    with pytest.raises(ValueError):
        CNNMethod(opt={"name": "rmsprop"})
    with pytest.raises(ValueError):
        CNNMethod(sched={"warmup": 3})


def test_balanced_batches_cover_everything_once():
    batches = balanced_batches(100, 32, torch.Generator().manual_seed(0))
    assert [len(b) for b in batches] == [25, 25, 25, 25]
    assert sorted(torch.cat(batches).tolist()) == list(range(100))


def test_lr_factor_warmup_then_cosine():
    assert lr_factor(0, 100, 10) == pytest.approx(0.1)
    assert lr_factor(9, 100, 10) == pytest.approx(1.0)
    assert lr_factor(10, 100, 10) == pytest.approx(1.0)
    assert lr_factor(100, 100, 10) == pytest.approx(0.0, abs=1e-12)


def test_fit_predict_contract_and_learns_toy_problem():
    X, y = toy_data()
    m = CNNMethod(arch=SMALL, aug={"crop": 4, "flip": True}, epochs=15, batch_size=20,
                  device="cpu").fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (len(X),) and pred.dtype == np.int64
    assert set(np.unique(pred)) <= {0, 1}
    assert (pred == y).mean() > 0.9
    assert m.n_trainable_params() > 0
    assert len(m.history_) == 1 and len(m.history_[0]) == 15


def test_same_seed_same_predictions():
    X, y = toy_data()
    kw = dict(arch=SMALL, aug={"crop": 4, "flip": True, "cutout": 8, "mixup": 0.2},
              epochs=3, batch_size=20, device="cpu")
    p1 = CNNMethod(seed=1, **kw).fit(X, y).predict_proba(X)
    p2 = CNNMethod(seed=1, **kw).fit(X, y).predict_proba(X)
    p3 = CNNMethod(seed=2, **kw).fit(X, y).predict_proba(X)
    assert np.allclose(p1, p2)
    assert not np.allclose(p1, p3)


def test_ensemble_tta_adamw_onecycle_label_smoothing_run():
    X, y = toy_data()
    m = CNNMethod(arch={**SMALL, "norm": "gn", "dropout": 0.3},
                  aug={"randaugment": {"n": 1, "m": 5}, "cutmix": 1.0},
                  opt={"name": "adamw", "lr": 1e-3, "wd": 0.05}, sched={"name": "onecycle"},
                  epochs=2, batch_size=20, label_smoothing=0.1, tta_flip=True, ensemble=3,
                  device="cpu").fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(1), 1, atol=1e-5)
    assert len(m.models_) == 3
    assert m.n_trainable_params() == 3 * count_params(m.models_[0])


def test_normalization_stats_come_from_training_images():
    X, y = toy_data()
    m = CNNMethod(arch=SMALL, epochs=1, batch_size=20, device="cpu").fit(X, y)
    assert np.allclose(m.mean_, (X / 255.0).mean(axis=(0, 1, 2)), atol=1e-5)


def test_callback_sees_every_epoch_of_every_member():
    X, y = toy_data()
    seen = []
    CNNMethod(arch=SMALL, epochs=3, batch_size=20, ensemble=2, device="cpu").fit(
        X, y, callback=lambda m, e, model: seen.append((m, e)))
    assert seen == [(m, e) for m in range(2) for e in range(3)]


def test_build_backbone_is_the_final_arch_without_its_head():
    from c1.cnn import configs
    from c1.cnn.models import FINAL_ARCH, build_backbone
    assert FINAL_ARCH == configs.get("best")["arch"] == configs.get("fast")["arch"]
    for arch in (None, {"type": "resnet", "width": 16, "n": 1}):
        f = build_backbone(arch)(torch.zeros(4, 3, 32, 32))
        assert f.shape == (4, 64)
