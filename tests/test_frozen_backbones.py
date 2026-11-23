import numpy as np
import pytest
import timm
import torch
from PIL import Image

from c2.frozen_features import backbones as B

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VIT = "test_vit.r160_in1k"      # tiny timm test models; pretrained=False -> no download
CNN = "test_resnet.r160_in1k"


@pytest.fixture
def X():
    return np.random.RandomState(0).randint(0, 256, (4, 32, 32, 3), dtype=np.uint8)


@pytest.fixture(autouse=True)
def _fresh_cache():
    B.clear_cache()
    yield
    B.clear_cache()


def _cfg(name):
    return timm.data.resolve_model_data_config(timm.create_model(name, pretrained=False))


def test_resolve():
    assert B.resolve("vitti") == "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    assert B.resolve(VIT) == VIT
    assert len(B.BACKBONES) == 7
    assert set(B.BACKBONES) == set(B.PRETRAINING)


def test_preprocess_matches_timm_transform(X):
    cfg = _cfg("vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    ours = B.preprocess(X, cfg, "cpu").numpy()
    assert ours.shape == (4, 3, 224, 224)
    tf = timm.data.create_transform(**cfg, is_training=False)
    ref = np.stack([tf(Image.fromarray(x)).numpy() for x in X])
    diff = np.abs(ours - ref)
    assert diff.mean() < 0.02
    assert diff.max() < 0.25


def test_preprocess_crop_override(X):
    cfg = _cfg("vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    default = B.preprocess(X, cfg, "cpu")
    full = B.preprocess(X, cfg, "cpu", crop_pct=1.0)
    assert full.shape == default.shape
    assert not torch.allclose(full, default)


def test_vit_layers1_cls_equals_prelogits(X):
    m, cfg = B.load(VIT, DEVICE, pretrained=False)
    ours = B.extract(m, X, cfg, pool="cls", layers=1)
    with torch.no_grad():
        ref = m(B.preprocess(X, cfg, DEVICE)).float().cpu().numpy()
    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_cnn_layers1_equals_prelogits(X):
    m, cfg = B.load(CNN, DEVICE, pretrained=False)
    ours = B.extract(m, X, cfg)
    with torch.no_grad():
        ref = m(B.preprocess(X, cfg, DEVICE)).float().cpu().numpy()
    np.testing.assert_allclose(ours, ref, atol=1e-4)


def test_feature_dims(X):
    vit, vcfg = B.load(VIT, DEVICE, pretrained=False)
    assert B.extract(vit, X, vcfg).shape == (4, 64)  # default pool for a ViT = cls
    assert B.extract(vit, X, vcfg, pool="mean").shape == (4, 64)
    assert B.extract(vit, X, vcfg, pool="cls+mean").shape == (4, 128)
    assert B.extract(vit, X, vcfg, pool="cls", layers=2).shape == (4, 128)
    assert B.extract(vit, X, vcfg, pool="cls+mean", layers=2).shape == (4, 256)
    cnn, ccfg = B.load(CNN, DEVICE, pretrained=False)
    assert B.extract(cnn, X, ccfg, layers=2).shape == (4, 192 + 96)


def test_vit_pools_differ(X):
    vit, cfg = B.load(VIT, DEVICE, pretrained=False)
    assert not np.allclose(B.extract(vit, X, cfg, pool="cls"), B.extract(vit, X, cfg, pool="mean"))


def test_cnn_rejects_token_pooling(X):
    cnn, cfg = B.load(CNN, DEVICE, pretrained=False)
    with pytest.raises(ValueError):
        B.extract(cnn, X, cfg, pool="cls")


def test_unknown_pool_raises(X):
    vit, cfg = B.load(VIT, DEVICE, pretrained=False)
    with pytest.raises(ValueError):
        B.extract(vit, X, cfg, pool="max")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast needs CUDA")
def test_fp16_finite_float32(X):
    vit, cfg = B.load(VIT, "cuda", pretrained=False)
    out = B.extract(vit, X, cfg, precision="fp16")
    assert out.dtype == np.float32 and np.isfinite(out).all()


def test_extract_batches(X):
    vit, cfg = B.load(VIT, DEVICE, pretrained=False)
    np.testing.assert_allclose(B.extract(vit, X, cfg, batch_size=3),
                               B.extract(vit, X, cfg, batch_size=128), atol=1e-5)


def test_load_uses_local_weights_file(monkeypatch):
    calls = []
    real_create = timm.create_model

    def fake_create(name, **kw):
        calls.append((name, kw))
        return real_create(VIT, pretrained=False, num_classes=0)

    monkeypatch.setattr(B, "weights_file", lambda name: "/tmp/w.safetensors")
    monkeypatch.setattr(B.timm, "create_model", fake_create)
    m1, _ = B.load("vitti", "cpu")
    name, kw = calls[0]
    assert name == "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    assert kw["pretrained"] is True and kw["num_classes"] == 0
    # custom_load=False: augreg ViT configs default to the JAX .npz loader, which can't read safetensors
    assert kw["pretrained_cfg_overlay"] == {"file": "/tmp/w.safetensors", "custom_load": False}
    m2, _ = B.load("vitti", "cpu")
    assert m2 is m1 and len(calls) == 1
    assert not any(p.requires_grad for p in m1.parameters())
    assert not m1.training
    B.clear_cache()
    B.load("vitti", "cpu")
    assert len(calls) == 2
