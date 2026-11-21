import numpy as np
import pytest

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


@pytest.mark.parametrize("kw", [
    {"tokenizer": "spt"},
    {"lsa": True},
    {"tokenizer": "spt", "lsa": True},
    {"tokenizer": "conv"},
    {"tokenizer": "conv", "pool": "seq"},
    {"pool": "seq"},
])
def test_small_data_options_fit_and_learn(kw):
    X, y = _toy_data()
    m = ViTScratch(epochs=10, **kw).fit(X, y)
    pred = m.predict(X)
    assert pred.shape == (40,)
    assert (pred == y).mean() >= 0.9


def test_tokenizers_keep_token_count():
    import torch
    from c1.vit_scratch.layers import ConvTokenizer, ShiftedPatchTokenizer
    x = torch.zeros(2, 3, 32, 32)
    assert ShiftedPatchTokenizer(4, 128)(x).shape == (2, 64, 128)
    assert ConvTokenizer(4, 128)(x).shape == (2, 64, 128)
    assert ConvTokenizer(2, 64)(x).shape == (2, 256, 64)


def test_lsa_masks_self_attention():
    import torch
    from c1.vit_scratch.layers import LSAAttention
    from timm.layers import Attention
    lsa = LSAAttention(Attention(16, num_heads=2, qkv_bias=True))
    out = lsa(torch.randn(3, 5, 16))
    assert out.shape == (3, 5, 16) and torch.isfinite(out).all()
    assert lsa.log_scale.requires_grad      # temperature is learned


def test_final_configs_resolve():
    from c1.vit_scratch import configs
    assert configs.get("best") == configs.get(configs.FINAL_SOURCE["best"])
    ViTScratch(**configs.get("best"))       # builds without error


def test_bad_options_rejected():
    with pytest.raises(ValueError):
        ViTScratch(tokenizer="nope")
    with pytest.raises(ValueError):
        ViTScratch(pool="nope")


def test_random_crop_keeps_shape_and_learns():
    X, y = _toy_data()
    m = ViTScratch(epochs=10, crop_pad=4).fit(X, y)
    assert (m.predict(X) == y).mean() >= 0.9
    import torch
    xb = torch.zeros(5, 3, 32, 32)
    assert m._random_crop(xb, torch.Generator().manual_seed(0)).shape == (5, 3, 32, 32)
