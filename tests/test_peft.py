import numpy as np
import pytest

from c2.peft.method import PEFTMethod
from common.method import Method


def _toy_data():
    rng = np.random.RandomState(0)
    dark = rng.randint(0, 100, (8, 32, 32, 3), dtype=np.uint8)
    bright = rng.randint(150, 256, (8, 32, 32, 3), dtype=np.uint8)
    return np.concatenate([dark, bright]), np.array([0] * 8 + [1] * 8)


def test_is_a_method():
    assert issubclass(PEFTMethod, Method)


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        PEFTMethod(mode="nope")


@pytest.mark.parametrize("kw,expected", [
    ({"mode": "head"}, 386),
    ({"mode": "bitfit"}, 386 + 25728),
    ({"mode": "lora", "lora_rank": 4}, 386 + 12 * 4 * (192 + 576)),
    ({"mode": "vpt", "vpt_prompts": 10}, 386 + 10 * 192),
    ({"mode": "vpt", "vpt_prompts": 10, "vpt_deep": True}, 386 + 12 * 10 * 192),
])
def test_trainable_params_and_predict(kw, expected):
    X, y = _toy_data()
    m = PEFTMethod(epochs=3, batch_size=8, lr=1e-3, **kw).fit(X, y)
    assert m.n_trainable_params() == expected
    pred = m.predict(X)
    assert pred.shape == (16,)
    assert np.issubdtype(pred.dtype, np.integer)
    assert (pred == y).mean() >= 0.9          # dark vs bright is easy


def test_frozen_weights_do_not_change():
    import timm
    X, y = _toy_data()
    ref = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True,
                            num_classes=2).state_dict()
    m = PEFTMethod(mode="bitfit", epochs=2, batch_size=8, lr=1e-2).fit(X, y)
    new = m.model.state_dict()
    w = "blocks.0.attn.qkv.weight"
    b = "blocks.0.attn.qkv.bias"
    assert (new[w].cpu() == ref[w]).all()      # weights frozen
    assert not (new[b].cpu() == ref[b]).all()  # biases trained


def test_lora_and_vpt_start_as_the_pretrained_model():
    """LoRA's B starts at zero, so before training it must equal the plain model.
    VPT with prompts must at least keep the class-token output shape."""
    import torch
    import timm
    from c2.peft.layers import PromptedViT, add_lora
    torch.manual_seed(0)
    ref = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True).eval()
    lora = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=True)
    add_lora(lora, rank=4)
    lora.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(ref(x), lora(x), atol=1e-5)
        out = PromptedViT(ref, n_prompts=5, deep=True)(x)
    assert out.shape == (2, 1000)
