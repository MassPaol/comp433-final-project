import numpy as np
import pytest
import torch

from common import augment as A


def batch(n=8, h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, h, w), generator=g, dtype=torch.uint8)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_to_float_normalizes_per_channel():
    x = torch.full((2, 3, 4, 4), 255, dtype=torch.uint8)
    out = A.to_float(x, mean=(0.5, 0.5, 1.0), std=(0.5, 0.25, 1.0))
    assert out.dtype == torch.float32
    assert torch.allclose(out[:, 0], torch.tensor(1.0))
    assert torch.allclose(out[:, 1], torch.tensor(2.0))
    assert torch.allclose(out[:, 2], torch.tensor(0.0))


def test_crop_padding_zero_is_identity():
    x = A.to_float(batch(), (0.5,) * 3, (0.25,) * 3)
    assert torch.equal(A.random_crop(x, 0, gen()), x)


def test_crop_is_a_shifted_window_of_the_padded_image():
    x = torch.arange(2 * 1 * 6 * 6, dtype=torch.float32).view(2, 1, 6, 6)
    out = A.random_crop(x, 2, gen(3), mode="constant")
    assert out.shape == x.shape
    xp = torch.nn.functional.pad(x, (2,) * 4)
    for i in range(len(x)):
        found = any(torch.equal(out[i, 0], xp[i, 0, oy:oy + 6, ox:ox + 6])
                    for oy in range(5) for ox in range(5))
        assert found


def test_crop_works_at_224():
    x = torch.zeros(2, 3, 224, 224)
    assert A.random_crop(x, 16, gen()).shape == x.shape


def test_hflip_p1_flips_everything_and_twice_is_identity():
    x = torch.randn(4, 3, 8, 8)
    once = A.hflip(x, gen(), p=1.0)
    assert torch.equal(once, x.flip(3))
    assert torch.equal(A.hflip(once, gen(), p=1.0), x)
    assert torch.equal(A.hflip(x, gen(), p=0.0), x)


def test_cutout_erases_at_most_size_squared_per_image():
    x = torch.ones(16, 3, 32, 32)
    out = A.cutout(x, 8, gen(1))
    erased = (out[:, 0] == 0).flatten(1).sum(1)
    assert (erased <= 64).all() and (erased > 0).all()
    assert torch.equal(out[:, 0] == 0, out[:, 2] == 0)  # same mask on every channel
    assert torch.equal(A.cutout(x, 0, gen()), x)


def test_mixup_targets_are_convex_combinations():
    x = torch.randn(8, 3, 4, 4)
    y = torch.nn.functional.one_hot(torch.tensor([0, 1] * 4), 2).float()
    xm, ym = A.mixup(x, y, 0.7, gen())
    assert xm.shape == x.shape
    assert torch.allclose(ym.sum(1), torch.ones(8))
    assert ((ym >= 0) & (ym <= 1)).all()


def test_cutmix_targets_match_pasted_area():
    x = torch.zeros(2, 1, 10, 10)
    x[1] = 1.0
    y = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    for seed in range(20):
        xm, ym = A.cutmix(x, y, 0.5, gen(seed))
        assert torch.allclose(ym.sum(1), torch.ones(2))
        # image 0 is all zeros, so its "ones" fraction is exactly the part pasted from image 1
        # (or nothing, if the permutation mapped it onto itself)
        frac_ones = xm[0].mean().item()
        assert ym[0, 1].item() == pytest.approx(frac_ones, abs=1e-6) or frac_ones == 0


def test_rand_augment_keeps_shape_and_dtype():
    x = batch(4)
    torch.manual_seed(0)
    out = A.rand_augment(x, 2, 9)
    assert out.shape == x.shape and out.dtype == torch.uint8


def test_augmenter_shapes_and_soft_targets():
    cfg = {"randaugment": {"n": 2, "m": 9}, "crop": 4, "flip": True, "cutout": 8,
           "mixup": 0.2, "cutmix": 1.0}
    aug = A.Augmenter(cfg, mean=(0.5,) * 3, std=(0.25,) * 3, num_classes=2, seed=0)
    y = torch.tensor([0, 1] * 4)
    for _ in range(5):
        x, ys = aug(batch(8), y)
        assert x.shape == (8, 3, 32, 32) and x.dtype == torch.float32
        assert ys.shape == (8, 2)
        assert torch.allclose(ys.sum(1), torch.ones(8))


def test_augmenter_without_mixing_returns_one_hot():
    aug = A.Augmenter({"crop": 4, "flip": True}, seed=0)
    y = torch.tensor([0, 1, 1, 0])
    _, ys = aug(batch(4), y)
    assert torch.equal(ys, torch.nn.functional.one_hot(y, 2).float())


def test_augmenter_is_deterministic_for_a_seed():
    cfg = {"crop": 4, "flip": True, "cutout": 8, "mixup": 0.4}
    x, y = batch(8), torch.tensor([0, 1] * 4)
    a1, b1 = A.Augmenter(cfg, seed=5)(x, y)
    a2, b2 = A.Augmenter(cfg, seed=5)(x, y)
    a3, _ = A.Augmenter(cfg, seed=6)(x, y)
    assert torch.equal(a1, a2) and torch.equal(b1, b2)
    assert not torch.equal(a1, a3)


def test_augmenter_rejects_unknown_keys():
    with pytest.raises(ValueError):
        A.Augmenter({"crops": 4})


def test_augmenter_eval_is_plain_normalization():
    aug = A.Augmenter({"crop": 4, "flip": True}, mean=(0.4,) * 3, std=(0.2,) * 3)
    x = batch(2)
    assert torch.allclose(aug.eval(x), A.to_float(x, (0.4,) * 3, (0.2,) * 3))
    assert np.isclose(aug.eval(torch.full((1, 3, 2, 2), 102, dtype=torch.uint8)).mean().item(), 0.0, atol=1e-6)
