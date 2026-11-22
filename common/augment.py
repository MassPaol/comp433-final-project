"""Shared augmentation for training from scratch (32x32) and fine-tuning (224x224).

OWNER: Ryan. Used by c1/cnn (Ryan), c1/vit_scratch (Ammar) and c2/finetune (Ryan).

Everything works on whole batches of image tensors (N, C, H, W) that already live
on the training device, so there is no DataLoader and no per-image CPU work
(except RandAugment, see below). With only 100 training images the whole set fits
on the GPU, and per-image CPU pipelines would dominate the step time.

Two layers:
  * Low-level functions (`random_crop`, `hflip`, `cutout`, `mixup`, `cutmix`,
    `rand_augment`): stateless, size-agnostic, randomness from a `torch.Generator`
    (CPU generator; the few random numbers are moved to the data's device).
  * `Augmenter`: builds a pipeline from a config dict and owns the RNGs, so a
    method only does `x, y_soft = aug(x_uint8, y)` once per batch.

Pipeline order (fixed): [RandAugment on uint8] -> float + normalize -> crop ->
flip -> cutout -> mixup / cutmix. Cutout zeroes *normalized* pixels, i.e. fills
with the training mean colour.

Reproducibility: all randomness comes from `seed`, except RandAugment, which
uses torchvision's internal calls to the global torch RNG; seed it with
`torch.manual_seed` (CNNMethod.fit does).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

# Keys an Augmenter config may contain, with their "off" defaults.
DEFAULTS = {
    "randaugment": None,   # None or {"n": ops per image, "m": magnitude 0..30}
    "crop": 0,             # padding in pixels for random crop (0 = off)
    "crop_mode": "reflect",  # "reflect" or "constant" (zero = mean colour after normalize)
    "flip": False,         # random horizontal flip, p = 0.5
    "cutout": 0,           # side of the square erased per image (0 = off)
    "mixup": 0.0,          # Beta(alpha, alpha) for mixup (0 = off)
    "cutmix": 0.0,         # Beta(alpha, alpha) for CutMix (0 = off)
    "mix_prob": 1.0,       # probability a batch gets mixup/CutMix at all
}


def _uniform(n, gen, device):
    return torch.rand(n, generator=gen).to(device)


def _randint(high, n, gen, device):
    return torch.randint(0, high, (n,), generator=gen).to(device)


def to_float(x_uint8, mean, std):
    """uint8 (N, C, H, W) -> float32 normalized with per-channel `mean`/`std` (in [0, 1] units)."""
    x = x_uint8.float().div_(255.0)
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return (x - mean) / std


def random_crop(x, padding, gen, mode="reflect"):
    """Pad each side by `padding`, then take a random HxW crop per image."""
    if padding <= 0:
        return x
    n, _, h, w = x.shape
    xp = F.pad(x, (padding,) * 4, mode=mode)
    oy = _randint(2 * padding + 1, n, gen, x.device)
    ox = _randint(2 * padding + 1, n, gen, x.device)
    rows = oy[:, None] + torch.arange(h, device=x.device)          # (N, H)
    cols = ox[:, None] + torch.arange(w, device=x.device)          # (N, W)
    idx = torch.arange(n, device=x.device)[:, None, None]
    out = xp[idx, :, rows[:, :, None], cols[:, None, :]]            # (N, H, W, C)
    return out.permute(0, 3, 1, 2).contiguous()


def hflip(x, gen, p=0.5):
    """Flip each image left-right with probability `p`."""
    mask = _uniform(len(x), gen, x.device) < p
    return torch.where(mask[:, None, None, None], x.flip(3), x)


def cutout(x, size, gen, p=1.0):
    """Zero one `size`x`size` square per image (DeVries & Taylor 2017).

    The centre is uniform over the image, so the square may be partly outside it.
    """
    if size <= 0:
        return x
    n, _, h, w = x.shape
    cy = _randint(h, n, gen, x.device)
    cx = _randint(w, n, gen, x.device)
    y0, x0 = cy - size // 2, cx - size // 2
    ys = torch.arange(h, device=x.device)
    xs = torch.arange(w, device=x.device)
    in_y = (ys[None] >= y0[:, None]) & (ys[None] < (y0 + size)[:, None])   # (N, H)
    in_x = (xs[None] >= x0[:, None]) & (xs[None] < (x0 + size)[:, None])   # (N, W)
    apply = _uniform(n, gen, x.device) < p
    erase = in_y[:, :, None] & in_x[:, None, :] & apply[:, None, None]
    return x.masked_fill(erase[:, None], 0.0)


def mixup(x, y_soft, lam, gen):
    """Convex combination with a shuffled copy of the batch (Zhang et al. 2018)."""
    perm = torch.randperm(len(x), generator=gen).to(x.device)
    return lam * x + (1 - lam) * x[perm], lam * y_soft + (1 - lam) * y_soft[perm]


def cutmix(x, y_soft, lam, gen):
    """Paste a box from a shuffled copy of the batch (Yun et al. 2019).

    The box covers about (1 - lam) of the image; targets are mixed by the exact
    pasted area after clipping to the image.
    """
    n, _, h, w = x.shape
    perm = torch.randperm(n, generator=gen).to(x.device)
    r = math.sqrt(1.0 - lam)
    ch, cw = int(h * r), int(w * r)
    cy = int(torch.randint(0, h, (1,), generator=gen))
    cx = int(torch.randint(0, w, (1,), generator=gen))
    y0, y1 = max(cy - ch // 2, 0), min(cy - ch // 2 + ch, h)
    x0, x1 = max(cx - cw // 2, 0), min(cx - cw // 2 + cw, w)
    out = x.clone()
    out[:, :, y0:y1, x0:x1] = x[perm, :, y0:y1, x0:x1]
    lam_adj = 1.0 - (y1 - y0) * (x1 - x0) / (h * w)
    return out, lam_adj * y_soft + (1 - lam_adj) * y_soft[perm]


def rand_augment(x_uint8, n_ops, magnitude):
    """torchvision RandAugment applied independently to each uint8 image (Cubuk et al. 2020).

    torchvision's v2 transform applies one sampled op to a whole batch, so this loops
    over images. It is the only per-image loop here; its cost shows up in train time.
    """
    from torchvision.transforms import v2
    ra = v2.RandAugment(num_ops=n_ops, magnitude=magnitude)
    return torch.stack([ra(img) for img in x_uint8])


class Augmenter:
    """Training-time augmentation pipeline built from a config dict (keys: see DEFAULTS).

    Usage, once per batch:
        aug = Augmenter(cfg, mean, std, num_classes=2, seed=0)
        x, y_soft = aug(x_uint8, y)    # x: float (N, C, H, W); y_soft: (N, num_classes)

    `aug.eval(x_uint8)` returns the same normalization without any augmentation.
    """

    def __init__(self, cfg=None, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25),
                 num_classes=2, seed=0):
        cfg = dict(cfg or {})
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown augmentation keys {sorted(unknown)}; expected {sorted(DEFAULTS)}")
        self.cfg = {**DEFAULTS, **cfg}
        self.mean, self.std = tuple(mean), tuple(std)
        self.num_classes = num_classes
        self.gen = torch.Generator().manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)  # Beta samples for mixup / CutMix

    @property
    def mixes(self):
        return self.cfg["mixup"] > 0 or self.cfg["cutmix"] > 0

    def eval(self, x_uint8):
        return to_float(x_uint8, self.mean, self.std)

    def __call__(self, x_uint8, y):
        c = self.cfg
        if c["randaugment"]:
            x_uint8 = rand_augment(x_uint8, c["randaugment"]["n"], c["randaugment"]["m"])
        x = to_float(x_uint8, self.mean, self.std)
        x = random_crop(x, c["crop"], self.gen, mode=c["crop_mode"])
        if c["flip"]:
            x = hflip(x, self.gen)
        x = cutout(x, c["cutout"], self.gen)
        y_soft = F.one_hot(y, self.num_classes).float()
        if self.mixes and self.np_rng.random() < c["mix_prob"]:
            use_cutmix = c["cutmix"] > 0 and (c["mixup"] <= 0 or self.np_rng.random() < 0.5)
            alpha = c["cutmix"] if use_cutmix else c["mixup"]
            lam = float(self.np_rng.beta(alpha, alpha))
            x, y_soft = (cutmix if use_cutmix else mixup)(x, y_soft, lam, self.gen)
        return x, y_soft
