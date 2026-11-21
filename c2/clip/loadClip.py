from functools import lru_cache
import time

import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image  # no longer used by image_features; kept for compatibility

ARCH, PRETRAINED = "ViT-B-32", "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 100.0

TEMPLATES = (
    'a photo of a {}.', 'a blurry photo of a {}.', 'a black and white photo of a {}.',
    'a low contrast photo of a {}.', 'a high contrast photo of a {}.', 'a bad photo of a {}.',
    'a good photo of a {}.', 'a photo of a small {}.', 'a photo of a big {}.',
    'a photo of the {}.', 'a blurry photo of the {}.', 'a black and white photo of the {}.',
    'a low contrast photo of the {}.', 'a high contrast photo of the {}.', 'a bad photo of the {}.',
    'a good photo of the {}.', 'a photo of the small {}.', 'a photo of the big {}.',
)


@lru_cache(maxsize=1)
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
    return model.to(DEVICE).eval(), preprocess, open_clip.get_tokenizer(ARCH)


def fresh_clip():
    """Reload CLIP from scratch. Every method's fit calls this, so train time includes
    model loading for all methods, not just the first run in a process."""
    load_clip.cache_clear()
    return load_clip()


@lru_cache(maxsize=1)
def _norm_constants():
    """Mean and std from the model's own transform, so they can't drift from open_clip."""
    _, preprocess, _ = load_clip()
    norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    mean = torch.tensor(norm.mean, device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, device=DEVICE).view(1, 3, 1, 1)
    return mean, std


@torch.no_grad()
def class_text_features(names):
    model, _, tokenizer = load_clip()
    rows = []
    for name in names:
        tokens = tokenizer([t.format(name) for t in TEMPLATES]).to(DEVICE)
        emb = F.normalize(model.encode_text(tokens).float(), dim=-1).mean(dim=0)
        rows.append(F.normalize(emb, dim=0))
    return torch.stack(rows)


def _preprocess_batch(x_uint8, size=224):
    """uint8 (B, H, W, 3) -> normalized float (B, 3, size, size), all on DEVICE.

    One host-to-device copy and one batched bicubic resize, instead of a PIL call per image.
    Same steps as open_clip's transform: resize to size, center crop (a no-op for square
    inputs already at size), scale to [0, 1], normalize.
    """
    mean, std = _norm_constants()
    x = torch.from_numpy(x_uint8).to(DEVICE, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    x = x.clamp_(0.0, 1.0)
    return (x - mean) / std


def _sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def image_features(X, batch_size=256, timing=None):
    """Normalized CLIP image embeddings.

    If `timing` is a dict, it accumulates:
      preprocess_s: batched resize and normalize (device)
      encode_s:     the CLIP forward pass
    """
    model, _, _ = load_clip()
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size]
        _sync()
        t0 = time.perf_counter()
        batch = _preprocess_batch(xb)
        _sync()
        t1 = time.perf_counter()
        out.append(F.normalize(model.encode_image(batch).float(), dim=-1))
        _sync()
        t2 = time.perf_counter()
        if timing is not None:
            timing["preprocess_s"] = timing.get("preprocess_s", 0.0) + (t1 - t0)
            timing["encode_s"] = timing.get("encode_s", 0.0) + (t2 - t1)
    return torch.cat(out)