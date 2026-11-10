"""Frozen ImageNet-pretrained backbones: loading, preprocessing and feature extraction.

Every backbone is a timm checkpoint that was never trained on CIFAR-10 (C2 rule).
Images are preprocessed on the GPU with each checkpoint's own data config
(resize, center crop, mean/std), the same math as timm's eval transform.
Weights are read from a local file, so no network call ever lands inside a
timed `fit`; `weights_file` downloads them once, before timing starts.
"""
import math

import numpy as np
import timm
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as TF

# key -> timm checkpoint
BACKBONES = {
    "vitti": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
    "vits": "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vits1k": "vit_small_patch16_224.augreg_in1k",
    "vitb": "vit_base_patch16_224.augreg_in21k_ft_in1k",
    "deits": "deit_small_patch16_224.fb_in1k",
    "rn50": "resnet50.tv_in1k",
    "cnxt": "convnext_tiny.fb_in22k_ft_in1k",
}

IN21K = "ImageNet-21k, fine-tuned on ImageNet-1k"
IN22K = "ImageNet-22k, fine-tuned on ImageNet-1k"
IN1K = "ImageNet-1k only"

# key -> (pretraining data, model card)
PRETRAINING = {
    key: (data, f"https://huggingface.co/timm/{BACKBONES[key]}")
    for key, data in {"vitti": IN21K, "vits": IN21K, "vits1k": IN1K, "vitb": IN21K,
                      "deits": IN1K, "rn50": IN1K, "cnxt": IN22K}.items()
}

POOLS_VIT = ("cls", "mean", "cls+mean")
PRECISIONS = ("fp32", "fp16")

_WEIGHTS = {}  # timm name -> local weights path (kept for the whole process)
_MODELS = {}   # (timm name, device, pretrained) -> (model, data config); emptied by clear_cache()


def resolve(name):
    """A backbone key ("vitti") -> its timm name; a timm name passes through."""
    return BACKBONES.get(name, name)


def weights_file(timm_name):
    """Local path of the checkpoint's weights, downloaded from the Hugging Face hub once."""
    if timm_name not in _WEIGHTS:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        repo = timm.models.get_pretrained_cfg(timm_name).hf_hub_id
        try:
            _WEIGHTS[timm_name] = hf_hub_download(repo, "model.safetensors")
        except EntryNotFoundError:
            _WEIGHTS[timm_name] = hf_hub_download(repo, "pytorch_model.bin")
    return _WEIGHTS[timm_name]


def load(name, device, pretrained=True):
    """(frozen model in eval mode on `device`, its data config). Memoized until clear_cache().

    The model has no classifier (num_classes=0): calling it returns its pooled
    pre-logit features.
    """
    timm_name = resolve(name)
    key = (timm_name, str(device), pretrained)
    if key not in _MODELS:
        kw = {"pretrained": pretrained, "num_classes": 0}
        if pretrained:
            # custom_load=False: augreg ViT configs point timm at the original JAX .npz
            # loader; the hub's safetensors file needs the normal state-dict loader
            # (which also drops the classifier for num_classes=0).
            kw["pretrained_cfg_overlay"] = {"file": weights_file(timm_name), "custom_load": False}
        model = timm.create_model(timm_name, **kw)
        model.eval().requires_grad_(False).to(device)
        _MODELS[key] = (model, timm.data.resolve_model_data_config(model))
    return _MODELS[key]


def clear_cache():
    """Forget the loaded models (the next load reads the weights from disk again)."""
    _MODELS.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_vit(model):
    return isinstance(model, timm.models.vision_transformer.VisionTransformer)


def preprocess(X, cfg, device, crop_pct=None):
    """uint8 (N, 32, 32, 3) -> normalized float (N, 3, S, S) on `device`.

    Same steps as timm's eval transform: resize to floor(S / crop_pct) with the
    checkpoint's interpolation, center crop S, normalize. The resized image is
    clamped and rounded to 8 bits, as PIL does. `crop_pct` overrides the
    checkpoint's value (1.0 keeps the whole image).
    """
    size = cfg["input_size"][-1]
    crop = cfg["crop_pct"] if crop_pct is None else crop_pct
    scale = math.floor(size / crop)
    interp = {"bicubic": InterpolationMode.BICUBIC,
              "bilinear": InterpolationMode.BILINEAR}[cfg["interpolation"]]
    x = torch.from_numpy(np.ascontiguousarray(X)).to(device).permute(0, 3, 1, 2).float()
    x = TF.resize(x, [scale, scale], interpolation=interp, antialias=True)
    x = x.round().clamp(0, 255) / 255.0
    x = TF.center_crop(x, [size, size])
    return TF.normalize(x, list(cfg["mean"]), list(cfg["std"]))


def _features(model, x, vit, pool, layers):
    if vit:
        # Each of the last `layers` blocks, passed through the final norm (as DINO's linear eval)
        blocks = model.forward_intermediates(x, indices=layers, norm=True, return_prefix_tokens=True,
                                             output_fmt="NLC", intermediates_only=True)
        parts = []
        for patches, prefix in blocks:
            if pool in ("cls", "cls+mean"):
                parts.append(prefix[:, 0])
            if pool in ("mean", "cls+mean"):
                parts.append(patches.mean(1))
        return torch.cat(parts, 1)
    stages = model.forward_intermediates(x, indices=layers, norm=True, intermediates_only=True)
    # Earlier stages: global average pool. Last stage: the model's own pooled head
    # features (e.g. ConvNeXt's LayerNorm after pooling), so layers=1 = pre-logits.
    parts = [s.mean((2, 3)) for s in stages[:-1]]
    parts.append(model.forward_head(stages[-1], pre_logits=True))
    return torch.cat(parts, 1)


def extract(model, X, cfg, *, pool=None, layers=1, precision="fp32", crop_pct=None, batch_size=128):
    """Frozen features of uint8 images X (N, 32, 32, 3) -> float32 (N, D).

    ViTs: pool "cls" (default), "mean" (patch tokens) or "cls+mean", for each of
    the last `layers` blocks, concatenated. CNNs: global average pooling only
    (pool None or "avg"), for each of the last `layers` stages.
    """
    vit = is_vit(model)
    pool = pool or ("cls" if vit else "avg")
    if (vit and pool not in POOLS_VIT) or (not vit and pool != "avg"):
        raise ValueError(f"pool {pool!r} not supported for {type(model).__name__}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    device = next(model.parameters()).device
    outs = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16,
                                         enabled=precision == "fp16"):
        for start in range(0, len(X), batch_size):
            x = preprocess(X[start:start + batch_size], cfg, device, crop_pct)
            outs.append(_features(model, x, vit, pool, layers).float().cpu())
    return torch.cat(outs).numpy().astype(np.float32)
