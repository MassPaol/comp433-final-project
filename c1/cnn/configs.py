"""Every configuration the C1 CNN sweep runs, written out explicitly.

Each value is the kwargs for `CNNMethod` (minus `seed`, which run.py sets per rep).
Stages run on the tuning seeds one after another: a stage is written only after
the previous stage's results are in, and the comment above it records why its
candidates were chosen (details in NOTES.md). `FINAL` holds the selected `best`
and `fast` configs that run on the report seeds.
"""
import copy


def cfg(base, **changes):
    """Deep copy of `base` with top-level keys replaced (dict values merged one level deep)."""
    out = copy.deepcopy(base)
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and v:
            out[k] = {**out[k], **v}
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---- S0: baseline recipe, with and without the two basic augmentations -----
# Starting point chosen before any results (no tuning yet):
#   ConvNet-6 (3 stages x 2 convs, widths 32/64/128, BatchNorm), ~0.29M params;
#   SGD + Nesterov, lr 0.05, wd 5e-4, cosine with 5 warmup epochs; batch 32
#   (4 balanced batches of 25 per epoch); 200 epochs (= 800 steps).
# These are standard CIFAR-from-scratch settings (He et al. 2016; Zagoruyko &
# Komodakis 2016), scaled down for 100 images. S0 checks the pipeline end to end
# and gives the two anchors every later stage is compared against.
BASE = {
    "arch": {"type": "convnet", "width": 32, "convs": 2, "norm": "bn", "dropout": 0.0},
    "aug": {},
    "opt": {"name": "sgd", "lr": 0.05, "wd": 5e-4, "momentum": 0.9, "nesterov": True},
    "sched": {"name": "cosine", "warmup_epochs": 5},
    "epochs": 200,
    "batch_size": 32,
    "label_smoothing": 0.0,
    "tta_flip": False,
    "ensemble": 1,
}
FLIPCROP = {"crop": 4, "flip": True}

STAGES = {
    "s0": {
        "s0_noaug": cfg(BASE),
        "s0_flipcrop": cfg(BASE, aug=FLIPCROP),
    },
}



# ---- S1: architecture screen on the S0 flip+crop recipe (200 epochs) -------
# S0 result (local CPU, 2 reps): noaug 85.37, flipcrop 86.17 (+0.80 pp paired, 3/5 seeds).
# Without augmentation the net memorizes the 100 images by ~epoch 40; flip+crop keeps
# improving until ~epoch 140. -> keep flip+crop as the reference recipe and screen
# backbones on it. ConvNet-6 w32 (= s0_flipcrop) is the reference, not re-run.
# Rep noise (~1 pp between reps) is as large as the effects, so S1 uses 2 reps.
FLIPCROP_BASE = cfg(BASE, aug=FLIPCROP)
S1_ARCHS = {
    "convnet3_w16": {"type": "convnet", "width": 16, "convs": 1},
    "convnet3_w32": {"type": "convnet", "width": 32, "convs": 1},
    "convnet3_w64": {"type": "convnet", "width": 64, "convs": 1},
    "convnet6_w16": {"type": "convnet", "width": 16, "convs": 2},
    "convnet6_w64": {"type": "convnet", "width": 64, "convs": 2},
    "resnet8_w16": {"type": "resnet", "width": 16, "n": 1},
    "resnet8_w32": {"type": "resnet", "width": 32, "n": 1},
    "resnet20_w16": {"type": "resnet", "width": 16, "n": 3},
    "resnet20_w32": {"type": "resnet", "width": 32, "n": 3},
    # GroupNorm: one representative per family at its standard width
    "convnet6_w32_gn": {"type": "convnet", "width": 32, "convs": 2, "norm": "gn"},
    "resnet20_w16_gn": {"type": "resnet", "width": 16, "n": 3, "norm": "gn"},
}
# arch is replaced, not merged, so a ResNet config doesn't carry the ConvNet's "convs" key
STAGES["s1"] = {f"s1_{name}": {**cfg(FLIPCROP_BASE), "arch": {"norm": "bn", "dropout": 0.0, **a}}
                for name, a in S1_ARCHS.items()}



# ---- S2a: augmentation / regularization, one change at a time --------------
# S1 result (local CPU, 2 reps): ConvNet-3 w16 + BN best (87.59, 9.6 s), > ConvNet-6 w32
# (+1.41 pp, 4/5), > ConvNet-3 w32 (+0.57, 5/5); ResNets worse at every size; GroupNorm
# -3 to -6 pp vs BN (5/5). -> backbone ConvNet-3 w16 BN; width re-checked in S4.
# Epochs 200 -> 300: S0 curves showed flip+crop still improving at ~140 epochs and
# stronger augmentation fits more slowly. s2a_ref re-runs flip+crop at 300 epochs so
# every S2a difference is measured at the same length.
S2_ARCH = {"type": "convnet", "width": 16, "convs": 1, "norm": "bn", "dropout": 0.0}
S2_REF = {**cfg(FLIPCROP_BASE, epochs=300), "arch": dict(S2_ARCH)}


def _s2a():
    ref = S2_REF
    out = {"s2a_ref": cfg(ref)}
    # decompose the basic augmentations
    out["s2a_noaug"] = {**cfg(ref), "aug": {}}
    out["s2a_flip_only"] = {**cfg(ref), "aug": {"flip": True}}
    out["s2a_crop_only"] = {**cfg(ref), "aug": {"crop": 4}}
    # one addition on top of flip+crop
    for size in (8, 16):
        out[f"s2a_cutout{size}"] = cfg(ref, aug={"cutout": size})
    for alpha, tag in ((0.2, "02"), (1.0, "1")):
        out[f"s2a_mixup{tag}"] = cfg(ref, aug={"mixup": alpha})
    out["s2a_cutmix1"] = cfg(ref, aug={"cutmix": 1.0})
    for m in (5, 9, 14):
        out[f"s2a_ra_n2_m{m}"] = cfg(ref, aug={"randaugment": {"n": 2, "m": m}})
    out["s2a_ls01"] = cfg(ref, label_smoothing=0.1)
    out["s2a_dropout03"] = cfg(ref, arch={"dropout": 0.3})
    for wd, tag in ((1e-4, "1e-4"), (5e-3, "5e-3")):
        out[f"s2a_wd{tag}"] = cfg(ref, opt={"wd": wd})
    return out


STAGES["s2a"] = _s2a()

# ---- how each stage is run --------------------------------------------------
# Reps (independent training runs per (config, seed)) for each stage; run.py uses
# these when --reps is not given, so the notebook can run every stage the same way.
STAGE_REPS = {"s0": 2, "s1": 2, "s2a": 2}
# Configs whose per-epoch curves (curves.py, tuning seeds only) go in the report.
CURVE_CONFIGS = ["s0_noaug", "s0_flipcrop", "s2a_noaug", "s2a_ref"]

# ---- final configs (filled in after S4) ------------------------------------
FINAL_SOURCE = {}   # {"best": "<stage config id>", "fast": "<stage config id>"}
FINAL = {name: None for name in FINAL_SOURCE}


def _resolve_final():
    for name, cid in FINAL_SOURCE.items():
        FINAL[name] = get(cid)


def get(config_id):
    """kwargs for `config_id`, looked up in FINAL and every stage."""
    if config_id in FINAL and FINAL[config_id] is not None:
        return copy.deepcopy(FINAL[config_id])
    for stage in STAGES.values():
        if config_id in stage:
            return copy.deepcopy(stage[config_id])
    raise KeyError(f"unknown config {config_id!r}")


_resolve_final()
