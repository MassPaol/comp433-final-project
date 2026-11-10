"""Every configuration the C2 frozen-features sweep runs, written out explicitly.

Each value is the kwargs for `FrozenFeatureMethod`. Stages run on the tuning
seeds, one after another: a stage is written only after the previous stage's
results are in, and the comment above it records why those candidates were
chosen. `FINAL` holds the configs that run on the report seeds: the selected
`best` and `fast`, and the testbed `baseline` (fixed in advance).
"""

# ---- S1: every backbone x every head, default settings --------------------
# Backbone keys -> timm checkpoints in backbones.BACKBONES. Default features are each
# model's own pre-logits (ViT: CLS token after the final norm; CNN: pooled features).
S1_BACKBONES = ["vitti", "vits", "vits1k", "vitb", "deits", "rn50", "cnxt"]
S1_HEADS = ["adamw", "lr", "linsvm", "rbfsvm", "ncm", "knn"]

STAGES = {
    "s1": {f"s1_{b}_{h}": {"backbone": b, "head": h} for b in S1_BACKBONES for h in S1_HEADS},
}


# ---- S2: head hyperparameters and feature normalization --------------------
# S1 result (mean tuning acc, best head per backbone): vitb 99.77 (lr), vits 99.46 (lr),
# cnxt 99.43 (linsvm), vits1k 99.29 (ncm), deits 99.29 (adamw), vitti 99.09 (ncm), rn50 97.97.
# -> top 3 backbones (vitb, vits, cnxt), plus vitti, the testbed backbone (how far does a
#    better head take it?). Every head's main knob; the S1 default of each knob is not
#    repeated here (its S1 config is the default point of the curve).
S2_BACKBONES = ["vitb", "vits", "cnxt", "vitti"]
S2_C = [1e-3, 1e-2, 1e-1, 10, 100]  # + S1's C = 1


def _s2(b):
    out = {}
    for C in S2_C:
        out[f"s2_{b}_lr_C{C:g}"] = {"backbone": b, "head": "lr", "head_params": {"C": C}}
        out[f"s2_{b}_linsvm_C{C:g}"] = {"backbone": b, "head": "linsvm", "head_params": {"C": C}}
    out[f"s2_{b}_lrcv"] = {"backbone": b, "head": "lr_cv"}
    for C in [0.1, 10, 100]:
        out[f"s2_{b}_rbfsvm_C{C:g}"] = {"backbone": b, "head": "rbfsvm", "head_params": {"C": C}}
    for epochs in [10, 30, 100]:
        for lr in [1e-3, 1e-2]:
            if (epochs, lr) != (10, 1e-3):  # the testbed recipe = S1's adamw
                out[f"s2_{b}_adamw_e{epochs}_lr{lr:g}"] = {
                    "backbone": b, "head": "adamw", "head_params": {"epochs": epochs, "lr": lr}}
    for norm in ["none", "l2", "cl2n"]:  # S1's lr uses "standard"
        out[f"s2_{b}_lr_{norm}"] = {"backbone": b, "head": "lr", "norm": norm}
    for norm in ["none", "standard", "cl2n"]:  # S1's ncm uses "l2"
        out[f"s2_{b}_ncm_{norm}"] = {"backbone": b, "head": "ncm", "norm": norm}
    for k in [1, 3, 9, 15]:  # S1's knn uses k = 5
        out[f"s2_{b}_knn_k{k}"] = {"backbone": b, "head": "knn", "head_params": {"n_neighbors": k}}
    return out


STAGES["s2"] = {cid: kw for b in S2_BACKBONES for cid, kw in _s2(b).items()}


# ---- S3: feature source (which tokens, which blocks, which crop) -----------
# S2 result (best config per backbone): vitb linsvm C=0.001 99.83; vits lr norm=none 99.51;
# cnxt lr C=100 99.49; vitti lr C=0.01 99.23. Head tuning moved accuracy by <= 0.14 pp.
# -> top 2 backbones with their best head. Both are ViTs: pool (CLS / mean of patch tokens /
#    both) x last-k blocks concatenated, each block after the final norm (DINO's linear-eval
#    recipe; Head2Toe: intermediate layers), plus the whole image instead of the checkpoint's
#    center crop (crop_pct = 1.0). pool=cls, layers=1 is the S2 config itself.
S3_BASES = {
    "vitb": {"backbone": "vitb", "head": "linsvm", "head_params": {"C": 0.001}},
    "vits": {"backbone": "vits", "head": "lr", "norm": "none"},
}


def _s3(b, base):
    out = {}
    for pool in ["cls", "mean", "cls+mean"]:
        for layers in [1, 2, 4]:
            if (pool, layers) != ("cls", 1):
                out[f"s3_{b}_{pool}_L{layers}"] = {**base, "pool": pool, "layers": layers}
    out[f"s3_{b}_crop1"] = {**base, "crop_pct": 1.0}
    return out


STAGES["s3"] = {cid: kw for b, base in S3_BASES.items() for cid, kw in _s3(b, base).items()}


# ---- S4: flip augmentation, backbone concatenation, fp16 --------------------
# S3 result: no pool/layers variant beats the plain CLS token of the last block (all within
# -0.23..+0.03 pp); the whole image (crop_pct 1.0) lifts vits to 99.77 (+0.26), not vitb (-0.06).
# Bases: the top config (vitb + linsvm), the best small config (vits + LR, whole image) and the
# fast candidate (vitti + LR C=0.01, the best vitti config in S2).
S4_BASES = {
    "vitb": S3_BASES["vitb"],
    "vits": {**S3_BASES["vits"], "crop_pct": 1.0},
    "vitti": {"backbone": "vitti", "head": "lr", "head_params": {"C": 0.01}},
}


def _s4():
    out = {}
    for b, base in S4_BASES.items():
        # flip: train = add mirrored training images; tta = average scores of image and mirror
        for ft, tta in [(True, False), (False, True), (True, True)]:
            tag = "_".join(t for t, on in (("fliptrain", ft), ("fliptta", tta)) if on)
            out[f"s4_{b}_{tag}"] = {**base, "flip_train": ft, "flip_tta": tta}
        # fp16 (autocast) feature extraction: speed knob
        out[f"s4_{b}_fp16"] = {**base, "precision": "fp16"}
    # the whole image on vitti too (helped vits in S3)
    out["s4_vitti_crop1"] = {**S4_BASES["vitti"], "crop_pct": 1.0}
    # concatenated features of two backbones (each with its own preprocessing), vitb's head
    for other in ["vits", "cnxt"]:
        out[f"s4_vitb+{other}"] = {**S3_BASES["vitb"], "backbone": ["vitb", other]}
    return out


STAGES["s4"] = _s4()


# ---- FINAL: chosen by `python -m c2.frozen_features.select` (rule in select.py) ----
# Over all 191 tuning configs (s1-s4), local run (hw=local-3070):
# top:  s4_vitb_fp16   99.83 +/- 0.23  2.80 s
# best: s4_vits_fp16   99.77 +/- 0.17  0.96 s  (within 0.5 pp of top, fastest of those;
#                                               crop_pct 1.0 = deviates from the checkpoint's
#                                               preprocessing, see README "Preprocessing deviation")
# fast: s4_vitti_fp16  99.23 +/- 0.38  0.47 s  (fastest with acc >= best - 2 pp)
# baseline: the testbed's C2 example (ViT-Tiny, CLS features, AdamW head, 10 epochs),
# fixed before any results, so best and fast can be compared with it on the report seeds.
FINAL_SOURCE = {"best": "s4_vits_fp16", "fast": "s4_vitti_fp16", "baseline": "s1_vitti_adamw"}


def get(config_id):
    """kwargs for a config id from FINAL or any stage. Raises KeyError if unknown."""
    if config_id in FINAL_SOURCE:
        config_id = FINAL_SOURCE[config_id]
    for stage in STAGES.values():
        if config_id in stage:
            return stage[config_id]
    raise KeyError(config_id)


FINAL = {name: get(cid) for name, cid in FINAL_SOURCE.items()}
