"""Every configuration the C1 self-supervised sweep runs (k-means and SimSiam), written out explicitly.

Each value is the kwargs for `KMeansMethod` (stages k1, k2) or `SimSiamMethod` (stages ss1-ss4).
Stages run on the tuning seeds, one after another. A stage is only written after the previous
stage's results are in, and the comment above it records why those candidates were chosen.

>>> Stages marked EDIT use defaults that you should update from the previous stage's results
>>> (K2_BASES, SS_BASE) and then record the reasoning in NOTES.md.
`FINAL_SOURCE` maps the four final names (kmeans_best, kmeans_fast, simsiam_best, simsiam_fast)
to config ids; the values below are PLACEHOLDERS until `python -m c1.ssl.select` has run.
"""

# Same tuned fixed features as c1.classical (copied so this file does not depend on it).
SCAT = ("scattering", {"J": 3, "L": 4, "color": True})
HOG = ("hog", {"orientations": 9, "pixels_per_cell": 8, "cells_per_block": 2, "color": True})
HSV = ("hsv_hist", {"bins": 8})

K_FEATURES = {
    "raw": [("raw", {})],
    "hog": [HOG],
    "hsv": [HSV],
    "scat": [SCAT],
    "hog+hsv": [HOG, HSV],
    "scat+hog+hsv": [SCAT, HOG, HSV],
}
HEADS = {
    "lr": {"head": "lr", "head_params": {}},
    "knn": {"head": "knn", "head_params": {}},
    "kmeans": {"head": "kmeans", "head_params": {"k": 8, "mode": "vote"}},
}

METHOD_OF_STAGE = {"k1": "kmeans", "k2": "kmeans",
                   "ss1": "simsiam", "ss2": "simsiam", "ss3": "simsiam", "ss4": "simsiam"}
STAGES = {}


# ---- K1: every feature set x number of clusters x cluster head --------------
# "vote": clusters named by majority label; "dist": distances to centroids -> logistic regression.
STAGES["k1"] = {
    f"k1_{fname}_k{k}_{mode}": {"features": feats, "k": k, "mode": mode}
    for fname, feats in K_FEATURES.items()
    for k in [2, 4, 8, 16, 32]
    for mode in ["vote", "dist"]
}

# ---- K2: preprocessing variants on the top feature sets of K1 ---------------
# EDIT after K1: put the 3 best feature sets here with their best (k, mode).
K2_BASES = {
    "scat+hog+hsv": {"k": 16, "mode": "dist"},
    "hog+hsv": {"k": 16, "mode": "dist"},
    "raw": {"k": 16, "mode": "dist"},
}


def _k2():
    out = {}
    for fname, base in K2_BASES.items():
        b = {"features": K_FEATURES[fname], **base}
        out[f"k2_{fname}_fliptrain"] = {**b, "flip_train": True}
        out[f"k2_{fname}_fliptrain_fliptta"] = {**b, "flip_train": True, "flip_tta": True}
        out[f"k2_{fname}_noscale"] = {**b, "scale": False}
        for p in [20, 50]:
            out[f"k2_{fname}_pca{p}"] = {**b, "pca": p}
    return out


STAGES["k2"] = _k2()

# ---- SimSiam ----------------------------------------------------------------
# EDIT after SS1/SS2/SS3: SS_BASE holds the settings that beat the method defaults
# (defaults: epochs=200, width=32, proj_dim=128, batch_size=50, lr=0.05, wd=5e-4,
#  crop_min_scale=0.3, jitter=0.4, head="lr").
SS_BASE = {}

# SS1: pretraining length x head. epochs=0 is the random-init backbone baseline.
STAGES["ss1"] = {
    f"ss1_e{e}_{h}": {**SS_BASE, "epochs": e, **HEADS[h]}
    for e in [0, 50, 100, 200, 400] for h in HEADS
}

# SS2: architecture size (backbone width x projector dimension)
STAGES["ss2"] = {
    f"ss2_w{w}_p{p}": {**SS_BASE, "width": w, "proj_dim": p}
    for w in [16, 32, 64] for p in [64, 128, 256]
}

# SS3: optimization and augmentation strength, one factor at a time
_SS3 = {"lr": [0.01, 0.05, 0.2], "weight_decay": [0, 5e-4, 5e-3],
        "crop_min_scale": [0.2, 0.5, 0.8, 1.0], "jitter": [0, 0.4, 0.8],
        "batch_size": [25, 50, 100]}
STAGES["ss3"] = {f"ss3_{key}_{v:g}": {**SS_BASE, key: v}
                 for key, vals in _SS3.items() for v in vals}


# SS4: heads on the learned embeddings (incl. k-means as the head, and flip augmentation of the head)
def _ss4():
    out = {}
    for C in [0.01, 0.1, 1, 10, 100]:
        out[f"ss4_lr_C{C:g}"] = {**SS_BASE, "head": "lr", "head_params": {"C": C}}
    out["ss4_lr_C1_fliphead"] = {**SS_BASE, "head": "lr", "flip_head": True}
    for k in [1, 3, 5, 9]:
        out[f"ss4_knn_k{k}"] = {**SS_BASE, "head": "knn", "head_params": {"n_neighbors": k}}
    for k in [2, 4, 8, 16, 32]:
        for mode in ["vote", "dist"]:
            out[f"ss4_kmeans_k{k}_{mode}"] = {
                **SS_BASE, "head": "kmeans", "head_params": {"k": k, "mode": mode}}
    return out


STAGES["ss4"] = _ss4()


# ---- FINAL: chosen by `python -m c1.ssl.select` (rule in select.py) ----
# PLACEHOLDERS: replace with the ids printed by select.py, then record them in NOTES.md.
FINAL_SOURCE = {
    "kmeans_best": "k2_scat+hog+hsv_fliptrain",
    "kmeans_fast": "k1_hog+hsv_k8_vote",
    "simsiam_best": "ss1_e200_lr",
    "simsiam_fast": "ss1_e50_knn",
}


def method_of(config_id):
    """'kmeans' or 'simsiam' for a config id or a FINAL name; None if unknown."""
    if config_id in FINAL_SOURCE:
        return method_of(FINAL_SOURCE[config_id])
    return METHOD_OF_STAGE.get(config_id.split("_", 1)[0])


def get(config_id):
    """(method name, kwargs) for a config id from FINAL or any stage. Raises KeyError if unknown."""
    cid = FINAL_SOURCE.get(config_id, config_id)
    for stage in STAGES.values():
        if cid in stage:
            return method_of(cid), stage[cid]
    raise KeyError(config_id)
