"""Every C1 self-supervised configuration (k-means Coates pipeline and SimSiam), written out.

Each value is the kwargs for KMeansMethod (stages k1, k2) or SimSiamMethod (ss1-ss3).
Stages run on the tuning seeds (0-4). Fill KMEANS_BASE / SS_BASE from the previous stage's
winner, record the choice in NOTES.md, then run the next stage. The SVM / probe C is chosen by
cross-validation inside each fit, so it is not a config axis.
"""
STAGES = {}
METHOD_OF_STAGE = {"k1": "kmeans", "k2": "kmeans",
                   "ss1": "simsiam", "ss2": "simsiam", "ss3": "simsiam"}

# K1: dictionary size x patch size (normalization, whitening and stride at their defaults)
STAGES["k1"] = {f"k1_k{k}_p{p}": {"n_clusters": k, "patch_size": p}
                for k in [50, 100, 200, 400, 800] for p in [4, 6, 8]}

# K2: preprocessing on the best K1 setting (EDIT KMEANS_BASE after K1)
KMEANS_BASE = {"n_clusters": 200, "patch_size": 6}
STAGES["k2"] = {
    **{f"k2_eps{e:g}": {**KMEANS_BASE, "epsilon": e} for e in [0.01, 0.1, 1.0, 10.0]},
    **{f"k2_wzca{w:g}": {**KMEANS_BASE, "whitening_epsilon": w} for w in [1e-5, 1e-3, 1e-1, 1.0]},
}

# SimSiam (EDIT SS_BASE after each stage). Defaults are the values in method_simsiam.py.
SS_BASE = {}

# SS1: pretraining length (epochs=0 is the random-init backbone baseline)
STAGES["ss1"] = {f"ss1_e{e}": {**SS_BASE, "epochs": e} for e in [0, 50, 100, 200, 400]}

# SS2: architecture size (backbone width x projection dimension)
STAGES["ss2"] = {f"ss2_w{w}_p{p}": {**SS_BASE, "width": w, "proj_dim": p}
                 for w in [16, 32, 64] for p in [64, 128, 256]}

# SS3: optimization and augmentation strength, one factor at a time
_SS3 = {"lr": [0.01, 0.05, 0.2], "weight_decay": [0, 5e-4, 5e-3],
        "min_scale": [0.2, 0.5, 0.8], "cj_strength": [0, 0.4, 0.8],
        "batch_size": [25, 50, 100]}
STAGES["ss3"] = {f"ss3_{k}_{v:g}": {**SS_BASE, k: v}
                 for k, vals in _SS3.items() for v in vals}

# FINAL: placeholders until `python -m c1.ssl.select` picks the winners on the tuning seeds
FINAL_SOURCE = {
    "kmeans_best": "k2_eps0.1",
    "kmeans_fast": "k1_k100_p6",
    "simsiam_best": "ss1_e200",
    "simsiam_fast": "ss2_w16_p64",
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