"""Every C1 self-supervised configuration (Coates k-means and SimSiam), written out.

Each value is the kwargs for KMeansMethod (stages k1, k2) or SimSiamMethod (ss1-ss6).
Stages run on the tuning seeds (0-4). The SVM / probe C is chosen by cross-validation inside
each fit, so it is not a config axis.

Workflow (two passes, see the notebook):
  pass 1: k1, ss1, ss6       (no edits needed; ss6 are hand-picked recipes that ignore SS_BASE)
  edit:   KMEANS_BASE <- winner of k1, SS_BASE <- winner of ss1 ("epochs"), record it in NOTES.md
  pass 2: k2, ss2, ss3, ss4, ss5   (one factor at a time around the base)
  then:   `python -m c1.ssl.select` -> copy its four picks into FINAL_SOURCE.

SimSiam backbone: Ryan's selected CNN (ConvNet-3 w16 + BatchNorm), built inside SimSiamMethod
by c1.cnn.models.build_backbone. Augmentation: common.augment.Augmenter (+ a random resized crop).
"""
STAGES = {}
METHOD_OF_STAGE = {"k1": "kmeans", "k2": "kmeans",
                   "ss1": "simsiam", "ss2": "simsiam", "ss3": "simsiam",
                   "ss4": "simsiam", "ss5": "simsiam", "ss6": "simsiam"}

# ---------------------------------------------------------------- Coates k-means (fast variant)
# K1: dictionary size x patch size (normalization, whitening and stride at their defaults)
STAGES["k1"] = {f"k1_k{k}_p{p}": {"n_clusters": k, "patch_size": p}
                for k in [5, 10, 15, 25, 50, 100] for p in [4, 6, 8]}

# K2: preprocessing on the best K1 setting (EDIT KMEANS_BASE after K1)
KMEANS_BASE = {"n_clusters": 10, "patch_size": 4}
STAGES["k2"] = {
    **{f"k2_eps{e:g}": {**KMEANS_BASE, "epsilon": e} for e in [0.01, 0.1, 1.0, 10.0]},
    **{f"k2_wzca{w:g}": {**KMEANS_BASE, "whitening_epsilon": w} for w in [1e-5, 1e-3, 1e-1, 1.0]},
}

# ---------------------------------------------------------------- SimSiam (deep variant)
# Method defaults (method_simsiam.py): 200 epochs, ConvNet-3 w16 (64-d features), proj 64,
# batch 32 (4 balanced batches of 25), SGD lr 0.05, RRC min_scale 0.3, pad-crop 4 + flip,
# no RandAugment, no labels during pretraining, frozen features + linear SVM.
SS_BASE = {"epochs": 100}   # EDIT after SS1: {"epochs": <winner>}; keep other keys at their defaults

# SS1: pretraining length (epochs=0 is the random-init backbone baseline)
STAGES["ss1"] = {f"ss1_e{e}": {**SS_BASE, "epochs": e} for e in [0, 50, 100, 200, 400]}
SS_REF = f"ss1_e{SS_BASE.get('epochs', 200)}"   # base config every later SimSiam stage is paired against
assert SS_REF in STAGES["ss1"], f"SS_BASE epochs must be one of the SS1 values, got {SS_REF}"

# SS2: augmentation strength (Ryan's Augmenter + random resized crop), one factor at a time
STAGES["ss2"] = {
    **{f"ss2_scale{s:g}": {**SS_BASE, "min_scale": s} for s in [0.15, 0.6, 1.0]},  # 1.0 = no RRC
    **{f"ss2_ra{n}_m{m}": {**SS_BASE, "ra_n": n, "ra_m": m} for n, m in [(1, 9), (2, 9), (2, 14)]},
    "ss2_cutout8": {**SS_BASE, "cutout": 8},
    "ss2_nocrop": {**SS_BASE, "crop": 0},
}

# SS3: architecture (Ryan's ConvNet family) and projector size
STAGES["ss3"] = {
    **{f"ss3_w{w}": {**SS_BASE, "width": w} for w in [32, 64]},
    "ss3_c2": {**SS_BASE, "convs": 2},
    **{f"ss3_proj{p}": {**SS_BASE, "proj_dim": p} for p in [32, 128, 256]},
}

# SS4: optimization
STAGES["ss4"] = {
    **{f"ss4_lr{v:g}": {**SS_BASE, "lr": v} for v in [0.01, 0.2]},
    **{f"ss4_wd{v:g}": {**SS_BASE, "weight_decay": v} for v in [0, 5e-3]},
    **{f"ss4_bs{v}": {**SS_BASE, "batch_size": v} for v in [50, 100]},
}

# SS5: using the 100 labels: joint SSL + cross-entropy loss, or fine-tuning after pretraining
STAGES["ss5"] = {
    **{f"ss5_joint{w:g}": {**SS_BASE, "sup_weight": w} for w in [0.3, 1.0, 3.0]},
    **{f"ss5_ft{e}": {**SS_BASE, "ft_epochs": e} for e in [50, 150]},
}

# SS6: hand-picked recipes (independent of SS_BASE). Written before any results, as bets on:
#   best accuracy: SimSiam as a regularizer on top of Ryan's supervised recipe (RandAugment views
#                  + labels, either joint or pretrain-then-fine-tune)
#   fast:          short GPU-only augmentation (no RandAugment loop), frozen features + SVM
STAGES["ss6"] = {
    "ss6_joint_ra": {"epochs": 200, "ra_n": 2, "ra_m": 9, "sup_weight": 1.0},
    "ss6_ft_ra": {"epochs": 100, "ra_n": 2, "ra_m": 9, "ft_epochs": 150},
    "ss6_fast_svm": {"epochs": 50, "batch_size": 50},
    "ss6_fast_joint": {"epochs": 100, "sup_weight": 1.0},
}

# FINAL: PROVISIONAL until `python -m c1.ssl.select --hw colab-t4` has run over ALL stages.
# Copy its four picks here (the notebook refuses to continue if they disagree).
FINAL_SOURCE = {
    "kmeans_best": "k2_wzca1",
    "kmeans_fast": "k2_wzca1",
    "simsiam_best": "ss6_ft_ra",
    "simsiam_fast": "ss5_ft150",
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
            return method_of(cid), dict(stage[cid])
    raise KeyError(config_id)