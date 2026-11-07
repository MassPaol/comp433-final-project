"""Every configuration the C1 classical sweep runs, written out explicitly.

Each value is the kwargs for `ClassicalMethod`. Stages run on the tuning seeds,
one after another: a stage is written only after the previous stage's results
are in, and the comment above it records why those candidates were chosen.
`FINAL` holds the selected `best` and `fast` configs that run on the report seeds.
"""

# ---- S1: every feature family x every classifier, default settings --------
S1_FEATURES = {
    "raw": {"features": [("raw", {})]},
    "rawpca": {"features": [("raw", {})], "pca": 50},
    "hsv": {"features": [("hsv_hist", {"bins": 16})]},
    "hog": {"features": [("hog", {})]},
    "lbp": {"features": [("lbp", {})]},
    "scat": {"features": [("scattering", {"J": 2, "L": 8})]},
}
S1_CLASSIFIERS = ["lr", "linsvm", "rbfsvm", "rf", "hgb", "xgb", "knn"]

STAGES = {
    "s1": {f"s1_{f}_{c}": {**fkw, "clf": c}
           for f, fkw in S1_FEATURES.items() for c in S1_CLASSIFIERS},
}


# ---- S2: classifier hyperparameters on the top-3 feature sets -------------
# S1 result (mean tuning acc, best classifier per feature set):
#   scat 85.7 (lr), hog 84.5 (rbfsvm), raw 80.0 (rbfsvm), rawpca 79.1, hsv 78.3, lbp 71.3.
# -> scat, hog, raw. Tree models only where their S1 mean was within 5 pp of the
#    S1 best (85.7 - 5 = 80.7): scat rf/hgb/xgb and hog rf.
C_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 100]


def _s2(feat, trees):
    base = S1_FEATURES[feat]
    out = {}
    for C in C_GRID:
        out[f"s2_{feat}_lr_C{C:g}"] = {**base, "clf": "lr", "clf_params": {"C": C}}
        out[f"s2_{feat}_linsvm_C{C:g}"] = {**base, "clf": "linsvm", "clf_params": {"C": C}}
    out[f"s2_{feat}_lrcv"] = {**base, "clf": "lr_cv"}
    for C in [0.1, 1, 10, 100]:
        for g in [0.25, 1, 4]:
            out[f"s2_{feat}_rbfsvm_C{C:g}_g{g:g}"] = {
                **base, "clf": "rbfsvm", "clf_params": {"C": C, "gamma_mult": g}}
    for k in [1, 3, 5, 9, 15]:
        for metric in ["euclidean", "cosine"]:
            out[f"s2_{feat}_knn_k{k}_{metric[:3]}"] = {
                **base, "clf": "knn", "clf_params": {"n_neighbors": k, "metric": metric}}
    if "rf" in trees:
        for mf in ["sqrt", 0.1]:
            out[f"s2_{feat}_rf_mf{mf}"] = {**base, "clf": "rf", "clf_params": {"max_features": mf}}
    if "hgb" in trees:
        for lr in [0.05, 0.1]:
            for depth in [3, None]:
                out[f"s2_{feat}_hgb_lr{lr:g}_d{depth}"] = {
                    **base, "clf": "hgb", "clf_params": {"learning_rate": lr, "max_depth": depth}}
    if "xgb" in trees:
        for depth in [2, 4]:
            for n in [100, 300]:
                out[f"s2_{feat}_xgb_d{depth}_n{n}"] = {
                    **base, "clf": "xgb", "clf_params": {"max_depth": depth, "n_estimators": n}}
    return out


STAGES["s2"] = {**_s2("scat", trees=("rf", "hgb", "xgb")),
                **_s2("hog", trees=("rf",)),
                **_s2("raw", trees=())}


# ---- S3: feature hyperparameters, each family with its best classifier so far ----
# S2 result: scat best = lr_cv 86.09 (all linear/RBF within 0.5 pp); hog best = rbfsvm
# C=10 gamma_mult=1 84.66; raw best = rbfsvm C=1 gamma_mult=1 80.00. Not tuned in S2,
# so S1 best: hsv -> rf 78.26, lbp -> rbfsvm 71.29.
# lr_cv picks C per feature variant on the training images, which keeps the
# scattering comparison fair across very different feature dimensions.
_HOG_CLF = {"clf": "rbfsvm", "clf_params": {"C": 10, "gamma_mult": 1}}
_RAW_CLF = {"clf": "rbfsvm", "clf_params": {"C": 1, "gamma_mult": 1}}


def _s3():
    out = {}
    for J in [1, 2, 3]:
        for L in [4, 6, 8]:
            for color in [False, True]:
                out[f"s3_scat_J{J}_L{L}_{'rgb' if color else 'gray'}"] = {
                    "features": [("scattering", {"J": J, "L": L, "color": color})], "clf": "lr_cv"}
        # pool=2 at L=8 RGB for every J (instead of only the best J/L, so the
        # stage has no internal dependency)
        out[f"s3_scat_J{J}_L8_rgb_pool2"] = {
            "features": [("scattering", {"J": J, "L": 8, "color": True, "pool": 2})], "clf": "lr_cv"}
    for ppc in [4, 8]:
        for cpb in [1, 2, 3]:
            for ori in [9, 12]:
                for color in [False, True]:
                    params = {"orientations": ori, "pixels_per_cell": ppc,
                              "cells_per_block": cpb, "color": color}
                    out[f"s3_hog_p{ppc}_b{cpb}_o{ori}_{'rgb' if color else 'gray'}"] = {
                        "features": [("hog", params)], **_HOG_CLF}
    for bins in [8, 16, 32]:
        out[f"s3_hsv_b{bins}"] = {"features": [("hsv_hist", {"bins": bins})], "clf": "rf"}
    for P, R in [(8, 1), (16, 2), (24, 3)]:
        for grid in [1, 2, 4]:
            out[f"s3_lbp_P{P}_R{R}_g{grid}"] = {
                "features": [("lbp", {"P": P, "R": R, "grid": grid})], "clf": "rbfsvm"}
    for pca in [10, 20, 50, None]:
        for gray in [False, True]:
            out[f"s3_raw_pca{pca}_{'gray' if gray else 'rgb'}"] = {
                "features": [("raw", {"gray": gray})], "pca": pca, **_RAW_CLF}
    return out


STAGES["s3"] = _s3()


# ---- S4: combinations, flip augmentation, scaling ---------------------------
# S3 result (best variant per family): scat J=3 L=4 RGB 86.51 (lr_cv); hog 8px cells,
# 2x2 blocks, 9 orient., RGB 85.94 (rbfsvm C=10); raw 80.00; hsv 8 bins 78.66; lbp
# P8 R1 4x4 grid 75.17.
# Part 1, concatenations: scattering + HOG (top 2) + HSV. HSV replaces raw (3rd) because
# raw is redundant with scattering (its order-0 coefficients are a low-pass copy of the
# image), while HSV adds color statistics neither shape feature has. Each combination is
# tried with both winning classifiers.
SCAT = ("scattering", {"J": 3, "L": 4, "color": True})
HOG = ("hog", {"orientations": 9, "pixels_per_cell": 8, "cells_per_block": 2, "color": True})
HSV = ("hsv_hist", {"bins": 8})
_LRCV = {"clf": "lr_cv"}


def _s4_concat():
    combos = {"scat+hog": [SCAT, HOG], "scat+hsv": [SCAT, HSV],
              "hog+hsv": [HOG, HSV], "scat+hog+hsv": [SCAT, HOG, HSV]}
    out = {}
    for name, feats in combos.items():
        out[f"s4_{name}_lrcv"] = {"features": feats, **_LRCV}
        out[f"s4_{name}_rbfsvm"] = {"features": feats, **_HOG_CLF}
    return out


# Part 2, written after part 1: every concatenation with scattering + HOG beats every single
# feature (best: scat+hog+hsv LR-CV 88.00; best single: scat 86.51). hog+hsv RBF-SVM reaches 87.83
# with a 0.04 s fit, the natural fast candidate. Flip augmentation (train: append mirrored
# images; test: average scores over image and mirror) on the best single feature, the best
# concatenation and the fast candidate; scale=False on the top 2 concatenations.
_S4_FLIP_BASES = {
    "scat": {"features": [SCAT], **_LRCV},
    "scat+hog+hsv_lrcv": {"features": [SCAT, HOG, HSV], **_LRCV},
    "hog+hsv_rbfsvm": {"features": [HOG, HSV], **_HOG_CLF},
}


def _s4_flip_scale():
    out = {}
    for name, base in _S4_FLIP_BASES.items():
        for ft, tta in [(True, False), (False, True), (True, True)]:
            tag = "_".join(t for t, on in (("fliptrain", ft), ("fliptta", tta)) if on)
            out[f"s4_{name}_{tag}"] = {**base, "flip_train": ft, "flip_tta": tta}
    out["s4_scat+hog+hsv_lrcv_noscale"] = {"features": [SCAT, HOG, HSV], **_LRCV, "scale": False}
    out["s4_scat+hog_lrcv_noscale"] = {"features": [SCAT, HOG], **_LRCV, "scale": False}
    return out


STAGES["s4"] = {**_s4_concat(), **_s4_flip_scale()}


# ---- FINAL: chosen by `python -m c1.classical.select` (rule in select.py), hw=local-3070 ----
# top:  s4_scat+hog+hsv_lrcv_fliptrain_fliptta  90.14 +/- 2.64  1.95 s
# best: s4_scat+hog+hsv_lrcv_fliptrain          89.80 +/- 3.15  1.43 s  (within 0.5 pp of top, faster)
# fast: s4_hog+hsv_rbfsvm                       87.83 +/- 4.78  0.37 s  (fastest with acc >= best - 2 pp)
# Both picks hinge on time margins < 2x, so the Colab T4 re-run re-checks the rule.
FINAL = {
    "best": STAGES["s4"]["s4_scat+hog+hsv_lrcv_fliptrain"],
    "fast": STAGES["s4"]["s4_hog+hsv_rbfsvm"],
}


def get(config_id):
    """kwargs for a config id from FINAL or any stage. Raises KeyError if unknown."""
    if config_id in FINAL:
        return FINAL[config_id]
    for stage in STAGES.values():
        if config_id in stage:
            return stage[config_id]
    raise KeyError(config_id)
