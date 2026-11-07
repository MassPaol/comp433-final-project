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

FINAL = {}


def get(config_id):
    """kwargs for a config id from FINAL or any stage. Raises KeyError if unknown."""
    if config_id in FINAL:
        return FINAL[config_id]
    for stage in STAGES.values():
        if config_id in stage:
            return stage[config_id]
    raise KeyError(config_id)
