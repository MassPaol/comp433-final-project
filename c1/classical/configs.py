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

FINAL = {}


def get(config_id):
    """kwargs for a config id from FINAL or any stage. Raises KeyError if unknown."""
    if config_id in FINAL:
        return FINAL[config_id]
    for stage in STAGES.values():
        if config_id in stage:
            return stage[config_id]
    raise KeyError(config_id)
