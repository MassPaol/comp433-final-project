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


# ---- FINAL ------------------------------------------------------------------
# baseline: the testbed's C2 example (ViT-Tiny, CLS features, AdamW head, 10 epochs),
# fixed before any results, so best and fast can be compared with it on the report seeds.
FINAL_SOURCE = {"baseline": "s1_vitti_adamw"}


def get(config_id):
    """kwargs for a config id from FINAL or any stage. Raises KeyError if unknown."""
    if config_id in FINAL_SOURCE:
        config_id = FINAL_SOURCE[config_id]
    for stage in STAGES.values():
        if config_id in stage:
            return stage[config_id]
    raise KeyError(config_id)


FINAL = {name: get(cid) for name, cid in FINAL_SOURCE.items()}
