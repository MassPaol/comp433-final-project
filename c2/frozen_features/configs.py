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
