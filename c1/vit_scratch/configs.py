"""Every configuration of the C1 ViT-from-scratch experiments.

Each value is the kwargs for `ViTScratch`. Add a new stage after looking at the
previous stage's results, and write a comment above it saying why.
"""

# ---- S0: the course testbed baseline, unchanged ---------------------------
S0 = {
    "baseline": {},                     # exact testbed settings
    "baseline_flip": {"flip": True},    # + random horizontal flips
}

# ---- S1: change ONE thing at a time from the baseline ---------------------
# S0 result (tuning seeds, local CPU): baseline 77.3%, +flip 77.8% (3/5 seeds won).
# Flip alone barely helps, so S1 asks which single knob matters most:
# model size (depth, width), token size (patch), training length, and a
# stronger augmentation (random crop = random shifts of up to 4 px).
S1 = {
    "s1_depth2": {"depth": 2},
    "s1_depth4": {"depth": 4},
    "s1_depth6": {"depth": 6},
    "s1_patch2": {"patch_size": 2},     # 256 tokens: slower, finer detail
    "s1_patch8": {"patch_size": 8},     # 16 tokens: faster, coarser
    "s1_dim64": {"embed_dim": 64},
    "s1_dim192": {"embed_dim": 192},
    "s1_ep200": {"epochs": 200},
    "s1_ep300": {"epochs": 300},
    "s1_crop": {"crop_pad": 4},
    "s1_flip_crop": {"flip": True, "crop_pad": 4},
}

# ---- S2: small-data ViT techniques, on top of flip + crop -----------------
# S1 result: augmentation was the only clear gain (flip + crop 79.3%, +2.1 pp,
# 4/5 seeds); size knobs were within noise; deeper / coarser hurt. So flip + crop
# becomes the base, and S2 tests the techniques built for small datasets:
# SPT and LSA (Lee et al. 2021), alone and together, and the CCT conv tokenizer
# with and without sequence pooling (Hassani et al. 2021). depth4 is re-checked
# on top of augmentation since it was the best size knob in S1.
AUG = {"flip": True, "crop_pad": 4}
S2 = {
    "s2_spt": {**AUG, "tokenizer": "spt"},
    "s2_lsa": {**AUG, "lsa": True},
    "s2_spt_lsa": {**AUG, "tokenizer": "spt", "lsa": True},
    "s2_conv": {**AUG, "tokenizer": "conv"},
    "s2_conv_seq": {**AUG, "tokenizer": "conv", "pool": "seq"},
    "s2_seq": {**AUG, "pool": "seq"},
    "s2_depth4": {**AUG, "depth": 4},
}

STAGES = {"s0": S0, "s1": S1, "s2": S2}


def all_configs():
    out = {}
    for stage in STAGES.values():
        out.update(stage)
    return out


def get(config_id):
    return all_configs()[config_id]
