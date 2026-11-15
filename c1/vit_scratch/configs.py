"""Every configuration of the C1 ViT-from-scratch experiments.

Each value is the kwargs for `ViTScratch`. Add a new stage after looking at the
previous stage's results, and write a comment above it saying why.
"""

# ---- S0: the course testbed baseline, unchanged ---------------------------
STAGES = {
    "s0": {
        "baseline": {},              # exact testbed settings
        "baseline_flip": {"flip": True},  # + random horizontal flips
    },
}


def all_configs():
    out = {}
    for stage in STAGES.values():
        out.update(stage)
    return out


def get(config_id):
    return all_configs()[config_id]
