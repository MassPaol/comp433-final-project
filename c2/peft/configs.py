"""Every configuration of the C2 parameter-efficient fine-tuning experiments.

Each value is the kwargs for `PEFTMethod`. Add a new stage after looking at the
previous stage's results, and write a comment above it saying why.
"""

# ---- P0: first numbers ----------------------------------------------------
# `head` is the course testbed's C2 example (frozen backbone, new linear head,
# AdamW lr 1e-3, wd 0.01, 10 epochs, batch 32): the control every PEFT method
# is compared against. BitFit adds the backbone's bias terms; biases usually
# need a larger learning rate than full fine-tuning, so try 3 values.
P0 = {
    "head": {"mode": "head"},
    "bitfit_lr1e-3": {"mode": "bitfit", "lr": 1e-3},
    "bitfit_lr3e-3": {"mode": "bitfit", "lr": 3e-3},
    "bitfit_lr1e-2": {"mode": "bitfit", "lr": 1e-2},
}

# ---- P1: tune BitFit, add LoRA and VPT ------------------------------------
# P0 result: head 98.74%, bitfit lr 1e-3 98.91% (+0.17 pp, within noise),
# lr 3e-3 slightly worse, lr 1e-2 breaks (91.6%). Best lr was the lowest tried,
# so BitFit goes lower (3e-4) and longer (20 epochs), plus random flips.
# LoRA on attention qkv: rank 2 / 4 / 8 at lr 1e-3, and rank 4 at 3e-3.
# VPT: 10 prompts, shallow vs deep, lr 1e-3 / 3e-3 (a toy check showed deep VPT
# fails to learn at 1e-2, like BitFit).
P1 = {
    "bitfit_lr3e-4": {"mode": "bitfit", "lr": 3e-4},
    "bitfit_lr3e-4_ep20": {"mode": "bitfit", "lr": 3e-4, "epochs": 20},
    "bitfit_lr1e-3_ep20": {"mode": "bitfit", "lr": 1e-3, "epochs": 20},
    "bitfit_lr1e-3_flip": {"mode": "bitfit", "lr": 1e-3, "flip": True},
    "lora_r2": {"mode": "lora", "lora_rank": 2, "lr": 1e-3},
    "lora_r4": {"mode": "lora", "lora_rank": 4, "lr": 1e-3},
    "lora_r8": {"mode": "lora", "lora_rank": 8, "lr": 1e-3},
    "lora_r4_lr3e-3": {"mode": "lora", "lora_rank": 4, "lr": 3e-3},
    "vpt_shallow10_lr1e-3": {"mode": "vpt", "vpt_prompts": 10, "lr": 1e-3},
    "vpt_shallow10_lr3e-3": {"mode": "vpt", "vpt_prompts": 10, "lr": 3e-3},
    "vpt_deep10_lr1e-3": {"mode": "vpt", "vpt_prompts": 10, "vpt_deep": True, "lr": 1e-3},
    "vpt_deep10_lr3e-3": {"mode": "vpt", "vpt_prompts": 10, "vpt_deep": True, "lr": 3e-3},
}

STAGES = {"p0": P0, "p1": P1}

# ---- FINAL: chosen by `python -m c2.peft.select --hw colab-t4` ------------------
# Team rule (common/sweep.py select) on the T4 tuning rows: the top config is
# lora_r2 (99.23%), but head (98.74%) is within 0.5 pp and faster (3.6 s vs 5.2 s
# train + inference), so head is best; it is also the fastest config, so fast too.
# tests/test_peft.py checks this matches results/ammar.csv.
# Report seeds (T4): head 96.56 +/- 4.07 % (train 1.5 s, predict 1.7 s / 700 images).
FINAL_SOURCE = {"best": "head", "fast": "head"}


def all_configs():
    out = {}
    for stage in STAGES.values():
        out.update(stage)
    return out


FINAL = {name: all_configs()[cid] for name, cid in FINAL_SOURCE.items()}


def get(config_id):
    """kwargs for a config id; also accepts the FINAL names ("best", "fast")."""
    if config_id in FINAL:
        return FINAL[config_id]
    return all_configs()[config_id]
