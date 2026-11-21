"""C2 CLIP search. Every stage runs on tuning seeds 0-4.
Set PROBE_BASE and TIP_BASE from the previous stage's best, and FINAL after selection."""
STAGES = {}
PROBE_BASE = {"lr": 0.001, "steps": 100}
TIP_BASE = {"alpha": 0.5, "beta": 1.0}

STAGES["zeroshot"] = {"c2_zeroshot": {}}
STAGES["probe"] = {f"c2_probe_lr{lr:g}_s{s}": {"lr": lr, "steps": s}
                   for lr in [1e-3, 1e-2] for s in [100, 300]}
STAGES["wiseft"] = {f"c2_wiseft_m{m:g}": {**PROBE_BASE, "mix": m} for m in [0.25, 0.5, 0.75]}
STAGES["tip"] = {f"c2_tip_a{a:g}_b{b:g}": {"alpha": a, "beta": b}
                 for a in [0.5, 1, 2] for b in [1, 5.5, 10]}
STAGES["tipf"] = {f"c2_tipf_lr{lr:g}_s{s}": {**TIP_BASE, "lr": lr, "steps": s}
                  for lr in [1e-4, 1e-3] for s in [100, 300]}

FINAL = {
    "zeroshot": "c2_zeroshot",
    "probe_best": "c2_probe_lr0.001_s300",
    "probe_fast": "c2_probe_lr0.01_s100",
    "wiseft_best": "c2_wiseft_m0.5",
    "wiseft_fast": "c2_wiseft_m0.75",
    "tip_best": "c2_tip_a0.5_b1",
    "tip_fast": "c2_tip_a0.5_b5.5",
    "tipf_best": "c2_tipf_lr0.0001_s100",
    "tipf_fast": "c2_tipf_lr0.001_s100",
}
