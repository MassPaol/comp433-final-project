"""Every configuration the C1 CNN sweep runs, written out explicitly.

Each value is the kwargs for `CNNMethod` (minus `seed`, which run.py sets per rep).
Stages run on the tuning seeds one after another: a stage is written only after
the previous stage's results are in, and the comment above it records why its
candidates were chosen. `FINAL` holds the selected `best`
and `fast` configs that run on the report seeds.
"""
import copy


def cfg(base, **changes):
    """Deep copy of `base` with top-level keys replaced (dict values merged one level deep)."""
    out = copy.deepcopy(base)
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and v:
            out[k] = {**out[k], **v}
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---- S0: baseline recipe, with and without the two basic augmentations -----
# Starting point chosen before any results (no tuning yet):
#   ConvNet-6 (3 stages x 2 convs, widths 32/64/128, BatchNorm), ~0.29M params;
#   SGD + Nesterov, lr 0.05, wd 5e-4, cosine with 5 warmup epochs; batch 32
#   (4 balanced batches of 25 per epoch); 200 epochs (= 800 steps).
# These are standard CIFAR-from-scratch settings (He et al. 2016; Zagoruyko &
# Komodakis 2016), scaled down for 100 images. S0 checks the pipeline end to end
# and gives the two anchors every later stage is compared against.
BASE = {
    "arch": {"type": "convnet", "width": 32, "convs": 2, "norm": "bn", "dropout": 0.0},
    "aug": {},
    "opt": {"name": "sgd", "lr": 0.05, "wd": 5e-4, "momentum": 0.9, "nesterov": True},
    "sched": {"name": "cosine", "warmup_epochs": 5},
    "epochs": 200,
    "batch_size": 32,
    "label_smoothing": 0.0,
    "tta_flip": False,
    "ensemble": 1,
}
FLIPCROP = {"crop": 4, "flip": True}

STAGES = {
    "s0": {
        "s0_noaug": cfg(BASE),
        "s0_flipcrop": cfg(BASE, aug=FLIPCROP),
    },
}



# ---- S1: architecture screen on the S0 flip+crop recipe (200 epochs) -------
# S0 result (T4, 2 reps): noaug 85.80, flipcrop 86.64 (+0.84 pp paired, 4/5 seeds).
# Without augmentation the net memorizes the 100 images by ~epoch 50; flip+crop keeps
# improving until ~epoch 150. (Laptop first look agreed: 85.37 / 86.17.) -> keep flip+crop as the reference recipe and screen
# backbones on it. ConvNet-6 w32 (= s0_flipcrop) is the reference, not re-run.
# Rep noise (~1 pp between reps) is as large as the effects, so S1 uses 2 reps.
FLIPCROP_BASE = cfg(BASE, aug=FLIPCROP)
S1_ARCHS = {
    "convnet3_w16": {"type": "convnet", "width": 16, "convs": 1},
    "convnet3_w32": {"type": "convnet", "width": 32, "convs": 1},
    "convnet3_w64": {"type": "convnet", "width": 64, "convs": 1},
    "convnet6_w16": {"type": "convnet", "width": 16, "convs": 2},
    "convnet6_w64": {"type": "convnet", "width": 64, "convs": 2},
    "resnet8_w16": {"type": "resnet", "width": 16, "n": 1},
    "resnet8_w32": {"type": "resnet", "width": 32, "n": 1},
    "resnet20_w16": {"type": "resnet", "width": 16, "n": 3},
    "resnet20_w32": {"type": "resnet", "width": 32, "n": 3},
    # GroupNorm: one representative per family at its standard width
    "convnet6_w32_gn": {"type": "convnet", "width": 32, "convs": 2, "norm": "gn"},
    "resnet20_w16_gn": {"type": "resnet", "width": 16, "n": 3, "norm": "gn"},
}
# arch is replaced, not merged, so a ResNet config doesn't carry the ConvNet's "convs" key
STAGES["s1"] = {f"s1_{name}": {**cfg(FLIPCROP_BASE), "arch": {"norm": "bn", "dropout": 0.0, **a}}
                for name, a in S1_ARCHS.items()}



# ---- S2a: augmentation / regularization, one change at a time --------------
# S1 result (T4, 2 reps): ConvNet-3 widths tie (w32 87.46, w16 86.99, w64 86.96, ~3.9 s);
# the rule picks w16 (within 0.5 pp of the top, faster). ConvNet-6 and ResNets are worse,
# GroupNorm -2.8 to -7.0 pp vs BN (5/5). -> backbone ConvNet-3 w16 BN; width re-checked in S4.
# Epochs 200 -> 300: S0 curves showed flip+crop still improving at ~150 epochs and
# stronger augmentation fits more slowly. s2a_ref re-runs flip+crop at 300 epochs so
# every S2a difference is measured at the same length.
S2_ARCH = {"type": "convnet", "width": 16, "convs": 1, "norm": "bn", "dropout": 0.0}
S2_REF = {**cfg(FLIPCROP_BASE, epochs=300), "arch": dict(S2_ARCH)}


def _s2a():
    ref = S2_REF
    out = {"s2a_ref": cfg(ref)}
    # decompose the basic augmentations
    out["s2a_noaug"] = {**cfg(ref), "aug": {}}
    out["s2a_flip_only"] = {**cfg(ref), "aug": {"flip": True}}
    out["s2a_crop_only"] = {**cfg(ref), "aug": {"crop": 4}}
    # one addition on top of flip+crop
    for size in (8, 16):
        out[f"s2a_cutout{size}"] = cfg(ref, aug={"cutout": size})
    for alpha, tag in ((0.2, "02"), (1.0, "1")):
        out[f"s2a_mixup{tag}"] = cfg(ref, aug={"mixup": alpha})
    out["s2a_cutmix1"] = cfg(ref, aug={"cutmix": 1.0})
    for m in (5, 9, 14):
        out[f"s2a_ra_n2_m{m}"] = cfg(ref, aug={"randaugment": {"n": 2, "m": m}})
    out["s2a_ls01"] = cfg(ref, label_smoothing=0.1)
    out["s2a_dropout03"] = cfg(ref, arch={"dropout": 0.3})
    for wd, tag in ((1e-4, "1e-4"), (5e-3, "5e-3")):
        out[f"s2a_wd{tag}"] = cfg(ref, opt={"wd": wd})
    return out


STAGES["s2a"] = _s2a()


# ---- S2b: combine the S2a winners, then leave each component out ------------
# S2a result (T4, 2 reps, paired vs s2a_ref): RandAugment N=2 +1.6/+2.1/+1.9 pp at M=5/9/14
# (5/5 seeds each); mixup a=1 +1.37 (4/5), a=0.2 +1.03 (5/5); label smoothing 0.1 +0.61 (3/5,
# inconclusive); Cutout, CutMix, weight decay 1e-4/5e-3 within +/-0.2 pp; dropout 0.3 -0.77 (0/5);
# no augmentation -1.07 (0/5). Flip carries the flip+crop gain (flip-only +0.21, crop-only -0.41).
# -> s2b_full = flip + crop + RandAugment(2, 9) + mixup(a=1) + label smoothing 0.1; each
# s2b_no_<x> removes one component (the leave-one-out ablation, paired vs s2b_full), and two
# variants swap in the runner-up setting of the two strong components. Cutout, CutMix, dropout
# and the weight-decay changes are dropped. Same backbone, optimizer and 300 epochs as S2a.
S2B_FULL = cfg(S2_REF, aug={"randaugment": {"n": 2, "m": 9}, "mixup": 1.0}, label_smoothing=0.1)


def _s2b():
    full = S2B_FULL
    out = {"s2b_full": cfg(full)}
    out["s2b_no_flip"] = cfg(full, aug={"flip": False})
    out["s2b_no_crop"] = cfg(full, aug={"crop": 0})
    out["s2b_no_ra"] = cfg(full, aug={"randaugment": None})
    out["s2b_no_mixup"] = cfg(full, aug={"mixup": 0.0})
    out["s2b_no_ls"] = cfg(full, label_smoothing=0.0)
    out["s2b_mixup02"] = cfg(full, aug={"mixup": 0.2})
    out["s2b_ra_m14"] = cfg(full, aug={"randaugment": {"n": 2, "m": 14}})
    return out


STAGES["s2b"] = _s2b()


# ---- S3: optimization and training length, on the two S2b recipes ----------
# S2b result (T4, 2 reps, paired vs s2b_full 88.94): the components don't add up.
# Leave-one-out: no mixup +1.03 (4/5), no label smoothing +0.69 (5/5), no RandAugment -0.27
# (2/5), no flip -0.50, no crop -0.29; mixup a=0.2 +0.36, RA M=14 -0.16. The full combination
# is no better than RandAugment alone (-0.50 vs s2a_ra_n2_m9): RandAugment and mixup are
# substitutes, and stacking them over-regularizes.
# Two recipes go on, one per role (selection rule over S2b; RA costs ~28 s vs ~6 s):
#   R ("best"): s2b_no_mixup = flip + crop + RandAugment(2, 9) + label smoothing 0.1, 89.97
#   F ("fast"): s2b_no_ra    = flip + crop + mixup(a=1) + label smoothing 0.1,    88.67, 5.9 s
# S3 changes one optimization setting at a time on R (paired vs s2b_no_mixup, not re-run),
# and sweeps the epoch count on F (paired vs s2b_no_ra), since F is the candidate for `fast`.
S3_R = cfg(S2B_FULL, aug={"mixup": 0.0})        # == s2b_no_mixup
S3_F = cfg(S2B_FULL, aug={"randaugment": None})  # == s2b_no_ra


def _s3():
    r, f = S3_R, S3_F
    out = {}
    for lr, tag in ((0.025, "0025"), (0.1, "01")):
        out[f"s3_r_lr{tag}"] = cfg(r, opt={"lr": lr})
    for lr, tag in ((1e-3, "1e-3"), (3e-3, "3e-3")):
        out[f"s3_r_adamw{tag}"] = cfg(r, opt={"name": "adamw", "lr": lr, "wd": 0.05})
    out["s3_r_onecycle"] = cfg(r, sched={"name": "onecycle"})
    for e in (150, 600):
        out[f"s3_r_e{e}"] = cfg(r, epochs=e)
    for bs in (16, 100):
        out[f"s3_r_bs{bs}"] = cfg(r, batch_size=bs)
    for e in (100, 150, 600):
        out[f"s3_f_e{e}"] = cfg(f, epochs=e)
    return out


STAGES["s3"] = _s3()


# ---- S4: width, flip TTA and ensembles -> the best / fast candidates --------
# S3 result (T4, 2 reps): no optimization change beats the S0 settings. On R (paired vs
# s2b_no_mixup 89.97): lr 0.025 -0.54, lr 0.1 -0.43, AdamW 3e-3 -0.31 (3/5), AdamW 1e-3 -1.41,
# one-cycle -0.41, batch 16 -0.51, full batch -1.01, 150 epochs -1.36 (0/5), 600 epochs -0.17
# (3/5, twice the time). On F (paired vs s2b_no_ra 88.67): 100 epochs -0.96 (1/5, 2.0 s),
# 150 -0.94 (0/5), 600 +0.17 (4/5, 12 s).
# -> keep SGD lr 0.05, cosine, batch 32, 300 epochs. S4 changes one thing at a time on R
# (paired vs s2b_no_mixup): width 32 / 64; flip TTA; ensembles of 3 and 5 (member 0 is the
# single model, so each ensemble contains its reference); and 5 members + TTA as the combined
# candidate. On F (paired vs s2b_no_ra / s3_f_e100): TTA at 300 and 100 epochs, 200 epochs, and
# 3 x 100 epochs (same compute as one 300-epoch run: ensemble vs longer training).
def _s4():
    r, f = S3_R, S3_F
    out = {}
    for w in (32, 64):
        out[f"s4_r_w{w}"] = cfg(r, arch={"width": w})
    out["s4_r_tta"] = cfg(r, tta_flip=True)
    for k in (3, 5):
        out[f"s4_r_ens{k}"] = cfg(r, ensemble=k)
    out["s4_r_ens5_tta"] = cfg(r, ensemble=5, tta_flip=True)
    out["s4_f_tta"] = cfg(f, tta_flip=True)
    out["s4_f_e100_tta"] = cfg(f, epochs=100, tta_flip=True)
    out["s4_f_e200"] = cfg(f, epochs=200)
    out["s4_f_e100_ens3"] = cfg(f, epochs=100, ensemble=3)
    return out


STAGES["s4"] = _s4()

# ---- how each stage is run --------------------------------------------------
# Reps (independent training runs per (config, seed)) for each stage; run.py uses
# these when --reps is not given, so the notebook can run every stage the same way.
STAGE_REPS = {"s0": 2, "s1": 2, "s2a": 2, "s2b": 2, "s3": 2, "s4": 2}
# Configs whose per-epoch curves (curves.py, tuning seeds only) go in the report.
CURVE_CONFIGS = ["s0_noaug", "s0_flipcrop", "s2a_noaug", "s2a_ref", "s2b_full"]


# ---- final configs: chosen by `python -m c1.cnn.select --hw colab-t4` -------
# S4 result (T4, 2 reps): nothing added at this stage helps. On R (paired vs s2b_no_mixup
# 89.97): width 32 -0.23, width 64 -0.69 (0/5), flip TTA -0.13, ensembles of 3 / 5 +0.09 / +0.04
# (2/5), 5 members + TTA +0.09 at ~130 s. On F: TTA -0.24 at 300 epochs and -0.19 at 100,
# 3 x 100 epochs ties 1 x 100 (+0.03); 200 epochs 88.14 in 3.7 s (-0.53 vs 300, +0.41 vs 150).
# Selection rule over all T4 tuning rows (stages S0-S4): top mean s4_r_ens3 90.06;
#   best = s4_r_tta (89.84, 25.5 s): within 0.5 pp of the top and the fastest such config.
#          It trains exactly like s2b_no_mixup; it is faster only because S4 ran in a faster
#          session, so the pick between the two is a tie either way.
#   fast = s4_f_e200 (88.14, 3.7 s): the fastest config within 2 pp of best
#          (flip + crop + mixup + label smoothing, 200 epochs).
FINAL_SOURCE = {"best": "s4_r_tta", "fast": "s4_f_e200"}
FINAL = {name: None for name in FINAL_SOURCE}


def _resolve_final():
    for name, cid in FINAL_SOURCE.items():
        FINAL[name] = get(cid)


def get(config_id):
    """kwargs for `config_id`, looked up in FINAL and every stage."""
    if config_id in FINAL and FINAL[config_id] is not None:
        return copy.deepcopy(FINAL[config_id])
    for stage in STAGES.values():
        if config_id in stage:
            return copy.deepcopy(stage[config_id])
    raise KeyError(f"unknown config {config_id!r}")


_resolve_final()
