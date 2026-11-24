# C1: Small CNN from scratch

**Owner:** Ryan

## What this is
A small convolutional network (a VGG-style ConvNet or a CIFAR-style ResNet) trained from random
initialization on only the 100 training images of each problem instance. It uses no pretrained weights
and no other data. Convolutions build in locality and translation equivariance, which is useful prior
knowledge when there are only 50 images per class. That's the contrast with the ViT trained from
scratch in `c1/vit_scratch`. With 100 images the network can memorize the training set within a few
epochs, so the main lever is **augmentation and regularization**: crop, flip, Cutout, mixup, CutMix,
RandAugment, label smoothing, weight decay, and dropout. The augmentation runs on the GPU from
`common/augment.py`. The reason for each stage's candidates is in the comment above it in
`configs.py`.

## Tuning stages
Hyperparameters are searched in stages on the tuning seeds 0–4. Each stage is written only after the
previous one's results are in, and config ids are prefixed with their stage (e.g. `s2a_cutout8`).

| stage | question |
|---|---|
| S0 | Does the pipeline work, and does basic augmentation (flip + crop) help? Sets the baseline. |
| S1 | Which backbone? Small screen of ConvNet / ResNet depth, width, BatchNorm vs GroupNorm. |
| S2a | What does each augmentation / regularizer add on its own? One change at a time vs flip + crop. |
| S2b | Combine the ones that helped, then leave each out of the combination (ablation). |
| S3 | Optimizer, learning rate, schedule, epochs, batch size for that recipe. |
| S4 | Width with the final recipe, flip TTA, ensembles → the `best` and `fast` configs. |

`best` and `fast` then run once on the report seeds 10–19.

## Files
| file | what it does |
|---|---|
| `models.py` | `ConvNet` (3 stages × 1–2 convs) and CIFAR `ResNet` (ResNet-8/20), with BatchNorm or GroupNorm |
| `method.py` | `CNNMethod`: normalization stats from the training images → fixed-length training with on-device augmentation → softmax averaged over an optional ensemble and flip TTA |
| `configs.py` | every config of every sweep stage, plus `FINAL` (`best`, `fast`) |
| `run.py` | runs configs × seeds × reps through `common.harness.run`. Resumable; rows are tagged `hw=<tag> rep=<k>` |
| `select.py` | ranks configs (averaging reps within each seed first), applies the team's best/fast rule, and does paired comparisons (`--pair A B`) |
| `curves.py` | diagnostic loss and validation-accuracy curves, **tuning seeds only** (it refuses report seeds) |
| `figures.py` | report figures → `report/figures/c1_cnn_*` |

## Setup (local, CPU is enough for tests and smoke runs)
```bash
py -m venv .venv && .venv\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e . -r requirements.txt pytest
python -m pytest tests/ -q
```

## Running
```bash
python -m c1.cnn.run --stage s0 --seeds tune --reps 2 --hw colab-t4     # one tuning stage
python -m c1.cnn.select --hw colab-t4 --stage s0                        # summary + best/fast rule
python -m c1.cnn.select --hw colab-t4 --pair s0_flipcrop s0_noaug       # paired comparison
python -m c1.cnn.curves --config s0_noaug s0_flipcrop --hw colab-t4     # training curves
python -m c1.cnn.figures --hw colab-t4                                  # figures
```
`--reps` defaults to the stage's entry in `configs.STAGE_REPS`, and `curves.py` defaults to
`configs.CURVE_CONFIGS`. Both are resumable.

**All reported numbers, figures and decisions come from a Colab T4**, through
[`notebooks/ryan/c1_cnn.ipynb`](../../notebooks/ryan/c1_cnn.ipynb). The notebook does the following:
1. Runs a GPU smoke test.
2. Runs every stage written in `configs.py`.
3. Records the training curves.
4. Prints each stage's summary.
5. Runs `best` and `fast` on the report seeds once they're chosen.
6. Draws the figures.

**Where rows go.** `run.py` and `curves.py` write `hw=colab-t4` rows to `results/ryan.csv` and
`results/ryan_curves.csv`, which are committed. Rows with any other hardware tag (laptop runs,
`hw=local-cpu`) go to `results/local/`, which is git-ignored and never committed. Laptop runs are only
a first look and a pipeline check. `select.py` and `figures.py` read both locations by default and
filter on `--hw`, so the T4-vs-laptop check runs locally:
`python -m c1.cnn.select --hw colab-t4 --stage s1 --compare-hw local-cpu`. If a laptop decision doesn't
hold on the T4, the T4 result wins.

## Reproduce the reported numbers
```bash
python -m c1.cnn.run --config best fast --seeds report --hw colab-t4 --results-dir <empty folder>
python -m c1.cnn.select --hw colab-t4 --seeds report --results <empty folder>/ryan.csv --pair best fast
```
Runs already in the results file are skipped, so point `--results-dir` at an empty folder to re-run
them (about 5 minutes on a T4). Accuracy should match to within run-to-run noise (~1 pp per pair):
GPU training is not bit-reproducible.

## Configs
Chosen by the team rule (`python -m c1.cnn.select --hw colab-t4`) over every tuning config, then
run once on report seeds 10–19 on a Colab T4. Accuracy is mean ± std over the 10 class pairs; times
are per problem instance.

| config_id | description | accuracy (report seeds) | train time | inference time |
|---|---|---|---|---|
| best (`s4_r_tta`) | ConvNet-3 w16 + BN (24k params); flip + crop + RandAugment (N=2, M=9) + label smoothing 0.1; SGD + Nesterov lr 0.05, wd 5e-4, cosine with 5 warmup epochs, batch 32, 300 epochs; flip TTA | 86.0 ± 10.5% | 26.0 s | 0.02 s |
| fast (`s4_f_e200`) | same network and optimizer; flip + crop + mixup (α=1) + label smoothing 0.1; 200 epochs | 84.4 ± 9.4% | 3.7 s | 0.01 s |

Paired on the report seeds: best − fast +1.6 pp (best wins 9/10); best − classical best +0.9 pp
(7/10); fast − classical fast +0.6 pp (5/10).
