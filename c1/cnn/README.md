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
`common/augment.py`. The experiment log (results and the reason for every decision) and the sources
are in [NOTES.md](NOTES.md).

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
| `NOTES.md` | experiment log + source log |

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
4. Prints each stage's summary, plus a T4-vs-laptop check
   (`python -m c1.cnn.select --stage s1 --hw colab-t4 --compare-hw local-cpu`).
5. Runs `best` and `fast` on the report seeds once they're chosen.
6. Draws the figures.

Rows tagged `hw=local-cpu` are laptop runs. They're used for a first look and for checking that the
pipeline works. If a laptop decision doesn't hold on the T4, the T4 result wins, and the flip is
recorded in NOTES.md.

## Reproduce the reported numbers
```bash
python -m c1.cnn.run --config best fast --seeds report --hw colab-t4
```

## Configs
| config_id | description | accuracy (report seeds) | train time | inference time |
|---|---|---|---|---|
| best | TBD after S4 | | | |
| fast | TBD after S4 | | | |
