# C2: Frozen pretrained features + lightweight heads

**Owner:** Massimo

## What this is
An ImageNet-pretrained backbone from timm is kept frozen and turns each image (upsampled from 32 to
224 px with the checkpoint's own preprocessing) into a feature vector; only a small head (logistic
regression, linear/RBF SVM, nearest class mean, kNN, or the testbed's AdamW-trained linear layer) is fit
on the 100 training images' features. With 100 images there is far too little data to train a network,
but enough to fit 200–800 head parameters on top of a representation learned from millions of images.
It is also cheap: one forward pass per image, and the whole `fit` takes under half a second. None of the
checkpoints saw CIFAR-10 (C2 rule; pretraining data per checkpoint in [NOTES.md](NOTES.md)). The search
covers 7 backbones, 7 heads, feature normalization, feature source (CLS / patch tokens / last-k blocks),
crop, flip augmentation, fp16 and backbone concatenation: 191 configs in 4 stages, with the reasoning for
each step in [NOTES.md](NOTES.md).

## Files
| file | what it does |
|---|---|
| `backbones.py` | the 7 checkpoints, loading from a local weights file, GPU preprocessing, feature extraction (pooling, last-k blocks, fp16) |
| `heads.py` | heads (`adamw`, `lr`, `lr_cv`, `linsvm`, `rbfsvm`, `ncm`, `knn`) and feature normalizations (`none`, `l2`, `standard`, `cl2n`) |
| `method.py` | `FrozenFeatureMethod`: [flip] → frozen backbone(s) → features → normalization → head |
| `configs.py` | every config of every sweep stage, plus `FINAL` (`best`, `fast`, `baseline`) |
| `run.py` | runs configs through `common.harness.run`; resumable; untimed warm-up per config |
| `select.py` | ranks configs, applies the best/fast selection rule (same rule as C1, in `common/sweep.py`) |
| `figures.py` | report figures → `report/figures/c2_frozen_*` |
| `NOTES.md` | experiment log, backbone/pretraining table, source log |

## Setup (local)
```bash
pip install -e . -r requirements.txt pytest       # plus torch/torchvision with CUDA (see the top-level README)
export COMP433_DATA=/path/to/cifar                # folder holding cifar-10-python.tar.gz
python -m pytest tests/ -q                        # no downloads: tests use timm's tiny test models
```
The first run of a config downloads its checkpoint from the Hugging Face hub (all 7: about 1 GB) into the
local HF cache. `run.py` does this in an untimed warm-up; inside the timed `fit`, weights are always read
from the local file, never from the network.

**Timing.** Train time = the whole `fit`: loading the frozen backbone from local disk + extracting the
100 training images' features + fitting the head. Each row's notes also log `load_s` and `extract_s`, so
train time can be split. Inference time = `predict` on the 700 validation images (feature extraction +
head).

## Reproduce the reported numbers
On Colab (T4): open [`notebooks/massimo/c2_frozen_features.ipynb`](../../notebooks/massimo/c2_frozen_features.ipynb)
and run all cells (~2 h). It runs every stage, re-checks the selection with T4 times, runs `best`/`fast`/
`baseline` on the report seeds and redraws the figures, all with `hw=colab-t4`. Rows are logged to a copy
of `results/massimo.csv` on Google Drive (survives a runtime reset; re-run all cells to resume), merged
back into the repo and pushed by the last two code cells (needs a GitHub token with **Contents: Read and
write**; cell 1 checks it before the long run).

Step by step (any machine; `--hw` tags the rows, default `local-3070`):
```bash
python -m c2.frozen_features.run --stage s1 --seeds tune --hw colab-t4   # then s2, s3, s4
python -m c2.frozen_features.select --hw colab-t4                        # prints best / fast
python -m c2.frozen_features.run --config best fast baseline --seeds report --hw colab-t4
python -m c2.frozen_features.figures --hw colab-t4
```
Re-running a command skips (config, seed) pairs that already have a row for that `hw` tag.

## Configs
Report seeds 10–19, `hw=local-3070` (RTX 3070 Laptop). The Colab T4 numbers replace these once the
notebook has run.

| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| best | ViT-S/16 (AugReg, IN-21k → IN-1k), CLS token of the whole image (no center crop), fp16 extraction → logistic regression (C = 1, raw features) | 98.69 ± 2.65 | 0.40 s (load 0.32 + extract 0.06) | 0.42 s (0.60 ms/img) |
| fast | ViT-Ti/16 (the testbed checkpoint), CLS token, fp16 extraction → logistic regression (C = 0.01, z-scored features) | 97.33 ± 3.96 | 0.16 s (load 0.11 + extract 0.03) | 0.21 s (0.30 ms/img) |
| baseline | the testbed's C2 example: ViT-Ti/16, CLS token → new linear layer trained with AdamW (lr 1e-3, wd 0.01, 10 epochs, batch 32) | 96.04 ± 5.02 | 0.24 s (load 0.11 + extract 0.07) | 0.45 s (0.65 ms/img) |

Paired over the 10 report seeds: best − baseline = +2.64 pp (best wins 10/10, Wilcoxon p = 0.002);
best − fast = +1.36 pp (10/10, p = 0.002). The hardest pair, cat-dog, gives 90.9 / 86.0 / 81.9 %.
`best` was chosen on the tuning seeds by the fixed selection rule: it is 0.06 pp below the most accurate
config (ViT-B/16, 99.83 %) and 2.9× faster.
