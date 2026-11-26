# C1: Self-supervised methods (Coates k-means and SimSiam)

**Owner:** Alvin

## What this is
Two methods that learn from the 100 training images only (no pretrained models, no external data,
no other CIFAR images):
- **SimSiam (deep):** Ryan's selected CNN backbone (built inside the model) is pretrained with a
  SimSiam objective on two augmented views per image (views from `common/augment.py` plus a random
  resized crop). Then a linear SVM on the frozen features, or (options) a joint SimSiam + cross-entropy
  loss, or supervised fine-tuning.
- **Coates k-means (fast):** random patches → contrast normalization → ZCA → k-means dictionary →
  triangle encoding → quadrant sum pooling → linear SVM.

The classifier's C is chosen by 5-fold cross-validation on the training features inside `fit`.
Selection of `best` / `fast` uses the shared rule in `common/sweep.py`. The search stages and the reasoning
behind each step are in [NOTES.md](NOTES.md); the stage definitions are in `configs.py`.

## Stages
| stage | what it varies |
|---|---|
| k1 / k2 | k-means: dictionary size × patch size / contrast and whitening constants |
| ss1 | SimSiam pretraining epochs (0 = random-init baseline) |
| ss2 | augmentation: resized-crop scale, RandAugment, Cutout, pad-crop |
| ss3 | backbone width / depth (Ryan's ConvNet family), projector size |
| ss4 | learning rate, weight decay, batch size |
| ss5 | using labels: joint SimSiam + CE loss, or fine-tuning |
| ss6 | four hand-picked recipes (two for best accuracy, two for speed), independent of `SS_BASE` |

## Files
| file | what it does |
|---|---|
| `method_kmeans.py` | `KMeansMethod`: Coates pipeline, fit on training images; logs `dict_s`, `extract_s` |
| `method_simsiam.py` | `SimSiamMethod` (Ryan's backbone inside, shared `Augmenter`), `random_resized_crop` |
| `configs.py` | every config of every stage, plus `FINAL_SOURCE` (best/fast per method) |
| `run.py` | runs configs through `common.harness.run`; resumable via `common/sweep.py` |
| `select.py` | ranks configs, applies the shared best/fast rule per method, `--pair A B` |
| `figures.py` | report figures → `report/figures/c1_ssl_*` |
| `NOTES.md` | experiment log + source log |

Also: `tests/test_ssl.py` and `notebooks/alvin/c1_ssl.ipynb`. Depends on `c1/cnn/models.py`
(`build_backbone`, `FINAL_ARCH`), `c1/cnn/method.py` (`balanced_batches`, `lr_factor`) and `common/`.
`lightly` is no longer needed in `requirements.txt` for this folder.

## Setup (local)
```bash
pip install -e . -r requirements.txt pytest
export COMP433_DATA=/path/to/cifar
python -m pytest tests/test_ssl.py -q
```

## Reproduce
```bash
# pass 1
python -m c1.ssl.run --stage k1  --seeds tune --hw colab-t4
python -m c1.ssl.run --stage ss1 --seeds tune --hw colab-t4
python -m c1.ssl.run --stage ss6 --seeds tune --hw colab-t4
# set KMEANS_BASE and SS_BASE in configs.py from k1 / ss1, then pass 2
for s in k2 ss2 ss3 ss4 ss5; do python -m c1.ssl.run --stage $s --seeds tune --hw colab-t4; done
python -m c1.ssl.select --hw colab-t4        # prints best / fast per method -> FINAL_SOURCE
python -m c1.ssl.run --config kmeans_best kmeans_fast simsiam_best simsiam_fast --seeds report --hw colab-t4
python -m c1.ssl.figures --hw colab-t4
```
Re-running a command skips (config, seed) pairs that already have a row for that `hw` tag. All reported
numbers come from a Colab T4 (`notebooks/alvin/c1_ssl.ipynb` runs everything above).

## Results
Chosen by the shared rule over the tuning seeds (`python -m c1.ssl.select --hw colab-t4`), then run once on
report seeds 10–19 on a Colab T4. Accuracy is mean ± std over the 10 class pairs; times are per problem instance.

| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| kmeans_best (`k2_wzca1`) | Coates pipeline: K = 10, patch 4, stride 1, 20,000 random patches, contrast ε 0.1, ZCA ε 1.0, triangle encoding, quadrant sum pooling, linear SVM (C by 5-fold CV from 0.01–10) | 75.97 ± 10.32 | 1.9 s | 0.63 s |
| kmeans_fast (`k2_wzca1`) | same config: no faster k-means config is within 2 pp of best | 75.97 ± 10.32 | 1.9 s | 0.64 s |
| simsiam_best (`ss6_ft_ra`) | ConvNet-3 w16 + BN (Ryan's backbone). SimSiam pretraining 100 epochs (projector 64 / hidden 256, predictor hidden 32, batch 32, SGD lr 0.05, wd 5e-4, momentum 0.9, 5 warmup epochs + cosine; views: RandAugment N=2 M=9, pad-crop 4, flip, random resized crop min scale 0.3), then supervised fine-tuning of backbone + linear head for 150 epochs (SGD Nesterov, lr 0.01) | 83.56 ± 9.99 | 33.7 s | 0.016 s |
| simsiam_fast (`ss5_ft150`) | same, without RandAugment | 81.40 ± 9.43 | 7.0 s | 0.014 s |

Paired on the report seeds: simsiam_best − simsiam_fast +2.16 pp (best wins 8/10, p = 0.014);
simsiam_best − kmeans_best +7.59 pp (10/10, p = 0.002).

## What the sweep showed
- **k-means:** small dictionaries and small patches win (K = 10–25, patch 4). Stronger ZCA regularization
  (ε 0.1–1.0 instead of 1e-3) gives about +1 pp; contrast ε does not matter.
- **SimSiam with a frozen backbone does not beat random init** (81.9 % for a random backbone + SVM vs
  at most 80.7 % for any pretrained setting). Pretraining length, augmentation strength, projector size
  and optimizer settings moved the result by about ±1 pp.
- **Using the 100 labels is what helps:** joint SimSiam + cross-entropy gives about +5.6 pp, fine-tuning
  gives +7.8 pp, and RandAugment views add a further +0.8 pp (at 5× the time). The control with a
  fine-tune from random init was not run, so the share due to SimSiam pretraining is not isolated.
- Report-seed accuracies are 3.7–5.1 pp below the tuning-seed ones: configs were selected on the tuning seeds.
  Details, the per-stage tables and the caveats are in [NOTES.md](NOTES.md).