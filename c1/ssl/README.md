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

## Configs
Fill in after the report-seed run (seeds 10–19).

| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| kmeans_best | | | | |
| kmeans_fast | | | | |
| simsiam_best | | | | |
| simsiam_fast | | | | |