# C1: Self-supervised methods (Coates k-means and SimSiam)

**Owner:** Alvin

## What this is
Two methods that learn from the 100 training images only (no pretrained models, no external data):

1. **Coates et al. (2011)** single-layer features: random patches, contrast normalization, ZCA
   whitening, k-means dictionary, triangle encoding, quadrant sum pooling, linear SVM. This is the
   fast variant.
2. **SimSiam** (Chen & He 2021): a small CNN pretrained without labels on the 100 training images
   (two augmented views, negative cosine similarity, stop-gradient), then frozen, with a linear SVM
   on the embeddings. This is the deep variant.

In both, the classifier's C is chosen by 5-fold cross-validation on the training features inside
`fit`. The search (stages k1, k2 for Coates; ss1–ss3 for SimSiam) and the reasoning behind each step
go in [NOTES.md](NOTES.md).

## Files
| file | what it does |
|---|---|
| `method_kmeans.py` | `KMeansMethod`: Coates pipeline, fit on training images |
| `method_simsiam.py` | `SimSiamMethod`, `SimSiamNetwork`, small CNN placeholder: SimSiam pretraining, then linear SVM |
| `configs.py` | every config of every stage, plus `FINAL_SOURCE` (best/fast per method) |
| `run.py` | runs configs through `common.harness.run`; resumable |
| `select.py` | ranks configs, applies the best/fast selection rule per method |
| `figures.py` | report figures → `report/figures/c1_ssl_*` |
| `NOTES.md` | experiment log + source log |

Also: `tests/test_ssl.py` and `notebooks/<you>/c1_ssl.ipynb`.

## Setup (local)
```bash
pip install -e . -r requirements.txt pytest
export COMP433_DATA=/path/to/cifar
python -m pytest tests/test_ssl.py -q
```

## Reproduce
```bash
python -m c1.ssl.run --stage k1 --seeds tune --hw colab-t4     # then k2, ss1, ss2, ss3
python -m c1.ssl.select --hw colab-t4                          # prints best / fast per method
python -m c1.ssl.run --config kmeans_best kmeans_fast simsiam_best simsiam_fast --seeds report --hw colab-t4
python -m c1.ssl.figures --hw colab-t4
```
Re-running a command skips (config, seed) pairs that already have a row for that `hw` tag.

Workflow: k2 and ss2–ss3 use defaults (`KMEANS_BASE`, `SS_BASE` in `configs.py`) that you update from
the previous stage's results before running them. After selection, update `FINAL_SOURCE`.

## Configs
Fill in after the report-seed run (seeds 10–19).

| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| kmeans_best | | | | |
| kmeans_fast | | | | |
| simsiam_best | | | | |
| simsiam_fast | | | | |