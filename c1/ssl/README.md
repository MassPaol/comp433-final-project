# C1: Self-supervised methods (k-means and SimSiam)

**Owner:** <your name>

## What this is
Two methods that learn from the 100 training images only (no pretrained models, no external data):

1. **k-means** on fixed, untrained features (scattering / HOG / HSV / raw pixels, reused from
   `c1.classical.features`). Labels only name the clusters ("vote") or feed a logistic regression on
   the centroid distances ("dist").
2. **SimSiam** (Chen & He 2021): a small CNN pretrained without labels on the 100 training images
   (two augmented views, negative cosine similarity, stop-gradient), then frozen; a head
   (logistic regression, kNN, or k-means) is fit on the embeddings.

The search (stages k1, k2 for k-means; ss1-ss4 for SimSiam) and the reasoning behind each step go in
[NOTES.md](NOTES.md).

## Files
| file | what it does |
|---|---|
| `heads.py` | cluster / logistic-regression / kNN heads, shared by both methods |
| `method_kmeans.py` | `KMeansMethod`: [flip] → features → scaler → [PCA] → k-means head |
| `method_simsiam.py` | `SimSiamMethod`: SimSiam pretraining → frozen embeddings → head |
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
python -m c1.ssl.run --stage k1 --seeds tune --hw colab-t4     # then k2, ss1, ss2, ss3, ss4
python -m c1.ssl.select --hw colab-t4                          # prints best / fast per method
python -m c1.ssl.run --config kmeans_best kmeans_fast simsiam_best simsiam_fast --seeds report --hw colab-t4
python -m c1.ssl.figures --hw colab-t4
```
Re-running a command skips (config, seed) pairs that already have a row for that `hw` tag.

Workflow: stages k2 and ss2-ss4 use defaults (`K2_BASES`, `SS_BASE` in `configs.py`) that you update
from the previous stage's results before running them. After selection, update `FINAL_SOURCE`.

## Configs
Fill in after the report-seed run (seeds 10–19).

| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| kmeans_best | | | | |
| kmeans_fast | | | | |
| simsiam_best | | | | |
| simsiam_fast | | | | |
