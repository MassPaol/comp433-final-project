# C1: Hand-crafted features + classical ML

**Owner:** Massimo

## What this is
Fixed, untrained feature extractors (wavelet scattering, HOG, HSV color histograms, plus raw pixels
and LBP that were explored) followed by a classical classifier (logistic regression, SVMs, trees,
kNN), all fit on the 100 training images only. Nothing is learned except the final classifier, so it
needs no pretraining or external data (C1 rules) and trains in 0.1–5 s on a Colab T4. The full search
(243 configs, 4 stages) and the reasoning behind each step are in [NOTES.md](NOTES.md).

## Files
| file | what it does |
|---|---|
| `features.py` | feature extractors: uint8 `(N,32,32,3)` → float32 `(N,D)` |
| `method.py` | `ClassicalMethod`: [flip] → features → scaler → [PCA] → classifier |
| `configs.py` | every config of every sweep stage, plus `FINAL` (`best`, `fast`) |
| `run.py` | runs configs through `common.harness.run`; resumable |
| `select.py` | ranks configs, applies the best/fast selection rule |
| `figures.py` | report figures → `report/figures/c1_classical_*` |
| `NOTES.md` | experiment log + source log |

## Setup (local)
```bash
pip install -e . -r requirements.txt pytest       # plus torch/torchvision (see the top-level README)
export COMP433_DATA=/path/to/cifar                # folder holding cifar-10-python.tar.gz
python -m pytest tests/ -q                        # 78 tests, ~15 s
```
`kymatio` is imported through its 2D torch frontend (`kymatio.scattering2d.frontend.torch_frontend`),
because `kymatio.torch` also loads the 3D frontend, which breaks with SciPy ≥ 1.17.

## Reproduce the reported numbers
On Colab (T4): open [`notebooks/massimo/c1_classical.ipynb`](../../notebooks/massimo/c1_classical.ipynb)
and run all cells. It runs every stage, re-checks the selection with T4 times, runs `best`/`fast`
on the report seeds and redraws the figures, all with `hw=colab-t4`. Rows are logged to a copy of
`results/massimo.csv` on Google Drive (survives a runtime reset; re-run all cells to resume), merged
back into the repo and pushed to GitHub by the last two code cells. Pushing needs a GitHub token with
**Contents: Read and write** on this repo; cell 1 explains how to make one and checks it before the
long run starts. If the push still fails, the notebook's last cell shows how to merge the Drive copy
on your laptop instead.

Step by step (any machine; `--hw` tags the rows, default `local-3070`):
```bash
python -m c1.classical.run --stage s1 --seeds tune --hw colab-t4   # then s2, s3, s4
python -m c1.classical.select --hw colab-t4                        # prints best / fast
python -m c1.classical.run --config best fast --seeds report --hw colab-t4
python -m c1.classical.figures --hw colab-t4
```
Re-running a command skips (config, seed) pairs that already have a row for that `hw` tag.

## Configs
Report seeds 10–19, `hw=colab-t4` (Colab T4 GPU, 2 CPU cores).

| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| best | scattering (J=3, L=4, RGB) + HOG (8 px cells, 2×2 blocks, RGB) + HSV hist (8 bins) → standardize → LR-CV; training set doubled with horizontal flips | 85.09 ± 8.22 | 5.01 s | 0.69 s (0.98 ms/img) |
| fast | HOG (8 px cells, 2×2 blocks, RGB) + HSV hist (8 bins) → standardize → RBF-SVM (C=10) | 83.80 ± 7.99 | 0.095 s | 0.45 s (0.64 ms/img) |

Paired over the 10 report seeds: best − fast = +1.29 pp, best wins 8/10 (Wilcoxon p = 0.033).
The first run on an RTX 3070 Laptop (`hw=local-3070`, rows kept in the results file) picked the same
two configs; see [NOTES.md](NOTES.md) for how the two runs compare.
