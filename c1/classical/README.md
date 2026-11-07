# C1: Hand-crafted features + classical ML

**Owner:** Massimo

## What this is
Fixed, untrained feature extractors (wavelet scattering, HOG, HSV color histograms, plus raw pixels
and LBP that were explored) followed by a classical classifier (logistic regression, SVMs, trees,
kNN), all fit on the 100 training images only. Nothing is learned except the final classifier, so it
needs no pretraining or external data (C1 rules) and trains in about a second. The full search
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
python -m pytest tests/ -q                        # 69 tests, ~10 s
```
`kymatio` is imported through its 2D torch frontend (`kymatio.scattering2d.frontend.torch_frontend`),
because `kymatio.torch` also loads the 3D frontend, which breaks with SciPy ≥ 1.17.

## Reproduce the reported numbers
On Colab (T4): open [`notebooks/massimo/c1_classical.ipynb`](../../notebooks/massimo/c1_classical.ipynb)
and run all cells. It runs every stage, re-checks the selection with T4 times, runs `best`/`fast`
on the report seeds and redraws the figures, all with `hw=colab-t4`. Rows are logged to a copy of
`results/massimo.csv` on Google Drive (survives a runtime reset; re-run all cells to resume) and
merged back into the repo by the last code cell.

Step by step (any machine; `--hw` tags the rows, default `local-3070`):
```bash
python -m c1.classical.run --stage s1 --seeds tune --hw colab-t4   # then s2, s3, s4
python -m c1.classical.select --hw colab-t4                        # prints best / fast
python -m c1.classical.run --config best fast --seeds report --hw colab-t4
python -m c1.classical.figures --hw colab-t4
```
Re-running a command skips (config, seed) pairs that already have a row for that `hw` tag.

## Configs
Report seeds 10–19, `hw=local-3070` (RTX 3070 Laptop). **T4 times: pending the Colab run.**

| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| best | scattering (J=3, L=4, RGB) + HOG (8 px cells, 2×2 blocks, RGB) + HSV hist (8 bins) → standardize → LR-CV; training set doubled with horizontal flips | 85.13 ± 8.10 | 0.92 s | 0.53 s (0.75 ms/img) |
| fast | HOG (8 px cells, 2×2 blocks, RGB) + HSV hist (8 bins) → standardize → RBF-SVM (C=10) | 83.80 ± 7.99 | 0.05 s | 0.35 s (0.50 ms/img) |

Paired over the 10 report seeds: best − fast = +1.33 pp, best wins 9/10 (Wilcoxon p = 0.020).
