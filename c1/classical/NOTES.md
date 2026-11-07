# C1 classical: experiment log

All runs: tuning seeds 0–4 unless stated, `hw=local-3070` (RTX 3070 Laptop GPU for scattering, CPU
for everything else; the machine was also running other software, so local times are indicative only).
Accuracy = mean ± std of validation accuracy over the seeds (std = spread across class pairs).
Time = train + inference seconds per seed. Raw rows: `results/massimo.csv`.

## 2026-11-06 · S1: every feature family × every classifier (42 configs, defaults)

**Why:** find which feature families are worth tuning before spending compute on hyperparameters.

Mean tuning accuracy (%):

| feature | lr | linsvm | rbfsvm | rf | hgb | xgb | knn |
|---|---|---|---|---|---|---|---|
| scattering (J=2, L=8, RGB) | **85.7** | 85.5 | 84.1 | 83.7 | 81.9 | 81.3 | 79.6 |
| HOG (9 orient., 4 px cells, 2×2 blocks, gray) | 82.7 | 81.8 | **84.5** | 83.3 | 76.9 | 75.0 | 78.4 |
| raw pixels | 74.2 | 73.4 | **80.0** | 77.9 | 74.9 | 74.4 | 64.4 |
| raw + PCA(50) | 72.4 | 71.8 | 79.1 | 77.0 | 73.0 | 74.1 | 64.3 |
| HSV histograms (16 bins) | 70.5 | 67.0 | 77.5 | **78.3** | 76.4 | 77.3 | 70.4 |
| LBP (P=8, R=1, 2×2 grid) | 66.9 | 64.8 | 71.3 | 71.3 | 69.0 | 69.2 | 68.3 |

**Observations:**
- Scattering wins with plain linear models: the fixed wavelet filters already give a
  near-linearly-separable representation, and it is the only family where linear beats RBF.
- HOG is close behind, best with an RBF-SVM.
- Color (HSV) and texture (LBP) histograms alone are weak, but HSV favors trees/RBF. It may still help
  as a complement to shape features (tested in S4).
- PCA(50) on raw pixels slightly hurts every classifier.
- Boosted trees are slow on 15k-dim scattering features (HGB ≈ 10.6 s/seed, XGB ≈ 6.5 s) without
  being better.

**Decision → S2:** tune classifiers on the top 3 feature sets (scattering, HOG, raw). Tree models only
where their S1 mean was within 5 pp of the S1 best (≥ 80.7): scattering rf/hgb/xgb, HOG rf.

## 2026-11-06 · S2: classifier hyperparameters on scattering, HOG, raw (117 configs)

**Why:** S1 used default settings; before comparing features further, give each classifier its fair
shot. Grids: LR and linear-SVM C ∈ {0.001 … 100}; LR-CV (C picked by 5-fold CV on the 100 training
images only); RBF-SVM C ∈ {0.1, 1, 10, 100} × gamma ∈ {0.25, 1, 4} × sklearn's "scale"; kNN k ∈ {1, 3,
5, 9, 15} × {euclidean, cosine}; tree models only where competitive in S1.

| feature | best config | acc (%) | time (s) | runner-up / note |
|---|---|---|---|---|
| scattering | LR-CV | **86.09** ± 4.4 | 1.98 | RBF-SVM C=10 γ×0.25: 85.91 (1.24 s); LR C=0.01: 85.80 (0.54 s) |
| HOG | RBF-SVM C=10, γ×1 | 84.66 ± 4.2 | 0.63 | LR C=0.001: 84.03; kNN k=9 cosine: 83.54 |
| raw | RBF-SVM C=1, γ×1 | 80.00 ± 3.9 | 0.05 | all linear models ≤ 76.8 |

**Observations:**
- On scattering, every linear/RBF classifier lands within 0.5 pp (85.6–86.1). The representation
  matters, the classifier barely does. Strong regularization (C = 0.001–0.01) is as good as C = 1,
  as expected with 100 samples in 15k dims.
- Cosine kNN beats euclidean kNN for every feature (the scale of the vector carries little class
  information).
- Tree models never beat the linear/RBF models: scattering RF 83.7, HGB 81.9, XGB 81.5, at 2–20×
  the cost.
- Differences of < 0.5 pp are far below the across-class-pair std (≈ 4–5 pp). They are not
  meaningful on 5 seeds.

**Decision → S3:** tune each family's feature parameters with its best classifier so far:
scattering → LR-CV (adapts C to each variant's dimensionality), HOG → RBF-SVM C=10, raw → RBF-SVM C=1,
HSV → RF, LBP → RBF-SVM.
