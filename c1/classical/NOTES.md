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
- Cosine kNN beats euclidean kNN in all 15 (feature, k) pairs, by up to +11 pp on raw pixels (k=9:
  74.2 vs 63.4): the direction of the feature vector carries the class information, its overall
  magnitude (brightness, contrast) mostly does not.
- Tree models never beat the linear/RBF models: scattering RF 83.7, HGB 81.9, XGB 81.5, at 2–20×
  the cost.
- Differences of < 0.5 pp are far below the across-class-pair std (≈ 4–5 pp). They are not
  meaningful on 5 seeds.

**Decision → S3:** tune each family's feature parameters with its best classifier so far:
scattering → LR-CV (adapts C to each variant's dimensionality), HOG → RBF-SVM C=10, raw → RBF-SVM C=1,
HSV → RF, LBP → RBF-SVM.

## 2026-11-06 · S3: feature hyperparameters (65 configs)

**Why:** S1/S2 used each feature's default parameters; the representation turned out to matter more
than the classifier, so tune it. Each family uses its best classifier so far (scattering → LR-CV, HOG →
RBF-SVM C=10, raw → RBF-SVM C=1, HSV → RF, LBP → RBF-SVM).

| family | searched | best | acc (%) | time (s) |
|---|---|---|---|---|
| scattering | J ∈ {1,2,3} × L ∈ {4,6,8} × {gray, RGB}, + 2×2 avg-pool at L=8 | J=3, L=4, RGB (2928 dims) | **86.51** ± 3.9 | 0.77 |
| HOG | cell {4,8} px × block {1,2,3} × orient. {9,12} × {gray, RGB} | 8 px cells, 2×2 blocks, 9 orient., RGB (324 dims) | 85.94 ± 4.7 | 0.25 |
| raw | PCA {10,20,50,none} × {gray, RGB} | no PCA, RGB | 80.00 ± 3.9 | 0.04 |
| HSV | bins {8,16,32} | 8 bins | 78.66 ± 4.1 | 0.77 (RF) |
| LBP | (P,R) {(8,1),(16,2),(24,3)} × grid {1,2,4} | P=8, R=1, 4×4 grid | 75.17 ± 5.9 | 0.17 |

**Observations:**
- Scattering is remarkably flat: all 21 variants within 84.1–86.5. Coarser scales (J=3) with fewer
  orientations (L=4) are as good as the default J=2, L=8 at 1/5 of the dimensions (2928 vs 15552) and
  a third of the time. RGB beats gray at every (J, L) setting (+0.1 to +1.1 pp).
- HOG gains +1.3 pp from coarser cells: with block normalization over 2×2 or 3×3 cells, 8×8 px cells
  (a 4×4 cell grid on a 32×32 image) beat 4×4 px cells in all 8 (block, orientation, color) settings.
  Without block normalization (1×1 blocks) it's a wash. Color HOG beats gray in all 12 settings
  (+0.1 to +1.9 pp).
- LBP needs spatial layout: a 4×4 grid of histograms beats one global histogram for every (P, R);
  best 75.2 vs 68–70 for global histograms.
- Raw pixels: PCA never helps (fewer dims = worse), gray is worse than RGB.

**Decision → S4:** concatenate scattering + HOG + HSV (raw dropped as redundant with scattering's
low-pass coefficients; HSV kept as the color complement promised in S1), with both LR-CV and the HOG
RBF-SVM; then flip augmentation and scaling on the winners.

## 2026-11-06 · S4: combinations, flip augmentation, scaling (19 configs)

**Why:** the best single features describe different things (scattering: multi-scale texture/shape;
HOG: edge orientations; HSV: color), so combining them may help. Mirrored images are a free, label-
preserving way to double the 100 training images (CIFAR objects are left-right symmetric).

Part 1, concatenations (scattering J=3 L=4 RGB, HOG 8px/2×2/RGB, HSV 8 bins):

| combination | LR-CV | RBF-SVM C=10 |
|---|---|---|
| scat + hog | 87.91 | 87.29 |
| scat + hsv | 86.89 | 86.37 |
| hog + hsv | 85.77 | **87.83** (0.04 s fit) |
| scat + hog + hsv | **88.00** | 87.46 |

Part 2, flip augmentation (train = append mirrored copies; TTA = average scores of image and mirror):

| base | none | flip-train | flip-TTA | both |
|---|---|---|---|---|
| scat (LR-CV) | 86.51 | 88.49 | 87.91 | 88.89 |
| scat+hog+hsv (LR-CV) | 88.00 | 89.80 | 89.26 | **90.14** |
| hog+hsv (RBF-SVM) | 87.83 | 88.54 | 88.34 | 88.57 |

No standardization (scale=False): scat+hog+hsv 87.17 (−0.8 vs scaled), scat+hog 86.91 (−1.0).

**Observations:**
- Every combination containing both scattering and HOG beats every single feature (+1.4 pp over the
  best single). HSV adds a little on top (+0.1 with LR-CV).
- Flip augmentation is the biggest single gain of the whole sweep for the LR-CV models: +1.8 to
  +2.0 pp from flip-train alone (only +0.7 for the hog+hsv RBF-SVM), and train + TTA together are
  best for all three bases.
- Standardization matters (−0.8 to −1.0 pp without it): the features live on very different scales
  (HSV fractions vs HOG block norms vs scattering energies).

## 2026-11-06 · Selection (rule in `select.py`, fixed before any results)

Over all 243 tuning configs (s1–s4):
- top: `s4_scat+hog+hsv_lrcv_fliptrain_fliptta` 90.14 ± 2.64, 1.95 s
- **best** = `s4_scat+hog+hsv_lrcv_fliptrain` 89.80 ± 3.15, 1.43 s: within 0.5 pp of top and
  faster (no TTA → half the inference feature extraction)
- **fast** = `s4_hog+hsv_rbfsvm` 87.83 ± 4.78, 0.37 s: fastest config with acc ≥ best − 2 pp
  (= 87.80; it clears that by 0.03 pp)

Close calls, to re-check with T4 times: best vs top is a 1.36× time margin, and fast vs its
flip-train variant (88.54, 0.43 s) is 1.16×. The Colab notebook re-applies the rule with
`hw=colab-t4`.

## 2026-11-06 · Report seeds 10–19 (`hw=local-3070`)

| config | acc (%) | train (s) | inference (s / 700 img) | per image |
|---|---|---|---|---|
| best | **85.13 ± 8.10** | 0.92 | 0.53 | 0.75 ms |
| fast | 83.80 ± 7.99 | 0.05 | 0.35 | 0.50 ms |

Paired (same seeds = same class pairs): best − fast = +1.33 ± 1.60 pp, best wins 9/10, Wilcoxon
signed-rank p = 0.020.

**Observations:**
- Both are 4.0–4.7 pp lower than on the tuning seeds. Two reasons: (1) the report seeds include
  cat-dog (seed 13), the hardest CIFAR pair, at 64.4% / 63.4%; without it best = 87.4, fast = 86.1;
  (2) picking the max of 236 configs on 5 seeds is optimistically biased (winner's curse). This is
  exactly why report numbers come from separate seeds.
- The std (≈ 8 pp) is variation across class pairs, not uncertainty of the mean.
- fast trains 20× faster (0.05 vs 0.92 s) and infers 1.5× faster (0.35 vs 0.53 s).

# Source log (paper → what was taken from it)

| source | used for |
|---|---|
| Dalal & Triggs 2005, *Histograms of Oriented Gradients for Human Detection*, CVPR | HOG features: cells, overlapping blocks, L2-Hys block normalization |
| Ojala, Pietikäinen & Mäenpää 2002, *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*, IEEE TPAMI 24(7) | uniform LBP codes with (P, R) neighborhoods |
| Swain & Ballard 1991, *Color indexing*, IJCV 7(1) | color histograms as an image descriptor |
| Mallat 2012, *Group invariant scattering*, Comm. Pure Appl. Math. 65(10) | scattering transform: fixed wavelet cascade, stable to small deformations |
| Bruna & Mallat 2013, *Invariant scattering convolution networks*, IEEE TPAMI 35(8) | scattering features + linear/SVM classifiers for images; the J (scale) and L (orientations) parameters |
| Oyallon, Belilovsky & Zagoruyko 2017, *Scaling the scattering transform: Deep hybrid networks*, ICCV | motivation: scattering helps most when training data is scarce (small CIFAR-10 subsets) |
| Andreux et al. 2020, *Kymatio: Scattering transforms in Python*, JMLR 21(60) | scattering implementation (`Scattering2D`, torch frontend, GPU) |
| Cortes & Vapnik 1995, *Support-vector networks*, Machine Learning 20(3) | linear and RBF SVMs |
| Breiman 2001, *Random forests*, Machine Learning 45(1) | random forest baseline |
| Chen & Guestrin 2016, *XGBoost: A scalable tree boosting system*, KDD | gradient-boosted trees (XGBoost) |
| Ke et al. 2017, *LightGBM*, NeurIPS | histogram-based gradient boosting (sklearn's HistGradientBoosting follows it) |
| Krizhevsky, Sutskever & Hinton 2012, *ImageNet classification with deep convolutional neural networks*, NeurIPS | horizontal-flip augmentation; averaging predictions over flipped copies at test time |
| Pedregosa et al. 2011, *Scikit-learn: Machine learning in Python*, JMLR 12 | all classifiers, StandardScaler, PCA, cross-validation |
