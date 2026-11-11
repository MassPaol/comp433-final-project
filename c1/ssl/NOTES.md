# C1 self-supervised: experiment log

Hardware tag, seeds and metric definitions are the same as in `c1/classical/NOTES.md`: tuning seeds
0–4, report seeds 10–19, accuracy = mean ± std over seeds (std = spread across class pairs), time =
train + inference seconds per seed. Raw rows: `results/<person>.csv`.

(Write one section per stage, only after its results are in. Template below.)

## <date> · K1: features × k × cluster head (60 configs)
**Why:** which fixed features cluster well, and does naming clusters by vote beat a logistic regression
on centroid distances?

| feature | k=2 vote | k=8 vote | k=32 vote | k=2 dist | k=8 dist | k=32 dist |
|---|---|---|---|---|---|---|
| ... | | | | | | |

**Observations:** (what changed with k, vote vs dist, which feature won)

**Decision → K2:** (top 3 feature sets and their best k / mode; copy them into `K2_BASES`)

## <date> · K2: flip augmentation, standardization, PCA
**Why / results / decision:**

## <date> · SS1: pretraining length × head
**Why:** does SimSiam pretraining on 100 images beat a random-init backbone (epochs=0)?
Check `collapse_std_` (≈ 1/sqrt(proj_dim) healthy, ≈ 0 collapsed) for the longest runs.

**Decision → SS2:** (write the epochs/head chosen into `SS_BASE`)

## <date> · SS2 / SS3 / SS4: architecture, optimization + augmentation, heads
**Results and decisions:**

## <date> · Selection (rule in `select.py`, fixed before any results)
- kmeans_best / kmeans_fast:
- simsiam_best / simsiam_fast:
- close calls:

## <date> · Report seeds 10–19
| config | acc (%) | train (s) | inference (s / 700 img) |
|---|---|---|---|
| kmeans_best | | | |
| kmeans_fast | | | |
| simsiam_best | | | |
| simsiam_fast | | | |

# Source log (paper → what was taken from it)

| source | used for |
|---|---|
| Chen & He 2021, *Exploring Simple Siamese Representation Learning*, CVPR | SimSiam: stop-gradient, predictor, negative cosine loss |
| Chen, Kornblith, Norouzi & Hinton 2020, *A simple framework for contrastive learning of visual representations*, ICML | augmentation recipe (random resized crop, color jitter, grayscale) |
| Ioffe & Szegedy 2015, *Batch normalization*, ICML | BatchNorm in backbone / projector / predictor |
| Arthur & Vassilvitskii 2007, *k-means++: the advantages of careful seeding*, SODA | k-means initialization (scikit-learn default) |
| Coates, Lee & Ng 2011, *An analysis of single-layer networks in unsupervised feature learning*, AISTATS | k-means centroid distances as features for a linear classifier ("dist" mode) |
| Pedregosa et al. 2011, *Scikit-learn: Machine learning in Python*, JMLR 12 | k-means, logistic regression, kNN, StandardScaler, PCA |
| (reused from `c1/classical/NOTES.md`) | HOG, HSV histograms, scattering features |
