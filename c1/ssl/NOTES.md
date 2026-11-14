# C1 self-supervised: experiment log

Hardware tag, seeds and metric definitions are the same as in `c1/classical/NOTES.md`: tuning seeds
0–4, report seeds 10–19, accuracy = mean ± std over seeds (std = spread across class pairs), time =
train + inference seconds per seed. Raw rows: `results/<person>.csv`.

Write one section per stage, only after its results are in.

## <date> · K1: dictionary size × patch size (15 configs)
**Why:** how does the dictionary size interact with the receptive field?

| patch | K=50 | K=100 | K=200 | K=400 | K=800 |
|---|---|---|---|---|---|
| 4 | | | | | |
| 6 | | | | | |
| 8 | | | | | |

**Observations:**

**Decision → K2:** (chosen K and patch; copy into `KMEANS_BASE`)

## <date> · K2: contrast and whitening constants
**Why / results / decision:**

## <date> · SS1: pretraining length
**Why:** does SimSiam pretraining on 100 images beat a random-init backbone (epochs=0)?
Check `collapse_std_` (≈ 1/sqrt(proj_dim) healthy, ≈ 0 collapsed) for the longest runs.

**Decision → SS2:** (epochs chosen; copy into `SS_BASE`)

## <date> · SS2 / SS3: architecture, optimization and augmentation
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

# Source log

| source | used for |
|---|---|
| Coates, Ng & Lee 2011, *An analysis of single-layer networks in unsupervised feature learning*, AISTATS | the whole k-means pipeline, quadrant sum pooling, linear SVM |
| Chen & He 2021, *Exploring Simple Siamese Representation Learning*, CVPR | SimSiam: stop-gradient, predictor, negative cosine loss |
| Chen, Kornblith, Norouzi & Hinton 2020, *A simple framework for contrastive learning*, ICML | augmentation recipe (random resized crop, color jitter, grayscale) |
| Ioffe & Szegedy 2015, *Batch normalization*, ICML | BatchNorm in backbone, projector and predictor |
| Pedregosa et al. 2011, *Scikit-learn*, JMLR | KMeans, LinearSVC, GridSearchCV, StandardScaler |
| Lightly (version pinned in `requirements.txt`) | SimSiamTransform augmentations and projection/prediction heads |