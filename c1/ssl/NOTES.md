# C1 self-supervised: experiment log

Hardware tag, seeds and metric definitions follow the team rules: tuning seeds 0–4, report seeds
10–19, accuracy = mean ± std over seeds (std = spread across class pairs), time = train + inference
seconds per seed. Raw rows: `results/alvin.csv` (`hw=colab-t4`). Feature-extraction / dictionary /
pretraining times are in each row's `notes` (`extract_s`, `dict_s`, `pretrain_s`, `finetune_s`).

Write one section per stage, only after its results are in.

## Rules this code follows (team plan, "Shared rules")
- Only the 100 training images are used: not the validation images, not other CIFAR-10 images.
  The input normalization statistics are computed from the 100 training images (the first version
  used CIFAR-wide constants, which come from 50 000 other images; removed).
- No pretrained weights, no external data; no early stopping: every epoch count is fixed by the config.
- Labels are not used during SimSiam pretraining unless `sup_weight` > 0 (joint SSL + supervised loss).
- **Selection rule (shared, `select` in `common/sweep.py`, fixed before any results):**
  best = highest mean tuning accuracy, but any config within 0.5 pp of it that is faster wins;
  fast = the fastest config within 2 pp of best. Applied separately to the k-means and SimSiam configs.
  (`c1/ssl/select.py` calls it directly; it has no rule of its own.)

## Design decisions (before any results)
- SimSiam backbone = Ryan's selected CNN (ConvNet-3 w16 + BN, 64-d features), built inside
  `SimSiamMethod` through `c1.cnn.models.build_backbone`; `width` / `convs` are the only knobs.
- Views come from `common.augment.Augmenter` (RandAugment, pad+crop, flip, Cutout). The shared module has
  no random resized crop, which SimSiam relies on, so `random_resized_crop` lives in `method_simsiam.py`
  (candidate to move into `common/augment.py`). No colour jitter / grayscale: RandAugment's colour ops stand in.
- Heads (projector, predictor) are plain PyTorch; Lightly is no longer needed.
- SS6 holds four recipes written before any result: two bets for best accuracy (joint loss with RandAugment
  views; pretrain then fine-tune) and two for speed (short GPU-only augmentation + SVM). They are
  candidates for `simsiam_best` / `simsiam_fast`, not pre-selected winners.

## <date> · K1: dictionary size × patch size (18 configs)
**Why:** how does the dictionary size interact with the receptive field?

| patch | K=5 | K=10 | K=15 | K=25 | K=50 | K=100 |
|---|---|---|---|---|---|---|
| 4 | | | | | | |
| 6 | | | | | | |
| 8 | | | | | | |

**Observations:**

**Decision → K2:** (chosen K and patch; copy into `KMEANS_BASE`)

## <date> · K2: contrast and whitening constants
**Why / results / decision:**

## <date> · SS1: pretraining length
**Why:** does SimSiam pretraining on 100 images beat a random-init backbone (epochs=0)?
Check `collapse_std` in the notes (≈ 1/sqrt(proj_dim) healthy, ≈ 0 collapsed) for the longest runs.

**Decision → SS2–SS5:** (epochs chosen; copy into `SS_BASE`)

## <date> · SS2–SS5: augmentation, architecture, optimization, using labels
Paired Δ vs the SS1 base (`configs.SS_REF`); figure `c1_ssl_simsiam_ablation`.

| stage | config | Δ acc (pp) | wins/seeds | time (s) |
|---|---|---|---|---|
| | | | | |

**Results and decisions:**

## <date> · SS6: recipes written before the sweep
| config | acc (%) | time (s) | note |
|---|---|---|---|
| ss6_joint_ra | | | |
| ss6_ft_ra | | | |
| ss6_fast_svm | | | |
| ss6_fast_joint | | | |

## <date> · Selection (shared rule above; `python -m c1.ssl.select --hw colab-t4`)
- kmeans_best / kmeans_fast:
- simsiam_best / simsiam_fast:
- close calls (picked by the 0.5 pp tie step):

## <date> · Report seeds 10–19
| config | acc (%) | train (s) | inference (s / 700 img) |
|---|---|---|---|
| kmeans_best | | | |
| kmeans_fast | | | |
| simsiam_best | | | |
| simsiam_fast | | | |

Paired differences (report seeds): simsiam_best − kmeans_best, best − fast per method, vs the other C1 methods.

# Source log

| source | used for |
|---|---|
| Coates, Ng & Lee 2011, *An analysis of single-layer networks in unsupervised feature learning*, AISTATS | the whole k-means pipeline, quadrant sum pooling, linear SVM |
| Chen & He 2021, *Exploring Simple Siamese Representation Learning*, CVPR | SimSiam: stop-gradient, predictor, negative cosine loss |
| Chen, Kornblith, Norouzi & Hinton 2020, *A simple framework for contrastive learning*, ICML | augmentation recipe (random resized crop, colour distortion) |
| Cubuk et al. 2020, *RandAugment*, CVPRW | RandAugment views via `common/augment.py` |
| Ioffe & Szegedy 2015, *Batch normalization*, ICML | BatchNorm in backbone, projector and predictor |
| Ryan's `c1/cnn` (`build_backbone`, `balanced_batches`, `lr_factor`) and `common/augment.py` | backbone, batching, LR schedule, views |
| Pedregosa et al. 2011, *Scikit-learn*, JMLR | KMeans, LinearSVC, GridSearchCV, StandardScaler |