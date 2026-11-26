# C1 self-supervised: experiment log

Hardware tag, seeds and metric definitions follow the team rules: tuning seeds 0–4, report seeds
10–19, accuracy = mean ± std over seeds (std = spread across class pairs), time = train + inference
seconds per seed. Raw rows: `results/alvin.csv` (`hw=colab-t4`). Feature-extraction / dictionary /
pretraining times are in each row's `notes` (`extract_s`, `dict_s`, `pretrain_s`, `finetune_s`).

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

## 2026-11-25 · K1: dictionary size × patch size (18 configs)
**Why:** how does the dictionary size interact with the receptive field?

Validation accuracy (%), mean over tuning seeds 0–4:

| patch | K=5 | K=10 | K=15 | K=25 | K=50 | K=100 |
|---|---|---|---|---|---|---|
| 4 | 77.1 | 79.8 | 79.2 | **80.1** | 77.2 | 76.8 |
| 6 | 77.6 | 78.6 | 78.1 | 77.0 | 73.9 | 74.6 |
| 8 | 75.5 | 76.9 | 74.6 | 75.0 | 71.9 | 74.2 |

**Observations:**
- Small patches win: patch 4 is best at every K except K=5, and accuracy falls as the patch grows (K=50: 77.2 → 73.9 → 71.9).
- Accuracy peaks at a small dictionary (K=10–25) and declines beyond it. With 100 training images, 4K pooled features at K ≥ 50 overfit the linear SVM; the paper's K in the hundreds does not transfer to this data size.
- Cost grows with both K and patch size (fit 1.7 s at K=5 / p=4 up to 37.7 s at K=100 / p=8), so the large settings are slower and worse.
- The std across class pairs is 4–7 pp, so K=10 vs K=25 at patch 4 is within noise.

**Decision → K2:** K=10, patch 4 (`KMEANS_BASE`). `k1_k25_p4` has the top accuracy (80.14 %), but `k1_k10_p4` is within 0.5 pp (79.77 %) and faster (2.89 s vs 4.21 s), so the shared rule picks it.

## 2026-11-25 · K2: contrast and whitening constants
**Why:** the Coates pipeline has two regularizers, a constant added to the patch std (contrast `epsilon`) and one added to the eigenvalues in ZCA (`whitening_epsilon`). Their defaults (0.1, 1e-3) come from the paper's larger-data setting.

| config | acc (%) |
|---|---|
| contrast ε = 0.01 / 0.1 (default) / 1 / 10 | 79.7 / 79.8 / 77.6 / 79.1 |
| whitening ε = 1e-5 / 1e-3 (default) / 0.1 / 1.0 | 76.9 / 79.8 / **81.0** / 80.5 |

**Observations:**
- Contrast ε does not help: nothing beats the default, and ε = 1 is worse (−2.2 pp) with a large spread across seeds (std 11.5 pp).
- Whitening ε matters more: a weak ε = 1e-5 hurts (−2.9 pp), while stronger regularization helps (+1.2 pp at 0.1, +0.7 pp at 1.0). With only 100 images, the covariance of 48-dimensional patches is estimated from noisy data, so whitening more gently suppresses noise directions.
- Time is unchanged (2.7–2.9 s for every config): these constants do not change the cost.

**Decision:** `k2_wzca0.1` has the top accuracy (81.00 %), but `k2_wzca1` is within 0.5 pp (80.51 %, a 0.49 pp gap) and faster (2.67 s vs 2.85 s), so the rule picks `k2_wzca1`.

## 2026-11-25 · SS1: pretraining length
**Why:** does SimSiam pretraining on 100 images beat a random-init backbone (epochs=0)?

| epochs | 0 (random init) | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|
| acc (%) | **81.9** | 77.7 | 78.7 | 77.5 | 78.2 |
| fit (s) | 0.2 | 2.4 | 4.8 | 9.4 | 18.8 |

**Observations:**
- Random init + linear SVM beats every pretrained setting. Paired, `ss1_e0 − ss1_e100` is +3.17 pp, random init wins on 4/5 seeds, Wilcoxon p = 0.125 (not significant with 5 seeds).
- Longer pretraining does not help (no trend from 50 to 400 epochs).
- Reading: with 100 images and a 24k-parameter backbone, SimSiam pretraining does not improve a frozen-feature linear probe. Random conv features are already a strong baseline.
- `collapse_std` was not inspected yet. TODO: check it for `ss1_e100` / `ss1_e400` (healthy ≈ 1/√64 = 0.125, collapsed ≈ 0). Run: `d = own_rows(pd.read_csv(RESULTS)); d = d[d.config_id.isin(["ss1_e100", "ss1_e400"])]; print(d.assign(cs=d.notes.str.extract(r"collapse_std=([\d.e-]+)")[0].astype(float)).groupby("config_id").cs.mean())`

**Decision → SS2–SS5:** `SS_BASE = {"epochs": 100}`: the best pretrained length (78.7 %). Epochs = 0 is not used as the base, because the later stages vary pretraining knobs and would change nothing on a model that does not pretrain.

## 2026-11-25 · SS2–SS5: augmentation, architecture, optimization, using labels
Change in mean accuracy vs the SS1 base `ss1_e100` (78.7 %, fit 4.8 s), one factor at a time. Wins out of 5 seeds are in the figure `c1_ssl_simsiam_ablation`. The time column is fit time only.

| stage | config | acc (%) | Δ acc (pp) | fit (s) |
|---|---|---|---|---|
| SS2 aug | RRC min scale 0.15 / 0.6 / none | 78.3 / 77.0 / 79.1 | −0.4 / −1.7 / +0.4 | 4.8 / 4.9 / 4.1 |
| | RandAugment N=1 M=9 | 78.7 | 0.0 | 12.9 |
| | RandAugment N=2 M=9 | 79.2 | +0.5 | 19.8 |
| | RandAugment N=2 M=14 | 77.8 | −0.9 | 19.7 |
| | Cutout 8 | 79.1 | +0.5 | 5.2 |
| | no pad-crop | 77.2 | −1.5 | 4.6 |
| SS3 arch | width 32 / 64 | 80.5 / 80.7 | +1.8 / +2.0 | 5.2 / 5.2 |
| | 2 convs per stage | 76.5 | −2.1 | 5.7 |
| | projector dim 32 / 128 / 256 | 78.2 / 77.8 / 78.4 | −0.5 / −0.9 / −0.3 | 4.8 / 4.6 / 4.7 |
| SS4 optim | lr 0.01 / 0.2 | 78.3 / 77.8 | −0.4 / −0.9 | 4.9 / 4.9 |
| | weight decay 0 / 5e-3 | 77.9 / 78.7 | −0.8 / 0.0 | 4.9 / 4.8 |
| | batch size 50 / 100 | 79.1 / 78.8 | +0.5 / +0.1 | 2.4 / 1.3 |
| SS5 labels | joint SimSiam + CE, weight 0.3 / 1 / 3 | 84.3 / 84.3 / 84.2 | +5.6 / +5.6 / +5.5 | 5.2 / 5.1 / 5.0 |
| | fine-tune 50 / 150 epochs | 84.4 / **86.5** | +5.7 / **+7.8** | 5.4 / 6.7 |

**Results:**
- **Using the labels is the only large effect.** Joint loss adds about 5.6 pp whatever the weight (0.3–3 are within 0.1 pp). Fine-tuning gains more as it gets longer (+5.7 pp at 50 epochs, +7.8 pp at 150).
- **Augmentation, projector and optimizer changes are small**, mostly within ±1 pp, against a spread of 4–6 pp across class pairs. RandAugment N=2 M=9 gives +0.5 pp for 4× the time. A strong pad-crop matters (removing it costs 1.5 pp).
- **Wider backbones help a little** (+1.8 / +2.0 pp at no extra time), but wider random-init features were not tested, so this cannot be credited to SimSiam.
- **Larger batches are much faster** for the same accuracy (batch 100: 1.3 s vs 4.8 s, +0.1 pp). Batch size was not combined with the other factors, so this is a possible speed-up that was left unexplored.
- Stages were run one factor at a time and not combined (apart from the SS6 recipes), so interactions were not tested.

## 2026-11-25 · SS6: recipes written before the sweep
| config | acc (%) | time (s) | note |
|---|---|---|---|
| ss6_ft_ra | **87.3** | 33.8 | pretrain 100 epochs with RandAugment N=2 M=9 views, then fine-tune 150 epochs |
| ss6_joint_ra | 86.1 | 41.3 | joint SimSiam + CE with RandAugment views, 200 epochs |
| ss6_fast_joint | 84.7 | 5.2 | joint loss, 100 epochs, no RandAugment |
| ss6_fast_svm | 77.6 | 1.2 (fit) | 50 epochs, frozen features + SVM: below random init (81.9 %) |

The two accuracy bets were right: using labels with RandAugment views gave the best configs. The speed bet with a frozen SVM was wrong, because it loses to random init. The fast recipe that works keeps the labels (`ss5_ft150`, `ss6_fast_joint`).

## 2026-11-25 · Selection (shared rule above; `python -m c1.ssl.select --hw colab-t4`)
`configs.FINAL_SOURCE` was updated to these picks and cell 8 reports that it agrees.
- kmeans_best / kmeans_fast: `k2_wzca1` for both (80.51 %, 2.67 s). No other k-means config is both faster and within 2 pp, so best and fast are the same config.
- simsiam_best: `ss6_ft_ra` (87.26 %, 33.8 s). simsiam_fast: `ss5_ft150` (86.49 %, 6.7 s).
- close calls:
  - k-means: `k2_wzca0.1` has the top accuracy (81.00 %), but `k2_wzca1` is 0.49 pp lower and faster, so the 0.5 pp tie step picks it.
  - SimSiam best: `ss5_ft150` is 0.77 pp below `ss6_ft_ra`, just outside the tie window, so the slower config stays best.
  - SimSiam fast: `ss6_fast_joint` (84.66 %, 5.2 s) misses the 2 pp window by 0.6 pp, so `ss5_ft150` is fast.

## 2026-11-25 · Report seeds 10–19
Accuracy is mean ± std over the 10 class pairs; times are per problem instance.

| config | acc (%) | train (s) | inference (s / 700 img) |
|---|---|---|---|
| kmeans_best (`k2_wzca1`) | 75.97 ± 10.32 | 1.88 | 0.626 (0.89 ms/img) |
| kmeans_fast (`k2_wzca1`) | 75.97 ± 10.32 | 1.91 | 0.639 (0.91 ms/img) |
| simsiam_best (`ss6_ft_ra`) | 83.56 ± 9.99 | 33.66 | 0.016 (0.02 ms/img) |
| simsiam_fast (`ss5_ft150`) | 81.40 ± 9.43 | 6.99 | 0.014 (0.02 ms/img) |

Paired differences on the report seeds (percentage points, wins out of 10):
- simsiam_best − kmeans_best: +7.59 pp, wins 10/10, Wilcoxon p = 0.0020.
- simsiam_best − simsiam_fast: +2.16 pp, wins 8/10, p = 0.0137.
- kmeans_best − kmeans_fast: +0.00 pp. They are the same config, so the two rows differ only in timing noise.

**Observations and caveats:**
- All report-seed accuracies are 3.7–5.1 pp lower than the tuning-seed numbers (k-means 80.5 → 76.0, SimSiam best 87.3 → 83.6, fast 86.5 → 81.4). The tuning numbers are optimistic because the configs were selected on them, and the class pairs differ. The std across class pairs (about 10 pp) is large, so the report-seed numbers are the ones to quote.
- SimSiam beats k-means on every class pair, but k-means inference is about 40× slower (0.63 s vs 0.016 s per 700 images), since it encodes every patch position on CPU.
- **The SimSiam winners are mostly a supervised result.** Frozen SimSiam features never beat random init (SS1), and the gains come from the labels (SS5/SS6). A fine-tune from random init with the same views (`epochs=0`) was not run, so how much the SimSiam pretraining contributes to `ss6_ft_ra` is not isolated. It would be the control to add.
- For reference, Ryan's `c1/cnn` README reports 86.0 ± 10.5 % (best) and 84.4 ± 9.4 % (fast) on the same report seeds, so our SimSiam configs are 2.4 and 3.0 pp below it, which is consistent with the SSL part not adding much. Paired numbers across the two methods still need the team comparison.

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