# C1 CNN: experiment log

Unless a stage says otherwise, all numbers below are on tuning seeds 0–4. Accuracy is the mean ± std of
validation accuracy over the seeds (std = spread across class pairs), after first averaging the reps
(independent training runs) within each seed. Time is train + inference seconds per seed. The raw T4
rows are in `results/ryan.csv` (and `results/ryan_curves.csv` for the training curves). Laptop rows are
kept out of the repo, in the git-ignored `results/local/`.

**Hardware tags.** `hw=colab-t4` rows are the ones the report uses. `hw=local-cpu` rows come from the
laptop (Intel Core Ultra 7 256V, CPU only, no CUDA). They exist to validate the pipeline and to get a
first look; times from them never go in the report.

**Selection rule** (the team's rule, fixed before any CNN results were seen, in `select.py`):
- best = the highest mean tuning accuracy, except that a config within 0.5 pp of it wins if it's faster.
- fast = the fastest config whose mean accuracy is at least best − 2 pp.

**Comparisons** between two configs are always paired over seeds: mean per-seed difference, wins out
of n, and a Wilcoxon p-value (`python -m c1.cnn.select --pair A B`).

## 2026-11-21 · S0: baseline recipe, with and without flip + crop (2 configs × 2 reps)

**Why:** check the pipeline end to end and set the two anchors that every later stage is compared
against.

**Setup (chosen before any results):**
- Model: ConvNet-6 (3 stages × 2 conv3×3-BN-ReLU, widths 32/64/128, max-pool after each stage, global
  average pool, linear head), 0.29M parameters.
- Optimization: SGD with Nesterov momentum 0.9, lr 0.05, weight decay 5e-4, cosine schedule with 5
  warmup epochs.
- Batches: batch size 32, which gives 4 balanced batches of 25 per epoch (`balanced_batches`), so no
  step sees a 4-image remainder batch.
- Training length: 200 epochs = 800 steps.
- Normalization: per-channel mean and std computed from the 100 training images, not from CIFAR-wide
  constants, because those come from the full training set.

Two configs: `s0_noaug` uses no augmentation; `s0_flipcrop` uses random crop (4 px reflect padding) and
a horizontal flip.

**Results** (`hw=colab-t4`, 2 reps):

| config | acc (%) | seed 0 | 1 | 2 | 3 | 4 | rep 0 | rep 1 | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| `s0_noaug` | 85.80 ± 5.56 | 86.4 | 87.3 | 90.9 | 75.1 | 89.3 | 86.60 | 85.00 | 5.05 |
| `s0_flipcrop` | 86.64 ± 6.60 | 86.6 | 90.0 | 92.6 | 74.0 | 89.9 | 87.20 | 86.09 | 5.24 |

The seeds are the class pairs bird–ship, bird–truck, deer–automobile, dog–deer and cat–ship.

Paired comparison, flipcrop − noaug: **+0.84 pp**, 4 wins and 1 loss out of 5 seeds, Wilcoxon p = 0.31.
Per-seed differences: +0.3, +2.7, +1.7, −1.1, +0.6 pp.

Training curves (`curves.py`, rep 0; accuracy is the mean over the 5 seeds, loss the median):

| epoch | 10 | 30 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|
| noaug val acc (%) | 72.7 | 85.4 | 86.0 | 86.5 | 86.7 | 86.7 |
| flipcrop val acc (%) | 75.7 | 84.5 | 82.1 | 85.6 | 86.8 | 86.5 |
| noaug train loss | 0.11 | 0.008 | 4e-4 | 4e-4 | 1e-4 | 9e-4 |
| flipcrop train loss | 0.20 | 0.093 | 0.058 | 0.007 | 0.002 | 0.006 |

**Laptop check** (`hw=local-cpu`, same configs, rows git-ignored): noaug 85.37, flipcrop 86.17,
+0.80 pp (3/5). The ranking is the same, and both accuracies are within 0.5 pp of the T4.

**Observations:**
- **Even the untuned CNN is at the level of the classical method.** The classical method's best tuning
  accuracy was 86.1% (`c1/classical/NOTES.md`, S2), and this baseline gets 85.8–86.6%.
- **Without augmentation the network memorizes the 100 images by about epoch 50,** with training loss
  around 4e-4. After that, validation accuracy is flat, because there's nothing left to learn from.
- **Flip + crop slows memorization down.** Training loss is still 0.06 at epoch 50, and validation
  accuracy catches up only by about epoch 150. Stronger augmentation should need more epochs, so S2
  uses 300 and S3 tests the epoch count.
- **The +0.84 pp gain from flip + crop is suggestive, not conclusive at n = 5** (4/5 seeds, p = 0.31).
- **Training noise is about the same size as the effect.** Across all T4 configs, the two reps of a
  config differ by about 1.0 pp on average. So every stage uses 2 reps.
- **Dog vs. deer (seed 3) is much harder** than the other pairs, at about 74–75% versus 86–93% for the
  rest. This is the class-pair variance that makes paired comparisons necessary.
- **Design note: the training seed is the rep number,** so rep 0 of every problem instance shares its
  weight init and batch-order RNG stream. That's why the loss spikes in the curves line up across
  seeds. It also gives common random numbers across configs, which reduces noise in the paired
  comparisons.

**Decision → S1:** keep flip + crop as the reference recipe. It's standard, it slows memorization, and
it's the base the S2 additions stack on. Compare architectures at the same 200-epoch recipe.

## 2026-11-21 · S1: architecture screen (11 configs × 2 reps; reference `s0_flipcrop` = ConvNet-6 w32)

**Why:** choose the backbone before tuning augmentation. The question is whether depth, width,
residual connections, or GroupNorm matter at n = 100.

**Candidates (S0 flip + crop recipe, 200 epochs):**
- ConvNet-3 at w ∈ {16, 32, 64}: 24k / 94k / 372k parameters.
- ConvNet-6 at w ∈ {16, 64}: 72k / 1.15M. The w32 version (288k) is `s0_flipcrop`, so it isn't re-run.
- ResNet-8 at w ∈ {16, 32}: 78k / 308k.
- ResNet-20 at w ∈ {16, 32}: 272k / 1.08M.
- GroupNorm instead of BatchNorm for one representative of each family at its standard width: ConvNet-6
  w32 and ResNet-20 w16.

**Results** (`hw=colab-t4`, 2 reps; time = T4 fit + inference per seed):

| config | params | acc (%) | seed 0 | 1 | 2 | 3 | 4 | time (s) |
|---|---|---|---|---|---|---|---|---|
| ConvNet-3 w32 | 94k | **87.46 ± 5.62** | 87.6 | 90.6 | 93.0 | 76.8 | 89.3 | 3.88 |
| **ConvNet-3 w16** | 24k | 86.99 ± 5.40 | 86.4 | 89.7 | 93.1 | 77.1 | 88.6 | **3.85** |
| ConvNet-3 w64 | 372k | 86.96 ± 5.52 | 86.9 | 89.8 | 93.1 | 76.7 | 88.4 | 3.98 |
| ConvNet-6 w32 (`s0_flipcrop`) | 288k | 86.64 ± 6.60 | 86.6 | 90.0 | 92.6 | 74.0 | 89.9 | 5.24 |
| ConvNet-6 w64 | 1.15M | 86.27 ± 6.75 | 86.4 | 89.1 | 92.9 | 73.4 | 89.5 | 5.37 |
| ConvNet-6 w16 | 72k | 85.64 ± 7.54 | 86.1 | 89.6 | 91.6 | 71.0 | 90.0 | 5.22 |
| ResNet-8 w32 | 308k | 85.33 ± 7.12 | 84.9 | 88.1 | 92.4 | 71.9 | 89.3 | 6.42 |
| ResNet-8 w16 | 78k | 85.07 ± 7.23 | 84.7 | 87.1 | 92.1 | 71.5 | 90.0 | 6.40 |
| ResNet-20 w16 | 272k | 84.69 ± 6.70 | 84.1 | 87.6 | 91.1 | 72.1 | 88.6 | 11.82 |
| ResNet-20 w32 | 1.08M | 83.99 ± 6.73 | 82.8 | 85.9 | 91.1 | 71.7 | 88.4 | 11.88 |
| ResNet-20 w16, GroupNorm | 272k | 81.86 ± 7.12 | 84.0 | 86.3 | 83.9 | 67.9 | 87.2 | 13.74 |
| ConvNet-6 w32, GroupNorm | 288k | 79.66 ± 6.52 | 82.9 | 85.4 | 76.1 | 68.5 | 85.4 | 5.74 |

Paired comparisons (mean difference; wins out of 5; Wilcoxon p):

| comparison | difference | wins | p |
|---|---|---|---|
| ConvNet-3 w32 − ConvNet-3 w16 | +0.47 pp | 3/5 | 0.31 |
| ConvNet-3 w64 − ConvNet-3 w16 | −0.03 pp | 2/5 | 1.00 |
| ConvNet-3 w16 − ConvNet-6 w32 | +0.34 pp | 2/5 | 1.00 |
| ConvNet-3 w16 − ConvNet-6 w16 | +1.34 pp | 4/5 | 0.31 |
| ConvNet-3 w16 − ResNet-8 w16 | +1.91 pp | 4/5 | 0.19 |
| ConvNet-6 w32 − ResNet-20 w16 | +1.96 pp | **5/5** | 0.06 |
| BatchNorm − GroupNorm, ConvNet-6 w32 | **+6.99 pp** | **5/5** | 0.06 |
| BatchNorm − GroupNorm, ResNet-20 w16 | **+2.83 pp** | **5/5** | 0.06 |

With n = 5 the smallest possible two-sided Wilcoxon p is 0.0625, so 5/5 wins is the strongest evidence
5 seeds can give. We therefore read wins out of n together with the effect size, rather than relying on
a p < 0.05 threshold.

**Laptop check** (`hw=local-cpu`):
- **ConvNet-3 order:** the laptop ranked ConvNet-3 w16 first (87.59), then w64 (87.16) and w32 (87.01).
  The T4 ranks w32 first.
- **That difference is training noise.** The three widths are within 0.5 pp on both machines, and
  w32 − w16 is +0.47 pp (3/5) on the T4 but −0.57 pp on the laptop.
- **Everything else agrees.** Every other config is within one rank position, and the backbone decision
  below is the same on both machines.

**Observations:**
- **Width doesn't matter within ConvNet-3.** w16, w32 and w64 are within 0.5 pp across a 16× range in
  parameters, and no pairwise difference is consistent across seeds.
- **Shallower is at least as good.** ConvNet-3 ties the 6-layer reference (+0.34 pp, 2/5), beats
  ConvNet-6 at the same width (+1.34 pp, 4/5), and is about 25% faster on the T4.
- **Residual networks are worse than plain ConvNets of similar size.** ResNet-8 w16 is −1.9 pp vs.
  ConvNet-3 w16, and ResNet-20 w16 is −2.0 pp vs. ConvNet-6 w32 (5/5). ResNets mainly help *optimize*
  deep networks, but with 100 images optimization isn't the bottleneck, generalization is. ResNet-20
  also costs about 3× the time of ConvNet-3.
- **BatchNorm clearly beats GroupNorm:** GN is −2.8 to −7.0 pp, on 5/5 seeds both times. Our balanced
  batches of 25 are evidently large enough for BN's batch statistics, and BN's batch noise may also act
  as a regularizer that GN doesn't provide. GN's biggest loss is on deer–automobile (seed 2): −16.5 pp
  for ConvNet-6.
- **On the T4, small networks are overhead-bound.** All ConvNet-3 widths take about 3.9 s, so the
  parameter count barely affects time at this scale.
- **The class-pair ordering is stable across architectures.** Dog–deer is always the hardest pair, at
  68–77%.

**Decision → S2:**
- **Backbone: ConvNet-3 w16 with BatchNorm.** w32 has the top mean (87.46), but w16 is within 0.5 pp
  (86.99) and faster (3.85 vs 3.88 s), so the selection rule picks w16. The time margin is negligible:
  in practice the three widths tie, and the smallest one is the natural choice.
- **Dropped:** GroupNorm (clearly worse) and ResNets (worse at every size, and slower).
- **Width is re-checked in S4.** Stronger augmentation makes the training task harder, which can favor
  more capacity, so S4 re-tests w32 and w64 with the final recipe.

## 2026-11-21 · S2a: augmentation and regularization, one at a time (16 configs × 2 reps)

**Why:** measure what each augmentation or regularizer adds on its own, on the S1 backbone. This is the
main ablation of the method.

**Setup:**
- Every config is the reference plus exactly one change. The reference `s2a_ref` is ConvNet-3 w16 + BN
  with flip + crop, using the S0 optimizer settings.
- **Training length goes from 200 to 300 epochs.** The S0 curves showed augmented runs still improving
  until about epoch 150, and stronger augmentation slows fitting further.
- The reference is re-run at 300 epochs, so every difference is measured at the same training length.

**Candidates:**
- Decomposing the basics: no augmentation, flip only, crop only.
- Cutout with size 8 or 16.
- mixup with α 0.2 or 1.0.
- CutMix with α 1.0.
- RandAugment with N = 2 and M ∈ {5, 9, 14}.
- Label smoothing 0.1.
- Dropout 0.3 before the head.
- Weight decay 1e-4 or 5e-3 (the reference uses 5e-4).

**Results** (`hw=colab-t4`, 2 reps; Δ = paired difference vs `s2a_ref`, which gets 87.36 ± 5.21%):

| config | acc (%) | Δ (pp) | wins | p | seed 0 | 1 | 2 | 3 | 4 | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| RandAugment M=9 | **89.44 ± 4.89** | **+2.09** | **5/5** | 0.06 | 88.4 | 91.9 | 94.9 | 80.6 | 91.5 | 27.9 |
| RandAugment M=14 | 89.30 ± 4.93 | +1.94 | 5/5 | 0.06 | 88.9 | 92.1 | 94.9 | 80.3 | 90.3 | 28.0 |
| RandAugment M=5 | 89.00 ± 5.39 | +1.64 | 5/5 | 0.06 | 88.2 | 91.5 | 95.3 | 79.2 | 90.8 | 28.0 |
| mixup α=1 | 88.73 ± 5.23 | +1.37 | 4/5 | 0.13 | 88.6 | 92.9 | 92.6 | 78.7 | 90.9 | 5.9 |
| mixup α=0.2 | 88.39 ± 5.53 | +1.03 | 5/5 | 0.06 | 88.1 | 92.2 | 93.4 | 77.9 | 90.4 | 5.9 |
| label smoothing 0.1 | 87.97 ± 5.37 | +0.61 | 3/5 | 0.31 | 88.2 | 91.7 | 92.7 | 77.7 | 89.5 | 5.6 |
| flip only (no crop) | 87.57 ± 6.21 | +0.21 | 3/5 | 0.81 | 87.7 | 92.1 | 92.9 | 75.7 | 89.4 | 5.1 |
| Cutout 16 | 87.53 ± 5.27 | +0.17 | 3/5 | 0.38 | 87.4 | 90.6 | 92.9 | 77.6 | 89.1 | 6.2 |
| CutMix α=1 | 87.47 ± 4.95 | +0.11 | 4/5 | 0.63 | 86.7 | 90.8 | 91.6 | 78.1 | 90.1 | 6.0 |
| *reference: flip + crop* | 87.36 ± 5.21 | 0 | | | 86.6 | 90.4 | 92.9 | 77.8 | 89.1 | 5.7 |
| Cutout 8 | 87.26 ± 5.40 | −0.10 | 2/5 | 0.81 | 86.6 | 91.1 | 92.9 | 77.4 | 88.3 | 6.1 |
| weight decay 1e-4 | 87.24 ± 5.54 | −0.11 | 1/5 | 0.75 | 87.6 | 90.4 | 92.9 | 76.8 | 88.6 | 5.6 |
| weight decay 5e-3 | 87.21 ± 5.45 | −0.14 | 1/5 | 0.44 | 87.2 | 90.1 | 92.8 | 76.9 | 89.1 | 5.6 |
| crop only (no flip) | 86.94 ± 5.96 | −0.41 | 1/5 | 0.44 | 86.4 | 91.4 | 92.6 | 75.9 | 88.4 | 5.5 |
| dropout 0.3 | 86.59 ± 5.85 | −0.77 | 0/5 | 0.13 | 86.0 | 90.4 | 92.5 | 75.7 | 88.3 | 5.6 |
| no augmentation | 86.29 ± 6.44 | −1.07 | 0/5 | 0.06 | 85.6 | 89.9 | 92.7 | 74.2 | 89.0 | 4.9 |

The reference at 300 epochs vs. the same model at 200 epochs (`s1_convnet3_w16`): +0.37 pp, 4/5.

Curves (rep 0, mean over seeds):

| epoch | 200 | 250 | 300 |
|---|---|---|---|
| `s2a_ref` val acc (%) | 86.9 | 87.3 | 87.4 |
| `s2a_noaug` val acc (%) | 86.7 | 87.0 | 87.0 |

**Observations:**
- **RandAugment is the strongest single addition:** +1.6 to +2.1 pp, better on every seed at every
  magnitude. The magnitude barely matters (M = 5, 9 and 14 are within 0.45 pp). It helps most on the
  hardest pair (dog–deer, +2.8 pp at M = 9).
- **RandAugment is 5× slower to train** (28 s vs 5.7 s). That comes from our implementation, not the
  method. torchvision's v2 RandAugment applies one op to a whole batch, so we call it per image, which
  makes it the only augmentation that loops over images one at a time. This matters for the `fast`
  config.
- **mixup is the second strongest,** at +1.0 to +1.4 pp, for almost no cost (+0.2 s). α = 1 has the
  larger mean, and α = 0.2 is more consistent (5/5).
- **Label smoothing may help a little** (+0.6 pp, 3/5). That's inconclusive on its own, so S2b's
  leave-one-out decides whether it stays.
- **Occlusion augmentations (Cutout, CutMix) and weight decay changes do nothing:** all are within
  ±0.2 pp. A possible reason for Cutout and CutMix, which we didn't test: at 32×32 a 16-px patch can
  remove most of the object, and with 50 images per class the network may not learn to use the
  remaining parts.
- **Dropout before the head hurts** (−0.77 pp, no seed improves). The network has only 64 features
  before the head, and dropping 30% of them removes useful signal.
- **Among the basics, flip is what helps.** Flip alone is as good as flip + crop (+0.21 pp), crop alone
  is worse than flip + crop (−0.41 pp), and with neither it's −1.07 pp (0/5).
- **300 epochs is worth a little more than 200** (+0.37 pp, 4/5), and the reference's curve is flat
  after about 250 epochs. With stronger augmentation the best length may differ, so S3 tests it.

**Decision → S2b:** combine the components that helped, then remove each one in turn.
- **Full recipe `s2b_full`:** flip + crop + RandAugment (N = 2, M = 9) + mixup (α = 1) + label
  smoothing 0.1. M = 9 and α = 1 are the best S2a settings of RandAugment and mixup.
- **Crop stays in the full recipe** even though it didn't help in S2a. The leave-one-out tests dropping
  it from the combined recipe, which is the question that matters.
- **Dropped:** Cutout, CutMix, dropout, and both weight decay changes (5e-4 stays).

## 2026-11-22 · S2b: combined recipe and leave-one-out (8 configs × 2 reps, to run on the T4)

**Why:** S2a measured each component on its own, but the components may not add up when combined. For
example, RandAugment and mixup could both be regularizing the same thing. The leave-one-out from the
full recipe gives the report's ablation table: how much each component contributes given all the
others.

**Configs** (ConvNet-3 w16 + BN, 300 epochs, S0 optimizer settings):
- `s2b_full`: flip + crop + RandAugment (2, 9) + mixup α = 1 + label smoothing 0.1.
- **Leave-one-out:** `s2b_no_flip`, `s2b_no_crop`, `s2b_no_ra`, `s2b_no_mixup` and `s2b_no_ls`. Each
  is `s2b_full` minus one component, compared paired against `s2b_full`.
- **Variants of the two strong components:** `s2b_mixup02` (mixup α = 0.2) and `s2b_ra_m14`
  (RandAugment M = 14), in case the milder or stronger setting combines better.
- **Curves:** `s2b_full` is added to the training curves, so the dynamics figure can show the final
  recipe against no augmentation.

**Results:** PENDING (T4)

## Source log

| Source | What we took from it |
|---|---|
| He et al. 2016, *Deep Residual Learning* (Sec. 4.2, CIFAR) | CIFAR ResNet-8/20 layout; SGD 0.1 / wd 1e-4 / crop+flip recipe as a starting point |
| Simonyan & Zisserman 2015 (VGG) | 3×3 conv stacks for the plain ConvNet |
| Ioffe & Szegedy 2015 (BatchNorm); Wu & He 2018 (GroupNorm) | BN vs. GN comparison: GN doesn't depend on batch statistics, which matters with batches of ~25 |
| DeVries & Taylor 2017 (Cutout) | Cutout; the centre can be anywhere, so the square may be partly outside the image |
| Zhang et al. 2018 (mixup) | mixup with Beta(α, α), α ∈ {0.2, 1} |
| Yun et al. 2019 (CutMix) | CutMix; targets mixed by the exact pasted area after clipping |
| Cubuk et al. 2020 (RandAugment) | RandAugment (N, M) via torchvision, applied per image |
| Szegedy et al. 2016 (Inception v3) | label smoothing |
| Loshchilov & Hutter 2017 (SGDR), 2019 (AdamW) | cosine schedule with warmup; AdamW |
| Smith & Topin 2019 (super-convergence / one-cycle) | one-cycle schedule |
| Huang et al. 2017 (Snapshot Ensembles); Lakshminarayanan et al. 2017 (deep ensembles) | averaging softmax over independently trained members |
