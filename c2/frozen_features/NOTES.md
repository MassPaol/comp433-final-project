# C2 frozen features: experiment log

All numbers below: tuning seeds 0–4 unless stated, `hw=local-3070` (RTX 3070 Laptop GPU; the Colab T4
re-run in `notebooks/massimo/c2_frozen_features.ipynb` gives the times for the report). Accuracy =
mean ± std of validation accuracy over the seeds (std = spread across class pairs). Time = train +
inference seconds per seed, where train includes loading the backbone from local disk and extracting
the 100 training images' features, and inference extracts the 700 validation images' features. Raw rows:
`results/massimo.csv` (`challenge=c2, method=frozen_features`; `load_s` / `extract_s` in the notes).

Tuning class pairs: seed 0 bird-ship, 1 bird-truck, 2 deer-automobile, 3 dog-deer, 4 cat-ship.

## 2026-11-10 · S1: every backbone × every head (42 configs, defaults)

**Why:** find which backbones are worth tuning before spending compute on heads and feature variants.
Default features are each model's own pre-logits (ViT: CLS token after the final norm; CNN: pooled
features); default normalization per head (AdamW: none, as in the testbed; NCM/kNN: L2; others: z-score).

Mean tuning accuracy (%):

| backbone (pretraining) | AdamW (testbed) | LR | lin. SVM | RBF SVM | NCM | kNN | time (s) |
|---|---|---|---|---|---|---|---|
| ViT-B/16 (IN-21k) | 99.40 | **99.77** | 99.74 | 99.43 | 99.69 | 99.63 | 6.1–6.6 |
| ViT-S/16 (IN-21k) | 99.23 | **99.46** | 99.26 | 99.11 | 99.40 | 99.37 | 1.6 |
| ConvNeXt-T (IN-22k) | 99.34 | 99.40 | **99.43** | 98.74 | 99.17 | 98.86 | 2.3 |
| ViT-S/16 (IN-1k only) | 98.89 | 99.23 | 99.09 | 99.03 | **99.29** | 99.09 | 1.7 |
| DeiT-S/16 (IN-1k) | **99.29** | 99.14 | 99.03 | 98.46 | 98.97 | 99.00 | 1.9–2.0 |
| ViT-Ti/16 (IN-21k, testbed) | 98.69 | 99.00 | 98.80 | 98.91 | **99.09** | 98.54 | 0.6 |
| ResNet-50 (IN-1k) | **97.97** | 97.63 | 97.37 | 97.43 | 97.51 | 97.71 | 1.6–1.7 |

Time per seed is dominated by feature extraction, not by the head: inference (700 images) takes 0.41 s
(ViT-Ti) to 4.46 s (ViT-B), loading 0.10 s to 1.26 s, training-feature extraction 0.06 s to 0.59 s;
every head trains in well under 0.1 s.

**Observations:**
- Near the ceiling: every config is between 97.4 and 99.8 %. Differences of 0.1 pp are 3–4 of the 3,500
  tuning predictions, far below the across-pair std (0.3–1.5 pp). Only dog-deer (seed 3) is hard
  (95.1–99.4 % for the best head per backbone; the other pairs are 98.4–100 %).
- The backbone matters more than the head: within a backbone the heads span 0.2–0.6 pp (except RBF
  SVM, which drops on DeiT-S and ConvNeXt-T), across backbones the best heads span 1.8 pp.
- Bigger and more pretraining data both help: ViT-B > ViT-S > ViT-Ti (99.77 > 99.46 > 99.09), and the
  same ViT-S/16 recipe pretrained on IN-21k beats its IN-1k-only twin by +0.2 pp (LR: 99.46 vs 99.23).
  ResNet-50 (IN-1k) is clearly last, 1.1 pp below the next backbone (ViT-Ti).
- The testbed head (AdamW, 10 epochs) is the worst or second-worst head on 4 of the 7 backbones: on
  the testbed's own ViT-Ti, LR/NCM on the same features beat it by +0.3/+0.4 pp, almost all of it on
  dog-deer (96.6 → 97.6/97.9). With 100 images and batch 32, 10 epochs are only 40 AdamW steps at
  lr 1e-3 (likely under-trained; tested in S2).
- `best`/`fast` by the selection rule on S1 alone: ViT-S + LR (99.46, 1.56 s; ViT-B + LR is 0.31 pp
  higher but 3.9× slower) and ViT-Ti + NCM (99.09, 0.58 s).

**Decision → S2:** tune heads on the top 3 backbones by best-head accuracy (ViT-B, ViT-S, ConvNeXt-T),
plus ViT-Ti (the testbed backbone: how far does a better head take it?).

## 2026-11-10 · S2: head hyperparameters and feature normalization (116 configs)

**Why:** S1 used each head's defaults; before comparing feature variants, give every head its fair
shot. Backbones: ViT-B, ViT-S, ConvNeXt-T (top 3 in S1) and ViT-Ti (the testbed backbone). Grids per
backbone: LR and linear-SVM C ∈ {0.001 … 100}; LR-CV (C chosen by 5-fold CV on the 100 training images
only); RBF-SVM C ∈ {0.1, 1, 10, 100}; testbed AdamW head epochs ∈ {10, 30, 100} × lr ∈ {1e-3, 1e-2};
normalization ∈ {none, L2, z-score, CL2N} for LR and NCM; kNN k ∈ {1, 3, 5, 9, 15}. Each knob's S1
default is not repeated (its S1 config is the default point of the curve).

| backbone | best config | acc (%) | S1 best | head tuning gain | testbed head: 10 → 100 epochs |
|---|---|---|---|---|---|
| ViT-B/16 | linear SVM, C = 0.001 | **99.83** ± 0.23 | 99.77 (LR) | +0.06 | 99.40 → 99.57 |
| ViT-S/16 | LR, no normalization | 99.51 ± 0.43 | 99.46 (LR) | +0.05 | 99.23 → 99.31 |
| ConvNeXt-T | LR, C = 100 | 99.49 ± 0.26 | 99.43 (lin. SVM) | +0.06 | 99.34 → 99.46 |
| ViT-Ti/16 | LR, C = 0.01 | 99.23 ± 0.38 | 99.09 (NCM) | +0.14 | 98.69 → 99.00 |

**Observations:**
- Tuning the head buys at most +0.14 pp (5 of 3,500 tuning predictions), within noise. Most grids are
  flat: on ViT-B every linear-SVM C from 0.01 to 100 gives the same 99.74 %; LR spans ≤ 0.2 pp over
  C ∈ [0.001, 100] on ViT-B, ViT-S and ConvNeXt-T. With 100 images in 384–768 dims the classes are
  linearly separable, so the regularization strength barely moves the boundary. Only the smallest
  backbone (ViT-Ti, 192 dims) prefers strong regularization: C = 0.01 → 99.23, C = 100 → 98.80.
  LR-CV never beats a fixed C.
- The testbed head is under-trained: 100 epochs instead of 10 (lr 1e-3) adds +0.08 to +0.31 pp on every
  backbone (ViT-Ti: 98.69 → 99.00), but even then it does not beat LR on the same features. A 10× larger
  learning rate (1e-2) is worse than 1e-3 at every epoch count on ViT-B and ViT-S.
- Normalization: for LR it changes ≤ 0.2 pp (z-score vs none vs L2 vs CL2N). For NCM, cosine geometry
  matters: z-scoring hurts NCM on ViT-B (99.43 vs 99.69 with L2), and CL2N is no better than plain L2.
- RBF-SVM never beats the linear heads; with C = 0.1 it underfits (ConvNeXt-T: 96.00). kNN prefers small
  k (k = 3 best or tied on 3 of 4 backbones; k = 15 is worst or tied-worst on every backbone).
- Timing: within a backbone every head trains in < 0.05 s, so time differences between configs of one
  backbone are run-to-run drift, not cost. S2 ran ViT-S and ViT-Ti ~25–30 % slower than S1 for identical
  work (inference 1.38 vs 1.08 s, 0.54 vs 0.41 s), while ViT-B and ConvNeXt-T stayed within 3 %; the
  small models are bound by CPU-side kernel launches, so laptop background load shows up there. The
  report's times come from the Colab T4 re-run.

**Decision → S3:** feature source on the top 2 backbones, each with its best S2 head: ViT-B + linear SVM
(C = 0.001), ViT-S + LR (no normalization). Both are ViTs, so: token pooling (CLS / mean of patch tokens
/ both) × number of last blocks concatenated (1, 2, 4), plus the full image instead of the checkpoint's
center crop (crop_pct = 1.0).

## 2026-11-10 · S3: feature source (18 configs)

**Why:** S1/S2 used each model's own pre-logits (CLS token of the last block). DINO's linear-eval recipe
concatenates the CLS tokens of the last 4 blocks (ViT-S) or adds the mean of the patch tokens (ViT-B),
and Head2Toe argues that intermediate layers carry information the last layer drops, most of all when
the target images differ from the pretraining images (as upsampled 32 px CIFAR images do). Also: the
checkpoints' preprocessing resizes to 248 px and center-crops 224 px (crop_pct 0.9), which cuts ~5 % of
each border; CIFAR objects fill the frame, so the whole image (crop_pct 1.0) may help. Bases: the best
S2 config of the top 2 backbones, ViT-B + linear SVM (C = 0.001) and ViT-S + LR (no normalization).

| feature source | ViT-B/16 | ViT-S/16 |
|---|---|---|
| CLS, last block (S2 base) | **99.83** ± 0.23 | 99.51 ± 0.43 |
| CLS, last 2 / 4 blocks | 99.77 / 99.71 | 99.54 / 99.51 |
| mean of patch tokens, last 1 / 2 / 4 blocks | 99.60 / 99.63 / 99.71 | 99.43 / 99.43 / 99.37 |
| CLS + patch mean, last 1 / 2 / 4 blocks | 99.69 / 99.74 / 99.74 | 99.51 / 99.51 / 99.49 |
| CLS, last block, whole image (crop_pct 1.0) | 99.77 ± 0.26 | **99.77** ± 0.17 |

**Observations:**
- Neither more blocks nor patch-token pooling helps at this ceiling: every variant is within −0.23 to
  +0.03 pp of the plain CLS token. Mean-pooled patch tokens are the weakest on both backbones (−0.1 to
  −0.2 pp); adding them to CLS recovers most of that, so the CLS token already carries what is needed for
  a 2-class problem. The intermediate-layer idea (Head2Toe) does not pay off here; it was proposed for
  harder, more varied VTAB tasks.
- Feature extraction cost is the same for every variant (one forward pass), so the choice is free in time.
- The whole image (no center crop) is the only change that moves ViT-S: +0.26 pp (99.51 → 99.77), mostly on
  dog-deer (98.71 → 99.43, 5 more of 700 images right). On ViT-B it is −0.06 (2 images). With 5 seeds and
  differences of a few images, this is suggestive, not established.

**Decision → S4:** keep CLS / last block. Flip augmentation (train / TTA / both) on three bases: ViT-B +
linear SVM (top config), ViT-S + LR on the whole image (best small config), ViT-Ti + LR C = 0.01 (best
ViT-Ti, the fast candidate). The crop is also tried on ViT-Ti. Feature concatenation of ViT-B with ViT-S
(the top 2) and with ConvNeXt-T (a different architecture family, 0.02 pp behind ViT-S in S2). fp16
inference on the three bases.

## 2026-11-10 · S4: flip augmentation, fp16, backbone concatenation (15 configs)

**Why:** mirrored images are a free, label-preserving way to double the 100 training images (and to
average two views at test time); in C1 flip-training was the biggest single gain. fp16 (autocast)
feature extraction is a pure speed knob: the backbone is frozen, so lower precision can only change
the features slightly. Concatenating two backbones' features tests whether they make different
mistakes. Bases: ViT-B + linear SVM (top config), ViT-S + LR on the whole image (best small config),
ViT-Ti + LR C = 0.01 (fast candidate).

| base | plain | flip-train | flip-TTA | both | fp16 | inference plain → fp16 (s / 700 img) |
|---|---|---|---|---|---|---|
| ViT-B + lin. SVM | 99.83 | 99.83 | 99.80 | 99.80 | **99.83** | 3.95 → 1.40 |
| ViT-S + LR, whole image | 99.77 | 99.77 | 99.77 | 99.77 | **99.77** | 1.33 → 0.51 |
| ViT-Ti + LR C = 0.01 | 99.23 | 99.17 | 99.20 | 99.26 | **99.23** | 0.53 → 0.27 |

Also: ViT-Ti on the whole image 99.17 (−0.06); ViT-B + ViT-S features 99.74, ViT-B + ConvNeXt-T 99.66
(both below ViT-B alone).

**Observations:**
- fp16 gives exactly the same per-seed accuracy on all three bases and makes inference 1.9–2.8× faster
  (more for bigger models: GPU tensor cores do the matrix products in half precision); training-image
  extraction drops the same way (ViT-B 0.58 → 0.18 s). It is the largest time saving of the sweep, at
  no accuracy cost. On ViT-B, loading the weights (1.2 s) now costs more than extracting the features.
- Flip augmentation does nothing at this ceiling (−0.06 to +0.03 pp), unlike C1 (+1.8 to +2.1 pp): the
  pretrained features are already close to invariant to a horizontal flip, so the mirrored copy adds
  almost no new information. Flip-TTA doubles inference time for nothing.
- Concatenating backbones is worse than the better backbone alone, and costs both backbones' time.
- The whole image helped ViT-S in S3 (+0.26) but not ViT-Ti (−0.06) or ViT-B (−0.06): the crop effect
  is within noise.

## 2026-11-10 · Selection (rule in `select.py`, fixed before any results; same rule as C1)

Over all 191 tuning configs (s1–s4), local times:
- top: `s4_vitb_fp16` (and `s2_vitb_linsvm_C0.001`, `s4_vitb_fliptrain`, same accuracy) 99.83 ± 0.23, 2.80 s
- **best** = `s4_vits_fp16`: ViT-S/16 (IN-21k) + LR (no normalization), whole image, fp16 —
  99.77 ± 0.17, 0.96 s. Within 0.5 pp of top (−0.06) and 2.9× faster; the next-fastest config within
  0.5 pp (`s1_vits_lr`, 99.46) takes 1.56 s.
- **fast** = `s4_vitti_fp16`: ViT-Ti/16 (the testbed checkpoint) + LR (C = 0.01), fp16 — 99.23 ± 0.38,
  0.47 s. The fastest config overall (next: 0.58 s), 0.54 pp below best (cutoff: best − 2 pp).
- **baseline** = `s1_vitti_adamw`, the testbed's C2 example, fixed in advance: 98.69 ± 1.10, 0.62 s.

Neither choice is a close call on time (fp16 roughly halves it), unlike C1's fast pick.

## 2026-11-10 · Report seeds 10–19 (`hw=local-3070`)

| config | acc (%) | train (s) = load + extract + head | inference (s / 700 img) | per image | trained params |
|---|---|---|---|---|---|
| best: ViT-S/16 + LR, whole image, fp16 | **98.69 ± 2.65** | 0.40 = 0.32 + 0.06 + 0.02 | 0.42 | 0.60 ms | 385 |
| fast: ViT-Ti/16 + LR (C = 0.01), fp16 | 97.33 ± 3.96 | 0.16 = 0.11 + 0.03 + 0.01 | 0.21 | 0.30 ms | 193 |
| baseline: testbed (ViT-Ti/16 + AdamW head, 10 epochs) | 96.04 ± 5.02 | 0.24 = 0.11 + 0.07 + 0.06 | 0.45 | 0.65 ms | 386 |

Per class pair (%): ship-bird 99.86 / 99.29 / 99.29 · horse-ship 99.86 / 99.57 / 98.57 · dog-ship 99.86 /
99.71 / 99.00 · **cat-dog 90.86 / 86.00 / 81.86** · cat-truck 99.86 / 98.86 / 98.71 · bird-frog 99.00 /
97.00 / 95.00 · frog-bird 98.43 / 95.86 / 94.29 · horse-bird 99.29 / 98.29 / 96.57 · horse-truck 99.86 /
99.43 / 98.14 · automobile-horse 100.00 / 99.29 / 99.00 (best / fast / baseline).

Paired over the 10 report seeds (same seeds = same class pairs):
- best − baseline = +2.64 ± 2.44 pp, best wins 10/10, Wilcoxon signed-rank p = 0.002
- best − fast = +1.36 ± 1.38 pp, best wins 10/10, p = 0.002
- fast − baseline = +1.29 pp, fast wins 9/10 (1 tie), p = 0.004
- vs Massimo's C1 best (hand-crafted features, `hw=colab-t4`, 85.09 %): C2 best − C1 best = +13.60 pp,
  10/10, p = 0.002 (from +6.3 pp on horse-ship to +26.4 pp on cat-dog).

**Observations:**
- The ranking from the tuning seeds holds on every report seed: best ≥ fast ≥ baseline on all 10 pairs
  (one tie). p = 0.002 is the smallest a two-sided Wilcoxon test can give with 10 pairs.
- The report seeds are harder than the tuning seeds (cat-dog, bird-frog, frog-bird), and that is where
  the configs separate: on cat-dog best is +9.0 pp over the testbed baseline and +4.9 pp over fast,
  while on the easy pairs all three are within ~1 pp. The near-ceiling tuning seeds hid most of this
  (tuning: best − fast = +0.54 pp; report: +1.36 pp).
- The std (2.7–5.0 pp) is variation across class pairs (driven by cat-dog), not uncertainty of the mean.
- Both best and fast are faster than the testbed baseline at inference (fp16) and train in under 0.5 s,
  most of which is loading the backbone from disk.
- Pretraining is worth +13.6 pp over the best method without it (C1), on every pair.

# Backbones and their pretraining data

None of these checkpoints was trained or fine-tuned on CIFAR-10 (C2 rule). Weights: the timm repos on
the Hugging Face hub, loaded with each checkpoint's own preprocessing (`timm.data.resolve_model_data_config`).

| key | timm checkpoint | params | pretraining data | model card |
|---|---|---|---|---|
| `vitti` | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (testbed) | 5.7M | ImageNet-21k, fine-tuned on ImageNet-1k (AugReg, Steiner et al. 2021) | https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k |
| `vits` | `vit_small_patch16_224.augreg_in21k_ft_in1k` | 22.1M | ImageNet-21k, fine-tuned on ImageNet-1k (AugReg) | https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k_ft_in1k |
| `vits1k` | `vit_small_patch16_224.augreg_in1k` | 22.1M | ImageNet-1k only (AugReg; same architecture and recipe as `vits`) | https://huggingface.co/timm/vit_small_patch16_224.augreg_in1k |
| `vitb` | `vit_base_patch16_224.augreg_in21k_ft_in1k` | 86.6M | ImageNet-21k, fine-tuned on ImageNet-1k (AugReg) | https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k_ft_in1k |
| `deits` | `deit_small_patch16_224.fb_in1k` | 22.1M | ImageNet-1k only (DeiT, Touvron et al. 2021) | https://huggingface.co/timm/deit_small_patch16_224.fb_in1k |
| `rn50` | `resnet50.tv_in1k` | 25.6M | ImageNet-1k only (original torchvision weights) | https://huggingface.co/timm/resnet50.tv_in1k |
| `cnxt` | `convnext_tiny.fb_in22k_ft_in1k` | 28.6M | ImageNet-22k, fine-tuned on ImageNet-1k (Liu et al. 2022) | https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k |

Params are the full models with their ImageNet classifier; the frozen feature extractors (classifier removed)
are slightly smaller.

# Source log (paper → what was taken from it)

| source | used for |
|---|---|
| Razavian, Azizpour, Sullivan & Carlsson 2014, *CNN features off-the-shelf: an astounding baseline for recognition*, CVPR Workshops | frozen pretrained features + a linear classifier (SVM) as a strong transfer baseline |
| Kornblith, Shlens & Le 2019, *Do better ImageNet models transfer better?*, CVPR | logistic regression on fixed features of many ImageNet backbones; the backbone sweep and comparing backbones by transfer accuracy |
| Dosovitskiy et al. 2021, *An image is worth 16x16 words: Transformers for image recognition at scale*, ICLR | ViT; the paper evaluates few-shot transfer on frozen features ("Few-shot accuracies are obtained by solving a regularized least-squares regression problem that maps the (frozen) representation of a subset of training images to {−1,1}^K target vectors", Sec. 4.1) |
| Steiner et al. 2022, *How to train your ViT? Data, augmentation, and regularization in vision transformers*, TMLR (arXiv 2021) | the AugReg checkpoints (incl. the testbed's ViT-Ti); more pretraining data (IN-21k vs IN-1k) transfers better, tested here with the `vits` / `vits1k` pair |
| Touvron et al. 2021, *Training data-efficient image transformers & distillation through attention*, ICML | DeiT-S (ViT-S trained on ImageNet-1k only) |
| He, Zhang, Ren & Sun 2016, *Deep residual learning for image recognition*, CVPR | ResNet-50 |
| Liu et al. 2022, *A ConvNet for the 2020s*, CVPR | ConvNeXt-T |
| Chen, Liu, Kira, Wang & Huang 2019, *A closer look at few-shot classification*, ICLR | a simple head on a frozen pretrained embedding is competitive with meta-learning |
| Tian, Wang, Krishnan, Tenenbaum & Isola 2020, *Rethinking few-shot image classification: a good embedding is all you need?*, ECCV | logistic regression on a frozen embedding as the few-shot method; L2-normalizing the features helps |
| Mensink, Verbeek, Perronnin & Csurka 2013, *Distance-based image classification: generalizing to new classes at near-zero cost*, IEEE TPAMI 35(11) | nearest class mean (NCM) classifier |
| Snell, Swersky & Zemel 2017, *Prototypical networks for few-shot learning*, NeurIPS | class prototypes (mean embedding per class) for few-shot classification |
| Wang, Chao, Weinberger & van der Maaten 2019, *SimpleShot: Revisiting nearest-neighbor classification for few-shot learning*, arXiv 1911.04623 | CL2N normalization: subtract a mean feature, then L2-normalize, before a nearest-neighbor / nearest-centroid classifier (they use the base-class mean; here the 100 training images' mean) |
| Caron et al. 2021, *Emerging properties in self-supervised vision transformers* (DINO), ICCV | cosine k-NN evaluation of frozen features; the linear-probe feature recipe: CLS tokens of the last n blocks, each after the final norm, concatenated (`eval_linear.py`: n = 4 for ViT-Small, n = 1 + average-pooled patch tokens for ViT-Base) → our `layers` and `pool = cls+mean` |
| Evci, Dumoulin, Larochelle & Mozer 2022, *Head2Toe: Utilizing intermediate representations for better transfer learning*, ICML | features from intermediate layers of a frozen backbone, not only the last; matches fine-tuning on VTAB-1k and helps most when the target data differs from the pretraining data (originality hook: multi-layer features) |
| Krizhevsky, Sutskever & Hinton 2012, *ImageNet classification with deep convolutional neural networks*, NeurIPS | horizontal-flip augmentation; averaging predictions over flipped copies at test time |
| Demšar 2006, *Statistical comparisons of classifiers over multiple data sets*, JMLR 7 | Wilcoxon signed-rank test for paired comparisons of two classifiers over many problem instances |
| Wightman 2019, *PyTorch Image Models (timm)*, GitHub | all checkpoints and their preprocessing configs |
| Pedregosa et al. 2011, *Scikit-learn: Machine learning in Python*, JMLR 12 | LR, LR-CV, SVMs, kNN, StandardScaler, grouped cross-validation |

Not from the literature (our choices, reported as such): the staged sweep and the best/fast selection rule
(shared with C1), the timing policy (backbone loading counted inside `fit`), fp16 inference, and
`crop_pct = 1.0`.
