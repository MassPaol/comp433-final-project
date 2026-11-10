# C2 frozen features: experiment log

All numbers below: tuning seeds 0–4 unless stated, `hw=local-3070` (RTX 3070 Laptop GPU). Accuracy =
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
- The backbone matters more than the head: within a backbone the heads span 0.3–0.7 pp (except RBF
  SVM, which drops on DeiT-S and ConvNeXt-T), across backbones the best heads span 1.8 pp.
- Bigger and more pretraining data both help: ViT-B > ViT-S > ViT-Ti (99.77 > 99.46 > 99.09), and the
  same ViT-S/16 recipe pretrained on IN-21k beats its IN-1k-only twin by +0.2 pp (LR: 99.46 vs 99.23).
  ResNet-50 (IN-1k) is clearly last, 1.3 pp below the next backbone.
- The testbed head (AdamW, 10 epochs) is the worst or second-worst head on 4 of the 7 backbones: on
  the testbed's own ViT-Ti, LR/NCM on the same features beat it by +0.3/+0.4 pp, almost all of it on
  dog-deer (96.6 → 97.6/97.9). With 100 images and batch 32, 10 epochs are only 40 AdamW steps at
  lr 1e-3 (likely under-trained; tested in S2).
- `best`/`fast` by the selection rule on S1 alone: ViT-S + LR (99.46, 1.56 s; ViT-B + LR is 0.31 pp
  higher but 3.9× slower) and ViT-Ti + NCM (99.09, 0.58 s).

**Decision → S2:** tune heads on the top 3 backbones by best-head accuracy (ViT-B, ViT-S, ConvNeXt-T),
plus ViT-Ti (the testbed backbone: how far does a better head take it?).
