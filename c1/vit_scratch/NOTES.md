# C1 ViT from scratch: experiment log + source log

All numbers: tuning seeds 0–4 (bird-ship, bird-truck, deer-automobile, dog-deer, cat-ship),
`hw=local-cpu` (laptop CPU, 16 threads). Times are for orientation only; the times in the report
will come from the Colab T4 runs. "Δ" = paired per-seed difference vs `baseline`, in percentage points
(pp); "wins" = seeds where the config beat the baseline.

## Experiment log

### 2026-11-14: S0, testbed baseline
**Why:** reproduce the course testbed's small ViT inside the team harness before changing anything.
Settings are copied unchanged (patch 4, dim 128, depth 3, 4 heads, MLP ratio 2, dropout 0.1, AdamW
lr 3e-4 / wd 0.01, batch 32, 100 epochs). Added one variant with random horizontal flips.

| config | acc (%) | Δ (pp) | wins | fit (s) |
|---|---|---|---|---|
| baseline | 77.3 ± 7.0 | – | – | 24.7 |
| baseline_flip | 77.8 ± 7.0 | +0.54 | 3/5 | 25.7 |

**Result:** baseline 77.3%. Dog-deer is by far the hardest pair (63.9%), deer-automobile the easiest
(83.3%). Flip alone gives a gain within noise. For reference, Massimo's C1 classical method gets about
85% on the same tuning seeds, so the ViT starts about 8 pp behind.

### 2026-11-14: S1, one knob at a time
**Why:** before adding new techniques, find out which basic knob the baseline is most sensitive to:
model size (depth, width), token size (patch), training length, and a stronger augmentation (random
crop: reflect-pad 4 px, then take a random 32 × 32 window, i.e. random shifts).

| config | change | acc (%) | Δ (pp) | wins | fit (s) |
|---|---|---|---|---|---|
| s1_flip_crop | flip + random crop (pad 4) | **79.3 ± 6.4** | **+2.06** | 4/5 | 28.4 |
| s1_crop | random crop (pad 4) | 79.0 ± 6.2 | +1.71 | 4/5 | 28.6 |
| s1_depth4 | depth 3 → 4 | 78.1 ± 6.1 | +0.77 | 4/5 | 38.3 |
| s1_dim64 | dim 128 → 64 | 77.5 ± 6.1 | +0.23 | 2/5 | 14.2 |
| s1_ep300 | 100 → 300 epochs | 77.4 ± 7.4 | +0.09 | 2/5 | 75.1 |
| s1_patch2 | patch 4 → 2 (256 tokens) | 77.3 ± 7.0 | +0.00 | 2/5 | 117.7 |
| s1_dim192 | dim 128 → 192 | 77.1 ± 7.0 | −0.23 | 2/5 | 44.9 |
| s1_ep200 | 100 → 200 epochs | 76.7 ± 6.7 | −0.54 | 1/5 | 49.9 |
| s1_depth2 | depth 3 → 2 | 76.1 ± 7.5 | −1.20 | 1/5 | 20.5 |
| s1_depth6 | depth 3 → 6 | 74.4 ± 6.0 | −2.89 | 1/5 | 57.8 |
| s1_patch8 | patch 4 → 8 (16 tokens) | 73.8 ± 5.5 | −3.49 | 0/5 | 9.5 |

**Observations:**
- **Augmentation is the only knob with a clear gain.** Random crop helps on 4/5 seeds (+1.7 pp); adding
  flip on top gives +2.1 pp. The one seed where both lose is deer-automobile, the easiest pair.
- **Model size barely matters, within a range.** Width 64–192 and depth 2–4 all land within about
  ±1 pp of the baseline. Too deep (6) or too coarse (patch 8, only 16 tokens per image) clearly hurts.
  Width 64 matches the baseline at about half the training time, so it is a "fast" candidate.
- **Training longer does not help** (200/300 epochs ≈ 100). The training loss is presumably already near
  zero at 100 epochs (the model has memorized 100 images), so extra epochs add nothing. Worth logging
  training loss to confirm.
- Patch 2 scores exactly the same as the baseline on seeds 0 and 1 (551/700 and 549/700 correct). This
  was checked: the configs differ (432,642 vs 412,674 parameters), so it is a coincidence, not a bug.
- All differences except augmentation are about the size of the seed-to-seed noise, with 5 seeds.

**Plan for S2 (written before S2 ran):** keep flip + crop as the new base. Then (a) stronger augmentation / regularization (more
padding, label smoothing, mixup / CutMix once Ryan's `common/augment.py` lands), (b) the small-data
ViT techniques: Shifted Patch Tokenization + Locality Self-Attention (Lee et al. 2021) and a
convolutional tokenizer (Hassani et al. 2021). Combine with depth 4 only if it still helps on top of
augmentation.

### 2026-11-15: S2, small-data ViT techniques on top of flip + crop
**Why:** S1 showed that augmentation was the only clear gain, so flip + crop (`s1_flip_crop`, 79.3%) is
the new base. The ViT's weakness with 100 images is its lack of locality bias; S2 tests the techniques
designed to add it back:
- **SPT** (Shifted Patch Tokenization, Lee et al. 2021): each token also sees 4 half-patch diagonal
  shifts of the image, so it covers its neighbours' pixels.
- **LSA** (Locality Self-Attention, same paper): a token may not attend to itself, and the softmax
  temperature is learned.
- **Conv tokenizer** (CCT, Hassani et al. 2021): 2 × [conv 3×3 → ReLU → max-pool /2] instead of the
  linear patch embedding, giving the same 8 × 8 = 64 tokens.
- **Sequence pooling** (CCT): learned attention-weighted average of all tokens instead of a class token.
- depth 4 again, since it was the best size knob in S1.

Implementation: `layers.py`. All are drop-in replacements inside the timm ViT; the default config still
reproduces the S0 baseline exactly (checked: seed 0 gives 78.71% before and after the change).

Δ is shown against both the testbed baseline and the new base (flip + crop):

| config | acc (%) | Δ vs baseline | Δ vs flip+crop (wins) | fit (s) | params |
|---|---|---|---|---|---|
| **s2_conv** | **82.0 ± 7.7** | +4.74 (5/5) | **+2.69 (4/5)** | 38.3 | 481,858 |
| s2_conv_seq | 81.4 ± 6.6 | +4.11 (5/5) | +2.06 (3/5) | 37.5 | 481,731 |
| s2_spt | 81.0 ± 5.5 | +3.69 (4/5) | +1.63 (**5/5**) | 35.1 | 437,730 |
| s2_depth4 | 79.3 ± 6.1 | +2.00 (4/5) | −0.06 (3/5) | 42.0 | 545,154 |
| s2_lsa | 78.9 ± 6.0 | +1.63 (4/5) | −0.43 (2/5) | 36.2 | 412,677 |
| s2_seq | 78.7 ± 6.2 | +1.37 (4/5) | −0.69 (0/5) | 30.1 | 412,547 |
| s2_spt_lsa | 78.6 ± 6.7 | +1.29 (3/5) | −0.77 (1/5) | 34.3 | 437,733 |

Per seed (%):

| config | bird-ship | bird-truck | deer-auto | dog-deer | cat-ship |
|---|---|---|---|---|---|
| s1_flip_crop | 81.3 | 82.1 | 81.4 | 66.9 | 85.0 |
| s2_conv | 84.1 | 86.6 | 87.4 | 66.7 | 85.3 |
| s2_conv_seq | 84.6 | 81.7 | 88.7 | 69.1 | 82.9 |
| s2_spt | 84.4 | 82.4 | 82.6 | 70.1 | 85.3 |

(Fit times in S2 are a little inflated: unit tests were running on the same CPU for part of the sweep.)

**Observations:**
- **Better tokenizers help; LSA does not.** Both changes to how the image becomes tokens beat the
  augmentation base: the conv tokenizer by +2.7 pp (best mean, 82.0%), SPT by +1.6 pp but on every
  seed (5/5) and with the smallest spread (± 5.5). Both give each token information from neighbouring
  pixels, which is exactly the locality bias the plain ViT lacks.
- **LSA hurts slightly here (−0.4 pp), and SPT + LSA is worse than SPT alone (−2.4 pp vs SPT).** Lee et
  al. report the two working together on larger "small" datasets (tens of thousands of images); with
  100 images and only 3 blocks, forbidding self-attention may just remove useful information. Worth one
  more check with depth 4 before dropping it.
- **Sequence pooling does not help** on its own (0/5 vs flip + crop) or with the conv tokenizer
  (81.4 vs 82.0).
- **Dog-deer stays at about 67–70% for everything.** The gains come from the other pairs. Dog-deer is
  where the ViT is still far behind Massimo's classical features.
- **depth 4 no longer helps** once augmentation is on (−0.06 pp), so the S1 gain was probably noise.
- Gap to C1 classical (about 85% on tuning seeds): from 8 pp (baseline) down to about 3 pp (conv).

**Next (S3):** build on `s2_conv` and `s2_spt`: training length with augmentation (more epochs may
help now that augmentation stops memorizing), learning rate, weight decay, label smoothing, and
stronger augmentation from `common/augment.py` once it lands. Candidates so far: best = `s2_conv`,
fast = `s1_dim64` (77.5%, 14 s) or `s2_spt` if the time gap matters less.

## Source log

| source | idea taken |
|---|---|
| Course testbed notebook | baseline ViT architecture and training settings |
| Dosovitskiy et al. 2021 (ViT) | patch tokens + Transformer encoder; ViTs need large data |
| Touvron et al. 2021 (DeiT) | strong augmentation + regularization make ViTs trainable on less data |
| Steiner et al. 2021 (AugReg) | augmentation / regularization matter more as data shrinks |
| Lee, Lee & Song 2021 | Shifted Patch Tokenization + Locality Self-Attention (S2) |
| Hassani et al. 2021 (CCT) | convolutional tokenizer + sequence pooling (S2) |
| He et al. 2016 / standard CIFAR practice | random crop with 4 px padding + horizontal flip |
