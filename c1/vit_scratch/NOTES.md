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

### 2026-11-20: T4 re-run of S0–S2 (tuning seeds), and selection
**Why:** every reported time has to come from a Colab T4 (team rule), and a second full run of
every config also shows how much a ViT result moves when nothing changes but the hardware.
All 20 configs were re-run on seeds 0–4 with `hw=colab-t4`
(`notebooks/ammar/ammar_colab_t4.ipynb`, cell 6).

**Run-to-run noise is large for the ViT.** The same config and seed gives a different accuracy on
CPU and on T4 (GPU kernels are not bit-identical, and 100 epochs on 100 images amplify the difference):
mean |T4 − CPU| = 1.7 pp per (config, seed), max 16.7 pp. The worst case is the *unaugmented* baseline,
seed 1 (bird-truck): 78.4% on CPU, 61.7% on T4, so the baseline's T4 mean drops to 73.8%. The
augmented configs move much less (flip + crop: within 1 pp on every seed). So augmentation also
makes training **more stable**, not just more accurate, and any difference below about 2 pp on 5
seeds should be treated as noise.

T4, tuning seeds, top configs vs flip + crop (paired):

| config | T4 acc (%) | Δ vs flip+crop (wins) | CPU + T4 pooled: Δ (wins / 10) | T4 fit (s) | T4 predict 700 (s) |
|---|---|---|---|---|---|
| s2_conv_seq | 82.7 | +3.69 (5/5) | +2.87 (8/10) | 3.3 | 0.04 |
| **s2_conv** | 82.2 | +3.14 (4/5) | **+2.91 (8/10)** | 3.2 | 0.04 |
| s2_spt_lsa | 80.5 | +1.40 (3/5) | | 4.0 | 0.04 |
| s2_spt | 80.1 | +1.09 (3/5) | +1.36 (8/10) | 3.1 | 0.03 |
| s1_crop | 79.7 | +0.60 (4/5) | | 2.9 | 0.03 |
| s1_flip_crop | 79.1 | – | – | 3.0 | 0.03 |
| baseline | 73.8 | | | 2.7 | 0.03 |

**What holds on both runs:** the conv tokenizer is the clear winner (+2.9 pp over flip + crop, better
in 8 of 10 paired runs); SPT helps less (+1.4 pp); sequence pooling on top of the conv tokenizer is a
tie (conv_seq is ahead on T4, behind on CPU). The S2 claim that LSA hurts does **not** hold on T4
(LSA alone +0.06, SPT + LSA ahead of SPT), so "LSA makes no reliable difference here" is the safe
conclusion.

**Time on the T4:** every config trains in 2.4–8.7 s and predicts 700 images in under 0.15 s. On the
GPU, model size barely changes the time (dim 64 is no faster than dim 128), so the CPU "fast"
candidate (`s1_dim64`) has no speed advantage on the reporting hardware.

**Selection (for the report seeds)** (superseded on 2026-11-21, see below):
- **best = `s2_conv`** (flip + crop, conv tokenizer, class token): top on both runs, simpler than
  conv_seq at equal accuracy.
- **No separate fast config needed**: best trains in about 3 s on a T4, and the fastest config
  (`s1_depth2`, 2.4 s) is 6 pp less accurate. For reference rows on the report seeds, also run
  `s1_flip_crop` (plain ViT + augmentation) and `baseline` (testbed), so the report can show the
  gain from each step on the same seeds.

### 2026-11-20: report seeds 10–19 (T4), final numbers
Configs fixed before this run (selection above). `hw=colab-t4`, notebook cell 8.

| pair | baseline | flip + crop | **s2_conv** |
|---|---|---|---|
| 10 ship-bird | 78.4 | 82.9 | 85.6 |
| 11 horse-ship | 79.9 | 85.3 | 87.4 |
| 12 dog-ship | 85.9 | 85.3 | 88.0 |
| 13 cat-dog | 51.9 | 54.3 | 58.9 |
| 14 cat-truck | 72.4 | 72.7 | 81.3 |
| 15 bird-frog | 67.6 | 66.3 | 71.4 |
| 16 frog-bird | 61.4 | 62.9 | 71.9 |
| 17 horse-bird | 65.0 | 66.9 | 74.3 |
| 18 horse-truck | 69.6 | 71.1 | 83.1 |
| 19 automobile-horse | 75.0 | 74.9 | 87.7 |
| **mean ± std** | 70.70 ± 9.38 | 72.24 ± 9.68 | **78.96 ± 9.11** |
| train / predict (s) | 2.95 / 0.026 | 3.07 / 0.026 | 3.32 / 0.037 |

- **s2_conv beats the testbed baseline on all 10 report seeds: +8.26 pp** (Wilcoxon p = 0.002), and
  beats plain augmentation on all 10: +6.71 pp (p = 0.002). The conv tokenizer's gain is much larger
  here than on the tuning seeds (+2.9 pp), mostly on pairs of two animals or animal vs vehicle with
  similar colours (horse-truck +12.0, automobile-horse +12.8 vs flip + crop).
- The report seeds are harder than the tuning seeds (baseline 70.7 vs 73.8–77.3%): they include
  cat-dog (58.9% at best, barely above chance), bird-frog and frog-bird.
- Augmentation alone helps less here (+1.5 pp over the baseline) than on the tuning seeds.
- For comparison, C1 classical (Massimo) on the same seeds: 85.09 ± 8.22%. The ViT closes about half
  of the original gap (from 14.4 pp to 6.1 pp) but stays behind hand-crafted features.

### 2026-11-21: selection redone with the team rule, and S3 (review on PR #4)
**Why:** the selection above was by judgement ("top on both runs"), not by the team rule
(`common/sweep.py select`): best = top mean accuracy, but a config within 0.5 pp of the top wins if it
is faster (train + inference); fast = fastest config within 2 pp of best. The review also pointed out
two gaps in the search: learning rate and weight decay were never varied, and longer training was only
tried *without* augmentation (S1), where extra epochs mostly memorize the 100 images.

`python -m c1.vit_scratch.select --hw colab-t4` on S0–S2: best = `s2_conv_seq` (82.74%), fast =
`s2_conv` (82.20%, 0.54 pp behind, just outside the tie margin, and faster). The two are within noise
of each other (the order flips on the CPU run), but the rule is fixed in advance, so it decides.

**S3** changes one setting at a time on both S2 winners (`s3_conv_*`, `s3_convseq_*`): lr 1e-4 / 1e-3
(base 3e-4), weight decay 0.05 / 0.1 (base 0.01), 200 / 300 epochs (base 100). 12 configs, tuning seeds,
T4 only (notebook cell 6). Then the rule is applied again over S0–S3 and the picks run on the report
seeds (cell 8).

S3 results (T4, tuning seeds, paired vs the S2 config it changes):

| config | acc (%) | Δ vs base (wins / 5) | T4 train + predict (s) |
|---|---|---|---|
| s3_convseq_ep200 | 83.71 | +0.97 (4/5) | 7.2 |
| s3_convseq_ep300 | 83.54 | +0.80 (3/5) | 11.0 |
| **s3_convseq_wd0.05** | 83.23 | +0.49 (3/5) | 3.7 |
| s3_convseq_lr1e-4 | 83.17 | +0.43 (4/5) | 3.6 |
| s3_conv_ep300 | 82.97 | +0.77 (2/5) | 10.6 |
| s3_conv_ep200 | 82.86 | +0.66 (2/5) | 7.0 |
| s3_convseq_wd0.1 | 82.86 | +0.11 (4/5) | 3.6 |
| s3_convseq_lr1e-3 | 81.94 | −0.80 (3/5) | 3.7 |
| s3_conv_lr1e-4 / lr1e-3 | 81.46 / 81.37 | −0.74 / −0.83 (2/5) | 4.1 / 3.5 |
| s3_conv_wd0.1 / wd0.05 | 80.43 / 80.31 | −1.77 / −1.89 (1/5) | 3.7 / 3.5 |

- **Longer training now helps a little** (+0.7 to +1.0 pp with 200–300 epochs), unlike in S1 without
  augmentation (where it did nothing), but it costs 2–3× the time.
- **Learning rate and weight decay:** the testbed values (3e-4, 0.01) are already about right. No
  change gains more than +0.5 pp, and lr 1e-3 or larger weight decay on the class-token model hurt.
- Every S3 difference is below 1 pp, inside the ~2 pp run-to-run noise measured above.

**Selection over S0–S3 (team rule):** top = `s3_convseq_ep200` (83.71%); `s3_convseq_wd0.05`
(83.23%) is within 0.5 pp and twice as fast, so **best = `s3_convseq_wd0.05`**. **fast = `s2_conv`**
(82.20%, fastest config within 2 pp of best). Both were run on the report seeds (notebook cell 8).

### 2026-11-21: report seeds 10–19 (T4) for the new picks
Picks fixed by the rule before this run. `s2_conv` rows are the ones from 2026-11-20.

| pair | baseline | flip + crop | **fast: s2_conv** | **best: s3_convseq_wd0.05** |
|---|---|---|---|---|
| 10 ship-bird | 78.4 | 82.9 | 85.6 | 80.6 |
| 11 horse-ship | 79.9 | 85.3 | 87.4 | 89.7 |
| 12 dog-ship | 85.9 | 85.3 | 88.0 | 88.4 |
| 13 cat-dog | 51.9 | 54.3 | 58.9 | 57.0 |
| 14 cat-truck | 72.4 | 72.7 | 81.3 | 83.3 |
| 15 bird-frog | 67.6 | 66.3 | 71.4 | 67.4 |
| 16 frog-bird | 61.4 | 62.9 | 71.9 | 70.0 |
| 17 horse-bird | 65.0 | 66.9 | 74.3 | 73.9 |
| 18 horse-truck | 69.6 | 71.1 | 83.1 | 83.1 |
| 19 automobile-horse | 75.0 | 74.9 | 87.7 | 85.4 |
| **mean ± std** | 70.70 ± 9.38 | 72.24 ± 9.68 | **78.96 ± 9.11** | **77.89 ± 9.98** |
| train / predict (s) | 2.95 / 0.026 | 3.07 / 0.026 | 3.32 / 0.037 | 3.82 / 0.035 |

- **best vs fast: −1.07 pp, 3 wins / 1 tie / 6 losses, p = 0.26.** A tie: the 0.5 pp tuning-seed
  lead of best did not carry over, as expected for a difference that small (see the noise above).
- **Both beat the testbed clearly:** best +7.19 pp (9/10, p = 0.004), fast +8.26 pp (10/10, p = 0.002);
  best vs flip + crop +5.64 pp (9/10, p = 0.006).
- So the conv tokenizer is what matters. Sequence pooling, weight decay and the other S3 changes are
  all within noise.

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
