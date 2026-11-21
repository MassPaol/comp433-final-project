# C2 parameter-efficient fine-tuning: experiment log + source log

All numbers: tuning seeds 0–4 (bird-ship, bird-truck, deer-automobile, dog-deer, cat-ship),
`hw=local-cpu` (laptop CPU, 16 threads). Times are for orientation only; the times in the report
will come from the Colab T4 runs. "Δ" = paired per-seed difference vs `head`, in percentage points (pp).
Backbone: `vit_tiny_patch16_224.augreg_in21k_ft_in1k` (ImageNet-21k → ImageNet-1k, no CIFAR-10),
checkpoint preprocessing (bicubic 32 → 248, center crop 224, mean = std = 0.5).

## Experiment log

### 2026-11-15: P0, first numbers (head control + BitFit learning rate)
**Why:** `head` is the testbed's C2 example (frozen backbone, new linear head, AdamW lr 1e-3, wd 0.01,
10 epochs, batch 32); every PEFT method is compared against it. BitFit adds every backbone bias term
(26,114 trainable vs 386). Biases are usually trained with a larger learning rate than full
fine-tuning, so three values were tried, everything else as in `head`. Biases get no weight decay.

| config | trainable | acc (%) | Δ (pp) | wins / ties | fit (s) | predict 700 (s) |
|---|---|---|---|---|---|---|
| head | 386 | 98.74 ± 0.84 | – | – | 11.6 | 9.6 |
| bitfit_lr1e-3 | 26,114 | **98.91 ± 1.05** | +0.17 | 3/5, 1 tie | 32.6 | 10.0 |
| bitfit_lr3e-3 | 26,114 | 98.37 ± 1.05 | −0.37 | 0/5, 2 ties | 31.4 | 9.3 |
| bitfit_lr1e-2 | 26,114 | 91.63 ± 6.41 | −7.11 | 0/5 | 30.7 | 9.2 |

Per seed (%):

| config | bird-ship | bird-truck | deer-auto | dog-deer | cat-ship |
|---|---|---|---|---|---|
| head | 98.6 | 99.0 | 99.9 | 97.3 | 99.0 |
| bitfit_lr1e-3 | 99.3 | 99.3 | 99.9 | 96.9 | 99.3 |
| bitfit_lr3e-3 | 98.6 | 98.4 | 99.9 | 96.6 | 98.4 |
| bitfit_lr1e-2 | 92.9 | 86.9 | 99.1 | 82.0 | 97.3 |

**Observations:**
- **The pretrained head-only baseline is already at 98.7%**, so there is very little room left on these
  seeds. BitFit at lr 1e-3 is +0.17 pp: a few images, within noise.
- **BitFit is sensitive to the learning rate.** 1e-2 is clearly too large (−7 pp, unstable: 82% on
  dog-deer): changing the biases too fast damages the pretrained features. 3e-3 is already slightly
  worse than the head alone. The best value is at the low end of the range, so lower values (3e-4)
  still need to be tried.
- **Cost:** BitFit trains about 3× slower than the head alone (32.6 vs 11.6 s on CPU), because gradients
  must flow back through the whole backbone even though only biases change. Inference time is the same
  (same network, same size).
- **The tuning seeds are mostly easy pairs.** Only dog-deer is hard. On the report seeds, Massimo's
  head baseline drops to 81.9% on cat-dog (seed 13), so the methods may separate more there. The
  tuning seeds cannot be changed (team rule), so decisions should look at dog-deer and at paired
  wins, not just the mean.

**Plan for P1 (written before P1 ran):** BitFit lr 3e-4 and 1e-3 with more epochs and random flips;
then LoRA on the attention qkv (rank 2 / 4 / 8) and Visual Prompt Tuning (shallow vs deep). Possibly
a `crop_pct = 1.0` variant, labelled as a deviation from the checkpoint preprocessing (Massimo found
the center crop cuts about 5% of each border of the upscaled image).

### 2026-11-15: P1, tune BitFit, add LoRA and VPT
**Why:** P0 left BitFit within noise of the head and showed the learning rate matters, so P1 goes
lower / longer for BitFit and adds the other two PEFT families, so the three methods adapt three
different places: biases (BitFit), attention weight matrices (LoRA, `layers.LoRALinear` on every
block's qkv, B initialised to 0 so training starts exactly from the pretrained model), and the input
sequence (VPT, `layers.PromptedViT`: 10 learned tokens inserted after the class token; shallow = only
before block 1, deep = a new set before every block). A toy check before the run showed deep VPT failing
to learn at lr 1e-2, so VPT uses 1e-3 / 3e-3.

| config | trainable | acc (%) | Δ vs head (pp) | wins / ties | median fit (s) |
|---|---|---|---|---|---|
| **lora_r2** | 18,818 | **99.23 ± 0.71** | **+0.49** | **5/5** | 34.4 |
| **vpt_shallow10_lr3e-3** | **2,306** | 99.20 ± 0.66 | +0.46 | 4/5 | 29.7 |
| lora_r4 | 37,250 | 99.17 ± 0.82 | +0.43 | 4/5, 1 tie | 34.4 |
| lora_r8 | 74,114 | 99.06 ± 0.74 | +0.31 | 3/5, 2 ties | 33.9 |
| vpt_shallow10_lr1e-3 | 2,306 | 99.03 ± 0.65 | +0.29 | 4/5, 1 tie | 30.1 |
| bitfit_lr1e-3_flip | 26,114 | 98.94 ± 1.06 | +0.20 | 3/5, 1 tie | 33.2 * |
| bitfit_lr1e-3_ep20 | 26,114 | 98.91 ± 1.00 | +0.17 | 2/5, 2 ties | 62.6 |
| bitfit_lr1e-3 (P0) | 26,114 | 98.91 ± 1.05 | +0.17 | 3/5, 1 tie | 32.6 |
| bitfit_lr3e-4_ep20 | 26,114 | 98.83 ± 0.89 | +0.09 | 2/5 | 61.8 |
| lora_r4_lr3e-3 | 37,250 | 98.80 ± 0.72 | +0.06 | 2/5, 2 ties | 32.8 |
| head (P0, control) | 386 | 98.74 ± 0.84 | – | – | 11.6 |
| vpt_deep10_lr3e-3 | 23,426 | 98.74 ± 0.88 | −0.00 | 4/5 | 29.0 |
| bitfit_lr3e-4 | 26,114 | 98.40 ± 1.05 | −0.34 | 1/5 | 34.1 |
| vpt_deep10_lr1e-3 | 23,426 | 98.23 ± 1.31 | −0.51 | 1/5 | 29.1 |

\* `bitfit_lr1e-3_flip`, seed 1: the logged `train_time_s` is 47,873 s because the laptop went to sleep
during that fit. Its accuracy is valid (sleep pauses training, it doesn't change it), but its time is
not, so the table shows the median of the 5 seeds (33.2 s). The row is kept in `results/ammar.csv` as
logged; the T4 re-run replaces it for the report. The figure uses median times for the same reason.

Dog-deer (seed 3), the hardest tuning pair: head 97.3, lora_r2 98.0, vpt_shallow10_lr3e-3 98.1,
bitfit_lr1e-3 96.9 (%).

**Observations:**
- **LoRA and shallow VPT beat the head-only control; BitFit doesn't, clearly.** LoRA rank 2 wins on all
  5 seeds (+0.49 pp); shallow VPT is +0.46 pp with only **2,306** trainable parameters, 8× fewer than
  LoRA r2 and 11× fewer than BitFit. The gains are small in absolute terms because the head is already
  at 98.7%: +0.5 pp is about 3–4 more correct images out of 700 per seed. Report seeds include cat-dog,
  where the head drops to about 82% (Massimo's report-seed numbers), so the gap may be larger there.
- **More trainable parameters do not help.** LoRA r2 > r4 > r8, and deep VPT (23k params) is worse than
  shallow VPT (2.3k). With 100 images, the smallest adaptation that changes the features at all works
  best: more capacity just fits the training images.
- **Learning rate matters more than the method.** BitFit's best stays lr 1e-3 (lower 3e-4 is worse,
  more epochs or flips add nothing). LoRA prefers 1e-3 (3e-3 is worse); shallow VPT prefers 3e-3 (prompts
  are new random tokens, so a larger step is fine); deep VPT is the most sensitive.
- **Cost:** every PEFT method costs about 3× the head's training time (about 30 s vs 12 s on CPU),
  because the backward pass goes through the whole backbone even though only a few parameters change.
  Inference time is the same for all (9–11 s for 700 images on CPU); LoRA adds a few small matrix
  products, which could be merged into the weights after training to remove that cost.

**Candidates:** best = `lora_r2` (highest mean, 5/5 wins) or `vpt_shallow10_lr3e-3` (equal within noise,
far fewer parameters). fast = `head` (the control, 3× faster to train). Decide after the T4 re-run.

**Next (P2), optional:** VPT with fewer / more prompts (5, 20) at lr 3e-3; LoRA r2 on qkv + proj or on
the last blocks only; the `crop_pct = 1.0` deviation. Then the chosen configs on the report seeds.

### 2026-11-20: T4 re-run of P0–P1 (tuning seeds), and selection
All 16 configs re-run on seeds 0–4 with `hw=colab-t4`. **Accuracies are identical to the CPU run**
for every config and seed (unlike C1: a frozen pretrained backbone with few trainable parameters and
10 epochs is far less sensitive to tiny numeric differences). Only the times change. The laptop-sleep
time in `bitfit_lr1e-3_flip` is replaced by a clean T4 measurement.

| config | trainable | acc (%) | T4 fit (s, median) | T4 predict 700 (s) |
|---|---|---|---|---|
| lora_r2 | 18,818 | 99.23 | 3.43 | 1.67 |
| vpt_shallow10_lr3e-3 | 2,306 | 99.20 | 3.27 | 1.61 |
| bitfit_lr1e-3 | 26,114 | 98.91 | 3.29 | 1.57 |
| head | 386 | 98.74 | 1.52 | 1.65 |

PEFT costs about 2.2× the head's training time on the T4 (3.3–3.4 s vs 1.5 s); inference is the same
for all (about 1.6 s per 700 images, 2.3 ms/image, dominated by the 224 px forward pass and
preprocessing).

**Selection (for the report seeds)** (superseded on 2026-11-21, see below: the team rule picks `head`):
- **best = `lora_r2`**: highest mean, wins 5/5 vs head.
- **fast = `head`**: 2.2× faster to train, −0.5 pp.
- also `vpt_shallow10_lr3e-3`: equal to LoRA within noise with 8× fewer trainable parameters; worth a
  report-seed row because it is the most parameter-efficient result.

### 2026-11-20: report seeds 10–19 (T4), final numbers
Configs fixed before this run (selection above). `hw=colab-t4`, notebook cell 8.

| pair | head | **lora_r2** | vpt_shallow10 |
|---|---|---|---|
| 10 ship-bird | 99.4 | 99.1 | 99.6 |
| 11 horse-ship | 98.6 | 99.3 | 99.6 |
| 12 dog-ship | 99.4 | 99.7 | 99.4 |
| 13 cat-dog | 85.1 | 85.6 | 84.4 |
| 14 cat-truck | 98.6 | 99.1 | 99.1 |
| 15 bird-frog | 96.0 | 96.9 | 96.7 |
| 16 frog-bird | 95.3 | 96.1 | 95.9 |
| 17 horse-bird | 96.0 | 97.1 | 98.0 |
| 18 horse-truck | 98.3 | 98.7 | 99.3 |
| 19 automobile-horse | 98.9 | 99.7 | 99.7 |
| **mean ± std** | 96.56 ± 4.07 | **97.14 ± 4.04** | 97.17 ± 4.43 |
| trainable params | 386 | 18,818 | 2,306 |
| train / predict (s) | 1.51 / 1.74 | 3.42 / 1.74 | 3.32 / 1.69 |

- **LoRA r2 beats the head-only control on 9/10 report seeds: +0.59 pp** (Wilcoxon p = 0.006). The
  gain is small but consistent, as on the tuning seeds.
- **LoRA and shallow VPT tie** (−0.03 pp, p = 0.77), with VPT training 8× fewer parameters (2,306,
  0.04% of the backbone).
- Cat-dog is the hard pair for every method (84–86%) and no PEFT variant fixes it; the other nine
  pairs are all at 95.3% or above.
- (A comparison with Massimo's C2 frozen-features numbers was removed: they were measured on an RTX
  3070 and his T4 re-run is still pending. Compare once his T4 rows are on main.)

### 2026-11-21: selection redone with the team rule (review on PR #4)
The selection above picked best by highest mean only. The team rule (`common/sweep.py select`, also
used by Massimo's methods) is: best = top mean accuracy, but a config within 0.5 pp of the top wins if
it is faster (train + inference time); fast = fastest config within 2 pp of best.
`python -m c2.peft.select --hw colab-t4` on the T4 tuning rows:

| config | acc (%) | T4 train + predict (s) |
|---|---|---|
| lora_r2 | 99.23 | 5.24 |
| vpt_shallow10_lr3e-3 | 99.20 | 4.99 |
| head | 98.74 | 3.56 |

head is 0.49 pp behind lora_r2 (inside the 0.5 pp margin) and the fastest config, so
**best = fast = `head`**. No new runs needed: head already has report-seed rows (96.56 ± 4.07%).
LoRA and VPT stay in the report as a finding (+0.59 pp on 9/10 report seeds for LoRA), not as the
submitted config. Plain reading: PEFT adapts the backbone a little, but on these tasks not by enough
to pay for about 2× the training time.

## Source log

| source | idea taken |
|---|---|
| Course testbed notebook | `head` control: frozen ViT-Tiny + new linear head, AdamW settings |
| Steiner et al. 2021 (AugReg) | the pretrained checkpoint and its preprocessing |
| Ben Zaken et al. 2022 (BitFit) | train only the bias terms |
| Hu et al. 2022 (LoRA) | low-rank updates to attention qkv, B = 0 init, alpha / r scaling (P1) |
| Jia et al. 2022 (VPT) | learned prompt tokens, shallow / deep, Xavier-uniform init (P1) |
| Massimo's C2 frozen-features notes | `crop_pct` observation, head baseline numbers on report seeds |
