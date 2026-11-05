# COMP 433 Final Project — Team Plan (4 people)

**Deadline:** Sun Dec 6, 2026 (double-check on the course page)
**Plan written:** Thu Nov 5, 2026
**Team:** Massimo · Ryan · Ammar · Alvin

---

## TL;DR

- Each person owns **one Challenge 1 method + one Challenge 2 method + one shared deliverable**. All research work runs in parallel. The only hard dependencies are the shared evaluation harness (week 1) and the report / video / final evaluation at the end.
- That gives us **4 methods for C1 and 4 for C2**, one more than the minimum for C2, so we're covered if the TA decides two of our methods count as one.
- **Weekly sync:** Sundays, 30 min. **Mid-week update:** post in chat every Wednesday (numbers so far + blockers).

---

## What the grade rewards (so we spend time in the right place)

| Criterion | Weight | What it means for us |
|---|---|---|
| Performance | 20% | Accuracy on the final splits + training/inference time. Part of it is for beating a baseline threshold (announced with the final eval instructions). |
| Report (excl. lit review) | 70% | What we tried and **why**, a clean validation protocol, detailed hyperparameter exploration, clear labelled figures, time-vs-accuracy tradeoffs. |
| Originality | 10% | At least one method beyond course material. Document where every idea came from. |

**Minimums for a team of 4 (our reading, confirm with TA):** ≥4 high-level methods in C1, ≥3 in C2, ≥1 page literature review, contributions statement, code + run instructions, video where every member is identified.

---

## Who does what

| | Challenge 1 method (no external data/models) | Challenge 2 method (external pretraining OK) | Shared deliverable |
|---|---|---|---|
| **Massimo** | Hand-crafted features + classical ML (HOG, color histograms, wavelet scattering → LR / SVM / boosting) | Frozen pretrained features + lightweight heads, backbone sweep | Eval harness, timing, results sheet, final-eval runs, protocol section, summary figures |
| **Ryan** | Small CNN from scratch with strong augmentation | Fine-tuning ViT-Tiny (partial → full, LP-FT, LR schedules) | Video lead + report introduction & conclusions |
| **Ammar** | ViT from scratch with small-data ViT tricks (SPT + LSA, conv tokenizer) | Parameter-efficient fine-tuning (BitFit, LoRA, visual prompt tuning) | Literature review editor + originality statement |
| **Alvin** | Self-supervised / unsupervised features learned from the 100 training images only (SimCLR/SimSiam, k-means features) | Vision-language few-shot (CLIP zero-shot, linear probe, Tip-Adapter). Needs TA OK, fallback below | Report assembly (template, style guide, tradeoffs section, contributions, final formatting, submission) |

**Buddy pairs** (review each other's code, cover if someone gets sick or falls behind):
- **Massimo ↔ Alvin**: both do feature extraction + linear/classical classifiers.
- **Ryan ↔ Ammar**: both train deep nets from scratch and fine-tune; they share one augmentation module.

---

## Shared rules (everyone read this before writing code)

### Data and integrity

1. Use the testbed's `split_indices` **unchanged**.
   - **Tuning seeds: 0–4.** Use these for all hyperparameter selection.
   - **Reporting seeds: 10–19.** Never used for tuning. Every number in the report comes from these.
2. Never train on the validation images. They are only for measuring and selecting. Don't look at the final test splits until methods are frozen.
3. **No per-run early stopping on validation.** The final training script only gets 50 images/class, so epoch counts and all other settings must be fixed in advance (chosen on the tuning seeds).
4. **C1:** no pretrained models, no external data, no other CIFAR-10 images (not even unlabeled ones, not even the validation images). Augmented/derived views of the 100 training images are fine.
5. **C2:** no extra CIFAR-10 images, no checkpoint trained or fine-tuned on CIFAR-10. For every model you use, write down its pretraining data with the model-card link (goes in the report). Use each checkpoint's own preprocessing (`timm.data.resolve_model_data_config`).

### Method interface (makes final evaluation plug-and-play)

```python
class Method:
    def __init__(self, **hparams): ...
    def fit(self, X, y):     # X: uint8 array (N, 32, 32, 3); y: int array with values in {0, 1}
        ...
    def predict(self, X):    # returns int array of shape (N,)
        ...
```

Each method does its own preprocessing inside `fit`/`predict`. The harness handles seeds, timing and logging. When the final split files arrive, Massimo only has to write a loader; nobody else has to touch their code.

### Timing

- **Train time** = wall-clock of `fit` on one problem instance. For frozen-feature methods, include feature extraction, and also log feature-extraction time separately.
- **Inference time** = wall-clock of `predict` on the 700 validation images, also reported per image.
- Call `torch.cuda.synchronize()` before stopping timers. Log `torch.cuda.get_device_name()`.
- Final reported times must all come from the **same GPU type** (default: Colab T4).

### Results logging

One shared sheet / `results.csv`, one row per (method, config, seed):

```
challenge, person, method, config_id, seed, class_pair, val_acc, train_time_s, infer_time_s, n_trainable_params, gpu, git_commit, notes
```

Log bad and failed configs too. They belong in the report.

### Statistics

- Report **mean ± std over the 10 reporting seeds** (std = variation across problem instances).
- When comparing two methods, use the **paired per-seed difference** (same seeds = same class pairs): mean difference, number of seeds won, optionally a paired t-test or Wilcoxon test. Difficulty varies hugely between class pairs (cat/dog vs ship/frog), so unpaired comparisons are misleading.

### Code and figures

- **Repo layout:** `common/` (data, harness, timing, plots, augment), `c1/<method>/`, `c2/<method>/`, `final_eval/`, `report/figures/`. Pinned `requirements.txt` (`timm==1.0.30`). Each method folder has a short README with the exact command that reproduces its reported numbers.
- **Figures:** all made with `common/plots.py`. Fixed color per method, labelled axes with units, error bars, readable font size.
- Keep an **experiment log** (date, what you changed, why, result) and a **source log** (paper/blog → idea you took from it). These become your report subsection and lit review paragraph almost directly.

### Definition of done for each method

- [ ] Runs through the harness on all 10 reporting seeds
- [ ] Hyperparameter search done on tuning seeds and documented (what was searched, ranges, chosen value, why)
- [ ] A "best" config and a "fast" config identified, with train and inference times
- [ ] Rows in the results sheet; figure(s) in `report/figures/`
- [ ] Report subsection drafted (~1 page per method: motivation, setup, exploration, results, takeaways)
- [ ] Lit review paragraph + sources sent to Ammar
- [ ] 2–3 min video segment recorded and sent to Ryan

---

## Timeline

The research parts are fully parallel, but the report, video and final eval depend on everyone's results, so we use internal deadlines well before Dec 6.

| Date | Milestone | Who |
|---|---|---|
| **Fri Nov 6** | Kickoff meeting. Roles confirmed, repo + results sheet created (Massimo), Overleaf created (Alvin), TA email sent (questions below). | All |
| **Mon Nov 9** | **Harness v1** shipped in `common/`: splits, Method interface, timing, CSV logging, plot helper. Everyone has the testbed running on Colab. | Massimo (others start from the testbed code in parallel and port over when the harness lands) |
| **Fri Nov 13** | TA answers in. Alvin confirms CLIP or switches to fallback. | Alvin (+ whoever sent the email) |
| **Sun Nov 15** | **Checkpoint 1: first numbers.** Untuned version of every C1 and C2 method runs on the tuning seeds, first rows in the sheet. Report template + figure style guide ready (Alvin). Shared augmentation module ready (Ryan). | All |
| **Sun Nov 22** | **Checkpoint 2: exploration done.** Hyperparameter sweeps finished, best + fast configs chosen and run on the reporting seeds. Lit review paragraph (≥¼ page each, with references) sent to Ammar. | All |
| **Sun Nov 29** | **Feature freeze.** Report subsections drafted. Team picks the final C1 and C2 submissions (best + fast). Final-eval script ready (Massimo). Lit review compiled (Ammar). Video slide template + script outline ready (Ryan). | All |
| **Wed Dec 2** | Full report draft assembled (Alvin). Video segments recorded and sent to Ryan. Final-eval results in, if splits have been released (Massimo). | All |
| **Fri Dec 4** | Review pass: each person proofreads one other person's sections (Massimo → Ryan, Ryan → Ammar, Ammar → Alvin, Alvin → Massimo). Video edited (Ryan). | All |
| **Sat Dec 5** | Buffer: fix figures, formatting, contributions statement, run through the submission checklist. | Alvin + all |
| **Sun Dec 6** | **Submit** report PDF, code, video. | Alvin submits, everyone confirms |

**If the final evaluation splits are released late:** everyone's final config is frozen behind `fit`/`predict` by Nov 29, and Massimo runs all final methods within 48 hours of release without needing anyone else.

**How we choose the final submissions (Nov 29):** for each challenge, pick (1) the best mean accuracy on the reporting seeds, and (2) a fast alternative if the best one is slow. Break near-ties with the paired comparison and training/inference time. An ensemble of top C1 methods is allowed as a candidate if it's clearly better and its time cost is reported.

---

## Massimo — Classical ML + frozen features + infrastructure

### C1: Hand-crafted features + classical classifiers

*Why:* strong, cheap baseline in the tiny-data regime; anchors the fast end of our time/accuracy tradeoff plot.

- **Features:** raw pixels (+ PCA), HSV color histograms, HOG, LBP, and concatenations.
- **Originality hook:** wavelet scattering transform (Kymatio). Fixed, untrained filters, so no data or pretraining involved (confirm with TA).
- **Classifiers:** logistic regression, linear and RBF SVM, random forest, gradient boosting (HistGradientBoosting / XGBoost), kNN.
- **Explore:** regularization strength, kernel width, HOG cell/block size, PCA dimensions, scattering scale J and orientations L, feature standardization, horizontal-flip augmentation (doubles the training set).

### C2: Frozen pretrained features + lightweight heads

*Why:* this is the testbed baseline, and cached features make exploration nearly free.

- Cache ViT-Tiny features (CLS token and mean-pooled patch tokens) once per image.
- **Heads:** testbed linear head (AdamW), logistic regression, SVM, nearest class mean (prototypes), kNN; L2-normalized vs raw features.
- Flip-augmented training features + test-time augmentation (average features of image and its flip).
- **Backbone sweep** (timm checkpoints trained only on ImageNet): ViT-Tiny (baseline), ViT-S/16 AugReg, DeiT-S, ResNet-50, ConvNeXt-T. Plot accuracy vs inference time.

### Shared deliverables

- Harness v1 by **Mon Nov 9**; maintain `common/` (others send PRs, Massimo reviews).
- Final-eval loader for the released split files; run all final methods; produce the final results table.
- Report: **Experimental protocol** section (splits, seeds, timing, hardware, statistics); C1 and C2 summary tables and accuracy-vs-time (Pareto) plots generated from the sheet; final submitted methods & final-split results section.

### Starting reading

Dalal & Triggs 2005 (HOG) · Bruna & Mallat 2013 (scattering networks) · Oyallon, Belilovsky & Zagoruyko 2017 (scattering hybrid networks, motivates scattering for small-sample settings) · Andreux et al. 2020 (Kymatio) · Tian et al. 2020 ("Rethinking few-shot image classification: a good embedding is all you need?") · Kornblith, Shlens & Le 2019 (do better ImageNet models transfer better?)

---

## Ryan — CNN from scratch + fine-tuning ViT-Tiny

### C1: Small CNN from scratch

*Why:* convolutions have locality and translation-equivariance biases that ViTs lack, which matters most with tiny data. Direct contrast with Ammar's ViT.

- **Architectures:** plain 4–6 layer ConvNet, CIFAR-style ResNet-8/20, width sweep; BatchNorm vs GroupNorm (batches are tiny).
- **Augmentation / regularization:** random crop with padding, flip, Cutout, mixup, CutMix, RandAugment, label smoothing, weight decay, dropout.
- **Optimization:** SGD + momentum vs AdamW, cosine / one-cycle schedules, 100–500 epochs (each epoch is only 100 images).
- Test-time flip augmentation; snapshot ensembles if time allows.

### C2: Fine-tuning ViT-Tiny

*Why:* the main direction the instructions suggest; tells us how much adapting the backbone helps over Massimo's frozen baseline.

- **Unfreezing levels:** head only → last block → last k blocks → full model.
- **LP-FT:** train the head with the backbone frozen first, then unfreeze (Kumar et al. 2022).
- Layer-wise LR decay, separate head/backbone learning rates, warmup + cosine, number of epochs, batch size.
- Augmentation at 224 (random resized crop, flip, RandAugment, mixup) while keeping the checkpoint's normalization; drop-path, label smoothing.
- Measure the training-time cost vs frozen features.

### Shared deliverables

- **Video lead:** slide template + script outline by Nov 29 (problem, main approaches, key results, conclusions; every member identified). Collect segments by Dec 2, edit by Dec 4. Check the required length.
- Report: **Introduction / problem setup** and **Conclusions**.
- Shared augmentation module `common/augment.py`, ready for Ammar by **Sun Nov 15**.

### Starting reading

He et al. 2016 (ResNet) · DeVries & Taylor 2017 (Cutout) · Zhang et al. 2018 (mixup) · Yun et al. 2019 (CutMix) · Cubuk et al. 2020 (RandAugment) · Steiner et al. 2021 ("How to train your ViT?", the AugReg paper behind our checkpoint) · Kumar et al. 2022 (LP-FT) · Bao et al. 2021 (BEiT, layer-wise LR decay)

---

## Ammar — ViT from scratch + parameter-efficient fine-tuning

### C1: Data-efficient ViT from scratch

*Why:* the testbed baseline. ViTs are known to be data-hungry, so the interesting question is how far small-data tricks close the gap to CNNs.

- Reproduce the testbed baseline on the reporting seeds first.
- **Architecture:** depth (2–6), embed dim (64–192), heads, patch size (2 / 4 / 8), MLP ratio.
- **Originality hook:** Shifted Patch Tokenization + Locality Self-Attention (Lee et al. 2021); convolutional tokenizer + sequence pooling from Compact Convolutional Transformers (Hassani et al. 2021).
- **Training:** Ryan's augmentation module, longer schedules, warmup + cosine, label smoothing, stochastic depth, weight decay sweep.

### C2: Parameter-efficient fine-tuning

*Why:* sits between frozen (Massimo) and full fine-tuning (Ryan). Very few trainable parameters may overfit less with only 100 images.

- BitFit (biases only), LoRA on attention qkv (rank 2–16, which blocks), Visual Prompt Tuning (shallow vs deep, prompt length).
- Plot accuracy vs number of trainable parameters vs training time, alongside Massimo's and Ryan's numbers.
- Ask the TA whether this counts as separate from fine-tuning. We have 4 C2 methods, so we're covered either way.

### Shared deliverables

- **Literature review editor:** collect everyone's paragraph + sources by Nov 22; turn them into one coherent ≥1-page section with consistent citations by Nov 29.
- **Originality statement:** which of our methods go beyond course material and where each came from.

### Starting reading

Dosovitskiy et al. 2021 (ViT) · Touvron et al. 2021 (DeiT) · Lee, Lee & Song 2021 (ViT for small-size datasets) · Hassani et al. 2021 (CCT) · Ben Zaken et al. 2022 (BitFit) · Hu et al. 2022 (LoRA) · Jia et al. 2022 (VPT)

---

## Alvin — Self-supervised features + vision-language few-shot

### C1: Representation learning from the 100 training images only

*Why:* labels are the bottleneck. Self-supervision extracts extra training signal from the same images without breaking the rules.

- **Deep SSL:** SimCLR or SimSiam pretraining of a small CNN (reuse Ryan's backbone) on the 100 training images with heavy augmentation, then linear probe or fine-tune with labels. Also try a joint SSL + supervised loss.
- **Fast variant:** single-layer unsupervised features (Coates, Ng & Lee 2011): random patches from the training images → whitening → k-means dictionary → triangle encoding + pooling → linear SVM.
- Only the 100 training images may be used: not the validation images and not other CIFAR images, even without labels.

### C2: Vision-language few-shot (CLIP)

*Why:* text embeddings of class names give a prior before seeing any image; few-shot adapters build on top of that.

- Zero-shot with prompt templates + prompt ensembling (no training at all).
- Linear probe on CLIP image embeddings; head initialized from the text embeddings; interpolation between zero-shot and probe heads (WiSE-FT style).
- Tip-Adapter (training-free cache of the 100 training embeddings) and Tip-Adapter-F.
- **Needs TA approval** because CLIP's web-scraped pretraining data isn't public and could contain CIFAR-like images. Read the CLIP paper's data-overlap analysis and cite it in the report.
- **Fallback if the TA says no:** same few-shot ideas on DINO ViT-S/16 (self-supervised, ImageNet-1k only): prototypes, Tip-Adapter-style cache, linear probe. Decide by **Fri Nov 13**.

### Shared deliverables

- Report: Overleaf template + figure style guide by **Sun Nov 15**; assemble the full draft by **Wed Dec 2**; write the **training/inference-time tradeoffs** discussion (using Massimo's Pareto plots) and the **contributions statement**; final formatting + submission.

### Starting reading

Chen et al. 2020 (SimCLR) · Chen & He 2021 (SimSiam) · Coates, Ng & Lee 2011 (single-layer unsupervised feature learning) · Radford et al. 2021 (CLIP) · Zhang et al. 2022 (Tip-Adapter) · Zhou et al. 2022 (CoOp) · Wortsman et al. 2022 (WiSE-FT) · Caron et al. 2021 (DINO)

---

## Report outline and owners

1. Introduction & problem setup — **Ryan**
2. Literature review (≥1 page) — **Ammar** edits, everyone contributes
3. Experimental protocol — **Massimo**
4. Challenge 1: one subsection per method (**Massimo, Ryan, Ammar, Alvin**) + comparison table/figures (**Massimo**)
5. Challenge 2: one subsection per method (**Massimo, Ryan, Ammar, Alvin**) + comparison (**Massimo**)
6. Training time vs inference time vs accuracy — **Alvin** (figures from Massimo)
7. Final submitted methods & final-split results (best + fast per challenge) — **Massimo**
8. Conclusions — **Ryan**
9. Member contributions — **Alvin** (everyone sends 2–3 sentences)

---

## Questions for the TA (send by Fri Nov 6)

1. For a team of 4, is the minimum 4 high-level methods for C1 and 3 for C2?
2. Is CLIP (or DINOv2) allowed for C2, given its web-scale pretraining data?
3. Does parameter-efficient fine-tuning (LoRA / VPT) count as a separate high-level method from fine-tuning?
4. For C1: is a wavelet scattering transform (fixed, untrained filters) OK? Is self-supervised pretraining on only the 100 training images OK?
5. When will the final evaluation split files be released, and in what format? What hardware should times be reported on? Required video length?

---

## Risks and what we do about them

| Risk | Plan |
|---|---|
| TA rejects CLIP | Alvin switches to the DINO fallback (decision by Nov 13). |
| Final splits released late | Methods frozen behind `fit`/`predict` by Nov 29; Massimo runs them within 48 h. |
| Colab disconnects / usage limits | Log a results row as soon as each seed finishes; save checkpoints. Runs are short anyway (100 training images). |
| Noisy comparisons | 10 reporting seeds, paired differences, never conclude anything from one seed. |
| Someone falls behind | Raise it in the Wednesday update; buddy helps get at least the "fast config" numbers in. |
| End-of-project crunch | Report subsections are due Nov 29, not Dec 5. |
