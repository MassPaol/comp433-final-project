# C2 CLIP: experiment log

Seeds: tuning 0–4 for every search decision, reporting 10–19 for every reported number.
Metric: validation accuracy on the 700 held-out images per problem instance. Mean ± std over seeds
(std = spread across class pairs). Time = train + inference seconds per seed.
Raw rows: `results/alvin.csv` (challenge = c2). Timing split: `results/alvin_timing.csv`.

## Setup decisions (fixed before any search)

- Checkpoint: ViT-B-32, pretrained `openai`. Approved by the prof. Model card link: (add before reporting).
- Preprocessing: open_clip's transform (PIL), not `timm.data.resolve_model_data_config`.
- Zero-shot templates: 18 templates in `loadClip.py`. Frozen before any run.
- Probe head: initialized from the text embeddings scaled by 100; AdamW (default `weight_decay=0.01`), full batch, fixed steps. No early stopping.
- Selection rule (shared, `common/sweep.py::select`): best = highest mean tuning accuracy, unless a config within 0.5 pp is faster (train + inference). Fast = fastest config within 2 pp of best. Per stage, tuning seeds 0–4.
  - **Changed during the project.** The first version used "fast = within 1 pp of best" with no tie step. It was replaced with the shared rule so C1 and C2 use the same selection code, after the review.

## Reviewer changes

- **Selection:** switched to the shared `select()`. See above.
- **Train time:** CLIP is reloaded inside every `fit` (`fresh_clip()`), so every method pays the same loading cost. The earlier train times (0.282 s, 0.887 s) were from a cached model and are invalid.
- **Inference:** a batched GPU bicubic resize was tested against PIL on seeds 0–4. It shifted validation accuracy by up to 0.71 pp, above the 0.5 pp limit, so it was rejected and the PIL path was kept. The timing split shows preprocessing is about 0.06 ms per image, about 2% of inference. The reviewer's 2.4–2.7 s preprocessing figure was not reproduced on this setup.

## Results from earlier runs

The 2026-11-18 results (zero-shot 96.81 ± 3.97, best 97.96 ± 2.67) used the cached-model timing and are not reported. Their CSVs are archived in `results/archive/`.

## Pass 1: zero-shot and linear probe (tuning seeds 0–4)

| config | acc. (%) | time (s) |
|---|---|---|
| c2_probe_lr0.001_s300 | 99.49 ± 0.62 | 4.76 |
| c2_probe_lr0.001_s100 | 99.40 ± 0.58 | 4.66 |
| c2_probe_lr0.01_s100 | 99.31 ± 0.76 | 4.86 |
| c2_probe_lr0.01_s300 | 99.31 ± 0.83 | 5.22 |
| c2_zeroshot | 99.23 ± 0.85 | 4.46 |

**Decision → `PROBE_BASE = {lr: 0.001, steps: 100}`.** The rule selected `c2_probe_lr0.001_s100` for both best and fast. Its accuracy is 0.09 pp below the top config, inside the tie window, and it is faster. Every probe is within 0.5 pp of zero-shot, so the probe adds little on tuning instances.

## Pass 2: WiSE-FT and Tip-Adapter (tuning seeds 0–4)

WiSE-FT was built on `PROBE_BASE` = `{lr: 0.001, steps: 100}`.

| config | acc. (%) | time (s) |
|---|---|---|
| c2_tip_a0.5_b1 | 99.49 ± 0.69 | 5.14 |
| c2_tip_a0.5_b10 | 99.49 ± 0.62 | 4.82 |
| c2_wiseft_m0.75 | 99.43 ± 0.59 | 4.70 |
| c2_tip_a0.5_b5.5 | 99.43 ± 0.67 | 4.84 |
| c2_tip_a2_b10 | 99.37 ± 0.73 | 4.76 |
| c2_tip_a1_b5.5 | 99.37 ± 0.79 | 4.96 |
| c2_tip_a1_b1 | 99.37 ± 0.77 | 4.81 |
| c2_wiseft_m0.5 | 99.37 ± 0.70 | 4.83 |
| c2_tip_a2_b5.5 | 99.34 ± 0.85 | 4.85 |
| c2_tip_a1_b10 | 99.34 ± 0.77 | 4.81 |
| c2_tip_a2_b1 | 99.31 ± 0.90 | 5.12 |
| c2_wiseft_m0.25 | 99.31 ± 0.75 | 4.80 |

**Decisions**
- WiSE-FT: `wiseft_best` = `wiseft_fast` = `c2_wiseft_m0.75`. All three mixes are within the tie window.
- Tip-Adapter: `tip_best` = `tip_fast` = `c2_tip_a2_b10`. Nine cache settings are within the tie window. The rule picks the fastest, which sits on the grid edge (alpha 2, beta 10).
- **Decision → `TIP_BASE = {alpha: 2.0, beta: 10.0}`** for pass 3.

## Pass 3: Tip-Adapter-F (tuning seeds 0–4)

**Base used:** `TIP_BASE` at the time of pass 3. 

| config | acc. (%) | time (s) |
|---|---|---|
| c2_tipf_lr0.001_s300 | 99.29 ± 0.75 | 5.36 |
| c2_tipf_lr0.001_s100 | 99.23 ± 0.72 | 4.92 |
| c2_tipf_lr0.0001_s100 | 99.17 ± 0.84 | 4.69 |
| c2_tipf_lr0.0001_s300 | 99.14 ± 0.90 | 4.95 |

**Decision → `FINAL` tipf_best = tipf_fast = `c2_tipf_lr0.0001_s100`.** This is the top-ranked tipf config under the rule. The stage's accuracies are the lowest of the three cache-based stages, which the report should state.

## Selection summary (tuning seeds)

- zeroshot: `c2_zeroshot`
- probe: best = fast = `c2_probe_lr0.001_s100`
- wiseft: best = fast = `c2_wiseft_m0.75`
- tip: best = fast = `c2_tip_a2_b10`
- tipf: best = fast = `c2_tipf_lr0.0001_s100`
- Close calls: every stage. The tie window (0.5 pp) is wider than seed noise (about 0.6–0.9 pp std), so the picks mostly reflect timing.

## Reporting seeds 10–19 (frozen, single run)

| config | acc. (%) | paired vs zero-shot | wins | Holm p |
|---|---|---|---|---|
| c2_zeroshot | 97.00 ± 3.74 | — | — | — |
| c2_probe_lr0.001_s100 | 97.80 ± 2.74 | +0.80 pp | 5/10 | 0.56 |
| c2_wiseft_m0.75 | 97.71 ± 2.97 | +0.71 pp | 7/10 | 0.28 |
| c2_tip_a2_b10 | 96.74 ± 4.23 | −0.26 pp | 5/10 | 0.88 |
| c2_tipf_lr0.0001_s100 | 97.94 ± 2.55 | +0.94 pp | 7/10 | 0.49 |

No method differs from zero-shot after Holm correction. Paired tests are Wilcoxon signed-rank over 10 seeds, with Holm correction across the four comparisons.

## Open items

- [ ] Verify the tipf base (see Pass 3).
- [ ] Add the model card link.
- [ ] State the QuickGELU mismatch in the report.
- [ ] Send the reviewer reply.