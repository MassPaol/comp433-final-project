# C2: Vision-language few-shot (CLIP)

**Owner:** Alvin

## What this is

Five methods on one frozen CLIP model (ViT-B-32, `openai` weights), each trained only on the 100 training images of a problem instance and evaluated on its 700 validation images. Each problem instance is a pair of CIFAR-10 classes.

1. **Zero-shot:** class embeddings from 18 prompt templates, averaged and renormalized. No training.
2. **Linear probe:** a linear head initialized from the zero-shot text embeddings (scaled by 100), trained with cross-entropy for a fixed number of AdamW steps.
3. **WiSE-FT-style interpolation:** the zero-shot head and the trained probe head are mixed, `W = (1 - mix) * W_zeroshot + mix * W_probe`. Head-level, not encoder-level. The selected setting uses mix 0.75, so 75% probe head and 25% zero-shot head.
4. **Tip-Adapter:** a training-free cache of the 100 training embeddings, added to the zero-shot logits.
5. **Tip-Adapter-F:** the same cache with its keys fine-tuned for a fixed number of steps.

No validation images are used in any `fit`, and no early stopping is used anywhere.

## Protocol

- **Tuning seeds 0–4** decide every search choice. Each stage runs on the full grid below.
- **Reporting seeds 10–19** run once, on the frozen `FINAL` configs only.
- **Selection rule** (shared, `common/sweep.py::select`): best is the highest mean tuning accuracy, unless a config within 0.5 pp is faster (train + inference), in which case that config is best. Fast is the fastest config within 2 pp of best. Applied per stage.
- **Stage order:** `probe` sets `PROBE_BASE` for `wiseft`; `tip` sets `TIP_BASE` for `tipf`. Each stage runs only after the base it depends on is fixed.

## Timing

- Every method's `fit` calls `fresh_clip()`, so train time includes the CLIP model load on every seed. Zero-shot's train time is therefore mostly model loading, not training.
- Inference is split into **preprocessing** (resize and normalize) and **encode** (the CLIP forward pass). The split is in `results/alvin_timing.csv`.
- **The reported runs used the batched GPU resize for preprocessing.** The PIL per-image path is not timed in these results. See Known limitations.
- Per-image inference time includes preprocessing.
- Hardware: Colab T4 (`hw=colab-t4`).

## Files

| file | what it does |
|---|---|
| `loadClip.py` | loads CLIP; `fresh_clip()` reloads it; prompt-ensembled class embeddings; image embeddings with the timing split. Current code uses the batched GPU resize (see Known limitations). |
| `zeroshot.py` | `ZeroShotMethod` |
| `linearProbe.py` | `LinearProbe` (initialized from text embeddings) |
| `wiseft.py` | `WiSEFT`: probe with a mix weight, inherits `fit` |
| `tipadapter.py` | `TipAdapter` and `TipAdapterF` |
| `configs.py` | search stages, `PROBE_BASE`, `TIP_BASE`, `FINAL` |
| `run.py` | runs stages through `common.harness`; selection via `common/sweep.py`; final runs; writes `alvin_timing.csv` |
| `figures.py` | C2 report figures to `report/figures/c2_clip_*` |
| `c2_report_table_local.py` | report table (accuracy, train, inference, encode) to `report/tables/` |
| `NOTES.md` | experiment log |

Shared code in `common/` is not modified for C2.

## Setup (Colab, T4)

```bash
pip install -e . -r requirements.txt open_clip_torch
export COMP433_DATA=/path/to/cifar
```

Run `notebooks/c2_clip.ipynb` top to bottom, using the pass structure in the notebook.

## Reproduce

```bash
python -m c2.clip.run --stage zeroshot --hw colab-t4
python -m c2.clip.run --stage probe --hw colab-t4
python -m c2.clip.run --select --hw colab-t4        # best and fast per stage
# set PROBE_BASE from probe_best, then:
python -m c2.clip.run --stage wiseft --hw colab-t4
python -m c2.clip.run --stage tip --hw colab-t4
python -m c2.clip.run --select --hw colab-t4
# set TIP_BASE from tip_best, then:
python -m c2.clip.run --stage tipf --hw colab-t4
python -m c2.clip.run --select --hw colab-t4
# fill FINAL from the --select output, then:
python -m c2.clip.run --final --hw colab-t4         # reporting seeds 10-19, run once
python -m c2.clip.figures --hw colab-t4
python c2/clip/c2_report_table_local.py --best <best_id> --fast <fast_id>
```

Stages must run in this order, because each base comes from the previous stage's selection.

## Results (reporting seeds 10–19, single run, batched GPU preprocessing)

Mean ± std over 10 class pairs. Best and fast coincide for every stage, so one config per method is shown.

| Method | Config | Val. acc. (%) | Train (s) | Infer (ms / img) | Encode (ms / img) | Preproc. (ms / img) | vs zero-shot | Wins | Holm p |
|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | c2_zeroshot | 97.00 ± 3.74 | 2.412 | 2.57 | 2.50 | 0.063 | — | — | — |
| Linear probe | c2_probe_lr0.001_s100 | 97.80 ± 2.74 | 2.802 | 2.71 | 2.65 | 0.061 | +0.80 pp | 5/10 | 0.56 |
| WiSE-FT | c2_wiseft_m0.75 | 97.71 ± 2.97 | 2.741 | 2.98 | 2.91 | 0.062 | +0.71 pp | 7/10 | 0.28 |
| Tip-Adapter | c2_tip_a2_b10 | 96.74 ± 4.23 | 2.654 | 3.24 | 3.17 | 0.065 | −0.26 pp | 5/10 | 0.88 |
| Tip-Adapter-F | c2_tipf_lr0.0001_s100 | 97.94 ± 2.55 | 2.827 | 3.15 | 3.09 | 0.062 | +0.94 pp | 7/10 | 0.49 |

**Reading the results**

- No method differs from zero-shot after Holm correction (smallest corrected p = 0.28).
- Standard deviations across class pairs (2.5–4.2 pp) are much larger than the method differences, so most of the variance comes from which pair is drawn.
- Zero-shot is already at 97%, which leaves little room for few-shot gains on these pairs.
- On the batched path, preprocessing is about 0.06 ms per image, roughly 2% of inference. Encode accounts for the differences in inference time. This does not describe the PIL path.
- Tip-Adapter's selected setting (alpha 2, beta 10) is at the edge of its grid, so stronger cache settings were not tested.

## Known limitations

- **The batched preprocessing was used in all reported runs, although it failed the accuracy gate.** A batched GPU bicubic resize shifted validation accuracy by up to 0.71 pp on seeds 0–4, above the 0.5 pp limit set before testing. The revert to PIL was not applied before the runs. [DECISION: either report these results as batched-path results with this limitation stated, or rerun on PIL in a new results folder, with the prof's approval for a second reporting run.]
- **PIL per-image cost is not measured in these results.** The reviewer's inference-time concern is about PIL. The 0.06 ms preprocessing figure applies to the batched GPU resize only.
- **QuickGELU mismatch.** The model is built with `ViT-B-32` (`quick_gelu=False`), but the `openai` weights were trained with QuickGELU. open_clip warns about this. All methods share the model, so comparisons hold, but absolute accuracies are below published ViT-B/32 values.
- **Selection ties.** The 0.5 pp tie window is wider than the seed-to-seed noise, so best and fast coincide for every stage and the rule mostly picks the fastest config that is not clearly worse.
- **Small reporting run.** Ten reporting seeds, one run. The reporting results cannot rank the methods.

## Preprocessing deviation

The plan asks for each checkpoint's preprocessing through `timm.data.resolve_model_data_config`. This project uses open_clip's transform instead, because CLIP's text encoder comes from open_clip, which timm does not provide.