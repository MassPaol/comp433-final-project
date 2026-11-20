# C2: Vision-language few-shot (CLIP)

**Owner:** Alvin

## What this is
Five methods on one frozen CLIP model (ViT-B-32, `openai`), trained only on the 100 training
images of each problem instance:

1. **Zero-shot**: class-prompt embeddings from nine templates, averaged and renormalized. No training.
2. **Linear probe**: a linear head initialized from the zero-shot text embeddings (scaled by 100),
   trained with cross-entropy for a fixed number of steps.
3. **WiSE-FT-style interpolation**: the zero-shot head and the trained probe head are mixed by
   weight, `W = (1 - mix) * W_zeroshot + mix * W_probe`. Head-level, not encoder-level.
4. **Tip-Adapter**: a training-free cache of the 100 training embeddings, added to the zero-shot logits.
5. **Tip-Adapter-F**: the same cache with its keys fine-tuned for a fixed number of steps.

Reference for the cache formula: Zhang et al. (2022), official code github.com/gaopengcuhk/Tip-Adapter.
No validation images are used in any `fit`, and no early stopping is used anywhere.

## Files
| file | what it does |
|---|---|
| `loadClip.py` | loads CLIP once; prompt-ensembled class embeddings; batched image embeddings |
| `zeroshot.py` | `ZeroShotMethod` |
| `linearProbe.py` | `LinearProbe` (initialized from text embeddings) |
| `wiseft.py` | `WiSEFT`: probe with a mix weight, inherits `fit` |
| `tipadapter.py` | `TipAdapter` and `TipAdapterF` |
| `configs.py` | search stages, `PROBE_BASE`, `TIP_BASE`, `FINAL` |
| `run.py` | runs stages through `common.harness`; selection; final runs |
| `figures.py` | C2 report figures → `report/figures/c2_clip_*` |
| `NOTES.md` | experiment log and source log |

## Setup (Colab, T4)
```bash
pip install -e . -r requirements.txt open_clip_torch
export COMP433_DATA=/path/to/cifar
```
Open `notebooks/c2_clip.ipynb` and run it top to bottom. It runs the search in three passes and
the final runs on the reporting seeds.

## Reproduce
```bash
python -m c2.clip.run --stage zeroshot --hw colab-t4
python -m c2.clip.run --stage probe --hw colab-t4
python -m c2.clip.run --select --hw colab-t4        # best and fast per method (tuning seeds)
python -m c2.clip.run --final --hw colab-t4         # FINAL configs, reporting seeds 10-19
python -m c2.clip.figures --hw colab-t4
```
Run the stages in order: `wiseft` and `tip` after `probe`, `tipf` after `tip`, because each stage's
base comes from the previous stage's best (`PROBE_BASE`, `TIP_BASE` in `configs.py`).

## Configs
Fill in after the reporting-seed run (seeds 10–19).

| config | acc (%) | train (s) | inference (ms / image) |
|---|---|---|---|---|
| c2_zeroshot | 96.81 ± 3.97 | 0.282 | 3.43 |
| c2_best (WiSE-FT-style) | 97.96 ± 2.67 | 0.887 | 3.88 |

## Preprocessing deviation
The plan asks for each checkpoint's preprocessing through `timm.data.resolve_model_data_config`.
This project uses open_clip's preprocessing instead, because CLIP's text encoder comes from open_clip,
which timm does not provide. Images are resized and normalized by the transform returned with the model.
