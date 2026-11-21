# C1: ViT from scratch

**Owner:** Ammar

## What this is
A small Vision Transformer trained from scratch on the 100 training images only (no pretrained
weights, no external data). The starting point is the course testbed's ViT: 32 × 32 input, 4 × 4
patches (64 tokens), embed dim 128, depth 3, 4 heads, MLP ratio 2, dropout 0.1, AdamW (lr 3e-4,
wd 0.01), batch 32, 100 epochs. ViTs have almost no built-in image priors, so they are known to need a
lot of data; the question here is how much of the gap to CNNs and classical features can be closed
with augmentation, architecture changes and small-data ViT techniques (Shifted Patch Tokenization +
Locality Self-Attention, convolutional tokenizer). The experiment log and the reasoning behind each
stage are in [NOTES.md](NOTES.md).

## Files
| file | what it does |
|---|---|
| `method.py` | `ViTScratch`: uint8 images → normalize → [flip / random crop] → ViT, trained with AdamW |
| `layers.py` | Shifted Patch Tokenizer, Locality Self-Attention, conv tokenizer, sequence pooling |
| `configs.py` | every config of every stage, plus `FINAL` (`best`, `fast`) |
| `run.py` | runs configs through `common.harness.run`, rows → `results/ammar.csv`; resumable |
| `select.py` | ranks configs on the tuning seeds and applies the team rule (`common/sweep.py`) |
| `figures.py` | report figures → `report/figures/c1_vit_*` |
| `NOTES.md` | experiment log + source log |

Options on top of the testbed ViT: `flip`, `crop_pad` (augmentation), `tokenizer="patch" | "spt" |
"conv"`, `lsa=True`, `pool="token" | "seq"`.

Tests: `python -m pytest tests/test_vit_scratch.py -q`

## Reproduce
Colab T4: [`notebooks/ammar/ammar_colab_t4.ipynb`](../../notebooks/ammar/ammar_colab_t4.ipynb) (runs every
stage, then the final configs on the report seeds, and downloads the results).

Step by step (`--hw` tags the rows; re-running skips pairs already logged for that tag):
```bash
python -m c1.vit_scratch.run --stage s0 --seeds tune --hw colab-t4      # then s1, s2
python -m c1.vit_scratch.run --config <best> <fast> --seeds report --hw colab-t4
python -m c1.vit_scratch.figures --hw colab-t4
```

## Configs
Report seeds 10–19, `hw=colab-t4` (Colab T4 GPU). Mean ± std across the 10 problem instances.

| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| **s2_conv** (best) | testbed ViT + flip + random crop (pad 4) + CCT conv tokenizer (2 × conv3×3–ReLU–maxpool), 481,858 params | **78.96 ± 9.11** | 3.32 s | 0.037 s (0.05 ms/img) |
| s1_flip_crop | testbed ViT + flip + random crop (reference) | 72.24 ± 9.68 | 3.07 s | 0.026 s (0.04 ms/img) |
| baseline | course testbed ViT, unchanged (reference) | 70.70 ± 9.38 | 2.95 s | 0.026 s (0.04 ms/img) |

Paired over the 10 report seeds: s2_conv − baseline = **+8.26 pp, wins 10/10** (Wilcoxon p = 0.002);
s2_conv − s1_flip_crop = +6.71 pp, wins 10/10 (p = 0.002). No separate fast config: best already
trains in about 3 s on a T4 (see [NOTES.md](NOTES.md), T4 re-run). Hardest pair: cat-dog, 58.9%.
