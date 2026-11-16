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
| `configs.py` | every config of every stage |
| `run.py` | runs configs through `common.harness.run`, rows → `results/ammar.csv`; resumable |
| `resume.py` | skip already-logged (config, seed, hw) rows; merge a results copy (also used by `c2/peft`) |
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
| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| best | TODO | | | |
| fast | TODO | | | |
