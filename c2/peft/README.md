# C2: Parameter-efficient fine-tuning

**Owner:** Ammar

## What this is
Start from a ViT-Tiny pretrained on ImageNet (`vit_tiny_patch16_224.augreg_in21k_ft_in1k`, timm;
ImageNet-21k pretraining then ImageNet-1k fine-tuning, never CIFAR-10) and train only a small part of
it on the 100 training images. This sits between a frozen backbone with a new head (Massimo) and full
fine-tuning (Ryan): changing very few parameters should overfit 100 images less than changing all of
them, while still adapting the features more than a head alone. Images go through the checkpoint's own
preprocessing (bicubic resize 32 → 248, center crop 224, mean = std = 0.5).

| mode | what trains | trainable params |
|---|---|---|
| `head` | new 2-class linear head only (the testbed's C2 example, kept as the control) | 386 |
| `bitfit` | head + every bias in the backbone (Ben Zaken et al. 2022) | 26,114 (0.47%) |
| `lora` | head + low-rank updates on every block's attention qkv (Hu et al. 2022) | 9,216 × rank + 386 (rank 4: 37,250) |
| `vpt` | head + learned prompt tokens, shallow or deep (Jia et al. 2022) | shallow 10: 2,306; deep 10: 23,426 |

Weight decay (0.01) applies to the head and LoRA matrices only, never to biases or prompts. The
experiment log and the reasoning behind each stage are in [NOTES.md](NOTES.md).

## Files
| file | what it does |
|---|---|
| `method.py` | `PEFTMethod(mode=...)`: preprocessing, picks the trainable parameters, trains, predicts |
| `layers.py` | `LoRALinear` / `add_lora`, `PromptedViT` (VPT) |
| `configs.py` | every config of every stage |
| `run.py` | runs configs through `common.harness.run`, rows → `results/ammar.csv`; resumable |
| `figures.py` | report figures → `report/figures/c2_peft_*` |
| `NOTES.md` | experiment log + source log |

Tests: `python -m pytest tests/test_peft.py -q`

## Reproduce
Colab T4: [`notebooks/ammar/ammar_colab_t4.ipynb`](../../notebooks/ammar/ammar_colab_t4.ipynb).

Step by step:
```bash
python -m c2.peft.run --stage p0 --seeds tune --hw colab-t4             # then p1
python -m c2.peft.run --config <best> <fast> --seeds report --hw colab-t4
python -m c2.peft.figures --hw colab-t4
```

## Configs
| config_id | description | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|
| best | TODO | | | |
| fast | TODO | | | |
