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
| `configs.py` | every config of every stage, plus `FINAL` (`best`, `fast`) |
| `run.py` | runs configs through `common.harness.run`, rows → `results/ammar.csv`; resumable |
| `select.py` | ranks configs on the tuning seeds and applies the team rule (`common/sweep.py`) |
| `figures.py` | report figures → `report/figures/c2_peft_*` |
| `NOTES.md` | experiment log + source log |

Tests: `python -m pytest tests/test_peft.py -q`

## Reproduce
Colab T4: [`notebooks/ammar/ammar_colab_t4.ipynb`](../../notebooks/ammar/ammar_colab_t4.ipynb).

Step by step:
```bash
python -m c2.peft.run --stage p0 --seeds tune --hw colab-t4             # then p1
python -m c2.peft.select --hw colab-t4                                   # team rule -> best, fast
python -m c2.peft.run --config <best> <fast> --seeds report --hw colab-t4
python -m c2.peft.figures --hw colab-t4
```

## Configs
Report seeds 10–19, `hw=colab-t4` (Colab T4 GPU). Mean ± std across the 10 problem instances.

| config_id | description | trainable | val acc (%) | train time | inference time (700 img) |
|---|---|---|---|---|---|
| **head** (best = fast) | testbed C2 example: frozen backbone, new linear head, AdamW lr 1e-3, 10 epochs | 386 | **96.56 ± 4.07** | 1.51 s | 1.74 s (2.48 ms/img) |
| lora_r2 | LoRA rank 2 on every block's attention qkv + new head, AdamW lr 1e-3, 10 epochs | 18,818 | 97.14 ± 4.04 | 3.42 s | 1.74 s (2.48 ms/img) |
| vpt_shallow10_lr3e-3 | 10 shallow prompt tokens + new head, lr 3e-3, 10 epochs | 2,306 | 97.17 ± 4.43 | 3.32 s | 1.69 s (2.42 ms/img) |

**Why head is both best and fast.** Picked with the team rule (`python -m c2.peft.select`, rule in
`common/sweep.py`) on the T4 tuning rows: the top config is lora_r2 (99.23%), but head (98.74%) is
within the 0.5 pp tie margin and faster (3.6 s vs 5.2 s train + inference), so head wins; it is also
the fastest config overall. So on these tasks PEFT does not beat a new head by enough to be worth
its extra training time.

LoRA and VPT were still run on the report seeds as a finding. Paired over the 10 report seeds:
lora_r2 − head = +0.59 pp, wins 9/10 (Wilcoxon p = 0.006); lora_r2 − vpt_shallow = −0.03 pp (4 wins,
2 ties; p = 0.77), i.e. a tie, with VPT training 8× fewer parameters. Hardest pair: cat-dog (head
85.1, LoRA 85.6, VPT 84.4%).
