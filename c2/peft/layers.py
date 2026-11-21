"""PEFT building blocks for a timm VisionTransformer.

LoRALinear: Hu et al. 2022, "LoRA: Low-Rank Adaptation of Large Language Models"
    (arXiv:2106.09685).
PromptedViT: Jia et al. 2022, "Visual Prompt Tuning" (arXiv:2203.12119).
"""
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A x, with W, b frozen and only A (r x in), B (out x r) trained.

    B starts at zero, so at step 0 the layer is exactly the pretrained one.
    """

    def __init__(self, base: nn.Linear, rank=4, alpha=None):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = (alpha if alpha is not None else rank) / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # same init as the LoRA paper's code

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


def add_lora(vit, rank=4, alpha=None, targets=("qkv",)):
    """Wrap the chosen Linear layers of every block's attention with LoRA."""
    for blk in vit.blocks:
        for name in targets:
            setattr(blk.attn, name, LoRALinear(getattr(blk.attn, name), rank, alpha))


class PromptedViT(nn.Module):
    """Visual Prompt Tuning around a frozen timm ViT.

    `n_prompts` learned tokens are inserted between the class token and the patch tokens.
    shallow (deep=False): inserted once, before the first block, then processed like any token.
    deep (deep=True): a fresh set of prompts replaces the previous layer's prompt outputs
    before every block.
    """

    def __init__(self, vit, n_prompts=10, deep=False):
        super().__init__()
        self.vit = vit
        self.n = n_prompts
        self.deep = deep
        n_sets = len(vit.blocks) if deep else 1
        self.prompts = nn.Parameter(torch.empty(n_sets, n_prompts, vit.embed_dim))
        # VPT init: uniform Xavier-style range based on the patch size and width.
        p = vit.patch_embed.patch_size[0]
        val = math.sqrt(6.0 / float(3 * p * p + vit.embed_dim))
        nn.init.uniform_(self.prompts, -val, val)

    @property
    def head(self):
        return self.vit.head

    def forward(self, x):
        v = self.vit
        x = v.patch_embed(x)
        x = v._pos_embed(x)            # adds the class token + position embeddings
        x = v.patch_drop(x)
        x = v.norm_pre(x)
        B = x.shape[0]
        for i, blk in enumerate(v.blocks):
            if i == 0:
                p = self.prompts[0].expand(B, -1, -1)
                x = torch.cat([x[:, :1], p, x[:, 1:]], dim=1)
            elif self.deep:
                p = self.prompts[i].expand(B, -1, -1)
                x = torch.cat([x[:, :1], p, x[:, 1 + self.n:]], dim=1)
            x = blk(x)
        x = v.norm(x)
        return v.forward_head(x)       # class-token pooling, so prompts never reach the head
