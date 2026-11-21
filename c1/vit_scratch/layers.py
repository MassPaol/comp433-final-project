"""Small-data ViT parts that plug into a timm VisionTransformer.

ShiftedPatchTokenizer + LSAAttention: Lee, Lee & Song 2021, "Vision Transformer for
    Small-Size Datasets" (arXiv:2112.13492).
ConvTokenizer + SeqPool: Hassani et al. 2021, "Escaping the Big Data Paradigm with
    Compact Transformers" (arXiv:2104.05704).

Each tokenizer maps (B, 3, 32, 32) -> (B, N, D) with the same N as the patch
embedding it replaces, so the ViT's position embeddings still fit.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftedPatchTokenizer(nn.Module):
    """SPT: concat the image with 4 copies shifted diagonally by half a patch, then patchify.

    The shifts let every token see pixels from its neighbouring patches, which gives
    the ViT a wider receptive field (a locality hint) from the first layer.
    Pipeline per the paper: shift -> concat (15 channels) -> patch split -> LayerNorm -> Linear.
    """

    def __init__(self, patch_size, embed_dim, in_chans=3):
        super().__init__()
        self.p = patch_size
        self.s = patch_size // 2
        in_dim = in_chans * 5 * patch_size * patch_size
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        s = self.s
        padded = F.pad(x, (s, s, s, s))                    # zero padding, as in the paper
        shifts = [(-s, -s), (-s, s), (s, -s), (s, s)]      # the 4 diagonal directions
        views = [x] + [padded[:, :, s + dy:s + dy + H, s + dx:s + dx + W] for dy, dx in shifts]
        x = torch.cat(views, dim=1)                        # (B, 15, H, W)
        x = F.unfold(x, kernel_size=self.p, stride=self.p)  # (B, 15*p*p, N)
        return self.proj(self.norm(x.transpose(1, 2)))     # (B, N, D)


class ConvTokenizer(nn.Module):
    """CCT-style tokenizer: [conv3x3 -> ReLU -> maxpool(3, stride 2)] x n_layers.

    n_layers = log2(patch_size), so a "patch 4" model gets 2 layers and 8x8 = 64 tokens,
    matching the plain patch embedding it replaces.
    """

    def __init__(self, patch_size, embed_dim, in_chans=3, hidden=64):
        super().__init__()
        n = int(math.log2(patch_size))
        if 2 ** n != patch_size or n < 1:
            raise ValueError("conv tokenizer needs patch_size to be a power of 2 (>= 2)")
        chans = [in_chans] + [hidden] * (n - 1) + [embed_dim]
        layers = []
        for i in range(n):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(3, stride=2, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).flatten(2).transpose(1, 2)      # (B, N, D)


class LSAAttention(nn.Module):
    """Locality Self-Attention: drop-in for timm's Attention.

    Two changes to standard self-attention (Lee et al. 2021):
    1. diagonal masking: a token may not attend to itself, so it must look at others;
    2. learnable temperature: the softmax scale is a trained parameter (init 1/sqrt(d_head)),
       letting attention become sharper than the fixed scale allows.
    Reuses the qkv / proj layers of the timm Attention it replaces.
    """

    def __init__(self, attn):
        super().__init__()
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        self.qkv = attn.qkv
        self.attn_drop = attn.attn_drop
        self.proj = attn.proj
        self.proj_drop = attn.proj_drop
        self.log_scale = nn.Parameter(torch.tensor(attn.head_dim ** -0.5).log())

    def forward(self, x, attn_mask=None, is_causal=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.log_scale.exp()
        diag = torch.eye(N, dtype=torch.bool, device=x.device)
        attn = attn.masked_fill(diag, float("-inf")).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj_drop(self.proj(x))


class ViTNet(nn.Module):
    """A timm VisionTransformer plus an optional sequence-pooling layer (CCT's SeqPool).

    pool="token": classify from the class token (standard ViT).
    pool="seq":   no class token; a learned attention weight per token, softmax over tokens,
                  weighted average of all tokens -> head.
    """

    def __init__(self, vit, pool="token"):
        super().__init__()
        self.vit = vit
        self.pool = pool
        self.seq_attn = nn.Linear(vit.embed_dim, 1) if pool == "seq" else None

    def forward(self, x):
        if self.pool != "seq":
            return self.vit(x)
        feats = self.vit.forward_features(x)               # (B, N, D)
        w = self.seq_attn(feats).softmax(dim=1)            # (B, N, 1)
        return self.vit.head((w * feats).sum(dim=1))
