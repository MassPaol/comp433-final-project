"""C1: a small Vision Transformer trained from scratch on the 100 training images.

The baseline settings are copied from the course testbed (cell "Small ViT
trained from scratch"), so `ViTScratch()` with no arguments reproduces it:
patch 4, embed 128, depth 3, 4 heads, MLP ratio 2, dropout 0.1,
AdamW lr 3e-4 / wd 0.01, batch 32, 100 epochs, no augmentation.

Optional augmentation (training only): `flip` (random horizontal flip) and
`crop_pad` (random crop after reflect-padding, i.e. random shifts).

Small-data ViT options (see layers.py):
    tokenizer="patch" | "spt" (Shifted Patch Tokenization) | "conv" (CCT conv tokenizer)
    lsa=True    Locality Self-Attention in every block
    pool="token" (class token) | "seq" (CCT sequence pooling, no class token)
"""
import numpy as np
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

from c1.vit_scratch.layers import ConvTokenizer, LSAAttention, ShiftedPatchTokenizer, ViTNet
from common.method import Method

TOKENIZERS = ("patch", "spt", "conv")
POOLS = ("token", "seq")

# Per-channel mean/std used by the testbed's `transforms.Normalize`.
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.247, 0.243, 0.261)


class ViTScratch(Method):
    name = "vit_scratch"

    def __init__(self, patch_size=4, embed_dim=128, depth=3, num_heads=4, mlp_ratio=2.0,
                 drop=0.1, epochs=100, lr=3e-4, weight_decay=0.01, batch_size=32,
                 flip=False, crop_pad=0, tokenizer="patch", lsa=False, pool="token", seed=0):
        if tokenizer not in TOKENIZERS:
            raise ValueError(f"tokenizer must be one of {TOKENIZERS}, got {tokenizer!r}")
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        super().__init__(patch_size=patch_size, embed_dim=embed_dim, depth=depth,
                         num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop, epochs=epochs,
                         lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                         flip=flip, crop_pad=crop_pad, tokenizer=tokenizer, lsa=lsa,
                         pool=pool, seed=seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    # ---- helpers ----------------------------------------------------------
    def _to_tensor(self, X):
        """uint8 (N, 32, 32, 3) -> normalized float tensor (N, 3, 32, 32) on the device."""
        x = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)
        x = x.permute(0, 3, 1, 2) / 255.0
        mean = torch.tensor(CIFAR_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD, device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _random_crop(self, xb, gen):
        """Pad by `crop_pad` pixels (reflect), then cut a random 32x32 window per image.

        Same as torchvision's RandomCrop(32, padding=p): the picture shifts by up to
        p pixels in each direction.
        """
        p = self.hparams["crop_pad"]
        padded = F.pad(xb, (p, p, p, p), mode="reflect")
        offs = torch.randint(0, 2 * p + 1, (len(xb), 2), generator=gen).tolist()
        return torch.stack([padded[i, :, dy:dy + 32, dx:dx + 32]
                            for i, (dy, dx) in enumerate(offs)])

    def _build_model(self):
        h = self.hparams
        seq = h["pool"] == "seq"
        vit = VisionTransformer(
            img_size=32, patch_size=h["patch_size"], in_chans=3, num_classes=2,
            embed_dim=h["embed_dim"], depth=h["depth"], num_heads=h["num_heads"],
            mlp_ratio=h["mlp_ratio"], qkv_bias=True,
            proj_drop_rate=h["drop"], pos_drop_rate=h["drop"],
            class_token=not seq, global_pool="avg" if seq else "token",
        )
        if h["tokenizer"] == "spt":
            vit.patch_embed = ShiftedPatchTokenizer(h["patch_size"], h["embed_dim"])
        elif h["tokenizer"] == "conv":
            vit.patch_embed = ConvTokenizer(h["patch_size"], h["embed_dim"])
        if h["lsa"]:
            for blk in vit.blocks:
                blk.attn = LSAAttention(blk.attn)
        return ViTNet(vit, pool=h["pool"]).to(self.device)

    # ---- Method interface -------------------------------------------------
    def fit(self, X, y):
        h = self.hparams
        torch.manual_seed(h["seed"])
        gen = torch.Generator().manual_seed(h["seed"])

        x_all = self._to_tensor(X)
        y_all = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        n = len(y_all)

        self.model = self._build_model()
        opt = torch.optim.AdamW(self.model.parameters(), lr=h["lr"],
                                weight_decay=h["weight_decay"])

        self.model.train()
        for _ in range(h["epochs"]):
            order = torch.randperm(n, generator=gen).to(self.device)
            for start in range(0, n, h["batch_size"]):
                idx = order[start:start + h["batch_size"]]
                xb, yb = x_all[idx], y_all[idx]
                if h["flip"]:
                    # Mirror a random half of the batch left-right (a flipped cat is still a cat).
                    mask = torch.rand(len(idx), generator=gen).to(self.device) < 0.5
                    xb = torch.where(mask.view(-1, 1, 1, 1), xb.flip(3), xb)
                if h["crop_pad"]:
                    xb = self._random_crop(xb, gen)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.model(xb), yb)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        x_all = self._to_tensor(X)
        preds = [self.model(x_all[i:i + 256]).argmax(dim=1)
                 for i in range(0, len(x_all), 256)]
        return torch.cat(preds).cpu().numpy().astype(np.int64)

    def n_trainable_params(self):
        if self.model is None:
            return None
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
