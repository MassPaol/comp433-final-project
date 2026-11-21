"""C2: parameter-efficient fine-tuning (PEFT) of a pretrained ViT-Tiny.

Only a small subset of parameters is trained; the pretrained weights stay frozen:

    mode="head"    only the new 2-class head trains. This is the course testbed's C2
                   example, kept as the control. 386 trainable params.
    mode="bitfit"  head + every bias term in the backbone (Ben Zaken et al. 2022).
                   ViT-Tiny: 25,728 backbone biases + 386 head params = 26,114
                   trainable, 0.47% of the 5.5M total.
    mode="lora"    head + low-rank updates on each block's attention qkv (Hu et al. 2022).
                   12 blocks x rank x (192 + 576) params, e.g. rank 4: 36,864 + 386.
    mode="vpt"     head + learned prompt tokens (Jia et al. 2022). Shallow: n_prompts x 192.
                   Deep: 12 x n_prompts x 192.

Pretrained checkpoint: vit_tiny_patch16_224.augreg_in21k_ft_in1k (timm), trained on
ImageNet-21k then fine-tuned on ImageNet-1k; never on CIFAR-10.
Preprocessing is the checkpoint's own (timm data config): 32 px -> bicubic resize
to 248 -> center crop 224 -> normalize with mean = std = 0.5.
"""
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from c2.peft.layers import PromptedViT, add_lora
from common.method import Method

MODEL_NAME = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
MODES = ("head", "bitfit", "lora", "vpt")


class PEFTMethod(Method):
    name = "peft"

    def __init__(self, mode="bitfit", model_name=MODEL_NAME, epochs=10, lr=1e-3,
                 weight_decay=0.01, batch_size=32, flip=False, crop_pct=None,
                 lora_rank=4, lora_alpha=None, vpt_prompts=10, vpt_deep=False, seed=0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        super().__init__(mode=mode, model_name=model_name, epochs=epochs, lr=lr,
                         weight_decay=weight_decay, batch_size=batch_size, flip=flip,
                         crop_pct=crop_pct, lora_rank=lora_rank, lora_alpha=lora_alpha,
                         vpt_prompts=vpt_prompts, vpt_deep=vpt_deep, seed=seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None      # the timm ViT
        self.net = None        # what is called on images (the ViT, or the ViT wrapped with prompts)
        self.transform = None

    # ---- preprocessing ----------------------------------------------------
    def _make_transform(self, model):
        cfg = timm.data.resolve_model_data_config(model)
        if self.hparams["crop_pct"] is not None:
            cfg["crop_pct"] = self.hparams["crop_pct"]   # labelled deviation, off by default
        return timm.data.create_transform(**cfg, is_training=False)

    def _prep(self, X):
        """uint8 (N, 32, 32, 3) -> float tensor (N, 3, 224, 224) on the device."""
        x = torch.stack([self.transform(Image.fromarray(img)) for img in np.asarray(X)])
        return x.to(self.device)

    # ---- build the network and choose what trains -------------------------
    def _build(self):
        h = self.hparams
        vit = timm.create_model(h["model_name"], pretrained=True, num_classes=2)
        vit.requires_grad_(False)
        vit.head.requires_grad_(True)
        net = vit
        if h["mode"] == "bitfit":
            for name, p in vit.named_parameters():
                if name.endswith(".bias"):
                    p.requires_grad_(True)
        elif h["mode"] == "lora":
            add_lora(vit, rank=h["lora_rank"], alpha=h["lora_alpha"])   # new params train
        elif h["mode"] == "vpt":
            net = PromptedViT(vit, n_prompts=h["vpt_prompts"], deep=h["vpt_deep"])
        self.model, self.net = vit, net.to(self.device)

    def _param_groups(self):
        """Weight decay on weight matrices (head, LoRA) only; never on biases or prompts."""
        decay, no_decay = [], []
        for name, p in self.net.named_parameters():
            if p.requires_grad:
                skip = name.endswith(".bias") or name.endswith("prompts")
                (no_decay if skip else decay).append(p)
        return [{"params": decay, "weight_decay": self.hparams["weight_decay"]},
                {"params": no_decay, "weight_decay": 0.0}]

    def _set_train_mode(self):
        if self.hparams["mode"] == "head":
            # Testbed behaviour: frozen backbone stays in eval mode.
            self.net.eval()
            self.model.head.train()
        else:
            self.net.train()

    # ---- Method interface -------------------------------------------------
    def fit(self, X, y):
        h = self.hparams
        torch.manual_seed(h["seed"])
        gen = torch.Generator().manual_seed(h["seed"])

        self._build()
        self.transform = self._make_transform(self.model)
        x_all = self._prep(X)
        y_all = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        n = len(y_all)
        opt = torch.optim.AdamW(self._param_groups(), lr=h["lr"])

        self._set_train_mode()
        for _ in range(h["epochs"]):
            order = torch.randperm(n, generator=gen).to(self.device)
            for start in range(0, n, h["batch_size"]):
                idx = order[start:start + h["batch_size"]]
                xb, yb = x_all[idx], y_all[idx]
                if h["flip"]:
                    mask = torch.rand(len(idx), generator=gen).to(self.device) < 0.5
                    xb = torch.where(mask.view(-1, 1, 1, 1), xb.flip(3), xb)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(self.net(xb), yb)
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict(self, X):
        self.net.eval()
        X = np.asarray(X)
        preds = []
        for i in range(0, len(X), 128):          # chunks keep memory small at 224 px
            preds.append(self.net(self._prep(X[i:i + 128])).argmax(dim=1))
        return torch.cat(preds).cpu().numpy().astype(np.int64)

    def n_trainable_params(self):
        if self.net is None:
            return None
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
