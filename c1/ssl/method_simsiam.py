"""Method 2: SimSiam self-supervised pretraining on the 100 training images, then a head.

Pipeline:
  1. SimSiam (Chen & He 2021): two augmented views -> backbone -> projector -> predictor,
     negative cosine similarity with stop-gradient. Labels are NOT used here, and the only
     images seen are the 100 training images (C1 forbids pretrained models and external data).
  2. Freeze the backbone, embed the training images, fit a head (lr / knn / kmeans) on the
     embeddings with the labels.

epochs=0 skips step 1 (random-init backbone): the baseline that shows what SimSiam adds.
Augmentations are written here per sample on the GPU, so this file does not depend on common.augment.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from c1.ssl.heads import make_head
from common.method import Method


# ---- augmentation (per-sample parameters, batch on the GPU) -----------------
def _gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def _augment(x, crop_min_scale, jitter):
    """x: float (B, 3, 32, 32) in [0, 1]. Random resized crop + flip (+ color jitter + gray)."""
    B, dev = x.shape[0], x.device

    def u(lo, hi):
        return torch.empty(B, device=dev).uniform_(lo, hi)

    area = u(crop_min_scale, 1.0)
    ratio = torch.exp(u(math.log(3 / 4), math.log(4 / 3)))
    w = torch.sqrt(area * ratio).clamp(max=1.0)
    h = torch.sqrt(area / ratio).clamp(max=1.0)
    cx, cy = u(-1, 1) * (1 - w), u(-1, 1) * (1 - h)
    sign = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1  # -1 = horizontal flip
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0], theta[:, 0, 2] = w * sign, cx
    theta[:, 1, 1], theta[:, 1, 2] = h, cy
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    if jitter > 0:
        def factor():
            return 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * jitter
        y = x * factor()                                   # brightness
        m = _gray(y).mean((1, 2, 3), keepdim=True)
        y = (y - m) * factor() + m                         # contrast
        g = _gray(y)
        y = ((y - g) * factor() + g).clamp(0, 1)           # saturation
        on = (torch.rand(B, 1, 1, 1, device=dev) < 0.8).float()
        x = on * y + (1 - on) * x
        gs = (torch.rand(B, 1, 1, 1, device=dev) < 0.2).float()  # random grayscale
        x = gs * _gray(x).expand(-1, 3, -1, -1) + (1 - gs) * x
    return x


# ---- network ----------------------------------------------------------------
def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                         nn.ReLU(inplace=True))


class _SimSiamNet(nn.Module):
    def __init__(self, width, proj_dim, pred_dim):
        super().__init__()
        w = width
        self.backbone = nn.Sequential(
            _block(3, w), _block(w, w), nn.MaxPool2d(2),
            _block(w, 2 * w), _block(2 * w, 2 * w), nn.MaxPool2d(2),
            _block(2 * w, 4 * w), _block(4 * w, 4 * w),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.projector = nn.Sequential(
            nn.Linear(4 * w, proj_dim, bias=False), nn.BatchNorm1d(proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False), nn.BatchNorm1d(proj_dim))
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True), nn.Linear(pred_dim, proj_dim))

    def embed(self, x):
        return self.backbone((x - 0.5) / 0.25)

    def project(self, x):
        return self.projector(self.embed(x))


def _neg_cosine(p, z):
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()  # stop-gradient on z


# ---- Method -----------------------------------------------------------------
class SimSiamMethod(Method):
    name = "simsiam"

    def __init__(self, epochs=200, width=32, proj_dim=128, pred_dim=None, batch_size=50,
                 lr=0.05, weight_decay=5e-4, crop_min_scale=0.3, jitter=0.4,
                 head="lr", head_params=None, flip_head=False, seed=0):
        super().__init__(epochs=epochs, width=width, proj_dim=proj_dim, pred_dim=pred_dim,
                         batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                         crop_min_scale=crop_min_scale, jitter=jitter, head=head,
                         head_params=head_params, flip_head=flip_head, seed=seed)
        self.epochs, self.width, self.proj_dim = epochs, width, proj_dim
        self.pred_dim = pred_dim or max(proj_dim // 4, 8)
        self.batch_size, self.lr, self.weight_decay = batch_size, lr, weight_decay
        self.crop_min_scale, self.jitter = crop_min_scale, jitter
        self.head, self.head_params, self.flip_head, self.seed = head, dict(head_params or {}), flip_head, seed

    @staticmethod
    def _to_tensor(X, dev):
        return torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).float().div(255).to(dev)

    def _embed(self, X):
        self.net_.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(X), 256):
                out.append(self.net_.embed(self._to_tensor(X[s:s + 256], self.dev_)).cpu().numpy())
        return np.concatenate(out)

    def _pretrain(self, X01):
        n = len(X01)
        bs = min(self.batch_size, n)
        steps = max(n // bs, 1)  # drop the last partial batch (BatchNorm needs > 1 sample)
        total = self.epochs * steps
        opt = torch.optim.SGD(self.net_.parameters(), lr=self.lr, momentum=0.9,
                              weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total, 1))
        gen = torch.Generator().manual_seed(self.seed)
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen).to(self.dev_)
            for i in range(steps):
                x = X01[perm[i * bs:(i + 1) * bs]]
                x1 = _augment(x, self.crop_min_scale, self.jitter)
                x2 = _augment(x, self.crop_min_scale, self.jitter)
                z1, z2 = self.net_.project(x1), self.net_.project(x2)
                p1, p2 = self.net_.predictor(z1), self.net_.predictor(z2)
                loss = 0.5 * (_neg_cosine(p1, z2) + _neg_cosine(p2, z1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        self.final_loss_ = float(loss.item()) if self.epochs else None

    def fit(self, X, y):
        self.dev_ = "cuda" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(self.seed)
        if self.dev_ == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        y = np.asarray(y)
        self.net_ = _SimSiamNet(self.width, self.proj_dim, self.pred_dim).to(self.dev_)
        if self.epochs > 0:
            self._pretrain(self._to_tensor(X, self.dev_))

        # collapse check: std of the L2-normalized projections is ~1/sqrt(proj_dim) when healthy, ~0 if collapsed
        self.net_.eval()
        with torch.no_grad():
            z = F.normalize(self.net_.project(self._to_tensor(X, self.dev_)), dim=1)
        self.collapse_std_ = float(z.std(0).mean().item())

        Z = self._embed(X)
        if self.flip_head:
            Z = np.concatenate([Z, self._embed(X[:, :, ::-1])])
            y = np.concatenate([y, y])
        self.scaler_ = StandardScaler().fit(Z)
        self.head_ = make_head(self.head, self.head_params, self.seed).fit(self.scaler_.transform(Z), y)
        return self

    def predict(self, X):
        scores = self.head_.scores(self.scaler_.transform(self._embed(X)))
        if self.flip_head:
            scores = (scores + self.head_.scores(self.scaler_.transform(self._embed(X[:, :, ::-1])))) / 2
        return (scores > 0).astype(np.int64)

    def n_trainable_params(self):
        net = sum(p.numel() for p in self.net_.parameters()) if self.epochs > 0 else 0
        return int(net + self.head_.n_params())
