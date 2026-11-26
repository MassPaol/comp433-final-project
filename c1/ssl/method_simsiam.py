"""Method 2 (C1, deep variant): SimSiam (Chen & He 2021) on the 100 training images.

Reuses the team's shared pieces instead of its own:
  * backbone: Ryan's selected C1 CNN, built INSIDE this class with c1.cnn.models.build_backbone
    (ConvNet-3, width 16, BatchNorm, 64-d features by default; `width` / `convs` change it);
  * augmentation: common.augment.Augmenter (RandAugment, pad + random crop, flip, Cutout) for
    each of the two views, plus `random_resized_crop` below, which the shared module lacks;
  * batching and LR schedule: c1.cnn.method.balanced_batches / lr_factor.

Pipeline, fit on the 100 training images only (never the validation images, never other CIFAR
images: the normalization statistics come from the 100 training images too):
  two augmented views -> backbone -> projector -> predictor, symmetric negative cosine loss
  with stop-gradient on the projector output.
  Labels are NOT used during pretraining unless `sup_weight` > 0 (joint SimSiam + cross-entropy).
  Then one of:
    ft_epochs == 0: freeze the backbone, embed, standardize, linear SVM (C by 5-fold CV);
    ft_epochs  > 0: fine-tune backbone + new linear head with cross-entropy.
epochs=0 skips pretraining (random-init backbone), the baseline for what SimSiam adds.
No early stopping: every epoch count is fixed by the config.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from c1.cnn.method import balanced_batches, lr_factor
from c1.cnn.models import FINAL_ARCH, build_backbone
from common.augment import Augmenter, to_float
from common.method import Method
from common.timing import Timer


def random_resized_crop(x, min_scale, gen, ratio=(3 / 4, 4 / 3)):
    """Per-image random resized crop of a float batch (N, C, H, W), on the batch's device.

    Crop area is uniform in [min_scale, 1] of the image, aspect ratio log-uniform in `ratio`;
    the crop is resized back to H x W (bilinear). min_scale >= 1 returns `x` unchanged.
    The shared common/augment.py has no resized crop, which SimSiam needs.
    """
    if min_scale >= 1.0:
        return x
    n, dev = len(x), x.device

    def u():
        return torch.rand(n, generator=gen).to(dev)

    area = min_scale + (1.0 - min_scale) * u()
    log_r = math.log(ratio[0]) + (math.log(ratio[1]) - math.log(ratio[0])) * u()
    r = torch.exp(log_r)
    w = torch.sqrt(area * r).clamp(max=1.0)   # crop width / image width
    h = torch.sqrt(area / r).clamp(max=1.0)
    cx = (2 * u() - 1) * (1 - w)              # crop centre, in [-1, 1] image coordinates
    cy = (2 * u() - 1) * (1 - h)
    theta = torch.zeros(n, 2, 3, device=dev, dtype=x.dtype)
    theta[:, 0, 0], theta[:, 0, 2] = w, cx
    theta[:, 1, 1], theta[:, 1, 2] = h, cy
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _projector(i, hidden, o):
    return nn.Sequential(
        nn.Linear(i, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, hidden, bias=False), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
        nn.Linear(hidden, o, bias=False), nn.BatchNorm1d(o, affine=False))


def _predictor(d, hidden):
    return nn.Sequential(nn.Linear(d, hidden, bias=False), nn.BatchNorm1d(hidden),
                         nn.ReLU(inplace=True), nn.Linear(hidden, d))


class SimSiamNet(nn.Module):
    """Backbone (Ryan's CNN without its classifier) + SimSiam projector and predictor."""

    def __init__(self, arch, proj_dim, proj_hidden, pred_hidden):
        super().__init__()
        self.backbone = build_backbone(arch)
        self.backbone.eval()
        with torch.no_grad():
            self.feat_dim = int(self.backbone(torch.zeros(1, 3, 32, 32)).shape[1])
        self.backbone.train()
        self.projector = _projector(self.feat_dim, proj_hidden, proj_dim)
        self.predictor = _predictor(proj_dim, pred_hidden)

    def forward(self, x):
        f = self.backbone(x)
        z = self.projector(f)
        return f, z, self.predictor(z)


def _neg_cosine(p, z):
    """-cos(p, stopgrad(z)), averaged over the batch (SimSiam, eq. 1)."""
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


class SimSiamMethod(Method):
    name = "simsiam"

    def __init__(self, epochs=200, width=16, convs=1, proj_dim=64, proj_hidden=256,
                 pred_hidden=32, batch_size=32, lr=0.05, weight_decay=5e-4, momentum=0.9,
                 warmup_epochs=5, min_scale=0.3, ra_n=0, ra_m=9, crop=4, flip=True, cutout=0,
                 sup_weight=0.0, ft_epochs=0, ft_lr=0.01, C_grid=(0.01, 0.1, 1.0, 10.0),
                 svm_max_iter=5000, seed=0, device=None):
        hp = dict(epochs=epochs, width=width, convs=convs, proj_dim=proj_dim,
                  proj_hidden=proj_hidden, pred_hidden=pred_hidden, batch_size=batch_size,
                  lr=lr, weight_decay=weight_decay, momentum=momentum,
                  warmup_epochs=warmup_epochs, min_scale=min_scale, ra_n=ra_n, ra_m=ra_m,
                  crop=crop, flip=flip, cutout=cutout, sup_weight=sup_weight,
                  ft_epochs=ft_epochs, ft_lr=ft_lr, C_grid=C_grid, svm_max_iter=svm_max_iter,
                  seed=seed)
        super().__init__(**hp)
        for k, v in hp.items():
            setattr(self, k, v)
        self.C_grid = tuple(C_grid)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # fail fast on a bad backbone / augmentation config
        build_backbone(self._arch())
        Augmenter(self._aug_cfg())

    # ---- config helpers ----------------------------------------------------
    def _arch(self):
        return {**FINAL_ARCH, "width": self.width, "convs": self.convs}

    def _aug_cfg(self):
        ra = {"n": self.ra_n, "m": self.ra_m} if self.ra_n > 0 else None
        return {"randaugment": ra, "crop": self.crop, "flip": self.flip, "cutout": self.cutout}

    def _make_aug(self, seed):
        return Augmenter(self._aug_cfg(), self.mean_, self.std_, num_classes=2, seed=seed)

    def _to_tensor(self, X):
        return torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).contiguous().to(self.device)

    # ---- training ----------------------------------------------------------
    def _pretrain(self, x, y):
        n = len(x)
        gen = torch.Generator().manual_seed(self.seed)
        aug = self._make_aug(self.seed)
        joint = self.sup_weight > 0
        if joint:
            self.cls_ = nn.Linear(self.net_.feat_dim, 2).to(self.device)
        params = list(self.net_.parameters()) + (list(self.cls_.parameters()) if joint else [])
        opt = torch.optim.SGD(params, lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)
        steps = math.ceil(n / self.batch_size)
        total = self.epochs * steps
        warmup = min(self.warmup_epochs * steps, total // 2)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, warmup))
        self.net_.train()
        for _ in range(self.epochs):
            for idx in balanced_batches(n, self.batch_size, gen):
                idx = idx.to(self.device)
                xb = x[idx]
                # labels only reach the loss when sup_weight > 0; otherwise a dummy is passed
                yb = y[idx] if joint else torch.zeros(len(idx), dtype=torch.long, device=self.device)
                v1 = random_resized_crop(aug(xb, yb)[0], self.min_scale, gen)
                v2 = random_resized_crop(aug(xb, yb)[0], self.min_scale, gen)
                f1, z1, p1 = self.net_(v1)
                f2, z2, p2 = self.net_(v2)
                loss = 0.5 * (_neg_cosine(p1, z2) + _neg_cosine(p2, z1))
                if joint:
                    ce = 0.5 * (F.cross_entropy(self.cls_(f1), yb) + F.cross_entropy(self.cls_(f2), yb))
                    loss = loss + self.sup_weight * ce
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()

    def _finetune(self, x, y):
        """Supervised fine-tuning of backbone + a new linear head (same views as pretraining, no RRC)."""
        n = len(x)
        gen = torch.Generator().manual_seed(self.seed + 1)
        aug = self._make_aug(self.seed + 1)
        self.cls_ = nn.Linear(self.net_.feat_dim, 2).to(self.device)
        params = list(self.net_.backbone.parameters()) + list(self.cls_.parameters())
        opt = torch.optim.SGD(params, lr=self.ft_lr, momentum=self.momentum, nesterov=True,
                              weight_decay=self.weight_decay)
        steps = math.ceil(n / self.batch_size)
        total = self.ft_epochs * steps
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, total, 0))
        self.net_.train()
        for _ in range(self.ft_epochs):
            for idx in balanced_batches(n, self.batch_size, gen):
                idx = idx.to(self.device)
                xb, y_soft = aug(x[idx], y[idx])
                loss = F.cross_entropy(self.cls_(self.net_.backbone(xb)), y_soft)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()

    @torch.no_grad()
    def _features(self, x_uint8, batch=500):
        self.net_.eval()
        return torch.cat([self.net_.backbone(to_float(x_uint8[i:i + batch], self.mean_, self.std_))
                          for i in range(0, len(x_uint8), batch)])

    # ---- Method interface --------------------------------------------------
    def fit(self, X, y):
        torch.manual_seed(self.seed)   # init + RandAugment's draws from the global RNG
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        x = self._to_tensor(X)
        xf = x.float().div(255.0)
        # Normalization stats from the 100 training images only (not CIFAR-wide constants).
        self.mean_ = tuple(xf.mean(dim=(0, 2, 3)).tolist())
        self.std_ = tuple(xf.std(dim=(0, 2, 3)).clamp_min(1e-3).tolist())
        y_np = np.asarray(y)
        y_t = torch.as_tensor(y_np, dtype=torch.long, device=self.device)

        self.net_ = SimSiamNet(self._arch(), self.proj_dim, self.proj_hidden,
                               self.pred_hidden).to(self.device)
        self.cls_ = None
        self.finetune_s_ = 0.0
        with Timer() as t:
            if self.epochs > 0:
                self._pretrain(x, y_t)
        self.pretrain_s_ = t.seconds

        self.net_.eval()
        with torch.no_grad():
            z = F.normalize(self.net_.projector(self.net_.backbone(to_float(x, self.mean_, self.std_))), dim=1)
        self.collapse_std_ = float(z.std(0).mean())   # ~1/sqrt(proj_dim) healthy, ~0 collapsed

        if self.ft_epochs > 0:
            with Timer() as t:
                self._finetune(x, y_t)
            self.finetune_s_ = t.seconds
            self.classifier_ = None
        else:
            Z = self._features(x).cpu().numpy()
            self.scaler_ = StandardScaler().fit(Z)
            search = GridSearchCV(LinearSVC(max_iter=self.svm_max_iter, random_state=self.seed),
                                  {"C": list(self.C_grid)}, cv=5)
            search.fit(self.scaler_.transform(Z), y_np)
            self.classifier_ = search.best_estimator_
            self.best_C_ = search.best_params_["C"]
        return self

    def predict(self, X):
        feats = self._features(self._to_tensor(X))
        if self.classifier_ is None:   # fine-tuned: linear head on the backbone
            with torch.no_grad():
                return self.cls_(feats).argmax(1).cpu().numpy().astype(np.int64)
        Z = self.scaler_.transform(feats.cpu().numpy())
        return self.classifier_.predict(Z).astype(np.int64)

    def n_trainable_params(self):
        n = sum(p.numel() for p in self.net_.backbone.parameters())
        if self.classifier_ is None:
            return int(n + sum(p.numel() for p in self.cls_.parameters()))
        return int(n + self.classifier_.coef_.size + self.classifier_.intercept_.size)

    def extra_notes(self):
        out = {"pretrain_s": self.pretrain_s_, "collapse_std": self.collapse_std_}
        if self.ft_epochs > 0:
            out["finetune_s"] = self.finetune_s_
        else:
            out["best_C"] = self.best_C_
        if self.device.type == "cuda":
            out["gpu"] = torch.cuda.get_device_name().replace(" ", "-")
        return out