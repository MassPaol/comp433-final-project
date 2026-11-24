"""Small CNN trained from scratch on the 100 training images, as one `Method`.

fit:  per-channel mean/std from the training images -> images to device once ->
      `epochs` epochs of shuffled mini-batches, each augmented on the device by
      common.augment.Augmenter -> cross-entropy on (possibly soft) targets.
      With ensemble=k, k models are trained independently (different seeds).
predict: softmax averaged over ensemble members and, with tta_flip, over each
      image and its mirror image.

The number of epochs is fixed by the config: nothing looks at validation data.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from c1.cnn.models import build_model, count_params
from common.augment import Augmenter, to_float
from common.method import Method

OPT_DEFAULTS = {"name": "sgd", "lr": 0.05, "wd": 5e-4, "momentum": 0.9, "nesterov": True,
                "betas": (0.9, 0.999)}
SCHED_DEFAULTS = {"name": "cosine", "warmup_epochs": 5, "pct_start": 0.3}


def _check_keys(given, defaults, what):
    unknown = set(given or {}) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {what} keys {sorted(unknown)}; expected {sorted(defaults)}")
    return {**defaults, **(given or {})}


def balanced_batches(n, batch_size, gen):
    """A random permutation of range(n) split into ceil(n / batch_size) near-equal batches.

    100 images at batch_size 32 -> 4 batches of 25 instead of 32/32/32/4, so no
    step sees a tiny batch (bad for BatchNorm and noisy gradients).
    """
    perm = torch.randperm(n, generator=gen)
    return torch.tensor_split(perm, math.ceil(n / batch_size))


def lr_factor(step, total, warmup):
    """Linear warmup over `warmup` steps, then cosine decay to 0 at `total`."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


class CNNMethod(Method):
    name = "cnn"

    def __init__(self, arch=None, aug=None, opt=None, sched=None, epochs=200, batch_size=32,
                 label_smoothing=0.0, tta_flip=False, ensemble=1, seed=0, amp=True,
                 device=None):
        super().__init__(arch=arch, aug=aug, opt=opt, sched=sched, epochs=epochs,
                         batch_size=batch_size, label_smoothing=label_smoothing,
                         tta_flip=tta_flip, ensemble=ensemble, seed=seed, amp=amp)
        self.arch = dict(arch or {})
        build_model(self.arch)  # fail fast on a bad arch config
        self.aug = dict(aug or {})
        Augmenter(self.aug)     # fail fast on a bad augmentation config
        self.opt = _check_keys(opt, OPT_DEFAULTS, "opt")
        self.sched = _check_keys(sched, SCHED_DEFAULTS, "sched")
        if self.opt["name"] not in ("sgd", "adamw"):
            raise ValueError(f"unknown optimizer {self.opt['name']!r}")
        if self.sched["name"] not in ("cosine", "onecycle"):
            raise ValueError(f"unknown schedule {self.sched['name']!r}")
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.label_smoothing = float(label_smoothing)
        self.tta_flip = bool(tta_flip)
        self.ensemble = int(ensemble)
        self.seed = int(seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = bool(amp) and self.device.type == "cuda"

    # ---- helpers -----------------------------------------------------------
    def _to_tensor(self, X):
        return torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).contiguous().to(self.device)

    def _make_optimizer(self, model):
        o = self.opt
        if o["name"] == "sgd":
            return torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"],
                                   nesterov=o["nesterov"], weight_decay=o["wd"])
        return torch.optim.AdamW(model.parameters(), lr=o["lr"], betas=tuple(o["betas"]),
                                 weight_decay=o["wd"])

    def _make_scheduler(self, optimizer, total_steps, steps_per_epoch):
        s = self.sched
        if s["name"] == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.opt["lr"], total_steps=total_steps, pct_start=s["pct_start"])
        warmup = min(s["warmup_epochs"] * steps_per_epoch, total_steps // 2)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: lr_factor(step, total_steps, warmup))

    def _train_one(self, x, y, member_seed, callback=None):
        torch.manual_seed(member_seed)  # weight init + RandAugment's global-RNG draws
        model = build_model(self.arch).to(self.device)
        aug = Augmenter(self.aug, self.mean_, self.std_, num_classes=2, seed=member_seed)
        gen = torch.Generator().manual_seed(member_seed)
        steps_per_epoch = math.ceil(len(x) / self.batch_size)
        total = self.epochs * steps_per_epoch
        optimizer = self._make_optimizer(model)
        scheduler = self._make_scheduler(optimizer, total, steps_per_epoch)
        scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        history = []
        for epoch in range(self.epochs):
            model.train()
            loss_sum = 0.0
            for idx in balanced_batches(len(x), self.batch_size, gen):
                idx = idx.to(self.device)
                xb, yb = aug(x[idx], y[idx])
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                    logits = model(xb)
                    loss = F.cross_entropy(logits.float(), yb, label_smoothing=self.label_smoothing)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                loss_sum += loss.detach() * len(idx)
            history.append(loss_sum)
            if callback is not None:
                callback(epoch, model)
        self.history_.append([float(v) / len(x) for v in history])  # one GPU sync per member
        return model

    @torch.no_grad()
    def _probs(self, model, x_uint8, batch=500):
        model.eval()
        out = []
        for i in range(0, len(x_uint8), batch):
            xb = to_float(x_uint8[i:i + batch], self.mean_, self.std_)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                p = model(xb).float().softmax(1)
                if self.tta_flip:
                    p = (p + model(xb.flip(3)).float().softmax(1)) / 2
            out.append(p)
        return torch.cat(out)

    # ---- Method interface --------------------------------------------------
    def fit(self, X, y, callback=None):
        """`callback(member, epoch, model)` is for diagnostics (c1/cnn/curves.py) only."""
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # fixed input shapes: let cuDNN pick fast kernels
        x = self._to_tensor(X)
        xf = x.float().div(255.0)
        # Normalization stats from the 100 training images only (not CIFAR-wide constants).
        self.mean_ = tuple(xf.mean(dim=(0, 2, 3)).tolist())
        self.std_ = tuple(xf.std(dim=(0, 2, 3)).clamp_min(1e-3).tolist())
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=self.device)
        self.history_ = []
        self.models_ = []
        for m in range(self.ensemble):
            cb = None if callback is None else (lambda e, mod, m=m: callback(m, e, mod))
            self.models_.append(self._train_one(x, y, self.seed + 1000 * m, cb))
        return self

    def predict_proba(self, X):
        x = self._to_tensor(X)
        p = sum(self._probs(m, x) for m in self.models_) / len(self.models_)
        return p.cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1).astype(np.int64)

    def n_trainable_params(self):
        return int(sum(count_params(m) for m in self.models_))
