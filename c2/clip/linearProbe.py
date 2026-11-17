import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from c2.clip.loadClip import DEVICE, SCALE, class_text_features, image_features
from common.method import Method


class LinearProbe(Method):
    name = "linear_probe_clip"

    def __init__(self, class_names, lr=1e-3, steps=100, mix=1.0, seed=0, **hparams):
        super().__init__(lr=lr, steps=steps, mix=mix, seed=seed)
        if len(class_names) != 2:
            raise ValueError("binary problems only")
        self.class_names = tuple(class_names)
        self.lr, self.steps, self.mix, self.seed = lr, steps, mix, seed
        self.init_ = self.head_ = None

    def fit(self, X, y):
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=DEVICE)
        feats = image_features(X)
        head = nn.Linear(feats.shape[1], 2, bias=False).to(DEVICE)
        with torch.no_grad():
            head.weight.copy_(SCALE * class_text_features(self.class_names))
        self.init_ = head.weight.detach().clone()
        opt = torch.optim.AdamW(head.parameters(), lr=self.lr)
        for _ in range(self.steps):
            loss = F.cross_entropy(head(feats), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.head_ = head.weight.detach().clone()
        return self

    @torch.no_grad()
    def predict(self, X):
        W = (1 - self.mix) * self.init_ + self.mix * self.head_
        return (image_features(X) @ W.T).argmax(dim=1).cpu().numpy().astype(np.int64)

    def n_trainable_params(self):
        return int(self.head_.numel())
