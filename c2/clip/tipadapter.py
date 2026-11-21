import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from c2.clip.loadClip import DEVICE, SCALE, class_text_features, fresh_clip, image_features
from common.method import Method


def tip_logits(feats, keys, values, text, alpha, beta):
    cache = torch.exp(-beta * (1 - feats @ keys.T)) @ values
    return SCALE * feats @ text.T + alpha * cache


class TipAdapter(Method):
    name = "tip_adapter_clip"

    def __init__(self, class_names, alpha=1.0, beta=5.5, seed=0, **hparams):
        super().__init__(alpha=alpha, beta=beta, seed=seed)
        if len(class_names) != 2:
            raise ValueError("binary problems only")
        self.class_names = tuple(class_names)
        self.alpha, self.beta, self.seed = alpha, beta, seed
        self.text_ = self.keys_ = self.values_ = None
        self.timing_ = {}

    def fit(self, X, y):
        fresh_clip()
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=DEVICE)
        self.text_ = class_text_features(self.class_names)
        self.keys_ = image_features(X)
        self.values_ = F.one_hot(y, num_classes=2).float()
        return self

    @torch.no_grad()
    def predict(self, X):
        feats = image_features(X, timing=self.timing_)
        logits = tip_logits(feats, self.keys_, self.values_, self.text_, self.alpha, self.beta)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    def n_trainable_params(self):
        return 0


class TipAdapterF(TipAdapter):
    name = "tip_adapter_f_clip"

    def __init__(self, class_names, lr=1e-4, steps=100, **kw):
        super().__init__(class_names, **kw)
        self.lr, self.steps = lr, steps

    def fit(self, X, y):
        super().fit(X, y)  # reloads CLIP and builds the cache
        feats, labels = self.keys_, self.values_.argmax(dim=1)
        keys = nn.Parameter(feats.clone())
        opt = torch.optim.AdamW([keys], lr=self.lr, weight_decay=0.0)
        for _ in range(self.steps):
            logits = tip_logits(feats, F.normalize(keys, dim=-1), self.values_,
                                self.text_, self.alpha, self.beta)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.keys_ = F.normalize(keys.detach(), dim=-1)
        return self

    def n_trainable_params(self):
        return int(self.keys_.numel())