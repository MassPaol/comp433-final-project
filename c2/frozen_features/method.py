"""Frozen pretrained features + a lightweight head, as one `Method`.

Pipeline: [flip-augment] -> frozen backbone(s) -> features -> normalization -> head.
Only the head is trained, on the training images' features. `fit` time
includes loading the backbone from local disk and extracting the features;
both are also reported separately through `extra_notes`.
"""
import numpy as np
import torch

from c2.frozen_features import backbones as B
from c2.frozen_features.heads import DEFAULT_NORM, HEADS, FeatureNorm, make_head
from common.method import Method
from common.timing import Timer


def _flip(X):
    return X[:, :, ::-1]


class FrozenFeatureMethod(Method):
    name = "frozen_features"

    def __init__(self, backbone, head, head_params=None, pool=None, layers=1, norm=None,
                 flip_train=False, flip_tta=False, precision="fp32", crop_pct=None,
                 pretrained=True, seed=0):
        super().__init__(backbone=backbone, head=head, head_params=head_params, pool=pool,
                         layers=layers, norm=norm, flip_train=flip_train, flip_tta=flip_tta,
                         precision=precision, crop_pct=crop_pct, pretrained=pretrained, seed=seed)
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
        self.backbones = [backbone] if isinstance(backbone, str) else list(backbone)
        self.head = head
        self.head_params = dict(head_params or {})
        self.pool = pool
        self.layers = layers
        self.norm_kind = norm or DEFAULT_NORM[head]
        FeatureNorm(self.norm_kind)  # fail at construction on an unknown norm
        self.flip_train = flip_train
        self.flip_tta = flip_tta
        self.precision = precision
        self.crop_pct = crop_pct
        self.pretrained = pretrained
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _features(self, X):
        parts = []
        for name in self.backbones:
            model, cfg = B.load(name, self.device, self.pretrained)
            parts.append(B.extract(model, X, cfg, pool=self.pool, layers=self.layers,
                                   precision=self.precision, crop_pct=self.crop_pct))
        return np.concatenate(parts, axis=1)

    # ---- Method interface -------------------------------------------------
    def fit(self, X, y):
        B.clear_cache()  # every fit pays for loading its backbone(s) from disk
        with Timer() as t:
            for name in self.backbones:
                B.load(name, self.device, self.pretrained)
        self.load_s_ = t.seconds

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        y = np.asarray(y)
        n = len(X)
        if self.flip_train:
            X = np.concatenate([X, _flip(X)])
            y = np.concatenate([y, y])
        self.groups_ = np.arange(len(y)) % n  # an image and its flip share a group

        with Timer() as t:
            Z = self._features(X)
        self.extract_s_ = t.seconds

        self.norm_ = FeatureNorm(self.norm_kind).fit(Z)
        self.train_features_ = self.norm_.transform(Z)
        self.head_ = make_head(self.head, self.head_params, self.seed)
        self.head_.fit(self.train_features_, y, self.groups_)
        return self

    def predict(self, X):
        if self.flip_tta:
            Z = self.norm_.transform(self._features(np.concatenate([X, _flip(X)])))
            scores = (self.head_.score(Z[:len(X)]) + self.head_.score(Z[len(X):])) / 2
        else:
            scores = self.head_.score(self.norm_.transform(self._features(X)))
        return (np.asarray(scores) > 0).astype(np.int64)

    def n_trainable_params(self):
        return int(self.head_.n_params())  # the backbone is frozen

    def extra_notes(self):
        return {"load_s": float(self.load_s_), "extract_s": float(self.extract_s_)}
