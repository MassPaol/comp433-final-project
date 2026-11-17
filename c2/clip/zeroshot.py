import numpy as np
import torch

from c2.clip.loadClip import class_text_features, image_features
from common.method import Method


class ZeroShotMethod(Method):
    name = "zeroshot_clip"

    def __init__(self, class_names, seed=0, batch_size=256, **hparams):
        super().__init__(seed=seed)
        if len(class_names) != 2:
            raise ValueError("binary problems only")
        self.class_names = tuple(class_names)
        self.seed, self.batch_size = seed, batch_size
        self.text_ = None

    def fit(self, X, y):
        self.text_ = class_text_features(self.class_names)
        return self

    @torch.no_grad()
    def predict(self, X):
        sims = image_features(X, self.batch_size) @ self.text_.T
        return sims.argmax(dim=1).cpu().numpy().astype(np.int64)

    def n_trainable_params(self):
        return 0
