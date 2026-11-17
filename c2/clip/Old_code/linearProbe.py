import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from common.method import Method
from loadClip import load_clip, DEVICE


class LinearProbe(Method):
    def __init__(self, class_names=None, C=0.316, max_iter=1000,batch_size=100, seed=0, **hparams):
        self.model, self.preprocess, self.tokenizer = load_clip()
        self.model = self.model.to(DEVICE).eval()
        self.device = DEVICE
        self.class_names = class_names
        self.C = C
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.seed = seed
        self.classifier = None

    @torch.no_grad()
    def _encode(self, images):
        dtype = next(self.model.parameters()).dtype
        feats = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size].to(self.device, dtype=dtype)
            feats.append(self.model.encode_image(batch).float().cpu())
        return torch.cat(feats).numpy()

    def fit(self, X, y):
        train_features = self._encode(X)
        train_labels = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        self.classifier = LogisticRegression(random_state=self.seed,C=self.C,max_iter=self.max_iter)
        self.classifier.fit(train_features, train_labels)
        return self

    @torch.no_grad()
    def predict(self, X):
        if self.classifier is None:
            raise RuntimeError("Call fit() before predict().")
        test_features = self._encode(X)
        return torch.from_numpy(self.classifier.predict(test_features))