"""Method 1: k-means on fixed (untrained) features, as one `Method`.

Pipeline: [flip-augment] -> features -> [StandardScaler] -> [PCA] -> k-means -> cluster head.
Everything is fit on the 100 training images only.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from c1.classical import features as F
from c1.ssl.heads import ClusterHead
from common.method import Method


class KMeansMethod(Method):
    name = "kmeans"

    def __init__(self, features, k=8, mode="vote", scale=True, pca=None,
                 flip_train=False, flip_tta=False, n_init=10, C=1.0, seed=0):
        super().__init__(features=features, k=k, mode=mode, scale=scale, pca=pca,
                         flip_train=flip_train, flip_tta=flip_tta, n_init=n_init, C=C, seed=seed)
        self.features = list(features)
        self.k, self.mode, self.scale, self.pca = k, mode, scale, pca
        self.flip_train, self.flip_tta = flip_train, flip_tta
        self.n_init, self.C, self.seed = n_init, C, seed

    def _transform(self, X):
        Z = F.extract(X, self.features)
        if self.scaler_ is not None:
            Z = self.scaler_.transform(Z)
        if self.pca_ is not None:
            Z = self.pca_.transform(Z)
        return Z

    def fit(self, X, y):
        F.clear_cache()
        y = np.asarray(y)
        if self.flip_train:
            X = np.concatenate([X, F.flip(X)])
            y = np.concatenate([y, y])
        Z = F.extract(X, self.features)
        self.scaler_ = StandardScaler().fit(Z) if self.scale else None
        if self.scaler_ is not None:
            Z = self.scaler_.transform(Z)
        self.pca_ = None
        if self.pca:
            self.pca_ = PCA(n_components=min(self.pca, *Z.shape), random_state=self.seed).fit(Z)
            Z = self.pca_.transform(Z)
        self.head_ = ClusterHead(k=self.k, mode=self.mode, n_init=self.n_init,
                                 C=self.C, seed=self.seed).fit(Z, y)
        return self

    def predict(self, X):
        scores = self.head_.scores(self._transform(X))
        if self.flip_tta:
            scores = (scores + self.head_.scores(self._transform(F.flip(X)))) / 2
        return (scores > 0).astype(np.int64)

    def n_trainable_params(self):
        return self.head_.n_params()
