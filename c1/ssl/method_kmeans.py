"""Method 1 (C1, fast variant): single-layer unsupervised features, Coates, Ng & Lee (2011).

Pipeline, fit on the 100 training images only:
  random patches -> contrast normalization -> ZCA whitening -> k-means dictionary
  -> triangle encoding at every patch position -> quadrant sum pooling
  -> standardization -> linear SVM, with C chosen by cross-validation on the training features.
"""
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common.method import Method


class KMeansMethod(Method):
    name = "kmeans"

    def __init__(self, patch_size=6, n_clusters=200, stride=1, num_patches=20000,
                 epsilon=0.1, whitening_epsilon=1e-3, C_grid=(0.01, 0.1, 1.0, 10.0),
                 n_init=10, kmeans_max_iter=100, svm_max_iter=5000, batch_size=200,
                 image_size=32, seed=0):
        super().__init__(patch_size=patch_size, n_clusters=n_clusters, stride=stride,
                         num_patches=num_patches, epsilon=epsilon,
                         whitening_epsilon=whitening_epsilon, C_grid=C_grid, n_init=n_init,
                         kmeans_max_iter=kmeans_max_iter, svm_max_iter=svm_max_iter,
                         batch_size=batch_size, image_size=image_size, seed=seed)
        self.patch_size, self.n_clusters, self.stride = patch_size, n_clusters, stride
        self.num_patches, self.epsilon = num_patches, epsilon
        self.whitening_epsilon, self.C_grid = whitening_epsilon, tuple(C_grid)
        self.n_init, self.kmeans_max_iter = n_init, kmeans_max_iter
        self.svm_max_iter, self.batch_size, self.seed = svm_max_iter, batch_size, seed
        self.valid_positions = image_size - patch_size + 1
        self.grid_size = (self.valid_positions - 1) // stride + 1
        self.patches_per_image = self.grid_size ** 2
        self.rng = np.random.default_rng(seed)

    def random_patches(self, X):
        idx = self.rng.integers(len(X), size=self.num_patches)
        h = self.rng.integers(self.valid_positions, size=self.num_patches)
        w = self.rng.integers(self.valid_positions, size=self.num_patches)
        p = self.patch_size
        patches = np.stack([X[i, a:a + p, b:b + p, :] for i, a, b in zip(idx, h, w)])
        return patches.reshape(self.num_patches, -1)

    @staticmethod
    def normalization(patches, epsilon):
        patches = patches.astype(np.float32)
        mean = patches.mean(axis=1, keepdims=True)
        std = patches.std(axis=1, keepdims=True)
        return (patches - mean) / (std + epsilon)

    def whitening(self, X, epsilon, fit=True):
        X = X.float()
        if fit:
            self.mean = X.mean(dim=0)
            Xc = X - self.mean
            cov = Xc.T @ Xc / Xc.shape[0]
            S, U = torch.linalg.eigh(cov)
            S = S.clamp(min=0)
            self.W = (U * torch.rsqrt(S + epsilon)) @ U.T
        return (X - self.mean) @ self.W

    def triangle_encoding(self, patches):
        distances = torch.cdist(patches.float(), self.centroids)
        mean_distances = torch.mean(distances, dim=1, keepdim=True)
        return torch.relu(mean_distances - distances)

    @staticmethod
    def pooling(y):
        h = y.shape[1] // 2
        w = y.shape[2] // 2
        quadrants = [y[:, :h, :w, :], y[:, :h, w:, :], y[:, h:, w:, :], y[:, h:, :w, :]]
        return torch.cat([q.sum(axis=(1, 2)) for q in quadrants], dim=1).numpy()

    def extract_patches(self, X):
        p = self.patch_size
        positions = range(0, self.valid_positions, self.stride)
        patches = np.array([X[:, h:h + p, w:w + p, :] for h in positions for w in positions])
        patches = patches.transpose(1, 0, 2, 3, 4)
        return patches.reshape(len(X) * self.patches_per_image, -1)

    def encode(self, X):
        feats = []
        for i in range(0, len(X), self.batch_size):
            xb = X[i:i + self.batch_size]
            p = self.normalization(self.extract_patches(xb), self.epsilon)
            p = self.whitening(torch.from_numpy(p), self.whitening_epsilon, fit=False)
            v = self.triangle_encoding(p).reshape(len(xb), self.grid_size, self.grid_size,
                                                  self.n_clusters)
            feats.append(self.pooling(v))
        return np.concatenate(feats)

    def fit(self, X, y):
        y = np.asarray(y)
        self.rng = np.random.default_rng(self.seed)
        patches = self.normalization(self.random_patches(X), self.epsilon)
        patches = self.whitening(torch.from_numpy(patches), self.whitening_epsilon, fit=True)
        km = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=self.n_init,
                    max_iter=self.kmeans_max_iter, random_state=self.seed).fit(patches.numpy())
        self.centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)

        F_tr = self.encode(X)
        self.scaler = StandardScaler().fit(F_tr)
        search = GridSearchCV(LinearSVC(max_iter=self.svm_max_iter, random_state=self.seed),
                              {"C": list(self.C_grid)}, cv=5)
        search.fit(self.scaler.transform(F_tr), y)
        self.classifier = search.best_estimator_
        self.best_C = search.best_params_["C"]
        return self

    def predict(self, X):
        return self.classifier.predict(self.scaler.transform(self.encode(X))).astype(np.int64)

    def n_trainable_params(self):
        return int(self.centroids.numel() + self.classifier.coef_.size
                   + self.classifier.intercept_.size)