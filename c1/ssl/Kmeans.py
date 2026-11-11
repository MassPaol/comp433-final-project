import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import random

class Kmeans:
    name = "base"

    def __init__(self, patch_size=6, C=1.0, num_patches=20000, epsilon=0.1,whitening_epsilon=1e-3, n_clusters=500, stride=1,
                 image_size=32, channels=3, batch_size=200,kmeans_n_init=10, kmeans_max_iter=100,
                 svm_max_iter=5000,seed=42, **hparams):
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.hparams = hparams
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.epsilon = epsilon
        self.whitening_epsilon = whitening_epsilon
        self.n_clusters = n_clusters
        self.stride = stride
        self.C = C

        self.image_size = image_size
        self.channels = channels
        self.batch_size = batch_size
        self.kmeans_n_init = kmeans_n_init
        self.kmeans_max_iter = kmeans_max_iter
        self.svm_max_iter = svm_max_iter

        self.valid_positions = image_size - patch_size + 1
        self.grid_size = (self.valid_positions - 1) // stride + 1
        self.patches_per_image = self.grid_size ** 2
        self.patch_dim = patch_size * patch_size * channels


    def random_patches(self,X):
        self.indice = self.rng.integers(len(X), size=self.num_patches)
        self.height = self.rng.integers(self.valid_positions , size=self.num_patches)
        self.width = self.rng.integers(self.valid_positions , size=self.num_patches)
        patches = np.array([X[i, h:h+self.patch_size, w:w+self.patch_size, :]
            for i, h, w in zip(
                self.indice,
                self.height,
                self.width
            )
        ])
        patches = patches.reshape(self.num_patches, -1)
        return patches

    def normalization(self,patches,epsilon=1e-5):
        patches = patches.astype(np.float32)
        mean = patches.mean(axis=1,keepdims=True)
        std = patches.std(axis=1, keepdims=True)
        patches = (patches - mean) / (std + epsilon)
        return patches

    def whitening(self,X, epsilon=1e-5, fit=True):
        X = X.float()
        if fit:
            self.mean = X.mean(dim=0)
            Xc = X - self.mean
            cov = Xc.T @ Xc / Xc.shape[0]
            S, U = torch.linalg.eigh(cov)
            S = S.clamp(min=0)
            self.W = (U * torch.rsqrt(S + epsilon)) @ U.T
        return (X - self.mean) @ self.W

    def triangle_encoding(self,patches):
        distances = torch.cdist(patches.float(),self.centroids)
        mean_distances = torch.mean(distances,dim=1,keepdim=True)
        return torch.relu(mean_distances-distances)

    def pooling(self,y):
       h = y.shape[1] // 2
       w = y.shape[2] // 2
       quadrant1= y[:,:h,:w,:].sum(axis=(1,2))
       quadrant2= y[:,:h,w:,:].sum(axis=(1,2))
       quadrant3= y[:,h:,w:,:].sum(axis=(1,2))
       quadrant4= y[:,h:,:w,:].sum(axis=(1,2))
       return  torch.cat((quadrant1, quadrant2, quadrant3, quadrant4), dim=1).numpy()

    def extract_patches(self, X):
        patches = np.array([
            X[:, h:h + self.patch_size, w:w + self.patch_size, :]
            for h in range(self.valid_positions,self.stride)
            for w in range(self.valid_positions,self.stride)
        ])

        patches = patches.transpose(1, 0, 2, 3, 4)
        return patches.reshape(len(X) * self.patches_per_image, -1)

    def encode(self, X, batch_size=200):
        feats = []
        for i in range(0, len(X), batch_size):
            xb = X[i:i + batch_size]
            p = self.extract_patches(xb)
            p = self.normalization(p, self.epsilon)
            p = self.whitening(torch.from_numpy(p), self.whitening_epsilon,fit=False)
            v = self.triangle_encoding(p).reshape(len(xb), self.grid_size, self.grid_size, self.n_clusters)
            feats.append(self.pooling(v))
        return np.concatenate(feats)

    def fit(self, X, y):
        self.rng = np.random.default_rng(self.seed)
        rng = np.random.default_rng(self.seed)
        patches = self.random_patches(X)
        patches = self.normalization(patches, self.epsilon)
        patches = self.whitening(torch.from_numpy(patches), self.whitening_epsilon)
        print("step done")
        self.kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init= self.kmeans_n_init, max_iter=self.kmeans_max_iter, random_state=self.seed)
        self.kmeans.fit(patches.numpy())
        self.centroids = torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(self.encode(X))
        self.classifier = LinearSVC(C=self.C, max_iter=self.svm_max_iter,random_state=self.seed)
        self.classifier.fit(scaled_features, y)
        print("training done")
        return self

    def predict(self, X):
        return self.classifier.predict(self.scaler.transform(self.encode(X)))
