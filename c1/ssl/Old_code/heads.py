"""Heads that sit on top of any feature matrix (hand-crafted features or SimSiam embeddings).

Every head has the same interface:
    fit(Z, y) -> self
    scores(Z) -> float array, > 0 means class 1
    n_params() -> number of stored/trained parameters
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class ClusterHead:
    """k-means on the (unlabeled) features, then labels enter only to name the clusters.

    mode="vote": each cluster predicts the smoothed fraction of class-1 training
                 images it contains; a test image gets the score of its nearest centroid.
    mode="dist": the k distances to the centroids become a new feature vector
                 (Coates & Ng 2011 style) and a logistic regression is fit on it.
    """

    def __init__(self, k=8, mode="vote", n_init=10, C=1.0, seed=0):
        self.k, self.mode, self.n_init, self.C, self.seed = k, mode, n_init, C, seed

    def fit(self, Z, y):
        y = np.asarray(y)
        k = min(self.k, len(Z))
        self.km_ = KMeans(n_clusters=k, n_init=self.n_init, random_state=self.seed).fit(Z)
        if self.mode == "vote":
            lab = self.km_.labels_
            # Laplace smoothing: a cluster with one class-1 image is not "100% class 1"
            self.p_ = np.array([(y[lab == c].sum() + 0.5) / ((lab == c).sum() + 1)
                                for c in range(k)])
        elif self.mode == "dist":
            D = self.km_.transform(Z)
            self.scaler_ = StandardScaler().fit(D)
            self.lr_ = LogisticRegression(C=self.C, max_iter=5000).fit(self.scaler_.transform(D), y)
        else:
            raise ValueError(f"unknown mode {self.mode!r}; expected 'vote' or 'dist'")
        return self

    def scores(self, Z):
        if self.mode == "vote":
            return self.p_[self.km_.predict(Z)] - 0.5
        return self.lr_.decision_function(self.scaler_.transform(self.km_.transform(Z)))

    def n_params(self):
        n = self.km_.cluster_centers_.size
        if self.mode == "vote":
            return int(n + len(self.p_))
        return int(n + self.lr_.coef_.size + self.lr_.intercept_.size)


class LRHead:
    def __init__(self, C=1.0, seed=0):
        self.C, self.seed = C, seed

    def fit(self, Z, y):
        self.m_ = LogisticRegression(C=self.C, max_iter=5000).fit(Z, y)
        return self

    def scores(self, Z):
        return self.m_.decision_function(Z)

    def n_params(self):
        return int(self.m_.coef_.size + self.m_.intercept_.size)


class KNNHead:
    def __init__(self, n_neighbors=5, metric="cosine", seed=0):
        self.n_neighbors, self.metric, self.seed = n_neighbors, metric, seed

    def fit(self, Z, y):
        k = min(self.n_neighbors, len(Z))
        self.m_ = KNeighborsClassifier(n_neighbors=k, metric=self.metric).fit(Z, y)
        return self

    def scores(self, Z):
        return self.m_.predict_proba(Z)[:, 1] - 0.5

    def n_params(self):
        return 0  # stores the training set, no trained parameters


HEADS = {"kmeans": ClusterHead, "lr": LRHead, "knn": KNNHead}


def make_head(name, params=None, seed=0):
    if name not in HEADS:
        raise ValueError(f"unknown head {name!r}; expected one of {sorted(HEADS)}")
    return HEADS[name](**{**dict(params or {}), "seed": seed})
