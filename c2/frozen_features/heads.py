"""Lightweight heads on frozen features, and the feature normalizations before them.

Every head exposes fit(Z, y, groups=None), score(Z) (> 0 means class 1) and
n_params(). Scores are centered at 0 so flip-TTA can average them for any head.
"""
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

NORMS = ("none", "l2", "standard", "cl2n")
HEADS = ("adamw", "lr", "lr_cv", "linsvm", "rbfsvm", "ncm", "knn")
# adamw: raw features, as in the testbed; distance heads: cosine geometry; the rest: standardized
DEFAULT_NORM = {"adamw": "none", "ncm": "l2", "knn": "l2",
                "lr": "standard", "lr_cv": "standard", "linsvm": "standard", "rbfsvm": "standard"}


def _l2(Z):
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


class FeatureNorm:
    """none | l2 (unit rows) | standard (z-score per dim) | cl2n (center on the training mean, then l2).

    cl2n is SimpleShot's CL2N (Wang et al. 2019). Statistics come from the
    training features only.
    """

    def __init__(self, kind):
        if kind not in NORMS:
            raise ValueError(f"unknown norm {kind!r}; expected one of {NORMS}")
        self.kind = kind

    def fit(self, Z):
        if self.kind == "standard":
            self.scaler_ = StandardScaler().fit(Z)
        if self.kind == "cl2n":
            self.mean_ = Z.mean(0)
        return self

    def transform(self, Z):
        if self.kind == "l2":
            return _l2(Z)
        if self.kind == "standard":
            return self.scaler_.transform(Z).astype(np.float32)
        if self.kind == "cl2n":
            return _l2(Z - self.mean_)
        return Z


def cv_splits(y, groups):
    """5 stratified folds that never split a group (an image and its flipped copy).

    Folds are drawn over the groups (one label each) with StratifiedKFold and then
    expanded to the groups' members. Not StratifiedGroupKFold(shuffle=True): in older
    scikit-learn (1.6, Colab's) it does not stratify, and on small inputs it left a
    one-class test fold, which crashes LR-CV's log-loss scoring.
    """
    y, groups = np.asarray(y), np.asarray(groups)
    uniq, first = np.unique(groups, return_index=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    folds = []
    for _, te in cv.split(uniq, y[first]):
        test = np.isin(groups, uniq[te])
        folds.append((np.flatnonzero(~test), np.flatnonzero(test)))
    return folds


class AdamWHead:
    """The testbed's head: a new nn.Linear(D, 2) trained with AdamW + cross-entropy.

    Same recipe as the testbed's frozen-backbone example (lr 1e-3, weight decay
    0.01, 10 epochs, batch 32, shuffled each epoch) and timm's head init
    (normal std 0.02, zero bias), but on the cached features, which is
    equivalent because the frozen backbone runs in eval mode.
    """

    def __init__(self, lr=1e-3, weight_decay=0.01, epochs=10, batch_size=32, seed=0):
        self.lr, self.weight_decay = lr, weight_decay
        self.epochs, self.batch_size, self.seed = epochs, batch_size, seed

    def fit(self, Z, y, groups=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator().manual_seed(self.seed)
        lin = torch.nn.Linear(Z.shape[1], 2)
        with torch.no_grad():
            torch.nn.init.trunc_normal_(lin.weight, std=0.02, generator=g)
            lin.bias.zero_()
        lin.to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        Zt = torch.as_tensor(Z, dtype=torch.float32, device=device)
        yt = torch.as_tensor(y, dtype=torch.long, device=device)
        for _ in range(self.epochs):
            for idx in torch.randperm(len(Zt), generator=g).split(self.batch_size):
                idx = idx.to(device)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(lin(Zt[idx]), yt[idx]).backward()
                opt.step()
        self.linear_ = lin.eval()
        return self

    def score(self, Z):
        with torch.no_grad():
            dev = self.linear_.weight.device
            out = self.linear_(torch.as_tensor(Z, dtype=torch.float32, device=dev))
        return (out[:, 1] - out[:, 0]).cpu().numpy()

    def n_params(self):
        return sum(p.numel() for p in self.linear_.parameters())


class NCMHead:
    """Nearest class mean (prototype) classifier: score = ||z - mu0||^2 - ||z - mu1||^2."""

    def fit(self, Z, y, groups=None):
        y = np.asarray(y)
        self.means_ = np.stack([Z[y == 0].mean(0), Z[y == 1].mean(0)])
        return self

    def score(self, Z):
        d = ((Z[:, None, :] - self.means_[None]) ** 2).sum(-1)
        return d[:, 0] - d[:, 1]

    def n_params(self):
        return int(self.means_.size)


class SklearnHead:
    """Wraps an sklearn classifier; `make_clf(Z, y, groups)` builds it at fit time."""

    def __init__(self, name, make_clf):
        self.name, self._make_clf = name, make_clf

    def fit(self, Z, y, groups=None):
        self.clf_ = self._make_clf(Z, y, groups)
        with warnings.catch_warnings():
            # sklearn >= 1.8 announces LogisticRegressionCV default changes
            warnings.simplefilter("ignore", FutureWarning)
            self.clf_.fit(Z, y)
        return self

    def score(self, Z):
        if hasattr(self.clf_, "decision_function"):
            return self.clf_.decision_function(Z)
        return self.clf_.predict_proba(Z)[:, 1] - 0.5

    def n_params(self):
        c = self.clf_
        if self.name == "knn":
            return 0  # stores the training set, trains nothing
        if self.name == "rbfsvm":
            return int(c.dual_coef_.size + c.intercept_.size)
        return int(c.coef_.size + c.intercept_.size)


def make_head(name, params=None, seed=0):
    """A fresh head. `params` override the defaults listed in the spec."""
    p = dict(params or {})
    if name == "adamw":
        return AdamWHead(**{"seed": seed, **p})
    if name == "ncm":
        return NCMHead()
    if name == "lr":
        return SklearnHead(name, lambda Z, y, g: LogisticRegression(**{"C": 1.0, "max_iter": 5000, **p}))
    if name == "lr_cv":
        def make(Z, y, g):
            groups = np.arange(len(y)) if g is None else g
            return LogisticRegressionCV(**{"Cs": np.logspace(-3, 2, 11), "cv": cv_splits(y, groups),
                                           "scoring": "neg_log_loss", "max_iter": 5000, **p})
        return SklearnHead(name, make)
    if name == "linsvm":
        return SklearnHead(name, lambda Z, y, g: LinearSVC(
            **{"C": 1.0, "max_iter": 20000, "random_state": seed, **p}))
    if name == "rbfsvm":
        def make(Z, y, g):
            q = dict(p)
            gamma_mult = q.pop("gamma_mult", 1.0)
            gamma = gamma_mult / (Z.shape[1] * (Z.var() or 1.0))  # multiple of sklearn's "scale"
            return SVC(**{"C": 1.0, "kernel": "rbf", **q, "gamma": gamma})
        return SklearnHead(name, make)
    if name == "knn":
        return SklearnHead(name, lambda Z, y, g: KNeighborsClassifier(
            **{"n_neighbors": 5, "metric": "cosine", **p}))
    raise ValueError(f"unknown head {name!r}; expected one of {HEADS}")
