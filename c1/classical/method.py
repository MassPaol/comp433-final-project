"""Hand-crafted features + classical classifier, as one `Method`.

Pipeline: [flip-augment] -> features -> [StandardScaler] -> [PCA] -> classifier.
Everything is fit on the training images only.
"""
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from c1.classical import features as F
from common.method import Method

CLASSIFIERS = ("lr", "lr_cv", "linsvm", "rbfsvm", "rf", "hgb", "xgb", "knn")


class ClassicalMethod(Method):
    name = "classical"

    def __init__(self, features, clf, clf_params=None, scale=True, pca=None,
                 flip_train=False, flip_tta=False, seed=0):
        super().__init__(features=features, clf=clf, clf_params=clf_params, scale=scale,
                         pca=pca, flip_train=flip_train, flip_tta=flip_tta, seed=seed)
        self.features = list(features)
        self.clf = clf
        self.clf_params = dict(clf_params or {})
        self.scale = scale
        self.pca = pca
        self.flip_train = flip_train
        self.flip_tta = flip_tta
        self.seed = seed

    # ---- pipeline ---------------------------------------------------------
    def _transform(self, X):
        Z = F.extract(X, self.features)
        if self.scaler_ is not None:
            Z = self.scaler_.transform(Z)
        if self.pca_ is not None:
            Z = self.pca_.transform(Z)
        return Z

    @staticmethod
    def _cv_splits(n, y, flip_train):
        """5 stratified folds where an image and its flipped copy share a group."""
        groups = np.arange(len(y)) % n if flip_train else np.arange(len(y))
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        return list(cv.split(np.zeros(len(y)), y, groups))

    def _make_classifier(self, Z, y, n_original):
        p = dict(self.clf_params)
        s = self.seed
        if self.clf == "lr":
            return LogisticRegression(**{"C": 1.0, "max_iter": 5000, **p})
        if self.clf == "lr_cv":
            cv = self._cv_splits(n_original, y, self.flip_train)
            return LogisticRegressionCV(**{"Cs": np.logspace(-3, 2, 11), "cv": cv,
                                           "scoring": "neg_log_loss", "max_iter": 5000, **p})
        if self.clf == "linsvm":
            return LinearSVC(**{"C": 1.0, "max_iter": 20000, "random_state": s, **p})
        if self.clf == "rbfsvm":
            gamma_mult = p.pop("gamma_mult", 1.0)
            var = Z.var() or 1.0
            self.gamma_ = gamma_mult / (Z.shape[1] * var)  # multiple of sklearn's "scale"
            return SVC(**{"C": 1.0, "kernel": "rbf", **p, "gamma": self.gamma_})
        if self.clf == "rf":
            return RandomForestClassifier(**{"n_estimators": 500, "random_state": s,
                                             "n_jobs": -1, **p})
        if self.clf == "hgb":
            return HistGradientBoostingClassifier(**{"random_state": s, **p})
        if self.clf == "xgb":
            from xgboost import XGBClassifier
            return XGBClassifier(**{"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1,
                                    "random_state": s, "n_jobs": -1, **p})
        if self.clf == "knn":
            return KNeighborsClassifier(**{"n_neighbors": 5, **p})
        raise ValueError(f"unknown clf {self.clf!r}; expected one of {CLASSIFIERS}")

    # ---- Method interface -------------------------------------------------
    def fit(self, X, y):
        F.clear_cache()  # every fit pays for building its feature operators
        y = np.asarray(y)
        n_original = len(X)
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
        self.clf_ = self._make_classifier(Z, y, n_original)
        with warnings.catch_warnings():
            # sklearn >= 1.8 announces LogisticRegressionCV default changes; the
            # parameters that silence them don't exist in older sklearn (Colab).
            warnings.simplefilter("ignore", FutureWarning)
            self.clf_.fit(Z, y)
        return self

    def _scores(self, X):
        Z = self._transform(X)
        if self.clf == "rbfsvm":
            # Same values as SVC.decision_function (checked in tests), but via BLAS:
            # libsvm's per-sample kernel loop is ~100x slower on 15k-dim features.
            c = self.clf_
            K = rbf_kernel(Z, c.support_vectors_, gamma=c.gamma)
            return K @ c.dual_coef_[0] + c.intercept_[0], 0.0
        if hasattr(self.clf_, "decision_function"):
            return self.clf_.decision_function(Z), 0.0
        return self.clf_.predict_proba(Z)[:, 1], 0.5

    def predict(self, X):
        scores, threshold = self._scores(X)
        if self.flip_tta:
            flipped, _ = self._scores(F.flip(X))
            scores = (scores + flipped) / 2
        return (scores > threshold).astype(np.int64)

    def n_trainable_params(self):
        c = self.clf_
        if self.clf in ("lr", "lr_cv", "linsvm"):
            return int(c.coef_.size + c.intercept_.size)
        if self.clf == "rbfsvm":
            return int(c.dual_coef_.size + c.intercept_.size)
        if self.clf == "rf":
            return int(sum(e.tree_.node_count for e in c.estimators_))
        if self.clf == "hgb":
            return int(sum(p.nodes.shape[0] for it in c._predictors for p in it))
        if self.clf == "xgb":
            return int(len(c.get_booster().trees_to_dataframe()))
        return 0  # knn: no trained parameters, it stores the training set
