"""Method 2 (C1, deep variant): SimSiam (Chen & He 2021) on the 100 training images.

Pipeline, fit on the 100 training images only. No labels are used during pretraining:
  two augmented views (lightly SimSiamTransform) -> backbone -> projector -> predictor,
  symmetric negative cosine loss with stop-gradient on the projector output.
  Then freeze the backbone, embed the images, standardize, and fit a linear SVM
  (C chosen by cross-validation on the training embeddings).
epochs=0 skips pretraining (random-init backbone), the baseline for what SimSiam adds.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.transforms import SimSiamTransform
from PIL import Image
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset

from common.method import Method

# CIFAR-10 channel statistics, used for both the training views and the frozen embeddings
NORM = {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2470, 0.2435, 0.2616]}


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o),
                         nn.ReLU(inplace=True))


def small_cnn(width=32):
    """Placeholder backbone. Replace with Ryan's architecture through backbone_fn."""
    w = width
    return nn.Sequential(_block(3, w), _block(w, w), nn.MaxPool2d(2),
                         _block(w, 2 * w), _block(2 * w, 2 * w), nn.MaxPool2d(2),
                         _block(2 * w, 4 * w), _block(4 * w, 4 * w),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten())


def _out_dim(backbone):
    backbone.eval()
    with torch.no_grad():
        return backbone(torch.zeros(1, 3, 32, 32)).flatten(1).shape[1]


class SimSiamNetwork(nn.Module):
    def __init__(self, backbone, feature_dim, proj_dim=128, proj_hidden=256, pred_hidden=64):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(input_dim=feature_dim,
                                                     hidden_dim=proj_hidden,
                                                     output_dim=proj_dim)
        self.prediction_head = SimSiamPredictionHead(input_dim=proj_dim,
                                                     hidden_dim=pred_hidden,
                                                     output_dim=proj_dim)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        return z, p


class TwoViews(Dataset):
    """The 100 training images as PIL images; each item is two augmented views."""
    def __init__(self, X, transform):
        self.X, self.transform = X, transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        v = self.transform(Image.fromarray(self.X[i]))
        return v[0], v[1]


def _neg_cosine(p, z):
    """-cos(p, stopgrad(z)), averaged over the batch (SimSiam, eq. 1)."""
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


class SimSiamMethod(Method):
    name = "simsiam"

    def __init__(self, epochs=200, width=32, proj_dim=128, proj_hidden=256, pred_hidden=64,
                 batch_size=25, lr=0.05, weight_decay=5e-4, momentum=0.9,
                 min_scale=0.2, cj_strength=0.4, C_grid=(0.01, 0.1, 1.0, 10.0),
                 svm_max_iter=5000, backbone_fn=None, seed=0):
        super().__init__(epochs=epochs, width=width, proj_dim=proj_dim,
                         proj_hidden=proj_hidden, pred_hidden=pred_hidden,
                         batch_size=batch_size, lr=lr, weight_decay=weight_decay,
                         momentum=momentum, min_scale=min_scale, cj_strength=cj_strength,
                         C_grid=C_grid, svm_max_iter=svm_max_iter, seed=seed)
        self.epochs, self.width, self.proj_dim = epochs, width, proj_dim
        self.proj_hidden, self.pred_hidden = proj_hidden, pred_hidden
        self.batch_size, self.lr, self.weight_decay = batch_size, lr, weight_decay
        self.momentum, self.min_scale, self.cj_strength = momentum, min_scale, cj_strength
        self.C_grid, self.svm_max_iter, self.seed = tuple(C_grid), svm_max_iter, seed
        self.backbone_fn = backbone_fn
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = SimSiamTransform(input_size=32, min_scale=min_scale,
                                          cj_strength=cj_strength, normalize=NORM)

    def _to_tensor(self, X):
        t = torch.from_numpy(np.ascontiguousarray(X)).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(NORM["mean"]).view(1, 3, 1, 1)
        std = torch.tensor(NORM["std"]).view(1, 3, 1, 1)
        return ((t - mean) / std).to(self.device)

    @torch.no_grad()
    def _embed(self, X):
        self.net_.eval()
        return np.concatenate([self.net_.backbone(self._to_tensor(X[s:s + 256])).flatten(1)
                               .cpu().numpy() for s in range(0, len(X), 256)])

    def _pretrain(self, X):
        bs = min(self.batch_size, len(X))
        gen = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(TwoViews(X, self.transform), batch_size=bs, shuffle=True,
                            drop_last=True, generator=gen)
        opt = torch.optim.SGD(self.net_.parameters(), lr=self.lr, momentum=self.momentum,
                              weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        self.net_.train()
        for _ in range(self.epochs):
            for x0, x1 in loader:
                z0, p0 = self.net_(x0.to(self.device))
                z1, p1 = self.net_(x1.to(self.device))
                loss = 0.5 * (_neg_cosine(p0, z1) + _neg_cosine(p1, z0))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
        self.final_loss_ = float(loss.item())

    def fit(self, X, y):
        y = np.asarray(y)
        torch.manual_seed(self.seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        backbone = self.backbone_fn() if self.backbone_fn else small_cnn(self.width)
        self.net_ = SimSiamNetwork(backbone, _out_dim(backbone), self.proj_dim,
                                   self.proj_hidden, self.pred_hidden).to(self.device)
        if self.epochs > 0:
            self._pretrain(X)

        self.net_.eval()
        with torch.no_grad():
            z = F.normalize(self.net_.projection_head(
                self.net_.backbone(self._to_tensor(X)).flatten(1)), dim=1)
        self.collapse_std_ = float(z.std(0).mean())

        Z = self._embed(X)
        self.scaler_ = StandardScaler().fit(Z)
        search = GridSearchCV(LinearSVC(max_iter=self.svm_max_iter, random_state=self.seed),
                              {"C": list(self.C_grid)}, cv=5)
        search.fit(self.scaler_.transform(Z), y)
        self.classifier_ = search.best_estimator_
        self.best_C_ = search.best_params_["C"]
        return self

    def predict(self, X):
        return self.classifier_.predict(self.scaler_.transform(self._embed(X))).astype(np.int64)

    def n_trainable_params(self):
        return int(sum(p.numel() for p in self.net_.backbone.parameters())
                   + self.classifier_.coef_.size + self.classifier_.intercept_.size)