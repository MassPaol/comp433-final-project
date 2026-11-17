import torch
import torch.nn as nn
import torch.nn.functional as F
from common.method import Method
from loadClip import load_clip, DEVICE


class tipAdapterF(Method):

    def __init__(self, class_names=None, alpha=0.5, beta=1.0, seed=0,lr=1e-3, train_epochs=20, batch_size=256, **hparams):
        self.model, self.preprocess, self.tokenizer = load_clip()
        self.model = self.model.to(DEVICE)
        self.model.eval()
        self.device = DEVICE
        self.class_names = class_names
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self.lr = lr
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.text_features = None
        self.cache_labels = None
        self.adapter = None

    def _encode_images(self, X):
        feats = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i + self.batch_size].to(self.device)
                f = self.model.encode_image(batch)
                feats.append(F.normalize(f, dim=-1))
        return torch.cat(feats, dim=0)

    @torch.no_grad()
    def _encode_text(self):
        tokens = self.tokenizer(
            [f"a photo of a {name}." for name in self.class_names]
        ).to(self.device)
        feats = self.model.encode_text(tokens)
        return F.normalize(feats, dim=-1)

    def _tip_logits(self, image_features, cache_keys):
        clip_logits = 100.0 * image_features @ self.text_features.T
        affinity = image_features @ cache_keys.T
        cache_logits = torch.exp(-self.beta + self.beta * affinity) @ self.cache_labels
        return clip_logits + cache_logits * self.alpha

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        self.text_features = self._encode_text()
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.cache_labels = F.one_hot(y, num_classes=len(self.class_names)).float()
        train_features = self._encode_images(X)
        self.adapter = nn.Linear(train_features.shape[1], train_features.shape[0], bias=False)
        self.adapter.weight = nn.Parameter(train_features.clone().to(self.device))
        self.adapter.to(self.device)
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=self.lr, eps=1e-4)
        steps_per_epoch = (len(train_features) + self.batch_size - 1) // self.batch_size
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.train_epochs * steps_per_epoch)
        self.adapter.train()
        for epoch in range(self.train_epochs):
            perm = torch.randperm(len(train_features), device=self.device)
            epoch_loss = 0.0
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                feats = train_features[idx]
                target = y[idx]
                tip_logits = self._tip_logits(feats, self.adapter.weight)
                loss = F.cross_entropy(tip_logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                epoch_loss += loss.item()

        self.adapter.eval()
        return self

    @torch.no_grad()
    def predict(self, X):
        test_features = self._encode_images(X)
        tip_logits = self._tip_logits(test_features, self.adapter.weight)
        return tip_logits.argmax(dim=1).cpu()