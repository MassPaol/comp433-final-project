import numpy as np
import torch
import torch.nn.functional as F

from common.method import Method
from loadClip import load_clip, DEVICE


class WiSEFT(Method):
    def __init__(self, class_names=None, alpha=0.5, epochs=5, lr=1e-5,
                 batch_size=16, template="a photo of a {}.", seed=0, **hparams):
        self.class_names = class_names
        self.alpha = alpha
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.template = template
        self.seed = seed
        self.device = DEVICE
        self.scale = 100.0
        self.model, self.preprocess, self.tokenizer = load_clip()
        self.model = self.model.float().to(DEVICE)
        self.model.eval()
        self.theta_0 = None
        self.theta_1 = None
        self.head_0 = None
        self.head_1 = None
        self.head = None

    def _image_feats(self, images):
        return F.normalize(self.model.encode_image(images).float(), dim=-1)

    @torch.no_grad()
    def _zeroshot_head(self):
        prompts = [self.template.format(c) for c in self.class_names]
        tokens = self.tokenizer(prompts).to(self.device)
        return F.normalize(self.model.encode_text(tokens).float(), dim=-1)

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = X.to(self.device)
        y = torch.as_tensor(y).long().to(self.device)

        head_0 = self._zeroshot_head()
        self.theta_0 = {k: v.detach().clone()
                        for k, v in self.model.visual.state_dict().items()}

        head = torch.nn.Parameter(head_0.clone())
        params = list(self.model.visual.parameters()) + [head]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.1)

        self.model.visual.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), device=self.device)
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                feats = self._image_feats(X[idx])
                logits = self.scale * feats @ F.normalize(head, dim=-1).T
                loss = F.cross_entropy(logits, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.model.visual.eval()

        self.theta_1 = {k: v.detach().clone()
                        for k, v in self.model.visual.state_dict().items()}
        self.head_0 = head_0
        self.head_1 = F.normalize(head.detach(), dim=-1)

        self.set_alpha(self.alpha)
        return self

    @torch.no_grad()
    def set_alpha(self, alpha):
        if self.theta_1 is None:
            raise RuntimeError("Call fit() before set_alpha().")
        a = float(alpha)
        merged = {
            k: (1 - a) * self.theta_0[k] + a * self.theta_1[k]
            if self.theta_0[k].is_floating_point() else self.theta_0[k]
            for k in self.theta_0
        }
        self.model.visual.load_state_dict(merged)
        self.head = F.normalize((1 - a) * self.head_0 + a * self.head_1, dim=-1)
        self.alpha = a
        return self

    @torch.no_grad()
    def predict(self, X):
        if self.head is None:
            raise RuntimeError("Call fit() before predict().")
        feats = []
        for i in range(0, len(X), self.batch_size):
            feats.append(self._image_feats(X[i:i + self.batch_size].to(self.device)))
        logits = self.scale * torch.cat(feats) @ self.head.T
        return logits.argmax(dim=1).cpu()