import torch
import torch.nn.functional as F
from common.method import Method
from loadClip import load_clip, DEVICE

class tipAdapter(Method):

    def __init__(self,class_names=None,alpha=0.5,beta=1.0,seed=0, **hparams):
        self.model, self.preprocess, self.tokenizer = load_clip()
        self.model = self.model.to(DEVICE)
        self.model.eval()
        self.device = DEVICE
        self.class_names = class_names
        self.cache_features = None
        self.cache_labels = None
        self.alpha = alpha
        self.beta = beta
        self.text_features = None
        self.seed = seed

    @torch.no_grad()
    def fit(self, X, y):
        X = X.to(self.device)
        text_tokens = self.tokenizer([f"a photo of a {name}." for name in self.class_names]).to(self.device)
        self.text_features = self.model.encode_text(text_tokens)
        self.text_features = F.normalize(self.text_features, dim=-1)
        features = self.model.encode_image(X)
        features = F.normalize(features, dim=-1)
        self.cache_features = features
        y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.cache_labels = F.one_hot(y, num_classes=len(self.class_names)).float()
        return self


    @torch.no_grad()
    def predict(self, X):
         X = X.to(self.device)
         test_features = self.model.encode_image(X)
         test_features = F.normalize(test_features, dim=-1)
         clip_logits = 100.0 * test_features @ self.text_features.T
         affinity = test_features @ self.cache_features.T
         cache_logits = torch.exp(-self.beta + self.beta * affinity) @ self.cache_labels
         tip_logits = clip_logits + cache_logits * self.alpha
         return tip_logits.argmax(dim=1).cpu()


