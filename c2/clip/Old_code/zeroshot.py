import torch
import torch.nn.functional as F
from common.method import Method
from loadClip import load_clip, DEVICE

class ZeroShotMethod(Method):

    def __init__(self,class_names=None,batch_size=100,seed=0, **hparams):
        self.model, self.preprocess, self.tokenizer = load_clip()
        self.classes = class_names
        self.batch_size = batch_size
        self.seed = 0

        self.templates = [
                 'a photo of a {}.',
                 'a blurry photo of a {}.',
                 'a black and white photo of a {}.',
                 'a low contrast photo of a {}.',
                 'a high contrast photo of a {}.',
                 'a bad photo of a {}.',
                 'a good photo of a {}.',
                 'a photo of a small {}.',
                 'a photo of a big {}.',
                 'a photo of the {}.',
                 'a blurry photo of the {}.',
                 'a black and white photo of the {}.',
                 'a low contrast photo of the {}.',
                 'a high contrast photo of the {}.',
                 'a bad photo of the {}.',
                 'a good photo of the {}.',
                 'a photo of the small {}.',
                 'a photo of the big {}.',
             ]

        self.text_features = None

    @torch.no_grad()
    def fit(self, X=None, y=None):
        self.text_features = []
        for cls in self.classes:
            prompts = [template.format(cls) for template in self.templates]
            tokens = self.tokenizer(prompts).to(DEVICE)
            embeddings = self.model.encode_text(tokens)
            embeddings = F.normalize(embeddings, dim=-1)
            class_embedding = embeddings.mean(dim=0)
            class_embedding = F.normalize(class_embedding, dim=0)
            self.text_features.append(class_embedding)
        self.text_features = torch.stack(self.text_features)
        return self

    @torch.no_grad()
    def predict(self, X):
        predictions = []
        for i in range(0, len(X), self.batch_size):
            X_batch = X[i:i + self.batch_size].to(DEVICE)
            image_features = self.model.encode_image(X_batch)
            image_features = F.normalize(image_features, dim=-1)
            similarity = image_features @ self.text_features.T
            batch_predictions = similarity.argmax(dim=-1)
            predictions.extend(batch_predictions.cpu().tolist())

        return predictions

