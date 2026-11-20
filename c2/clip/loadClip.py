from functools import lru_cache

import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

ARCH, PRETRAINED = "ViT-B-32", "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALE = 100.0

TEMPLATES = (
    'a photo of a {}.', 'a blurry photo of a {}.', 'a black and white photo of a {}.',
    'a low contrast photo of a {}.', 'a high contrast photo of a {}.', 'a bad photo of a {}.',
    'a good photo of a {}.', 'a photo of a small {}.', 'a photo of a big {}.',
    'a photo of the {}.', 'a blurry photo of the {}.', 'a black and white photo of the {}.',
    'a low contrast photo of the {}.', 'a high contrast photo of the {}.', 'a bad photo of the {}.',
    'a good photo of the {}.', 'a photo of the small {}.', 'a photo of the big {}.',
)


@lru_cache(maxsize=1)
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
    return model.to(DEVICE).eval(), preprocess, open_clip.get_tokenizer(ARCH)


@torch.no_grad()
def class_text_features(names):
    model, _, tokenizer = load_clip()
    rows = []
    for name in names:
        tokens = tokenizer([t.format(name) for t in TEMPLATES]).to(DEVICE)
        emb = F.normalize(model.encode_text(tokens).float(), dim=-1).mean(dim=0)
        rows.append(F.normalize(emb, dim=0))
    return torch.stack(rows)


@torch.no_grad()
def image_features(X, batch_size=256):
    model, preprocess, _ = load_clip()
    out = []
    for i in range(0, len(X), batch_size):
        batch = torch.stack([preprocess(Image.fromarray(x)) for x in X[i:i + batch_size]])
        out.append(F.normalize(model.encode_image(batch.to(DEVICE)).float(), dim=-1))
    return torch.cat(out)
