from functools import lru_cache
import open_clip
import torch

ARCH = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
    return model.to(DEVICE).eval(), preprocess, open_clip.get_tokenizer(ARCH)

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()