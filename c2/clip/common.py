import open_clip
import torch
import torch.nn.functional as F

ARCH, PRETRAINED = "ViT-B-32", "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED)
    return model.to(DEVICE).eval(), preprocess, open_clip.get_tokenizer(ARCH)
