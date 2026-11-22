"""Small CNNs for 32x32 inputs, trained from scratch.

    build_model({"type": "convnet", "width": 32, "convs": 2, "norm": "bn", "dropout": 0.0})
    build_model({"type": "resnet",  "width": 16, "n": 3,     "norm": "bn", "dropout": 0.0})

convnet: VGG-style. 3 stages of widths (w, 2w, 4w); each stage is `convs` x
         [conv3x3 - norm - ReLU] followed by 2x2 max-pool (32 -> 16 -> 8 -> 4),
         then global average pooling, dropout and a linear head.
         convs=1 -> 3 conv layers ("ConvNet-3"), convs=2 -> 6 ("ConvNet-6").
resnet:  CIFAR-style ResNet (He et al. 2016, Sec. 4.2). 3x3 stem, 3 stages of
         `n` BasicBlocks with widths (w, 2w, 4w), stride 2 at stages 2 and 3,
         global average pooling, dropout, linear head. Depth = 6n + 2:
         n=1 -> ResNet-8, n=3 -> ResNet-20.
norm:    "bn" (BatchNorm) or "gn" (GroupNorm, 8 groups). GroupNorm does not depend on
         batch statistics, which matters when a batch holds only ~25 images.
"""
import torch.nn as nn

ARCH_DEFAULTS = {"type": "convnet", "width": 32, "convs": 2, "n": 1, "norm": "bn",
                 "dropout": 0.0}


def _norm(kind, ch):
    if kind == "bn":
        return nn.BatchNorm2d(ch)
    if kind == "gn":
        return nn.GroupNorm(min(8, ch), ch)
    raise ValueError(f"unknown norm {kind!r}; expected 'bn' or 'gn'")


def _conv_bn_relu(cin, cout, norm, stride=1):
    return [nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            _norm(norm, cout), nn.ReLU(inplace=True)]


class ConvNet(nn.Module):
    def __init__(self, width=32, convs=2, norm="bn", dropout=0.0, num_classes=2):
        super().__init__()
        layers, cin = [], 3
        for stage in range(3):
            cout = width * 2 ** stage
            for _ in range(convs):
                layers += _conv_bn_relu(cin, cout, norm)
                cin = cout
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(cin, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride, norm):
        super().__init__()
        self.body = nn.Sequential(
            *_conv_bn_relu(cin, cout, norm, stride),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), _norm(norm, cout))
        self.shortcut = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                                          _norm(norm, cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, width=16, n=3, norm="bn", dropout=0.0, num_classes=2):
        super().__init__()
        layers, cin = _conv_bn_relu(3, width, norm), width
        for stage in range(3):
            cout = width * 2 ** stage
            for i in range(n):
                layers.append(BasicBlock(cin, cout, 2 if (stage > 0 and i == 0) else 1, norm))
                cin = cout
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(cin, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(arch=None, num_classes=2):
    a = {**ARCH_DEFAULTS, **(arch or {})}
    unknown = set(a) - set(ARCH_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown arch keys {sorted(unknown)}")
    if a["type"] == "convnet":
        return ConvNet(a["width"], a["convs"], a["norm"], a["dropout"], num_classes)
    if a["type"] == "resnet":
        return ResNet(a["width"], a["n"], a["norm"], a["dropout"], num_classes)
    raise ValueError(f"unknown arch type {a['type']!r}; expected 'convnet' or 'resnet'")


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
