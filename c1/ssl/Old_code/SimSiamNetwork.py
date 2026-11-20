import torch
from torch import nn

from lightly.loss import NegativeCosineSimilarity

from lightly.models.modules import (SimSiamProjectionHead,SimSiamPredictionHead)


class SimSiamNetwork(nn.Module):

    def __init__(self, backbone, feature_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(input_dim=feature_dim,hidden_dim=256,output_dim=128)
        self.prediction_head = SimSiamPredictionHead(input_dim=128,hidden_dim=64,output_dim=128)

    def forward(self, x):

        f = self.backbone(x)
        z = self.projection_head(f)
        p = self.prediction_head(z)

        return z, p