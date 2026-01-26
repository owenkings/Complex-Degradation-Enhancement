import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGGPerceptual(nn.Module):
    def __init__(self, layer="relu2_2"):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        if layer == "relu2_2":
            end_idx = 9
        elif layer == "relu3_3":
            end_idx = 16
        else:
            raise ValueError("Unsupported layer")
        self.features = nn.Sequential(*list(vgg)[:end_idx])
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.eval()
        self.criterion = nn.L1Loss()
    
    def forward(self, pred, target):
        feat_pred = self.features(pred)
        feat_target = self.features(target)
        return self.criterion(feat_pred, feat_target)
