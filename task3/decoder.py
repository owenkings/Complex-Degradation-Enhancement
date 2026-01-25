import torch
import torch.nn as nn

class FeatureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 128 x H/2 x W/2 (Assuming 224x224 input image -> 112x112 features)
        
        # Reverse of VGG Block 2 (Conv 64->128, Conv 128->128)
        # We go 128 -> 128 -> 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Reverse of MaxPool
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # Reverse of VGG Block 1 (Conv 3->64, Conv 64->64)
        # We go 64 -> 64 -> 3
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.upsample(x)
        x = self.layer2(x)
        return x
