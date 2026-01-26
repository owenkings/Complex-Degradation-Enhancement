import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class FeatureDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.res1 = ResBlock(128)
        self.down = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.res2 = ResBlock(128)
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.res3 = ResBlock(64)
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 3 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.up1(x)
        x = self.res3(x)
        x = self.up2(x)
        return x
