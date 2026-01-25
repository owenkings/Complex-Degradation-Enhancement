import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class VGG16FeatureWrapper(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        # Load VGG16 with default pretrained weights
        original_vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Split features into SPL (Shallow) and DPL (Deep)
        # SPL: Up to Relu2_2 (index 9 in features, exclusive means 0-8)
        # 0:Conv, 1:Relu, 2:Conv, 3:Relu, 4:Maxpool, 5:Conv, 6:Relu, 7:Conv, 8:Relu
        self.spl = original_vgg.features[:9] 
        
        # DPL: The rest of features + avgpool + classifier
        self.dpl_features = original_vgg.features[9:]
        self.avgpool = original_vgg.avgpool
        self.classifier = original_vgg.classifier
        
        # Freeze all VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        self.to(device)
        self.eval() # Set to eval mode by default (BatchNorm, Dropout behavior)

    def extract_shallow_features(self, x):
        """
        Extract features using SPL.
        """
        return self.spl(x)

    def predict_from_features(self, x):
        """
        Predict logits from features (input to DPL).
        """
        x = self.dpl_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward_from_shallow(self, x):
        return self.predict_from_features(x)

    def forward(self, x):
        # Standard VGG forward pass
        x = self.extract_shallow_features(x)
        x = self.predict_from_features(x)
        return x
