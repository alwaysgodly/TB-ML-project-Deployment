import torch.nn as nn
from torchvision import models

class CXR_EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(CXR_EfficientNetModel, self).__init__()
        self.model = models.efficientnet_b0(pretrained=True)

        
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)
