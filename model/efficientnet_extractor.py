import torch.nn as nn
from torchvision.models import efficientnet_b0


class EfficientNetExtractor(nn.Module):
    def __init__(self):
        super(EfficientNetExtractor, self).__init__()
        self.efficient_net = efficientnet_b0(pretrained=True)

        # Remove the last few layers to adapt to smaller input size
        self.feature_extractor = nn.Sequential(
            self.efficient_net.features[0],  # Conv2d
            self.efficient_net.features[1],  # BatchNorm2d
            self.efficient_net.features[2],  # SiLU
            self.efficient_net.features[3],  # MBConv block
            self.efficient_net.features[4],  # MBConv block
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return x
