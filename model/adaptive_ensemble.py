import torch
import torch.nn as nn

from model.efficientnet_extractor import EfficientNetExtractor


class AdaptiveEnsemble(nn.Module):
    def __init__(self, num_classes=10, num_clusters=10):
        super(AdaptiveEnsemble, self).__init__()
        self.feature_extractor1 = EfficientNetExtractor()
        self.feature_extractor2 = EfficientNetExtractor()

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Adjust the combined_features size based on the output of EfficientNetExtractor
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            dummy_output = self.feature_extractor1(dummy_input)
            combined_features = dummy_output.shape[1] * 2

        self.fc = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self.clustering_head = nn.Linear(combined_features, num_clusters)

    def forward(self, x):
        features1 = self.feature_extractor1(x)
        features2 = self.feature_extractor2(x)

        # Apply adaptive pooling
        features1 = self.adaptive_pool(features1).flatten(1)
        features2 = self.adaptive_pool(features2).flatten(1)

        # Combine features from both feature extractors
        combined_features = torch.cat([features1, features2], dim=1)
        classification_output = self.fc(combined_features)
        clustering_output = self.clustering_head(combined_features)

        return classification_output, clustering_output
