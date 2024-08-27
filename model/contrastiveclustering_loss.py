import torch
import torch.nn as nn


class ContrastiveClusteringLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveClusteringLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, labels):
        batch_size = features.shape[0]
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        contrast_feature = torch.cat([features, features], dim=0)  # Duplicate features for contrastive learning
        anchor_feature = contrast_feature
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask to remove self-comparisons
        mask = torch.eye(batch_size * 2, device=features.device)
        exp_logits = torch.exp(logits) * (1 - mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean of log-likelihood over positive samples
        loss = -torch.sum(mask * log_prob) / batch_size
        
        return loss