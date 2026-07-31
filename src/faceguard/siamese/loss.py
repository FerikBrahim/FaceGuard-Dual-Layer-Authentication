from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineTripletLoss(nn.Module):
    """Standard cosine triplet loss for preconstructed triplets."""

    def __init__(self, margin: float = 0.8) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        positive_distance = 1.0 - F.cosine_similarity(anchor, positive)
        negative_distance = 1.0 - F.cosine_similarity(anchor, negative)
        losses = F.relu(positive_distance - negative_distance + self.margin)
        active = losses > 0
        return losses[active].mean() if active.any() else losses.mean() * 0.0


class BatchSemiHardTripletLoss(nn.Module):
    """Online semi-hard triplet loss over a labeled embedding batch.

    A negative is semi-hard when it is farther than the positive but still lies
    inside the margin. If no semi-hard negative exists, the closest negative is
    used. This implementation supports the manuscript's optional mining claim;
    the original uploaded notebook used preconstructed triplets instead.
    """

    def __init__(self, margin: float = 0.8) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        distances = 1.0 - embeddings @ embeddings.T
        losses: list[torch.Tensor] = []
        for anchor_index in range(embeddings.size(0)):
            positive_indices = torch.where(labels == labels[anchor_index])[0]
            positive_indices = positive_indices[positive_indices != anchor_index]
            negative_indices = torch.where(labels != labels[anchor_index])[0]
            if positive_indices.numel() == 0 or negative_indices.numel() == 0:
                continue
            for positive_index in positive_indices:
                positive_distance = distances[anchor_index, positive_index]
                negative_distances = distances[anchor_index, negative_indices]
                candidates = negative_distances[
                    (negative_distances > positive_distance)
                    & (negative_distances < positive_distance + self.margin)
                ]
                negative_distance = candidates.min() if candidates.numel() else negative_distances.min()
                losses.append(F.relu(positive_distance - negative_distance + self.margin))
        if not losses:
            return embeddings.sum() * 0.0
        return torch.stack(losses).mean()
