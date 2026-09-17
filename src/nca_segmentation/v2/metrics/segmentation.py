from __future__ import annotations

import torch
from torch import Tensor


def mean_iou(prediction: Tensor, target: Tensor, num_classes: int, ignore_index: int = 255) -> dict[str, float]:
    """Return mean IoU and one IoU value per semantic class."""
    predicted = prediction.detach().reshape(-1)
    expected = target.detach().reshape(-1)
    valid = expected != ignore_index
    predicted = predicted[valid]
    expected = expected[valid]
    values: dict[str, float] = {}
    scores = []
    for label in range(num_classes):
        intersection = ((predicted == label) & (expected == label)).sum().item()
        union = ((predicted == label) | (expected == label)).sum().item()
        score = intersection / union if union else 0.0
        values[f"iou_{label}"] = score
        scores.append(score)
    values["mean_iou"] = sum(scores) / len(scores) if scores else 0.0
    return values
