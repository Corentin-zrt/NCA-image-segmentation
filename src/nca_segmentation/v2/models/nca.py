from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


@dataclass
class StreetSceneNCAConfig:
    num_classes: int = 19
    base_channels: int = 24
    state_channels: int = 48
    embedding_dim: int = 8
    nca_steps: int = 12
    update_probability: float = 0.5


class StreetSceneNCA(nn.Module):
    """CPU-friendly convolutional encoder followed by a shared NCA refinement loop."""

    def __init__(self, config: StreetSceneNCAConfig | None = None) -> None:
        super().__init__()
        self.config = config or StreetSceneNCAConfig()
        channels = self.config.base_channels
        state_channels = self.config.state_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.GELU(),
            nn.Conv2d(channels * 2, state_channels, 3, padding=1),
            nn.GELU(),
        )
        self.state_seed = nn.Conv2d(state_channels, state_channels, 1)
        self.perception = nn.Conv2d(state_channels, state_channels, 3, padding=1, groups=1)
        self.update = nn.Sequential(
            nn.Conv2d(state_channels * 2, state_channels, 1),
            nn.GELU(),
            nn.Conv2d(state_channels, state_channels, 1),
        )
        self.semantic_head = nn.Conv2d(state_channels, self.config.num_classes, 1)
        self.objectness_head = nn.Conv2d(state_channels, 1, 1)
        self.embedding_head = nn.Conv2d(state_channels, self.config.embedding_dim, 1)

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        features = self.encoder(image)
        state = self.state_seed(features)
        for _ in range(self.config.nca_steps):
            sensed = self.perception(state)
            delta = self.update(torch.cat((state, sensed), dim=1))
            state = state + self.config.update_probability * delta
        return {
            "semantic_logits": self.semantic_head(state),
            "objectness_logits": self.objectness_head(state),
            "embeddings": nn.functional.normalize(self.embedding_head(state), dim=1),
        }

    def export_config(self) -> dict[str, object]:
        return asdict(self.config)
