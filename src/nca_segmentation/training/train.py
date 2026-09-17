from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from nca_segmentation.nca.model import NeuralCellularAutomaton


def train_nca(
    dataset_path: Path,
    epochs: int = 10,
    model: NeuralCellularAutomaton | None = None,
    on_step: Callable[[int, float, np.ndarray], None] | None = None,
) -> tuple[NeuralCellularAutomaton, list[float]]:
    """Train an NCA from a compressed dataset and optionally report every step."""
    data = np.load(dataset_path)
    images = data["images"]
    masks = data["masks"]
    if epochs < 1:
        raise ValueError("epochs must be positive")

    model = model or NeuralCellularAutomaton()
    history: list[float] = []
    for epoch in range(1, epochs + 1):
        loss, prediction = model.train_step(images, masks)
        history.append(loss)
        if on_step is not None:
            on_step(epoch, loss, prediction)
    return model, history
