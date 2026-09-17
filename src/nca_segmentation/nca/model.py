from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .losses import segmentation_loss


@dataclass
class NCAConfig:
    channels: int = 5
    hidden_channels: int = 8
    update_probability: float = 1.0
    steps: int = 8
    learning_rate: float = 0.8
    classes: int = 4


class NeuralCellularAutomaton:
    """A small NumPy NCA for binary or multi-class pixel segmentation."""

    def __init__(self, config: NCAConfig | None = None, seed: int = 7) -> None:
        self.config = config or NCAConfig()
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.05, size=(5, self.config.classes)).astype(np.float32)
        self.weights[-1, 0] = 1.0
        self.weights[-1, 1:] = -1.0

    def forward(self, image: np.ndarray, steps: int | None = None) -> np.ndarray:
        """Return the most likely class index for every pixel."""
        return np.argmax(self.predict_proba(image, steps), axis=-1).astype(np.uint8)

    def predict_proba(self, image: np.ndarray, steps: int | None = None) -> np.ndarray:
        batch_image, was_single = self._batch_image(image)
        state = np.zeros((*batch_image.shape[:3], self.config.classes), dtype=np.float32)
        state[..., 0] = 1.0
        for _ in range(steps or self.config.steps):
            features = self._perception(batch_image, state)
            proposal = self._softmax(np.einsum("...f,fc->...c", features, self.weights))
            state += self.config.update_probability * (proposal - state)
        return state[0] if was_single else state

    def train_step(self, images: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
        batch_image, _ = self._batch_image(images)
        batch_target = targets.astype(np.int64)
        if batch_target.ndim == 2:
            batch_target = batch_target[None, ...]
        required_classes = int(batch_target.max()) + 1
        if required_classes > self.config.classes:
            self._resize_classes(required_classes)
        class_weights = self._class_weights(batch_target, self.config.classes)

        state = np.zeros((*batch_image.shape[:3], self.config.classes), dtype=np.float32)
        state[..., 0] = 1.0
        features = None
        for _ in range(self.config.steps):
            features = self._perception(batch_image, state)
            proposal = self._softmax(np.einsum("...f,fc->...c", features, self.weights))
            state += self.config.update_probability * (proposal - state)

        assert features is not None
        one_hot = np.eye(self.config.classes, dtype=np.float32)[batch_target]
        pixel_weights = class_weights[batch_target]
        gradient = np.mean(
            (state - one_hot)[..., None] * features[..., None, :] * pixel_weights[..., None, None],
            axis=(0, 1, 2),
        )
        self.weights -= self.config.learning_rate * gradient.T
        prediction = np.argmax(state[0], axis=-1).astype(np.uint8)
        return segmentation_loss(state, batch_target, class_weights), prediction

    @staticmethod
    def _class_weights(target: np.ndarray, classes: int) -> np.ndarray:
        """Balance the objective so the background cannot hide missed shapes."""
        counts = np.bincount(target.ravel(), minlength=classes)
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts.astype(np.float32))
        return np.clip(weights, 0.25, 4.0).astype(np.float32)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, config=np.array([asdict(self.config)], dtype=object))
        return path

    @classmethod
    def load(cls, path: Path) -> "NeuralCellularAutomaton":
        data = np.load(path, allow_pickle=True)
        config = NCAConfig(**data["config"].item())
        model = cls(config=config)
        model.weights = data["weights"].astype(np.float32)
        return model

    def _resize_classes(self, classes: int) -> None:
        old_weights = self.weights
        self.config.classes = classes
        self.weights = np.zeros((5, classes), dtype=np.float32)
        self.weights[:, : min(classes, old_weights.shape[1])] = old_weights[:, : min(classes, old_weights.shape[1])]
        self.weights[-1, 0] = 1.0

    @staticmethod
    def _batch_image(image: np.ndarray) -> tuple[np.ndarray, bool]:
        array = image.astype(np.float32) / 255.0
        was_single = array.ndim == 3
        if was_single:
            array = array[None, ...]
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError("image must have shape (height, width, 3) or (batch, height, width, 3)")
        return array, was_single

    @staticmethod
    def _perception(image: np.ndarray, state: np.ndarray) -> np.ndarray:
        foreground = 1.0 - state[..., 0]
        padded = np.pad(foreground, ((0, 0), (1, 1), (1, 1)), mode="edge")
        neighborhood = sum(
            padded[:, row : row + state.shape[1], col : col + state.shape[2]]
            for row in range(3)
            for col in range(3)
        ) / 9.0
        return np.concatenate((image, neighborhood[..., None], np.ones_like(neighborhood[..., None])), axis=-1)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = values - values.max(axis=-1, keepdims=True)
        exponential = np.exp(np.clip(shifted, -30, 30))
        return exponential / exponential.sum(axis=-1, keepdims=True)
