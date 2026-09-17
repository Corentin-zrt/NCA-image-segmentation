from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .shapes import create_multi_shape, create_shape


@dataclass
class Dataset:
    images: np.ndarray
    masks: np.ndarray
    class_names: tuple[str, ...] = ("background", "circle", "square", "triangle")


def generate_dataset(
    samples: int,
    image_size: int = 96,
    shape: str = "mixed",
    seed: int = 7,
) -> Dataset:
    if samples < 1:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)
    images = np.empty((samples, image_size, image_size, 3), dtype=np.uint8)
    masks = np.empty((samples, image_size, image_size), dtype=np.uint8)
    for index in range(samples):
        if shape == "multi":
            images[index], masks[index] = create_multi_shape(image_size, rng)
        else:
            images[index], masks[index] = create_shape(shape, image_size, rng)
    return Dataset(images=images, masks=masks)
