from __future__ import annotations

from typing import Literal

import numpy as np

ShapeName = Literal["circle", "square", "triangle", "mixed"]
SHAPE_LABELS = {"circle": 1, "square": 2, "triangle": 3}
SHAPE_COLORS = {
    1: (231, 92, 92),
    2: (76, 157, 242),
    3: (245, 183, 74),
}


def create_shape(
    shape: str,
    image_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one RGB image and a shape-specific class mask."""
    if image_size < 16:
        raise ValueError("image_size must be at least 16")

    selected_shape = shape
    if shape == "mixed":
        selected_shape = str(rng.choice(["circle", "square", "triangle"]))
    if selected_shape not in {"circle", "square", "triangle"}:
        raise ValueError(f"Unknown shape: {shape}")

    yy, xx = np.mgrid[:image_size, :image_size]
    center_x = int(rng.integers(image_size // 3, image_size * 2 // 3 + 1))
    center_y = int(rng.integers(image_size // 3, image_size * 2 // 3 + 1))
    radius = int(rng.integers(max(8, image_size // 8), max(9, image_size // 3)))

    if selected_shape == "circle":
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    elif selected_shape == "square":
        mask = (abs(xx - center_x) <= radius) & (abs(yy - center_y) <= radius)
    else:
        top = center_y - radius
        base = center_y + radius
        width = (yy - top) / max(1, base - top) * (2 * radius)
        mask = (yy >= top) & (yy <= base) & (abs(xx - center_x) <= width / 2)

    background = np.array([18, 25, 36], dtype=np.uint8)
    image = np.empty((image_size, image_size, 3), dtype=np.uint8)
    image[:] = background
    image[mask] = SHAPE_COLORS[SHAPE_LABELS[selected_shape]]
    return image, (mask.astype(np.uint8) * SHAPE_LABELS[selected_shape])


def create_multi_shape(
    image_size: int,
    rng: np.random.Generator,
    count: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one image containing several labeled, non-overlapping shapes."""
    if count < 2:
        raise ValueError("count must be at least 2 for a multi-shape sample")
    yy, xx = np.mgrid[:image_size, :image_size]
    background = np.array([18, 25, 36], dtype=np.uint8)
    image = np.empty((image_size, image_size, 3), dtype=np.uint8)
    image[:] = background
    labels = np.zeros((image_size, image_size), dtype=np.uint8)
    selected_shapes = list(SHAPE_LABELS)
    rng.shuffle(selected_shapes)
    radius = max(5, image_size // 8)

    for selected_shape in selected_shapes[:count]:
        label = SHAPE_LABELS[selected_shape]
        center_x = int(rng.integers(radius + 2, image_size - radius - 1))
        center_y = int(rng.integers(radius + 2, image_size - radius - 1))
        if selected_shape == "circle":
            shape_mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
        elif selected_shape == "square":
            shape_mask = (abs(xx - center_x) <= radius) & (abs(yy - center_y) <= radius)
        else:
            top = center_y - radius
            base = center_y + radius
            width = (yy - top) / max(1, base - top) * (2 * radius)
            shape_mask = (yy >= top) & (yy <= base) & (abs(xx - center_x) <= width / 2)
        shape_mask &= labels == 0
        labels[shape_mask] = label
        image[shape_mask] = SHAPE_COLORS[label]
    return image, labels
