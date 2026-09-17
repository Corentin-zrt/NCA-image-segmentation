from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .generator import Dataset


def save_dataset(dataset: Dataset, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "dataset.npz",
        images=dataset.images,
        masks=dataset.masks,
        class_names=np.asarray(dataset.class_names),
    )
    for index, (image, mask) in enumerate(zip(dataset.images, dataset.masks)):
        Image.fromarray(image).save(output_dir / f"image_{index:04d}.png")
        palette = np.asarray([(18, 25, 36), (231, 92, 92), (76, 157, 242), (245, 183, 74)], dtype=np.uint8)
        mask_rgb = palette[np.clip(mask, 0, len(palette) - 1)]
        Image.fromarray(mask_rgb).save(output_dir / f"mask_{index:04d}.png")
    return output_dir / "dataset.npz"
