from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CityscapesInstances(Dataset):
    """Read Cityscapes images, semantic labels and instance ids."""

    def __init__(self, root: Path, split: str = "train", size: tuple[int, int] = (256, 512)) -> None:
        self.root = Path(root)
        self.split = split
        self.size = size
        self.image_root = _find_dataset_folder(self.root, "leftImg8bit")
        self.label_root = _find_dataset_folder(self.root, "gtFine")
        image_root = self.image_root / split
        self.images = sorted(image_root.glob("*/*_leftImg8bit.png"))
        if not self.images:
            raise FileNotFoundError(f"No Cityscapes images found in {image_root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path = self.images[index]
        city = image_path.parent.name
        stem = image_path.name.replace("_leftImg8bit.png", "")
        label_path = self.label_root / self.split / city / f"{stem}_gtFine_labelIds.png"
        instance_path = self.label_root / self.split / city / f"{stem}_gtFine_instanceIds.png"
        if not label_path.exists() or not instance_path.exists():
            raise FileNotFoundError(f"Missing labels for {image_path.name}")

        image = Image.open(image_path).convert("RGB").resize((self.size[1], self.size[0]), Image.BILINEAR)
        semantic = Image.open(label_path).resize((self.size[1], self.size[0]), Image.NEAREST)
        instances = Image.open(instance_path).resize((self.size[1], self.size[0]), Image.NEAREST)
        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        semantic_array = _cityscapes_train_ids(np.asarray(semantic, dtype=np.int64))
        instance_array = np.asarray(instances, dtype=np.int64)
        return {
            "image": image_array,
            "semantic": semantic_array,
            "instances": instance_array,
            "path": str(image_path),
        }


def _cityscapes_train_ids(label_ids: np.ndarray) -> np.ndarray:
    """Map Cityscapes raw labelIds to stable trainIds, preserving ignore=255."""
    result = np.full(label_ids.shape, 255, dtype=np.int64)
    mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
        31: 16, 32: 17, 33: 18,
    }
    for raw_id, train_id in mapping.items():
        result[label_ids == raw_id] = train_id
    return result


def _find_dataset_folder(root: Path, name: str) -> Path:
    candidates = [
        root / name,
        root / f"{name}_trainvaltest" / name,
        root / f"{name}_trainval" / name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find {name}. Expected {root / name} or an extracted {name}_trainvaltest archive."
    )
