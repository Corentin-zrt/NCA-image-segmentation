from pathlib import Path

import numpy as np
from PIL import Image

from nca_segmentation.v2.data import CityscapesInstances


def test_cityscapes_reader_maps_labels(tmp_path: Path):
    city = "demo"
    (tmp_path / "leftImg8bit" / "train" / city).mkdir(parents=True)
    (tmp_path / "gtFine" / "train" / city).mkdir(parents=True)
    stem = f"{city}_000000_000000"
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(
        tmp_path / "leftImg8bit" / "train" / city / f"{stem}_leftImg8bit.png"
    )
    labels = np.array([[7, 11, 26, 255]] * 8, dtype=np.uint8)
    instances = np.array([[24001, 26001, 26002, 0]] * 8, dtype=np.uint16)
    Image.fromarray(labels).save(tmp_path / "gtFine" / "train" / city / f"{stem}_gtFine_labelIds.png")
    Image.fromarray(instances).save(tmp_path / "gtFine" / "train" / city / f"{stem}_gtFine_instanceIds.png")

    sample = CityscapesInstances(tmp_path, size=(8, 12))[0]

    assert tuple(sample["image"].shape) == (3, 8, 12)
    assert set(np.unique(sample["semantic"])) == {0, 2, 13, 255}
    assert sample["instances"].shape == (8, 12)
