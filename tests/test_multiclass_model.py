from pathlib import Path

import numpy as np

from nca_segmentation.data.generator import generate_dataset
from nca_segmentation.nca.model import NeuralCellularAutomaton


def test_multi_shape_dataset_has_four_labels():
    dataset = generate_dataset(3, image_size=48, shape="multi", seed=9)
    assert dataset.class_names == ("background", "circle", "square", "triangle")
    assert set(np.unique(dataset.masks)) == {0, 1, 2, 3}


def test_simple_shape_keeps_its_global_class_label():
    circle = generate_dataset(2, image_size=32, shape="circle", seed=3)
    square = generate_dataset(2, image_size=32, shape="square", seed=3)
    triangle = generate_dataset(2, image_size=32, shape="triangle", seed=3)
    assert set(np.unique(circle.masks)) == {0, 1}
    assert set(np.unique(square.masks)) == {0, 2}
    assert set(np.unique(triangle.masks)) == {0, 3}


def test_multiclass_model_can_save_and_reload(tmp_path: Path):
    dataset = generate_dataset(4, image_size=32, shape="multi", seed=2)
    model = NeuralCellularAutomaton()
    model.train_step(dataset.images, dataset.masks)
    model_path = model.save(tmp_path / "model.npz")
    restored = NeuralCellularAutomaton.load(model_path)

    assert restored.config.classes == 4
    assert restored.weights.shape == (5, 4)
    assert restored.forward(dataset.images[0]).shape == dataset.masks[0].shape