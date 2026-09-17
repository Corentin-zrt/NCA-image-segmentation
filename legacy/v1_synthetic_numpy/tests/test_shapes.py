import numpy as np

from nca_segmentation.data.generator import generate_dataset
from nca_segmentation.data.shapes import create_shape


def test_shape_has_expected_dimensions_and_foreground():
    image, mask = create_shape("circle", 64, np.random.default_rng(4))
    assert image.shape == (64, 64, 3)
    assert mask.shape == (64, 64)
    assert mask.any()
    assert set(np.unique(mask)) == {0, 1}
    assert image[mask.astype(bool)].mean() > image[~mask.astype(bool)].mean()


def test_dataset_is_reproducible():
    first = generate_dataset(3, image_size=32, seed=12)
    second = generate_dataset(3, image_size=32, seed=12)
    assert np.array_equal(first.images, second.images)
    assert np.array_equal(first.masks, second.masks)
