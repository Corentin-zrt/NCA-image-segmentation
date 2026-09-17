import numpy as np
import pytest

from nca_segmentation.control_panel.visualization import overlay_mask


def test_overlay_preserves_background_and_tints_foreground():
    image = np.full((2, 2, 3), 100, dtype=np.uint8)
    mask = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    result = overlay_mask(image, mask, color=(200, 0, 0), alpha=0.5)

    assert np.array_equal(result[0, 0], image[0, 0])
    assert result[0, 1, 0] == 150
    assert result[0, 1, 1] == 50


def test_overlay_rejects_incompatible_mask():
    with pytest.raises(ValueError):
        overlay_mask(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((3, 3)))