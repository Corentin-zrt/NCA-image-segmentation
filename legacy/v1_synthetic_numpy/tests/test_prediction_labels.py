import numpy as np

from nca_segmentation.control_panel.app import ControlPanel


def test_update_prediction_preserves_multiclass_labels(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    panel = ControlPanel()
    panel._generate()
    prediction = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    panel.dataset = panel.dataset.__class__(
        images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
        masks=prediction[None, ...],
        class_names=("background", "circle", "square", "triangle"),
    )
    panel.update_prediction(prediction, step=1)

    assert np.array_equal(panel.prediction_mask, prediction)
    panel.running = False