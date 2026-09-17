from __future__ import annotations

import numpy as np


def segmentation_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    class_weights: np.ndarray | None = None,
) -> float:
    """Mean cross-entropy for binary or integer multi-class segmentation."""
    if prediction.ndim == target.ndim + 1:
        classes = prediction.shape[-1]
        clipped = np.clip(prediction.astype(np.float32), 1e-6, 1)
        one_hot = np.eye(classes, dtype=np.float32)[target.astype(np.int64)]
        pixel_loss = -np.sum(one_hot * np.log(clipped), axis=-1)
        if class_weights is not None:
            pixel_loss *= class_weights[target.astype(np.int64)]
        return float(np.mean(pixel_loss))
    prediction = np.clip(prediction.astype(np.float32), 1e-6, 1 - 1e-6)
    target = target.astype(np.float32)
    return float(-np.mean(target * np.log(prediction) + (1 - target) * np.log(1 - prediction)))
