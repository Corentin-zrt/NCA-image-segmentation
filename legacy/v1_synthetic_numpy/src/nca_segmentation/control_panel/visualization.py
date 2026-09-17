from __future__ import annotations

import numpy as np


CLASS_COLORS = {
    1: (231, 92, 92),
    2: (76, 157, 242),
    3: (245, 183, 74),
}


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (53, 211, 153),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a binary or probabilistic segmentation mask over an RGB image."""
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask and image dimensions must match")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    image_float = image.astype(np.float32)
    mask_float = np.clip(mask.astype(np.float32), 0, 1)[..., None]
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    blended = image_float * (1 - mask_float * alpha) + color_array * mask_float * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def comparison_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.78,
) -> np.ndarray:
    """Show correct pixels and segmentation errors with distinct colors.

    Green is a true positive, red a false positive, and blue a false negative.
    Correct background pixels remain visible from the original image.
    """
    if prediction.shape != target.shape or prediction.shape != image.shape[:2]:
        raise ValueError("image, prediction and target dimensions must match")
    if not 0 <= threshold <= 1 or not 0 <= alpha <= 1:
        raise ValueError("threshold and alpha must be between 0 and 1")

    predicted = prediction >= threshold
    expected = target >= 0.5
    true_positive = predicted & expected
    false_positive = predicted & ~expected
    false_negative = ~predicted & expected
    colors = np.zeros_like(image, dtype=np.float32)
    colors[true_positive] = (44, 220, 130)
    colors[false_positive] = (245, 75, 82)
    colors[false_negative] = (54, 145, 255)
    visible = true_positive | false_positive | false_negative
    result = image.astype(np.float32)
    result[visible] = result[visible] * (1 - alpha) + colors[visible] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Return mean IoU, per-class IoU and predicted coverage."""
    predicted = prediction.astype(np.int64)
    expected = target.astype(np.int64)
    classes = max(int(predicted.max()), int(expected.max()), 1)
    ious = []
    metrics: dict[str, float] = {}
    for label in range(1, classes + 1):
        intersection = np.logical_and(predicted == label, expected == label).sum()
        union = np.logical_or(predicted == label, expected == label).sum()
        value = float(intersection / union) if union else 0.0
        ious.append(value)
        metrics[f"iou_{label}"] = value
    metrics["iou"] = float(np.mean(ious)) if ious else 1.0
    metrics["coverage"] = float(np.mean(predicted > 0))
    return metrics


def multiclass_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.78,
) -> np.ndarray:
    """Overlay multi-class results: correct classes are bright, errors muted."""
    if prediction.shape != target.shape or prediction.shape != image.shape[:2]:
        raise ValueError("image, prediction and target dimensions must match")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    result = image.astype(np.float32)
    predicted = prediction.astype(np.int64)
    expected = target.astype(np.int64)
    correct = (predicted == expected) & (expected > 0)
    wrong_prediction = (predicted > 0) & (predicted != expected)
    missed_shape = (expected > 0) & (predicted == 0)
    for label, color in CLASS_COLORS.items():
        result[correct & (expected == label)] = (
            result[correct & (expected == label)] * (1 - alpha) + np.asarray(color) * alpha
        )
    result[wrong_prediction] = result[wrong_prediction] * (1 - alpha) + np.asarray((245, 75, 82)) * alpha
    result[missed_shape] = result[missed_shape] * (1 - alpha) + np.asarray((54, 145, 255)) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def label_colors(mask: np.ndarray) -> np.ndarray:
    """Render class labels using one stable color per shape."""
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[mask == 0] = (18, 25, 36)
    for label, color in CLASS_COLORS.items():
        result[mask == label] = color
    return result