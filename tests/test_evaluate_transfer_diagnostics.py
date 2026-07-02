import cv2
import numpy as np

from evaluate_transfer import (
    _prediction_metrics,
    _save_blob_overlay,
    _save_prediction_comparison,
)


def test_prediction_metrics_are_zero_for_matching_next_state():
    image = np.zeros((3, 8, 8), dtype=np.uint8)
    image[1, 2:6, 2:6] = 255
    ir = np.linspace(0.0, 1.0, 8, dtype=np.float32)
    prediction = {
        "image": image.astype(np.float32) / 255.0,
        "ir": ir.copy(),
        "reward": 2.5,
        "continue_probability": 1.0,
        "value": 3.0,
    }

    metrics = _prediction_metrics(
        prediction,
        {"image": image, "ir": ir},
        reward=2.5,
        done=False,
    )

    assert metrics["image_mse"] == 0.0
    assert metrics["image_mae"] == 0.0
    assert metrics["green_saliency_mse"] == 0.0
    assert metrics["ir_mse"] == 0.0
    assert metrics["reward_error"] == 0.0
    assert metrics["continue_target"] == 1.0


def test_prediction_comparison_contains_three_labeled_panels(tmp_path):
    prediction = np.zeros((3, 8, 8), dtype=np.float32)
    actual = np.zeros((3, 8, 8), dtype=np.uint8)
    path = tmp_path / "comparison.png"

    _save_prediction_comparison(path, prediction, actual)

    saved = cv2.imread(str(path))
    assert saved is not None
    assert saved.shape == (30, 24, 3)


def test_sac_blob_overlay_is_saved_with_header(tmp_path):
    image = np.zeros((3, 16, 16), dtype=np.uint8)
    image[1, 6:10, 6:10] = 255
    path = tmp_path / "blob.png"

    _save_blob_overlay(
        path,
        image,
        np.array([0.5, 0.5, 0.0625, 1.0], dtype=np.float32),
    )

    saved = cv2.imread(str(path))
    assert saved is not None
    assert saved.shape == (38, 16, 3)
