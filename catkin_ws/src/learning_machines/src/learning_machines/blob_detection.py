from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class BlobDetectionConfig:
    hsv_low: tuple[int, int, int] = (35, 80, 80)
    hsv_high: tuple[int, int, int] = (85, 255, 255)
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.5
    morph_kernel_size: int = 5
    morph_iterations: int = 2


@dataclass
class BlobResult:
    x: float = 0.5
    y: float = 0.5
    area: float = 0.0
    found: bool = False


def detect_blob(
    image_bgr: np.ndarray,
    config: BlobDetectionConfig | None = None,
) -> BlobResult:
    if config is None:
        config = BlobDetectionConfig()

    if image_bgr is None or image_bgr.size == 0:
        return BlobResult()

    h, w = image_bgr.shape[:2]

    hsv_low = np.array(config.hsv_low, dtype=np.uint8)
    hsv_high = np.array(config.hsv_high, dtype=np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.morph_kernel_size, config.morph_kernel_size),
    )

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_iterations)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BlobResult()

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    img_area = h * w
    area_ratio = area / img_area

    if area_ratio < config.min_area_ratio or area_ratio > config.max_area_ratio:
        return BlobResult()

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return BlobResult(area=area_ratio, found=True)

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return BlobResult(
        x=cx / w,
        y=cy / h,
        area=area_ratio,
        found=True,
    )
