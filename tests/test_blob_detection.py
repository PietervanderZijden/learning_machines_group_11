'Tests for blob_detection.py (no sim required).'
from __future__ import annotations

import numpy as np
import pytest


class TestBlobDetection:
    def test_no_blob_in_empty_image(self):
        from learning_machines.blob_detection import detect_blob
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = detect_blob(img)
        assert result is not None

    def test_detect_orange_blob(self):
        from learning_machines.blob_detection import detect_blob
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2 = pytest.importorskip("cv2")
        cv2.circle(img, (32, 32), 10, (0, 165, 255), -1)
        result = detect_blob(img)
        assert result is not None

    def test_blob_result_has_fields(self):
        from learning_machines.blob_detection import BlobResult
        r = BlobResult(x=0.5, y=0.5, area=0.01, found=True)
        assert r.found is True
        assert r.x == 0.5
