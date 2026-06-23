from __future__ import annotations

import numpy as np
import pytest

from learning_machines.reference_wandb_media import (
    WandbLogger,
    image_to_hwc_uint8,
    reconstruction_media,
    video_to_btchw_uint8,
)


def _diagnostic() -> np.ndarray:
    value = np.empty((2, 6, 12, 4, 3), dtype=np.float32)
    value[:, :, :4] = 0.1
    value[:, :, 4:8] = 0.5
    value[:, :, 8:] = 0.9
    return value


def test_reference_reconstruction_media_splits_and_preserves_batch():
    media = reconstruction_media(_diagnostic())

    assert media.target.shape == (4, 4, 3)
    assert media.predicted.shape == (4, 4, 3)
    assert media.error.shape == (4, 4, 3)
    assert media.comparison.shape == (4, 12, 3)
    assert media.video.shape == (2, 6, 3, 4, 12)
    assert np.all(media.target == 25)
    assert np.all(media.predicted == 127)
    assert np.all(media.error == 229)


def test_reference_media_layout_conversion_is_explicit():
    thwc = np.zeros((5, 8, 9, 3), dtype=np.float32)
    btchw = video_to_btchw_uint8(thwc)
    assert btchw.shape == (1, 5, 3, 8, 9)

    chw = np.zeros((3, 8, 9), dtype=np.float32)
    assert image_to_hwc_uint8(chw).shape == (8, 9, 3)

    with pytest.raises(ValueError, match="video must"):
        video_to_btchw_uint8(np.zeros((5, 8, 9), dtype=np.float32))
    with pytest.raises(ValueError, match="BTHWC"):
        reconstruction_media(np.zeros((1, 5, 3, 8, 9), dtype=np.float32))


class _FakeImage:
    def __init__(self, data, caption=None):
        self.data = np.asarray(data)
        self.caption = caption


class _FakeVideo:
    def __init__(self, data, caption=None, fps=None, format=None):
        self.data = np.asarray(data)
        self.caption = caption
        self.fps = fps
        self.format = format


class _FakeWandb:
    Image = _FakeImage
    Video = _FakeVideo

    def __init__(self):
        self.run = object()
        self.logged = []
        self.finished = 0

    def log(self, payload, step=None):
        self.logged.append((payload, step))

    def finish(self):
        self.finished += 1
        self.run = None


def test_reference_logger_emits_images_and_channel_first_video():
    fake = _FakeWandb()
    logger = WandbLogger(100, wandb_module=fake)
    logger.video("eval_openl", _diagnostic())
    logger.video("eval_policy", np.zeros((1, 5, 8, 9, 3), dtype=np.float32))
    logger.write(step=100)

    payload, step = fake.logged[-1]
    assert step == 100
    assert {
        "eval/recon/target",
        "eval/recon/predicted",
        "eval/recon/error",
        "eval/recon/comparison",
        "eval/recon/open_loop",
        "video/eval_policy",
    } <= payload.keys()
    assert payload["eval/recon/target"].data.shape == (4, 4, 3)
    assert payload["eval/recon/open_loop"].data.shape == (2, 6, 3, 4, 12)
    assert payload["eval/recon/open_loop"].fps == 4
    assert payload["video/eval_policy"].data.shape == (1, 5, 3, 8, 9)


def test_reference_logger_no_wandb_mode_drops_media_without_importing_client():
    logger = WandbLogger(0, enabled=False)
    logger.video("eval_openl", _diagnostic())
    logger.write()
    logger.close()


def test_reference_logger_close_flushes_pending_media_once():
    fake = _FakeWandb()
    logger = WandbLogger(9, wandb_module=fake)
    logger.video("train_openl", _diagnostic())

    logger.close()
    logger.close()

    payload, step = fake.logged[-1]
    assert step == 9
    assert "recon/open_loop" in payload
    assert fake.finished == 1
