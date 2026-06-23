from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


def _to_uint8(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype == np.uint8:
        return value
    if not np.issubdtype(value.dtype, np.number):
        raise TypeError(f"media must be numeric, got {value.dtype}")
    return (np.clip(value, 0.0, 1.0) * 255.0).astype(np.uint8)


def image_to_hwc_uint8(value: np.ndarray) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"image must have 3 dimensions, got {image.shape}")
    if image.shape[-1] in (1, 3, 4):
        pass
    elif image.shape[0] in (1, 3, 4):
        image = image.transpose(1, 2, 0)
    else:
        raise ValueError(
            "image must use HWC or CHW layout with 1, 3, or 4 channels; "
            f"got {image.shape}"
        )
    return _to_uint8(image)


def video_to_btchw_uint8(value: np.ndarray) -> np.ndarray:
    video = np.asarray(value)
    if video.ndim == 4:
        video = video[None]
    if video.ndim != 5:
        raise ValueError(
            "video must use THWC/TCHW or BTHWC/BTCHW layout; "
            f"got {video.shape}"
        )
    if video.shape[-1] in (1, 3, 4):
        channel_first = video.transpose(0, 1, 4, 2, 3)
    elif video.shape[2] in (1, 3, 4):
        channel_first = video
    else:
        raise ValueError(
            "video must have 1, 3, or 4 channels on axis 2 or the final axis; "
            f"got {video.shape}"
        )
    return _to_uint8(channel_first)


@dataclass(frozen=True)
class ReconstructionMedia:
    target: np.ndarray
    predicted: np.ndarray
    error: np.ndarray
    comparison: np.ndarray
    video: np.ndarray


def reconstruction_media(
    value: np.ndarray,
    *,
    frame_index: int = 4,
) -> ReconstructionMedia:
    """Split NM512's [target; model; error] BTHWC diagnostic."""
    video = np.asarray(value)
    if video.ndim != 5 or video.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "Dreamer reconstruction diagnostic must use BTHWC layout; "
            f"got {video.shape}"
        )
    batch, frames, stacked_height, _width, _channels = video.shape
    if batch < 1 or frames < 1 or stacked_height % 3:
        raise ValueError(
            "Dreamer reconstruction diagnostic must contain a non-empty "
            f"target/model/error vertical stack; got {video.shape}"
        )
    height = stacked_height // 3
    target = video[:, :, :height]
    predicted = video[:, :, height : 2 * height]
    error = video[:, :, 2 * height :]
    comparison = np.concatenate([target, predicted, error], axis=3)
    selected = min(max(0, frame_index), frames - 1)
    return ReconstructionMedia(
        target=image_to_hwc_uint8(target[0, selected]),
        predicted=image_to_hwc_uint8(predicted[0, selected]),
        error=image_to_hwc_uint8(error[0, selected]),
        comparison=image_to_hwc_uint8(comparison[0, selected]),
        video=video_to_btchw_uint8(comparison),
    )


class WandbLogger:
    """NM512 logger that emits correctly shaped W&B images and videos."""

    ENV_KEYS = {
        "train_return",
        "train_length",
        "train_episodes",
        "eval_return",
        "eval_length",
        "eval_episodes",
        "dataset_size",
    }

    def __init__(self, step: int, *, enabled: bool = True, wandb_module=None):
        self.step = step
        self.enabled = bool(enabled)
        self._wandb = wandb_module
        self._last_step = None
        self._last_time = None
        self._scalars: dict[str, float] = {}
        self._images: dict[str, np.ndarray] = {}
        self._videos: dict[str, np.ndarray] = {}
        self._section_cache: dict[str, str] = {}
        self._closed = False

    def _client(self):
        if self._wandb is None:
            import wandb

            self._wandb = wandb
        return self._wandb

    def _section(self, name: str) -> str:
        if name in self._section_cache:
            return self._section_cache[name]
        if name.startswith(("recon/", "eval/recon/")):
            sectioned = name
        elif name in self.ENV_KEYS:
            sectioned = f"env/{name}"
        elif name.startswith("log_"):
            sectioned = f"env/{name}"
        elif name.startswith("expl_"):
            sectioned = f"model/{name}"
        else:
            sectioned = f"model/{name}"
        self._section_cache[name] = sectioned
        return sectioned

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def image(self, name, value):
        self._images[name] = np.asarray(value)

    def video(self, name, value):
        self._videos[name] = np.asarray(value)

    def _add_reconstruction(self, log: dict, name: str, value: np.ndarray) -> None:
        wandb = self._client()
        media = reconstruction_media(value)
        prefix = "eval/recon" if name == "eval_openl" else "recon"
        log[f"{prefix}/target"] = wandb.Image(media.target)
        log[f"{prefix}/predicted"] = wandb.Image(media.predicted)
        log[f"{prefix}/error"] = wandb.Image(media.error)
        log[f"{prefix}/comparison"] = wandb.Image(
            media.comparison,
            caption="left=target  center=prediction  right=error",
        )
        log[f"{prefix}/open_loop"] = wandb.Video(
            media.video,
            fps=4,
            format="gif",
            caption="left=target  center=prediction  right=error",
        )

    def write(self, fps=False, step=False):
        if not step:
            step = self.step
        if not self.enabled:
            self._scalars.clear()
            self._images.clear()
            self._videos.clear()
            return

        wandb = self._client()
        log = {
            self._section(name): value for name, value in self._scalars.items()
        }
        if fps:
            log["perf/fps"] = self._compute_fps(step)
        for name, value in self._images.items():
            log[name] = wandb.Image(image_to_hwc_uint8(value))
        for name, value in self._videos.items():
            name = name if isinstance(name, str) else name.decode("utf-8")
            if name in {"train_openl", "eval_openl"}:
                self._add_reconstruction(log, name, value)
            else:
                log[f"video/{name}"] = wandb.Video(
                    video_to_btchw_uint8(value),
                    fps=16,
                    format="gif",
                )
        if log:
            wandb.log(log, step=step)
        self._scalars.clear()
        self._images.clear()
        self._videos.clear()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return
        self.write(step=self.step)
        wandb = self._client()
        if getattr(wandb, "run", None) is not None:
            wandb.finish()

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / max(duration, 1e-9)
