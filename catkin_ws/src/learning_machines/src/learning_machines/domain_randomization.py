"""Hybrid persistent/per-step domain randomization shared by all algorithms."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

from learning_machines.transfer import CalibrationProfile
from learning_machines.transfer import SmoothingConfig


@dataclass(frozen=True)
class RandomizationRanges:
    ir_gain: tuple[float, float] = (0.9, 1.1)
    ir_bias: tuple[float, float] = (-0.04, 0.04)
    ir_saturation: tuple[float, float] = (0.9, 1.0)
    ir_exponent: tuple[float, float] = (0.85, 1.15)
    ir_lag: tuple[float, float] = (0.0, 0.25)
    motor_gain: tuple[float, float] = (0.9, 1.1)
    motor_bias: tuple[float, float] = (-0.04, 0.04)
    motor_deadband: tuple[float, float] = (0.0, 0.08)
    motor_latency_steps: tuple[int, int] = (0, 1)
    smoothing_previous_weight: tuple[float, float] = (0.55, 0.75)
    camera_exposure: tuple[float, float] = (-0.12, 0.12)
    camera_contrast: tuple[float, float] = (0.85, 1.15)
    camera_color_balance: tuple[float, float] = (0.9, 1.1)
    camera_crop_fraction: tuple[float, float] = (0.0, 0.04)
    camera_tilt_offset: tuple[int, int] = (-4, 4)
    camera_shift_pixels: tuple[float, float] = (-3.0, 3.0)
    camera_blur_kernel: tuple[int, ...] = (1, 1, 1, 3)
    ir_noise_std: float = 0.015
    image_noise_std: float = 0.01
    action_jitter_std: float = 0.015
    spike_probability: float = 0.002
    dropout_probability: float = 0.002

    @classmethod
    def from_calibration_profiles(
        cls,
        simulation: CalibrationProfile,
        hardware: CalibrationProfile,
    ) -> "RandomizationRanges":
        """Derive conservative IR ranges from measured sim/hardware endpoints."""
        sim_span = np.array([
            abs(sensor.near_obstacle - sensor.free_space)
            for sensor in simulation.sensors
        ])
        hw_span = np.array([
            abs(sensor.near_obstacle - sensor.free_space)
            for sensor in hardware.sensors
        ])
        valid = sim_span > 1e-6
        ratios = hw_span[valid] / sim_span[valid]
        if ratios.size < 4:
            return cls()
        gain_error = float(np.clip(np.percentile(np.abs(ratios - 1.0), 90), 0.1, 0.35))
        sim_free = np.array([sensor.free_space for sensor in simulation.sensors])
        hw_free = np.array([sensor.free_space for sensor in hardware.sensors])
        normalized_bias = np.abs(hw_free - sim_free) / np.maximum(sim_span, 1e-6)
        bias_error = float(np.clip(np.percentile(normalized_bias, 90), 0.04, 0.2))
        return cls(
            ir_gain=(1.0 - gain_error, 1.0 + gain_error),
            ir_bias=(-bias_error, bias_error),
        )


class DomainRandomizationWrapper(gym.Wrapper):
    """Samples physical/visual parameters once per episode and noise per step."""

    def __init__(
        self,
        env: gym.Env,
        enabled: bool = True,
        ranges: RandomizationRanges | None = None,
        seed: int | None = None,
        **legacy_kwargs,
    ) -> None:
        super().__init__(env)
        self.enabled = enabled
        self.ranges = ranges or RandomizationRanges(
            ir_noise_std=legacy_kwargs.get("ir_noise_std", 0.015),
            image_noise_std=legacy_kwargs.get("image_noise_std", 0.01),
        )
        self.rng = np.random.default_rng(seed)
        self.episode_parameters: dict[str, Any] = {}
        self._lagged_ir: np.ndarray | None = None
        self._previous_action = np.zeros(2, dtype=np.float32)
        self._action_queue: list[np.ndarray] = []
        config = getattr(self.env.unwrapped, "config", None)
        self._base_phone_tilt = int(getattr(config, "phone_tilt", 100))

    def _uniform(self, bounds, size=None):
        return self.rng.uniform(bounds[0], bounds[1], size=size)

    def _sample_episode_parameters(self) -> None:
        r = self.ranges
        self.episode_parameters = {
            "ir_gain": self._uniform(r.ir_gain, 8).astype(np.float32),
            "ir_bias": self._uniform(r.ir_bias, 8).astype(np.float32),
            "ir_saturation": self._uniform(r.ir_saturation, 8).astype(np.float32),
            "ir_exponent": self._uniform(r.ir_exponent, 8).astype(np.float32),
            "ir_lag": self._uniform(r.ir_lag, 8).astype(np.float32),
            "motor_gain": self._uniform(r.motor_gain, 2).astype(np.float32),
            "motor_bias": self._uniform(r.motor_bias, 2).astype(np.float32),
            "motor_deadband": self._uniform(r.motor_deadband, 2).astype(np.float32),
            "motor_latency_steps": int(self.rng.integers(
                r.motor_latency_steps[0], r.motor_latency_steps[1] + 1
            )),
            "smoothing_previous_weight": float(self._uniform(r.smoothing_previous_weight)),
            "camera_exposure": float(self._uniform(r.camera_exposure)),
            "camera_contrast": float(self._uniform(r.camera_contrast)),
            "camera_color_balance": self._uniform(r.camera_color_balance, 3).astype(np.float32),
            "camera_crop_fraction": float(self._uniform(r.camera_crop_fraction)),
            "camera_tilt_offset": int(self.rng.integers(
                r.camera_tilt_offset[0], r.camera_tilt_offset[1] + 1
            )),
            "camera_shift_pixels": self._uniform(r.camera_shift_pixels, 2).astype(np.float32),
            "camera_blur_kernel": int(self.rng.choice(r.camera_blur_kernel)),
        }
        self._lagged_ir = None
        self._previous_action.fill(0.0)
        self._action_queue.clear()

    def _randomize_action(self, action: np.ndarray) -> np.ndarray:
        p = self.episode_parameters
        action = np.asarray(action, dtype=np.float32)
        action = action * p["motor_gain"] + p["motor_bias"]
        action[np.abs(action) < p["motor_deadband"]] = 0.0
        action += self.rng.normal(0.0, self.ranges.action_jitter_std, 2)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._action_queue.append(action)
        latency = p["motor_latency_steps"]
        if len(self._action_queue) <= latency:
            return np.zeros(2, dtype=np.float32)
        return self._action_queue.pop(0)

    def _augment_ir(self, ir: np.ndarray) -> np.ndarray:
        p = self.episode_parameters
        transformed = np.asarray(ir, dtype=np.float32) * p["ir_gain"] + p["ir_bias"]
        transformed = np.clip(transformed, 0.0, p["ir_saturation"]) / p["ir_saturation"]
        transformed = np.power(np.clip(transformed, 0.0, 1.0), p["ir_exponent"])
        if self._lagged_ir is None:
            self._lagged_ir = transformed.copy()
        lag = p["ir_lag"]
        transformed = (1.0 - lag) * transformed + lag * self._lagged_ir
        self._lagged_ir = transformed.copy()
        transformed += self.rng.normal(0.0, self.ranges.ir_noise_std, 8)
        spike_mask = self.rng.random(8) < self.ranges.spike_probability
        dropout_mask = self.rng.random(8) < self.ranges.dropout_probability
        transformed[spike_mask] = 1.0
        transformed[dropout_mask] = 0.0
        return np.clip(transformed, 0.0, 1.0).astype(np.float32)

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        chw = image.ndim == 3 and image.shape[0] == 3
        img = np.transpose(image, (1, 2, 0)) if chw else image
        img = img.astype(np.float32) / 255.0
        p = self.episode_parameters
        img = (img - 0.5) * p["camera_contrast"] + 0.5 + p["camera_exposure"]
        img *= p["camera_color_balance"].reshape(1, 1, 3)
        crop = int(min(img.shape[:2]) * p["camera_crop_fraction"])
        if crop > 0 and img.shape[0] > 2 * crop and img.shape[1] > 2 * crop:
            cropped = img[crop:-crop, crop:-crop]
            img = cv2.resize(cropped, (img.shape[1], img.shape[0]))
        shift_x, shift_y = p["camera_shift_pixels"]
        transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(
            img,
            transform,
            (img.shape[1], img.shape[0]),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        kernel = p["camera_blur_kernel"]
        if kernel > 1:
            img = cv2.GaussianBlur(img, (kernel, kernel), 0)
        img += self.rng.normal(0.0, self.ranges.image_noise_std, img.shape)
        result = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        return np.transpose(result, (2, 0, 1)).copy() if chw else result

    def _augment_obs(self, obs):
        if not self.enabled or not isinstance(obs, dict):
            return obs
        obs = dict(obs)
        if "ir" in obs:
            obs["ir"] = self._augment_ir(obs["ir"])
        if "image" in obs:
            obs["image"] = self._augment_image(obs["image"])
        if "blob" in obs and obs["blob"][3] > 0.5:
            blob = np.asarray(obs["blob"], dtype=np.float32).copy()
            blob[:2] += self.rng.normal(0.0, 0.015, 2)
            blob[2] = max(0.0, blob[2] * (1.0 + self.rng.normal(0.0, 0.03)))
            obs["blob"] = blob
        return obs

    def reset(self, **kwargs):
        self._sample_episode_parameters()
        if self.enabled:
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "config"):
                unwrapped.config.phone_tilt = int(np.clip(
                    self._base_phone_tilt + self.episode_parameters["camera_tilt_offset"],
                    26,
                    109,
                ))
            if hasattr(unwrapped, "action_executor"):
                previous_weight = self.episode_parameters["smoothing_previous_weight"]
                unwrapped.action_executor.smoothing = SmoothingConfig(
                    previous_weight=previous_weight,
                    requested_weight=1.0 - previous_weight,
                    max_delta=unwrapped.config.max_action_delta,
                )
        obs, info = self.env.reset(**kwargs)
        obs = self._augment_obs(obs)
        info = dict(info)
        info["domain_randomization"] = self.serializable_parameters()
        return obs, info

    def step(self, action):
        policy_requested_action = np.asarray(action, dtype=np.float32).copy()
        randomized_action = (
            self._randomize_action(policy_requested_action)
            if self.enabled else policy_requested_action
        )
        obs, reward, terminated, truncated, info = self.env.step(randomized_action)
        obs = self._augment_obs(obs)
        info = dict(info)
        info["domain_randomization"] = self.serializable_parameters()
        info["policy_requested_action"] = policy_requested_action
        info["randomized_action"] = randomized_action.copy()
        return obs, reward, terminated, truncated, info

    def serializable_parameters(self) -> dict[str, Any]:
        result = {}
        for key, value in self.episode_parameters.items():
            result[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return result

    def schema(self) -> dict[str, Any]:
        return asdict(self.ranges)
