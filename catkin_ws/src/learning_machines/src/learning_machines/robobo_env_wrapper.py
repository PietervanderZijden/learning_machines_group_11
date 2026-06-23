"""Gym wrapper to adapt RoboboCompactEnv for NM512/dreamerv3-torch."""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import sys


class RoboboNM512Wrapper(gym.Env):
    """Wraps RoboboCompactEnv to match NM512's expected observation format.

    NM512 expects:
    - obs as a dict with "image" (uint8 HWC) + any vector keys
    - 4-tuple (obs, reward, done, info) from step()
    - "is_first" in obs dict after reset
    - continuous actions in [-1, 1]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        robobo_env: gym.Env,
        include_ir: bool = True,
    ):
        super().__init__()
        self._env = robobo_env
        self._include_ir = include_ir
        self._is_first = True
        self._episode_step = 0

        obs_spaces = {"image": gym.spaces.Box(
            low=0, high=255,
            shape=(64, 64, 3),
            dtype=np.uint8,
        )}
        if include_ir:
            obs_spaces["ir"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(obs_spaces)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

    def _convert_obs(self, obs: dict) -> dict:
        raw_image = obs["image"]
        if raw_image.ndim == 3 and raw_image.shape[0] == 1:
            gray = raw_image[0]
            hwc = np.stack([gray, gray, gray], axis=-1)
        elif raw_image.ndim == 2:
            hwc = np.stack([raw_image, raw_image, raw_image], axis=-1)
        elif raw_image.ndim == 3 and raw_image.shape[2] == 3:
            hwc = raw_image
        elif raw_image.ndim == 3 and raw_image.shape[0] == 3:
            hwc = np.transpose(raw_image, (1, 2, 0))
        else:
            hwc = np.stack([raw_image.squeeze()] * 3, axis=-1)

        result = {
            "image": hwc.astype(np.uint8),
            "is_first": self._is_first,
            "is_terminal": False,
        }
        if self._include_ir and "ir" in obs:
            result["ir"] = obs["ir"].astype(np.float32)
        return result

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        self._is_first = True
        self._episode_step = 0
        return self._convert_obs(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        info = dict(info)
        if done:
            info.setdefault(
                "discount",
                np.array(0.0 if terminated else 1.0, dtype=np.float32),
            )
        self._is_first = False
        self._episode_step += 1
        converted = self._convert_obs(obs)
        converted["is_terminal"] = terminated
        return converted, reward, done, info

    def render(self, *args, **kwargs):
        raise NotImplementedError("Render not supported")

    def close(self):
        self._env.close()
