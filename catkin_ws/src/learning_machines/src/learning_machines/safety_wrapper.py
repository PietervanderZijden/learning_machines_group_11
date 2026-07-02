from __future__ import annotations

import gymnasium as gym
import numpy as np

from learning_machines.transfer import PreActionSafetyFilter, SafetyConfig


class SafetyWrapper(gym.Wrapper):
    """Compatibility wrapper implementing safety before the action is executed."""

    def __init__(
        self,
        env: gym.Env,
        max_wheel_speed: int = 100,
        front_ir_indices: list[int] | None = None,
        danger_threshold: float = 0.7,
        critical_threshold: float = 0.9,
    ) -> None:
        super().__init__(env)
        self.max_wheel_speed = max_wheel_speed
        self.front_ir_indices = front_ir_indices or [2, 3, 4, 5, 7]
        self.danger_threshold = danger_threshold
        self.critical_threshold = critical_threshold
        self._latest_ir = np.zeros(8, dtype=np.float32)
        self._filter = PreActionSafetyFilter(SafetyConfig(
            warning_threshold=danger_threshold,
            critical_threshold=critical_threshold,
            front_indices=tuple(self.front_ir_indices),
        ))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if isinstance(obs, dict) and "ir" in obs:
            self._latest_ir = np.asarray(obs["ir"], dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        filtered, safety_event = self._filter.filter(action, self._latest_ir)
        obs, reward, terminated, truncated, info = self.env.step(filtered)
        if isinstance(obs, dict) and "ir" in obs:
            self._latest_ir = np.asarray(obs["ir"], dtype=np.float32).copy()
        if safety_event is not None:
            info["wrapper_safety_override"] = safety_event
        return obs, reward, terminated, truncated, info
