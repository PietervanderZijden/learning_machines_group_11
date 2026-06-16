from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np


class SafetyWrapper(gym.Wrapper):
    """Intercepts actions to prevent wall collisions via IR sensor monitoring.

    When front IR sensors exceed danger_threshold, backs away.
    When critical_threshold exceeded, full stop + spin.
    """

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

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        ir = obs["ir"] if isinstance(obs, dict) and "ir" in obs else None
        if ir is not None:
            front_vals = [ir[i] for i in self.front_ir_indices if i < len(ir)]
            max_front = max(front_vals) if front_vals else 0.0

            if max_front >= self.critical_threshold:
                action = np.array([0.7, -0.7], dtype=np.float32) * 0.5
                info["safety_override"] = "spin"
            elif max_front >= self.danger_threshold:
                action = np.array([-0.5, -0.5], dtype=np.float32)
                info["safety_override"] = "back_away"

        return obs, reward, terminated, truncated, info
