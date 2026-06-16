from __future__ import annotations

import gymnasium as gym
import numpy as np


class DomainRandomizationWrapper(gym.Wrapper):
    """Applies IR noise and pose jitter for sim-to-real transfer."""

    def __init__(
        self,
        env: gym.Env,
        ir_noise_std: float = 0.02,
        ir_noise_prob: float = 0.5,
        pose_jitter: float = 0.02,
        enabled: bool = True,
    ) -> None:
        super().__init__(env)
        self.ir_noise_std = ir_noise_std
        self.ir_noise_prob = ir_noise_prob
        self.pose_jitter = pose_jitter
        self.enabled = enabled

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.enabled and isinstance(obs, dict):
            if "ir" in obs:
                ir = obs["ir"].copy()
                noise_mask = np.random.random(len(ir)) < self.ir_noise_prob
                noise = np.random.normal(0, self.ir_noise_std, len(ir))
                ir = np.clip(ir + noise * noise_mask, 0.0, 1.0)
                obs["ir"] = ir

        return obs, reward, terminated, truncated, info
