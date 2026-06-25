"""Gym wrapper to adapt RoboboCompactEnv for NM512/dreamerv3-torch."""
from __future__ import annotations

from typing import Any
from pathlib import Path

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
        curriculum_controller=None,
        curriculum_state_path: str | Path | None = None,
        domain_randomization: bool = False,
    ):
        super().__init__()
        self._env = robobo_env
        self._include_ir = include_ir
        self._is_first = True
        self._episode_step = 0
        self._spin_streak = 0
        self._curriculum_controller = curriculum_controller
        self._curriculum_state_path = (
            Path(curriculum_state_path) if curriculum_state_path else None
        )
        self._domain_randomization = bool(domain_randomization)
        self._curriculum_tracking_enabled = True
        self._curriculum_promotion_callback = None

        source_image_space = robobo_env.observation_space.spaces["image"]
        source_shape = tuple(source_image_space.shape)
        if len(source_shape) != 3:
            raise ValueError(
                f"Robobo image observation must be 3D, got {source_shape}"
            )
        if source_shape[0] in (1, 3):
            image_shape = (source_shape[1], source_shape[2], 3)
        elif source_shape[-1] in (1, 3):
            image_shape = (source_shape[0], source_shape[1], 3)
        else:
            raise ValueError(
                f"Robobo image observation must have 1 or 3 channels, got {source_shape}"
            )
        obs_spaces = {"image": gym.spaces.Box(
            low=0, high=255,
            shape=image_shape,
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
        self._spin_streak = 0
        return self._convert_obs(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        info = dict(info)
        promotion = None
        episode_stage = None
        stage_steps = None
        stage_episodes = None
        stage_success = None
        if (
            self._curriculum_controller is not None
            and self._curriculum_tracking_enabled
        ):
            controller = self._curriculum_controller
            controller.record_transition()
            episode_stage = controller.stage
            if done:
                promotion = controller.record_episode(bool(terminated))
                if promotion is not None:
                    stage_steps = promotion["stage_steps"]
                    stage_episodes = promotion["episodes"]
                    stage_success = promotion["success_rate"]
                else:
                    stage_steps = controller.stage_steps
                    stage_episodes = len(controller.recent_outcomes)
                    stage_success = controller.rolling_success
                self.set_push_curriculum_stage(
                    controller.stage, self._domain_randomization
                )
                if self._curriculum_state_path is not None:
                    controller.save(self._curriculum_state_path)
                if (
                    promotion is not None
                    and self._curriculum_promotion_callback is not None
                ):
                    self._curriculum_promotion_callback(promotion)
        if done:
            info.setdefault(
                "discount",
                np.array(0.0 if terminated else 1.0, dtype=np.float32),
            )
        self._is_first = False
        self._episode_step += 1
        converted = self._convert_obs(obs)
        converted["is_terminal"] = terminated
        requested = np.asarray(
            info.get("requested_action", action), dtype=np.float32
        )
        executed = np.asarray(
            info.get("executed_action", action), dtype=np.float32
        )
        opposite = bool(
            executed.shape == (2,)
            and executed[0] * executed[1] < 0.0
        )
        self._spin_streak = self._spin_streak + 1 if opposite else 0
        converted.update({
            "log_avg_opposite_wheels": np.float32(opposite),
            "log_max_spin_sequence": np.float32(self._spin_streak),
            "log_avg_wheel_saturation": np.float32(
                np.mean(np.abs(executed) >= 0.999)
            ),
            "log_avg_action_execution_error": np.float32(
                np.max(np.abs(requested - executed))
            ),
            "log_avg_block_goal_distance": np.float32(
                info.get("block_goal_distance", 0.0)
            ),
            "log_sum_potential_shaping": np.float32(
                info.get("potential_shaping", 0.0)
            ),
            "log_sum_time_cost": np.float32(info.get("time_cost", 0.0)),
        })
        if episode_stage is not None:
            controller = self._curriculum_controller
            converted.update({
                "log_avg_curriculum_stage": np.float32(episode_stage),
                "log_max_curriculum_stage_steps": np.float32(
                    stage_steps if done else controller.stage_steps
                ),
                "log_max_curriculum_stage_episodes": np.float32(
                    stage_episodes
                    if done
                    else len(controller.recent_outcomes)
                ),
                "log_avg_curriculum_rolling_success": np.float32(
                    stage_success if done else controller.rolling_success
                ),
                "log_avg_curriculum_domain_randomization": np.float32(
                    self._domain_randomization and controller.stage == 2
                ),
            })
        return converted, reward, done, info

    def render(self, *args, **kwargs):
        raise NotImplementedError("Render not supported")

    def set_push_curriculum_stage(
        self,
        stage: int,
        domain_randomization: bool,
    ) -> None:
        """Apply a push stage before the next simulator reset."""
        if stage not in (0, 1, 2):
            raise ValueError("push curriculum stage must be 0, 1, or 2")
        base_env = self._env.unwrapped
        if not hasattr(base_env, "config"):
            raise RuntimeError("push curriculum requires an environment config")
        base_env.config.push_curriculum_stage = stage
        if hasattr(self._env, "enabled"):
            self._env.enabled = bool(domain_randomization and stage == 2)

    def set_curriculum_tracking_enabled(self, enabled: bool) -> None:
        self._curriculum_tracking_enabled = bool(enabled)

    def set_curriculum_promotion_callback(self, callback) -> None:
        self._curriculum_promotion_callback = callback

    def close(self):
        if (
            self._curriculum_controller is not None
            and self._curriculum_state_path is not None
        ):
            self._curriculum_controller.save(self._curriculum_state_path)
        self._env.close()
