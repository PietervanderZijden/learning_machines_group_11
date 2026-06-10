from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import gymnasium as gym
import numpy as np
from learning_machines.rl_robobo_env import (
    RoboboObstacleAvoidanceEnv,
    RoboboObstacleEnvConfig,
)
from robobo_interface import SimulationRobobo


class MultiRoboboObstacleAvoidanceEnv(gym.Env):
    """
    Gymnasium wrapper that trains on multiple Robobo instances in one
    CoppeliaSim scene.

    Each Robobo is selected by passing a different `identifier` to
    SimulationRobobo(identifier=...).

    The active robot/environment is selected at reset. After
    `switch_every_steps` steps, the episode is truncated so SB3 resets the env
    and a new robot/environment is sampled.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        identifiers: Sequence[int] = (0, 1, 2),
        config: Optional[RoboboObstacleEnvConfig] = None,
        switch_every_steps: int = 500,
        avoid_immediate_repeat: bool = True,
    ) -> None:
        super().__init__()

        if len(identifiers) == 0:
            raise ValueError("At least one Robobo identifier is required.")

        self.identifiers = tuple(int(identifier) for identifier in identifiers)
        self.config = config or RoboboObstacleEnvConfig()
        self.switch_every_steps = int(switch_every_steps)
        self.avoid_immediate_repeat = avoid_immediate_repeat

        if self.switch_every_steps <= 0:
            raise ValueError("switch_every_steps must be positive.")

        self._envs: dict[int, RoboboObstacleAvoidanceEnv] = {}

        for identifier in self.identifiers:
            rob = SimulationRobobo(identifier=identifier)
            env_config = copy.deepcopy(self.config)
            self._envs[identifier] = RoboboObstacleAvoidanceEnv(
                rob=rob,
                config=env_config,
            )

        first_env = self._envs[self.identifiers[0]]
        self.observation_space = first_env.observation_space
        self.action_space = first_env.action_space

        self._active_identifier: int | None = None
        self._active_env: RoboboObstacleAvoidanceEnv | None = None
        self._steps_since_switch = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        requested_identifier = None
        if options is not None:
            requested_identifier = options.get("identifier")

        if requested_identifier is not None:
            identifier = int(requested_identifier)
            if identifier not in self._envs:
                raise ValueError(
                    f"Unknown Robobo identifier {identifier}. "
                    f"Available identifiers: {self.identifiers}"
                )
        else:
            identifier = self._sample_identifier()

        self._active_identifier = identifier
        self._active_env = self._envs[identifier]
        self._steps_since_switch = 0

        obs, info = self._active_env.reset(seed=seed, options=options)

        info["active_robot_identifier"] = identifier
        info["robot_switch"] = True

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._active_env is None or self._active_identifier is None:
            raise RuntimeError("Environment must be reset before calling step().")

        obs, reward, terminated, truncated, info = self._active_env.step(action)

        self._steps_since_switch += 1

        switch_due = self._steps_since_switch >= self.switch_every_steps
        if switch_due:
            truncated = True

        info["active_robot_identifier"] = self._active_identifier
        info["robot_switch_due"] = switch_due
        info["steps_since_robot_switch"] = self._steps_since_switch

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        for env in self._envs.values():
            env.close()

    def _sample_identifier(self) -> int:
        if len(self.identifiers) == 1:
            return self.identifiers[0]

        choices = list(self.identifiers)

        if (
            self.avoid_immediate_repeat
            and self._active_identifier is not None
            and len(choices) > 1
        ):
            choices = [
                identifier
                for identifier in choices
                if identifier != self._active_identifier
            ]

        index = int(self.np_random.integers(0, len(choices)))
        return choices[index]


@dataclass
class DomainRandomizationConfig:
    """
    Training-time randomization to make the policy less dependent on exact
    simulator sensor values, camera appearance and perfect wheel commands.
    """

    enabled: bool = True

    # IR randomization.
    ir_scale_range: tuple[float, float] = (0.6, 1.4)
    ir_bias_range: tuple[float, float] = (-0.08, 0.08)
    ir_noise_std: float = 0.03
    ir_dropout_prob: float = 0.02

    # Image randomization.
    image_contrast_range: tuple[float, float] = (0.75, 1.25)
    image_brightness_range: tuple[float, float] = (-25.0, 25.0)
    image_noise_std: float = 6.0
    image_blur_prob: float = 0.10

    # Action / actuator randomization.
    action_scale_range: tuple[float, float] = (0.85, 1.15)
    action_bias_range: tuple[float, float] = (-0.04, 0.04)
    action_noise_std: float = 0.025
    action_latency_prob: float = 0.05


class RoboboDomainRandomizationWrapper(gym.Wrapper):
    """
    Applies domain randomization during training.

    Randomizes:
        - IR scale, bias, Gaussian noise and occasional dropout
        - image brightness, contrast, noise and blur
        - action scale, bias, Gaussian noise and occasional one-step latency
    """

    def __init__(
        self,
        env: gym.Env,
        config: Optional[DomainRandomizationConfig] = None,
    ) -> None:
        super().__init__(env)

        self.config = config or DomainRandomizationConfig()
        self._rng = np.random.default_rng()

        self._ir_scale: np.ndarray | None = None
        self._ir_bias: np.ndarray | None = None
        self._action_scale: np.ndarray | None = None
        self._action_bias: np.ndarray | None = None
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._sample_episode_randomization()
        self._last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._randomize_obs(obs)

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        randomized_action = self._randomize_action(action)

        obs, reward, terminated, truncated, info = self.env.step(
            randomized_action,
        )

        obs = self._randomize_obs(obs)

        info["domain_randomization_enabled"] = self.config.enabled
        info["executed_action_left"] = float(randomized_action[0])
        info["executed_action_right"] = float(randomized_action[1])

        return obs, reward, terminated, truncated, info

    def _sample_episode_randomization(self) -> None:
        ir_shape = self.observation_space.spaces["ir"].shape
        action_shape = self.action_space.shape

        self._ir_scale = self._rng.uniform(
            self.config.ir_scale_range[0],
            self.config.ir_scale_range[1],
            size=ir_shape,
        ).astype(np.float32)

        self._ir_bias = self._rng.uniform(
            self.config.ir_bias_range[0],
            self.config.ir_bias_range[1],
            size=ir_shape,
        ).astype(np.float32)

        self._action_scale = self._rng.uniform(
            self.config.action_scale_range[0],
            self.config.action_scale_range[1],
            size=action_shape,
        ).astype(np.float32)

        self._action_bias = self._rng.uniform(
            self.config.action_bias_range[0],
            self.config.action_bias_range[1],
            size=action_shape,
        ).astype(np.float32)

    def _randomize_obs(
        self,
        obs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if not self.config.enabled:
            return obs

        randomized_obs = {
            "image": self._randomize_image(obs["image"]),
            "ir": self._randomize_ir(obs["ir"]),
        }

        return randomized_obs

    def _randomize_ir(self, ir: np.ndarray) -> np.ndarray:
        if self._ir_scale is None or self._ir_bias is None:
            self._sample_episode_randomization()

        noisy_ir = ir.astype(np.float32).copy()

        noisy_ir = noisy_ir * self._ir_scale + self._ir_bias

        if self.config.ir_noise_std > 0.0:
            noisy_ir += self._rng.normal(
                0.0,
                self.config.ir_noise_std,
                size=noisy_ir.shape,
            ).astype(np.float32)

        if self.config.ir_dropout_prob > 0.0:
            dropout_mask = self._rng.random(noisy_ir.shape) < (
                self.config.ir_dropout_prob
            )
            noisy_ir[dropout_mask] = 0.0

        return np.clip(noisy_ir, 0.0, 1.0).astype(np.float32)

    def _randomize_image(self, image: np.ndarray) -> np.ndarray:
        # image shape is expected to be (1, H, W).
        img = image[0].astype(np.float32)

        contrast = self._rng.uniform(
            self.config.image_contrast_range[0],
            self.config.image_contrast_range[1],
        )
        brightness = self._rng.uniform(
            self.config.image_brightness_range[0],
            self.config.image_brightness_range[1],
        )

        img = img * contrast + brightness

        if self.config.image_noise_std > 0.0:
            img += self._rng.normal(
                0.0,
                self.config.image_noise_std,
                size=img.shape,
            ).astype(np.float32)

        if self._rng.random() < self.config.image_blur_prob:
            img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0.0)

        img = np.clip(img, 0.0, 255.0).astype(np.uint8)

        return img[None, :, :]

    def _randomize_action(self, action: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return action

        if self._action_scale is None or self._action_bias is None:
            self._sample_episode_randomization()

        action = np.asarray(action, dtype=np.float32)

        randomized_action = action * self._action_scale + self._action_bias

        if self.config.action_noise_std > 0.0:
            randomized_action += self._rng.normal(
                0.0,
                self.config.action_noise_std,
                size=randomized_action.shape,
            ).astype(np.float32)

        randomized_action = np.clip(
            randomized_action,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)

        if self._rng.random() < self.config.action_latency_prob:
            executed_action = self._last_action.copy()
            self._last_action = randomized_action.copy()
            return executed_action

        self._last_action = randomized_action.copy()
        return randomized_action
