"""Simulator-independent compact environment contract tests."""

from __future__ import annotations

import cv2
import numpy as np
from gymnasium.utils.env_checker import check_env

from learning_machines.rl_robobo_compact_env import (
    RoboboCompactEnv,
    RoboboCompactEnvConfig,
)


class FakeRobobo:
    """Provide deterministic hardware-style Robobo observations."""

    def __init__(self) -> None:
        """Initialize sensor and command state."""
        self._used_pids: set[int] = set()
        self.commands: list[tuple[int, int, int]] = []

    def set_wheel_speeds(self, left: float, right: float) -> None:
        """Accept a continuous wheel command."""

    def move_blocking(self, left: int, right: int, millis: int) -> None:
        """Record a blocking wheel command."""
        self.commands.append((left, right, millis))

    def read_irs(self) -> list[float]:
        """Return eight clear proximity readings."""
        return [0.0] * 8

    def read_image_front(self) -> np.ndarray:
        """Return an image containing one green target."""
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(image, (32, 32), 8, (0, 255, 0), -1)
        return image

    def read_phone_tilt(self) -> int:
        """Return the configured phone tilt."""
        return 100

    def set_phone_tilt(self, position: int, speed: int) -> None:
        """Accept a phone tilt command."""

    def sleep(self, seconds: float) -> None:
        """Accept a hardware settling interval."""

    def get_nr_food_collected(self) -> int:
        """Return a stable food count."""
        return 0


def make_env() -> RoboboCompactEnv:
    """Build a fast fake-hardware environment."""
    return RoboboCompactEnv(
        rob=FakeRobobo(),
        config=RoboboCompactEnvConfig(
            initialize_phone_tilt=False,
            reset_settle_time=0.0,
            settle_sleep=0.0,
            max_episode_steps=2,
            max_episode_seconds=0.8,
        ),
    )


def test_gymnasium_contract() -> None:
    """Satisfy Gymnasium reset and step requirements."""
    env = make_env()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_observation_and_action_contract() -> None:
    """Return bounded observations and five step values."""
    env = make_env()
    try:
        observation, info = env.reset(seed=4)
        assert env.observation_space.contains(observation)
        assert isinstance(info, dict)
        result = env.step(np.zeros(2, dtype=np.float32))
        assert len(result) == 5
        assert env.observation_space.contains(result[0])
    finally:
        env.close()
