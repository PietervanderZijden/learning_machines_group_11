from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from robobo_interface import SimulationRobobo
from robobo_interface.datatypes import Orientation, Position

FRONT_IR_INDICES = [2, 3, 4, 5, 7]
BACK_IR_INDICES = [0, 1, 6]


@dataclass
class RoboboObstacleEnvConfig:
    image_size: tuple[int, int] = (84, 84)

    max_wheel_speed: int = 100
    step_millis: int = 200

    max_episode_steps: int = 500

    max_ir_value: float = 400.0
    obstacle_penalty_threshold: float = 0.15
    collision_ir_threshold: float = 0.85

    progress_normalizer_m: float = 0.05
    progress_reward_scale: float = 2.0

    distance_bonus_scale: float = 0.0

    obstacle_penalty_scale: float = 0.15
    front_obstacle_penalty_scale: float = 0.25

    action_penalty_scale: float = 0.0

    turning_penalty_scale: float = 0.02
    spin_penalty_scale: float = 0.08

    alive_bonus: float = 0.01
    collision_penalty: float = 5.0

    idle_penalty_scale: float = 0.15
    idle_speed_threshold: float = 0.15

    movement_bonus_scale: float = 0.05

    low_displacement_penalty_scale: float = 0.08
    low_displacement_threshold_m: float = 0.01

    near_start_penalty_scale: float = 0.02
    near_start_distance_threshold_m: float = 0.10
    near_start_grace_steps: int = 20

    fc_early_penalty_scale: float = 0.1
    fc_early_threshold: float = 0.15

    reset_settle_seconds: float = 0.25

    start_position: Optional[Position] = None
    start_orientation: Optional[Orientation] = None


class RoboboObstacleAvoidanceEnv(gym.Env):
    """
    Observation:
        Dict:
            image:
                Grayscale camera image.
                Shape: (1, H, W)
                Dtype: uint8
            ir:
                Normalized IR sensor readings: [BackL, BackR, FrontL, FrontR,
                FrontC, FrontRR, BackC, FrontLL]
                Shape: (8,)
                Dtype: float32

    Action:
        Box(-1, 1, shape=(2,))
            action[0] = left wheel command
            action[1] = right wheel command

    Reward objective:
        Keep moving as far as possible from the episode start while avoiding
        collisions. The robot is penalised for idling so it cannot exploit the
        alive_bonus by sitting still.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rob: Optional[SimulationRobobo] = None,
        config: Optional[RoboboObstacleEnvConfig] = None,
    ) -> None:
        super().__init__()

        self.rob = rob or SimulationRobobo()
        self.config = config or RoboboObstacleEnvConfig()

        height, width = self.config.image_size

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(1, height, width),
                    dtype=np.uint8,
                ),
                "ir": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(8,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        if self.config.start_position is None:
            self._initial_position = self.rob.get_position()
        else:
            self._initial_position = self.config.start_position

        if self.config.start_orientation is None:
            self._initial_orientation = self.rob.get_orientation()
        else:
            self._initial_orientation = self.config.start_orientation

        self._episode_start_position = self._initial_position
        self._previous_distance_from_start = 0.0
        self._best_distance_from_start = 0.0
        self._previous_position = self._initial_position
        self._step_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        self._reset_simulation()

        self._episode_start_position = self.rob.get_position()
        self._previous_distance_from_start = 0.0
        self._best_distance_from_start = 0.0
        self._previous_position = self._episode_start_position
        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info(obs)

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        left_speed = int(action[0] * self.config.max_wheel_speed)
        right_speed = int(action[1] * self.config.max_wheel_speed)

        self.rob.move_blocking(
            left_speed,
            right_speed,
            self.config.step_millis,
        )

        obs = self._get_obs()
        info = self._get_info(obs)

        terminated = bool(info["collision"])
        truncated = self._step_count >= self.config.max_episode_steps

        reward = self._compute_reward(
            action=action,
            info=info,
            terminated=terminated,
            obs=obs,
        )

        info["reward"] = reward
        info["left_speed"] = left_speed
        info["right_speed"] = right_speed
        info["action_left"] = float(action[0])
        info["action_right"] = float(action[1])
        info["wheel_difference"] = float(abs(action[0] - action[1]))

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if not self.rob.is_stopped():
            self.rob.stop_simulation()

    def _reset_simulation(self) -> None:
        if not self.rob.is_stopped():
            self.rob.stop_simulation()

        self.rob.play_simulation()
        self.rob.sleep(self.config.reset_settle_seconds)

        self.rob.set_phone_tilt_blocking(100, 100)

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "image": self._read_grayscale_image(),
            "ir": self._read_normalized_irs(),
        }

    def _read_grayscale_image(self) -> np.ndarray:
        image_bgr = self.rob.read_image_front()

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray,
            self.config.image_size[::-1],
            interpolation=cv2.INTER_AREA,
        )

        return gray[None, :, :].astype(np.uint8)

    def _read_normalized_irs(self) -> np.ndarray:
        raw_irs = self.rob.read_irs()

        cleaned_irs: list[float] = []
        for value in raw_irs:
            if value is None or value is False:
                cleaned_irs.append(0.0)
            else:
                cleaned_irs.append(float(value))

        if len(cleaned_irs) != 8:
            cleaned_irs = (cleaned_irs + [0.0] * 8)[:8]

        irs = np.asarray(cleaned_irs, dtype=np.float32)
        irs = np.nan_to_num(
            irs,
            nan=0.0,
            posinf=self.config.max_ir_value,
            neginf=0.0,
        )

        irs = np.clip(irs / self.config.max_ir_value, 0.0, 1.0)

        return irs.astype(np.float32)

    def _get_info(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        position = self.rob.get_position()

        distance_from_start = self._xy_distance(
            self._episode_start_position,
            position,
        )

        step_displacement = self._xy_distance(
            self._previous_position,
            position,
        )
        ir = obs["ir"]

        front_obstacle_closeness = float(np.max(ir[FRONT_IR_INDICES]))
        back_obstacle_closeness = float(np.max(ir[BACK_IR_INDICES]))
        max_obstacle_closeness = float(np.max(ir))

        collision = front_obstacle_closeness >= self.config.collision_ir_threshold

        return {
            "x": float(position.x),
            "y": float(position.y),
            "z": float(position.z),
            "distance_from_start": float(distance_from_start),
            "step_displacement": float(step_displacement),
            "front_obstacle_closeness": front_obstacle_closeness,
            "back_obstacle_closeness": back_obstacle_closeness,
            "max_obstacle_closeness": max_obstacle_closeness,
            "collision": collision,
            "step_count": self._step_count,
        }

    def _compute_reward(
        self,
        action: np.ndarray,
        info: dict[str, Any],
        terminated: bool,
        obs: dict[str, np.ndarray],
    ) -> float:
        current_position = self.rob.get_position()
        current_distance = float(info["distance_from_start"])
        step_displacement = float(info["step_displacement"])

        record_progress = max(
            0.0,
            current_distance - self._best_distance_from_start,
        )

        self._best_distance_from_start = max(
            self._best_distance_from_start,
            current_distance,
        )

        self._previous_distance_from_start = current_distance
        self._previous_position = current_position

        normalized_progress = np.clip(
            record_progress / self.config.progress_normalizer_m,
            0.0,
            1.0,
        )

        progress_reward = self.config.progress_reward_scale * float(normalized_progress)

        distance_bonus = self.config.distance_bonus_scale * current_distance

        max_closeness = float(info["max_obstacle_closeness"])
        front_closeness = float(info["front_obstacle_closeness"])

        obstacle_penalty = 0.0
        if max_closeness > self.config.obstacle_penalty_threshold:
            obstacle_penalty = self.config.obstacle_penalty_scale * (
                max_closeness - self.config.obstacle_penalty_threshold
            )

        front_obstacle_penalty = 0.0
        if front_closeness > self.config.obstacle_penalty_threshold:
            front_obstacle_penalty = self.config.front_obstacle_penalty_scale * (
                front_closeness - self.config.obstacle_penalty_threshold
            )

        fc_early_penalty = 0.0
        fc_value = float(obs["ir"][4])
        if fc_value > self.config.fc_early_threshold:
            fc_early_penalty = self.config.fc_early_penalty_scale * (
                fc_value - self.config.fc_early_threshold
            )

        mean_abs_action = float(np.mean(np.abs(action)))

        action_penalty = self.config.action_penalty_scale * mean_abs_action

        wheel_difference = float(abs(action[0] - action[1]))
        spin_penalty = self.config.spin_penalty_scale * wheel_difference

        turning_penalty = 0.0
        if action[0] * action[1] < 0.0:
            turning_penalty = self.config.turning_penalty_scale * wheel_difference

        idle_penalty = 0.0
        if mean_abs_action < self.config.idle_speed_threshold:
            idle_penalty = self.config.idle_penalty_scale * (
                1.0 - mean_abs_action / self.config.idle_speed_threshold
            )

        low_displacement_penalty = 0.0
        if step_displacement < self.config.low_displacement_threshold_m:
            low_displacement_penalty = self.config.low_displacement_penalty_scale * (
                1.0 - step_displacement / self.config.low_displacement_threshold_m
            )

        near_start_penalty = 0.0
        if (
            self._step_count > self.config.near_start_grace_steps
            and current_distance < self.config.near_start_distance_threshold_m
        ):
            near_start_penalty = self.config.near_start_penalty_scale * (
                1.0 - current_distance / self.config.near_start_distance_threshold_m
            )

        forward_component = float(min(action[0], action[1]))

        if forward_component > 0.0:
            movement_bonus = self.config.movement_bonus_scale * forward_component
        else:
            movement_bonus = self.config.movement_bonus_scale * forward_component * 2.0

        reward = (
            progress_reward
            + distance_bonus
            + movement_bonus
            + self.config.alive_bonus
            - obstacle_penalty
            - front_obstacle_penalty
            - fc_early_penalty
            - idle_penalty
            - low_displacement_penalty
            - near_start_penalty
            - action_penalty
            - turning_penalty
            - spin_penalty
        )

        if terminated:
            reward -= self.config.collision_penalty

        return float(reward)

    @staticmethod
    def _xy_distance(a: Position, b: Position) -> float:
        dx = float(a.x) - float(b.x)
        dy = float(a.y) - float(b.y)

        return float(np.sqrt(dx * dx + dy * dy))
