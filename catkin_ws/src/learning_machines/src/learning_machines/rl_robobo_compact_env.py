from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass
class RoboboCompactEnvConfig:
    image_size: tuple[int, int] = (64, 64)
    max_wheel_speed: int = 100
    step_millis: int = 400
    phone_tilt: int = 100
    max_episode_steps: int = 300
    max_ir_value: float = 400.0
    collision_ir_threshold: float = 0.85
    collect_reward: float = 100.0
    speed_bonus_scale: float = 50.0
    collision_penalty: float = 0.0
    use_reward_shaping: bool = False
    randomize_food_positions: bool = True
    food_arena_radius: float = 0.95
    food_min_radius: float = 0.30
    arena_center: tuple[float, float] = (-3.125, 0.8)
    reset_settle_time: float = 1.5
    settle_sleep: float = 0.05


class RoboboCompactEnv(gym.Env):
    """
    Compact Robobo env for TD-MPC2 training.

    Observation:
        Dict:
            blob: [x, y, area, found] (4,) — green food blob detection
            ir: [BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL] (8,)
    Action:
        Box(-1, 1, shape=(2,)) — [left_speed, right_speed]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rob=None,
        config: RoboboCompactEnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RoboboCompactEnvConfig()

        if rob is None:
            from robobo_interface import SimulationRobobo
            rob = SimulationRobobo()
        self.rob = rob

        self.observation_space = spaces.Dict({
            "blob": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "ir": spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        self._step_count = 0
        self._prev_food_count = 0
        self._food_handles: list[int] = []
        self._arena_cx, self._arena_cy = self.config.arena_center
        self._num_food = 7

        self._blob_low = np.array([35, 80, 80], dtype=np.uint8)
        self._blob_high = np.array([85, 255, 255], dtype=np.uint8)
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        if self.rob.is_stopped():
            self.rob.play_simulation()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_simulation()
        self._step_count = 0
        self._prev_food_count = 0
        self.rob._used_pids.clear()
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self._step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        left_speed = float(action[0] * self.config.max_wheel_speed)
        right_speed = float(action[1] * self.config.max_wheel_speed)
        duration_s = self.config.step_millis / 1000.0

        try:
            self.rob.set_wheel_speeds(left_speed, right_speed, duration_s=duration_s)
            self.rob.sleep(duration_s)
        except RuntimeError:
            if not self.rob.is_running():
                obs = self._get_obs()
                info = self._get_info()
                info["simulation_stopped"] = True
                return obs, 0.0, False, True, info
            raise

        obs = self._get_obs()
        info = self._get_info()

        new_food = info["food_collected"]
        newly_collected = max(0, new_food - self._prev_food_count)
        self._prev_food_count = new_food

        reward = newly_collected * self.config.collect_reward

        if new_food >= self._num_food:
            speed_bonus = self.config.speed_bonus_scale * (
                self.config.max_episode_steps / max(1, self._step_count)
            )
            reward += speed_bonus

        terminated = new_food >= self._num_food
        truncated = self._step_count >= self.config.max_episode_steps

        info["newly_collected"] = newly_collected
        info["left_speed"] = left_speed
        info["right_speed"] = right_speed

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            self.rob.set_wheel_speeds(0, 0)
            if not self.rob.is_stopped():
                self.rob.stop_simulation()
        except Exception:
            pass

    def _reset_simulation(self) -> None:
        self.rob.set_wheel_speeds(0, 0)

        if not self.rob.is_stopped():
            self.rob.stop_simulation()
        time.sleep(0.1)

        self._discover_food_handles()
        if self.config.randomize_food_positions:
            self._randomize_food_positions()

        self.rob.play_simulation()
        self.rob.sleep(self.config.reset_settle_time)

        self._fix_lifted_food()

        pos = self.rob.get_position()
        self._initial_pos_x = pos.x
        self._initial_pos_y = pos.y

    def _discover_food_handles(self) -> None:
        self._food_handles = []
        sim = self.rob._sim
        names = ["/Food", "/Food0", "/Food1", "/Food2", "/Food3", "/Food4", "/Food5"]
        for name in names:
            try:
                h = sim.getObject(name)
                if h >= 0:
                    self._food_handles.append(h)
            except Exception:
                pass
        self._num_food = max(1, len(self._food_handles))

    def _randomize_food_positions(self) -> None:
        sim = self.rob._sim
        for h in self._food_handles:
            placed = False
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                r = random.uniform(self.config.food_min_radius, self.config.food_arena_radius)
                fx = self._arena_cx + r * math.cos(angle)
                fy = self._arena_cy + r * math.sin(angle)
                dx = fx - self._initial_pos_x if hasattr(self, '_initial_pos_x') else fx - self._arena_cx
                dy = fy - self._initial_pos_y if hasattr(self, '_initial_pos_y') else fy - self._arena_cy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist >= self.config.food_min_radius:
                    try:
                        sim.setObjectPosition(h, [fx, fy, 0.025])
                        placed = True
                        break
                    except Exception:
                        pass
            if not placed:
                try:
                    sim.setObjectPosition(h, [fx, fy, 0.025])
                except Exception:
                    pass

    def _fix_lifted_food(self) -> None:
        sim = self.rob._sim
        for h in self._food_handles:
            try:
                pos = sim.getObjectPosition(h, sim.handle_world)
                if pos[2] > 0.5:
                    sim.setObjectPosition(h, [pos[0], pos[1], 0.025])
            except Exception:
                pass

    def _safe_read_irs(self) -> list[float]:
        for _ in range(5):
            try:
                raw = self.rob.read_irs()
                cleaned = []
                for v in raw:
                    if v is None or v is False:
                        cleaned.append(0.0)
                    else:
                        cleaned.append(float(v))
                if len(cleaned) == 8:
                    return cleaned
                cleaned = (cleaned + [0.0] * 8)[:8]
                return cleaned
            except Exception:
                time.sleep(0.2)
        return [0.0] * 8

    def _get_obs(self) -> dict:
        raw_irs = self._safe_read_irs()
        irs = np.clip(
            np.array(raw_irs, dtype=np.float32) / self.config.max_ir_value,
            0.0, 1.0,
        )

        try:
            image_bgr = self.rob.read_image_front()
            blob = self._detect_blob(image_bgr)
        except Exception:
            blob = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        return {
            "blob": blob,
            "ir": irs,
        }

    def _detect_blob(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._blob_low, self._blob_high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = h * w
        area_ratio = area / img_area

        if area_ratio < 0.001 or area_ratio > 0.5:
            return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return np.array([0.5, 0.5, area_ratio, 1.0], dtype=np.float32)

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        return np.array([cx / w, cy / h, area_ratio, 1.0], dtype=np.float32)

    def _get_info(self) -> dict:
        pos = self.rob.get_position()
        try:
            food_collected = self.rob.get_nr_food_collected()
        except Exception:
            food_collected = self._prev_food_count

        raw_irs = self._safe_read_irs()
        ir_arr = np.array(raw_irs, dtype=np.float32) / self.config.max_ir_value
        front_ir = [ir_arr[i] for i in [2, 3, 4, 5, 7]]
        collision = max(front_ir) >= self.config.collision_ir_threshold

        return {
            "x": pos.x,
            "y": pos.y,
            "z": pos.z,
            "food_collected": food_collected,
            "collision": collision,
            "step_count": self._step_count,
        }
