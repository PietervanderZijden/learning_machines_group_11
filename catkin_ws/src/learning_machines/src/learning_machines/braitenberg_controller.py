"""Braitenberg controller for IR obstacle avoidance + green blob steering."""
from __future__ import annotations

import numpy as np


class BraitenbergController:
    """IR avoidance + green blob steering."""

    def __init__(
        self,
        max_speed: int = 100,
        avoid_gain: float = 1.0,
        steer_gain: float = 0.5,
        ir_indices_front: list[int] | None = None,
    ):
        self.max_speed = max_speed
        self.avoid_gain = avoid_gain
        self.steer_gain = steer_gain
        self.ir_indices_front = ir_indices_front or [2, 3, 4, 5, 7]

    def compute(self, ir: np.ndarray, blob: np.ndarray) -> tuple[int, int]:
        front_ir = np.array([ir[i] for i in self.ir_indices_front if i < len(ir)])
        obstacle_avoidance = float(np.mean(front_ir))

        left_correction = 0.0
        right_correction = 0.0

        if len(front_ir) >= 3:
            left_wall = float(np.mean(front_ir[:len(front_ir)//2]))
            right_wall = float(np.mean(front_ir[len(front_ir)//2:]))
            left_correction = (right_wall - left_wall) * self.avoid_gain
            right_correction = (left_wall - right_wall) * self.avoid_gain

        blob_x = blob[0] if len(blob) >= 4 and blob[3] > 0.5 else 0.5
        blob_offset = (blob_x - 0.5) * self.steer_gain

        base_speed = self.max_speed * (1.0 - obstacle_avoidance * 0.8)
        base_speed = max(10, min(self.max_speed, base_speed))

        left = int(np.clip(base_speed + left_correction * self.max_speed - blob_offset * self.max_speed, -self.max_speed, self.max_speed))
        right = int(np.clip(base_speed + right_correction * self.max_speed + blob_offset * self.max_speed, -self.max_speed, self.max_speed))

        return left, right
