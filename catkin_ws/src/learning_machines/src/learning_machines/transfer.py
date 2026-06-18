"""Shared sim-to-real contracts for Robobo training and deployment."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


OBSERVATION_CONTRACT_VERSION = "robobo-obs-v2"
REWARD_CONTRACT_VERSION = "robobo-reward-v4"
CONTROL_INTERVAL_SECONDS = 0.4
IR_LABELS = ("BackL", "BackR", "FrontL", "FrontR", "FrontC", "FrontRR", "BackC", "FrontLL")


@dataclass(frozen=True)
class IRSensorCalibration:
    free_space: float
    near_obstacle: float
    polarity: int = 1
    exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.polarity not in (-1, 1):
            raise ValueError("polarity must be -1 or 1")
        if self.exponent <= 0:
            raise ValueError("exponent must be positive")
        if self.free_space == self.near_obstacle:
            raise ValueError("free_space and near_obstacle must differ")


@dataclass(frozen=True)
class CalibrationProfile:
    name: str
    sensors: tuple[IRSensorCalibration, ...]
    version: str = OBSERVATION_CONTRACT_VERSION
    source: str = "default"

    def __post_init__(self) -> None:
        if self.version != OBSERVATION_CONTRACT_VERSION:
            raise ValueError(f"unsupported observation contract: {self.version}")
        if len(self.sensors) != 8:
            raise ValueError("calibration profile must contain exactly eight sensors")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationProfile":
        sensors = data["sensors"]
        if isinstance(sensors, dict):
            sensors = [sensors[label] for label in IR_LABELS]
        return cls(
            name=data["name"],
            version=data.get("version", OBSERVATION_CONTRACT_VERSION),
            source=data.get("source", "unknown"),
            sensors=tuple(IRSensorCalibration(**item) for item in sensors),
        )

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sensors"] = {
            label: asdict(sensor) for label, sensor in zip(IR_LABELS, self.sensors)
        }
        return data

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def default_calibration_profile(name: str = "simulation") -> CalibrationProfile:
    # Matches the old raw/400 behavior while making the contract explicit.
    return CalibrationProfile(
        name=name,
        source="conservative-default",
        sensors=tuple(IRSensorCalibration(0.0, 400.0) for _ in IR_LABELS),
    )


class ObservationAdapter:
    """Applies exactly the same IR and vector preprocessing in sim and hardware."""

    def __init__(self, profile: CalibrationProfile):
        self.profile = profile

    def normalize_ir(self, raw_ir: np.ndarray | list[float]) -> np.ndarray:
        raw = np.asarray(raw_ir, dtype=np.float32)
        if raw.shape != (8,):
            raise ValueError(f"expected 8 IR readings, got {raw.shape}")
        result = np.empty(8, dtype=np.float32)
        for i, sensor in enumerate(self.profile.sensors):
            direction = float(sensor.polarity)
            denominator = direction * (sensor.near_obstacle - sensor.free_space)
            if denominator <= 0:
                raise ValueError(
                    f"invalid calibration polarity for {IR_LABELS[i]}: "
                    "near-obstacle must map above free-space"
                )
            normalized = direction * (raw[i] - sensor.free_space) / denominator
            result[i] = np.clip(normalized, 0.0, 1.0) ** sensor.exponent
        return result

    @staticmethod
    def policy_vector(blob: np.ndarray, normalized_ir: np.ndarray) -> np.ndarray:
        blob = np.asarray(blob, dtype=np.float32)
        ir = np.asarray(normalized_ir, dtype=np.float32)
        if blob.shape != (4,) or ir.shape != (8,):
            raise ValueError("policy observation must contain 4 blob and 8 IR values")
        return np.concatenate([blob, ir]).astype(np.float32)


@dataclass
class RewardConfig:
    food_reward: float = 100.0
    time_penalty_per_second: float = 0.5
    completion_bonus_scale: float = 50.0
    completion_bonus_max: float = 200.0
    collision_penalty: float = 0.0
    action_change_penalty: float = 0.02
    max_episode_seconds: float = 60.0


def transfer_reward(
    newly_collected: int,
    elapsed_delta_seconds: float,
    elapsed_seconds: float,
    completed: bool,
    collision: bool,
    action_change: float,
    config: RewardConfig,
) -> tuple[float, dict[str, float]]:
    collect = float(newly_collected) * config.food_reward
    time_cost = max(0.0, elapsed_delta_seconds) * config.time_penalty_per_second
    completion = 0.0
    if completed:
        completion = min(
            config.completion_bonus_max,
            config.completion_bonus_scale
            * config.max_episode_seconds
            / max(CONTROL_INTERVAL_SECONDS, elapsed_seconds),
        )
    collision_cost = config.collision_penalty if collision else 0.0
    change_cost = config.action_change_penalty * max(0.0, action_change)
    reward = collect + completion - time_cost - collision_cost - change_cost
    return reward, {
        "collect_reward": collect,
        "completion_bonus": completion,
        "time_penalty": time_cost,
        "collision_penalty": collision_cost,
        "action_change_penalty": change_cost,
    }


def blob_progress_potential(blob: np.ndarray | list[float]) -> float:
    """Bounded navigation potential derived only from deployable camera features.

    The potential rewards changes in alignment and apparent proximity. Its
    transition difference is used for shaping, so merely keeping a food blob
    visible does not produce reward.
    """
    x, _y, area, found = np.asarray(blob, dtype=np.float32)
    if found < 0.5:
        return 0.0
    alignment = 1.0 - min(1.0, abs(float(x) - 0.5) / 0.5)
    proximity = min(1.0, np.sqrt(max(0.0, float(area)) / 0.05))
    return float(0.6 * alignment + 0.4 * proximity)


@dataclass(frozen=True)
class SmoothingConfig:
    previous_weight: float = 0.65
    requested_weight: float = 0.35
    max_delta: float = 0.5

    def __post_init__(self) -> None:
        if not np.isclose(self.previous_weight + self.requested_weight, 1.0):
            raise ValueError("smoothing weights must sum to one")
        if self.max_delta <= 0:
            raise ValueError("max_delta must be positive")


@dataclass(frozen=True)
class SafetyConfig:
    warning_threshold: float = 0.78
    critical_threshold: float = 0.9
    reduced_speed: float = 0.35
    reverse_speed: float = 0.35
    turn_speed: float = 0.45
    front_indices: tuple[int, ...] = (2, 3, 4, 5, 7)


class PreActionSafetyFilter:
    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()

    def filter(self, requested: np.ndarray, normalized_ir: np.ndarray) -> tuple[np.ndarray, str | None]:
        action = np.clip(np.asarray(requested, dtype=np.float32), -1.0, 1.0)
        ir = np.asarray(normalized_ir, dtype=np.float32)
        front = ir[list(self.config.front_indices)]
        maximum = float(front.max(initial=0.0))
        if maximum >= self.config.critical_threshold:
            left_pressure = float(ir[[2, 7]].max(initial=0.0))
            right_pressure = float(ir[[3, 5]].max(initial=0.0))
            turn = self.config.turn_speed if left_pressure >= right_pressure else -self.config.turn_speed
            return np.array([-self.config.reverse_speed - turn, -self.config.reverse_speed + turn],
                            dtype=np.float32).clip(-1.0, 1.0), "emergency_reverse_turn"
        if maximum >= self.config.warning_threshold:
            scale = self.config.reduced_speed / max(self.config.reduced_speed, float(np.max(np.abs(action))))
            return action * min(1.0, scale), "speed_reduction"
        return action, None


class ActionExecutor:
    """Produces the action that must be stored in replay and sent to the motors."""

    def __init__(
        self,
        smoothing: SmoothingConfig | None = None,
        safety: PreActionSafetyFilter | None = None,
    ):
        self.smoothing = smoothing or SmoothingConfig()
        self.safety = safety or PreActionSafetyFilter()
        self.previous = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self.previous.fill(0.0)

    def execute(
        self,
        requested: np.ndarray,
        normalized_ir: np.ndarray,
        emergency: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        requested = np.clip(np.asarray(requested, dtype=np.float32), -1.0, 1.0)
        safe, safety_event = self.safety.filter(requested, normalized_ir)
        if emergency or safety_event == "emergency_reverse_turn":
            executed = safe
        else:
            smoothed = (
                self.smoothing.previous_weight * self.previous
                + self.smoothing.requested_weight * safe
            )
            delta = np.clip(
                smoothed - self.previous,
                -self.smoothing.max_delta,
                self.smoothing.max_delta,
            )
            executed = np.clip(self.previous + delta, -1.0, 1.0)
        action_change = float(np.abs(executed - self.previous).sum())
        self.previous = executed.astype(np.float32, copy=True)
        return self.previous.copy(), {
            "requested_action": requested.copy(),
            "executed_action": self.previous.copy(),
            "safety_override": safety_event,
            "action_change": action_change,
            "action_saturation": float(np.mean(np.abs(self.previous) >= 0.999)),
        }


@dataclass(frozen=True)
class CheckpointManifest:
    algorithm: str
    calibration_profile: str
    algorithm_config: dict[str, Any]
    image_size: int = 64
    phone_tilt: int = 100
    control_interval_seconds: float = CONTROL_INTERVAL_SECONDS
    observation_contract: str = OBSERVATION_CONTRACT_VERSION
    reward_contract: str = REWARD_CONTRACT_VERSION
    smoothing: dict[str, float] = field(default_factory=lambda: asdict(SmoothingConfig()))

    def validate(
        self,
        algorithm: str,
        calibration_profile: str,
        image_size: int,
        phone_tilt: int = 100,
    ) -> None:
        expected = {
            "algorithm": algorithm,
            "calibration_profile": calibration_profile,
            "image_size": image_size,
            "phone_tilt": phone_tilt,
            "control_interval_seconds": CONTROL_INTERVAL_SECONDS,
            "observation_contract": OBSERVATION_CONTRACT_VERSION,
            "reward_contract": REWARD_CONTRACT_VERSION,
            "smoothing": asdict(SmoothingConfig()),
        }
        actual = asdict(self)
        mismatches = [f"{key}: expected {value!r}, got {actual[key]!r}"
                      for key, value in expected.items() if actual[key] != value]
        if mismatches:
            raise ValueError("incompatible checkpoint manifest: " + "; ".join(mismatches))

    @classmethod
    def load(cls, path: str | Path) -> "CheckpointManifest":
        return cls(**json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")


class FixedPeriod:
    """Monotonic fixed-period scheduler used by hardware deployment."""

    def __init__(self, period_seconds: float = CONTROL_INTERVAL_SECONDS):
        self.period_seconds = period_seconds
        self.deadline = time.monotonic()

    def reset(self) -> None:
        self.deadline = time.monotonic()

    def sleep(self) -> float:
        self.deadline += self.period_seconds
        remaining = self.deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
            return 0.0
        return -remaining
