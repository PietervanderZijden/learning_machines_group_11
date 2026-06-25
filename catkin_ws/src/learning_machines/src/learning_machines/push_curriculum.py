"""Shared phased dense-reward push curriculum state."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path


PUSH_REWARD_CONTRACT = "robobo-push-phased-dense-v1"
PUSH_STAGE_NAMES = ("approach", "push", "full")
PUSH_CURRICULUM_STATE_VERSION = 2


@dataclass(frozen=True)
class PushCurriculumConfig:
    enabled: bool = True
    start_stage: int = 0
    success_threshold: float = 0.80
    window: int = 100
    min_stage_steps: int = 20_000
    goal_jitter_radius: float = 0.20

    def __post_init__(self) -> None:
        if self.start_stage not in range(len(PUSH_STAGE_NAMES)):
            raise ValueError("curriculum start stage must be 0, 1, or 2")
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("curriculum success threshold must be in [0, 1]")
        if self.window <= 0:
            raise ValueError("curriculum window must be positive")
        if self.min_stage_steps < 0:
            raise ValueError("curriculum minimum stage steps must be non-negative")
        if self.goal_jitter_radius < 0.0:
            raise ValueError("curriculum goal jitter radius must be non-negative")


@dataclass
class PushCurriculumController:
    config: PushCurriculumConfig
    stage: int = field(init=False)
    stage_start_step: int = 0
    transition_count: int = 0
    recent_outcomes: deque[float] = field(init=False)
    promotion_history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stage = self.config.start_stage if self.config.enabled else 2
        self.recent_outcomes = deque(maxlen=self.config.window)

    @property
    def stage_name(self) -> str:
        return PUSH_STAGE_NAMES[self.stage]

    @property
    def stage_steps(self) -> int:
        return self.transition_count - self.stage_start_step

    @property
    def rolling_success(self) -> float:
        return (
            sum(self.recent_outcomes) / len(self.recent_outcomes)
            if self.recent_outcomes
            else 0.0
        )

    def record_transition(self) -> None:
        self.transition_count += 1

    def record_episode(self, success: bool) -> dict | None:
        self.recent_outcomes.append(float(bool(success)))
        if (
            not self.config.enabled
            or self.stage >= len(PUSH_STAGE_NAMES) - 1
            or self.stage_steps < self.config.min_stage_steps
            or len(self.recent_outcomes) < self.config.window
            or self.rolling_success < self.config.success_threshold
        ):
            return None
        event = {
            "from_stage": self.stage,
            "from_stage_name": self.stage_name,
            "to_stage": self.stage + 1,
            "to_stage_name": PUSH_STAGE_NAMES[self.stage + 1],
            "transition_count": self.transition_count,
            "stage_steps": self.stage_steps,
            "success_rate": self.rolling_success,
            "episodes": len(self.recent_outcomes),
        }
        self.stage += 1
        self.stage_start_step = self.transition_count
        self.recent_outcomes.clear()
        self.promotion_history.append(event)
        return event

    def metrics(self) -> dict[str, float | str]:
        return {
            "stage": self.stage,
            "stage_name": self.stage_name,
            "stage_steps": self.stage_steps,
            "stage_episodes": len(self.recent_outcomes),
            "rolling_success": self.rolling_success,
        }

    def to_dict(self) -> dict:
        return {
            "version": PUSH_CURRICULUM_STATE_VERSION,
            "config": asdict(self.config),
            "stage": self.stage,
            "stage_name": self.stage_name,
            "stage_start_step": self.stage_start_step,
            "transition_count": self.transition_count,
            "recent_outcomes": list(self.recent_outcomes),
            "promotion_history": self.promotion_history,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        temporary.replace(path)

    @classmethod
    def load(
        cls, path: str | Path, config: PushCurriculumConfig
    ) -> "PushCurriculumController":
        data = json.loads(Path(path).read_text())
        version = int(data.get("version", 1))
        if version != PUSH_CURRICULUM_STATE_VERSION:
            raise ValueError(
                "persisted push curriculum uses an incompatible reward/stage "
                f"contract (version {version}); start a fresh run"
            )
        controller = cls(config)
        controller.stage = int(data["stage"])
        if controller.stage not in range(len(PUSH_STAGE_NAMES)):
            raise ValueError("persisted curriculum stage must be 0, 1, or 2")
        controller.stage_start_step = int(data["stage_start_step"])
        controller.transition_count = int(data["transition_count"])
        controller.recent_outcomes.extend(
            float(value) for value in data.get("recent_outcomes", ())
        )
        controller.promotion_history = list(data.get("promotion_history", ()))
        return controller
