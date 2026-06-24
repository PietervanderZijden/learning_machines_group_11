import json
import math

from learning_machines.push_curriculum import (
    PushCurriculumConfig,
    PushCurriculumController,
)
from learning_machines.rl_robobo_compact_env import (
    RoboboCompactEnv,
    RoboboCompactEnvConfig,
)


class _LayoutSim:
    handle_world = -1

    def __init__(self):
        self.positions = {
            1: [-3.50, 0.80, 0.025],
            2: [-2.90, 0.80, 0.005],
        }

    def getObjectPosition(self, handle, _world):
        return list(self.positions[handle])

    def setObjectPosition(self, handle, position):
        self.positions[handle] = list(position)

    def resetDynamicObject(self, _handle):
        pass


def _layout_env(stage):
    env = RoboboCompactEnv.__new__(RoboboCompactEnv)
    env.config = RoboboCompactEnvConfig(
        task="push",
        push_curriculum_stage=stage,
        push_goal_jitter_radius=0.20,
        push_arena_radius=0.90,
        push_min_robot_distance=0.25,
        push_min_block_goal_distance=0.25,
    )
    env.rob = type("Rob", (), {"_sim": _LayoutSim()})()
    env._is_simulation = True
    env._red_block_handle = 1
    env._green_goal_handle = 2
    env._red_block_z = 0.025
    env._green_goal_z = 0.005
    env._arena_cx, env._arena_cy = env.config.arena_center
    env._initial_pos_x, env._initial_pos_y = env.config.arena_center
    env._push_layout_randomized = False
    env._push_layout_mode = "full"
    env._authored_red_block_pose = (-3.50, 0.80, 0.025)
    env._authored_green_goal_pose = (-2.90, 0.80, 0.005)
    return env


def test_curriculum_requires_steps_and_complete_success_window():
    controller = PushCurriculumController(
        PushCurriculumConfig(window=100, min_stage_steps=20_000)
    )
    for _ in range(19_999):
        controller.record_transition()
    for _ in range(100):
        assert controller.record_episode(True) is None
    controller.record_transition()
    event = controller.record_episode(True)
    assert event is not None
    assert controller.stage == 1
    assert controller.stage_steps == 0
    assert list(controller.recent_outcomes) == []


def test_curriculum_uses_last_window_and_stops_at_full_stage():
    controller = PushCurriculumController(
        PushCurriculumConfig(window=100, min_stage_steps=0)
    )
    for outcome in [False] * 20 + [True] * 80:
        event = controller.record_episode(outcome)
    assert event is not None
    assert controller.stage == 1
    for _ in range(100):
        event = controller.record_episode(True)
    assert event is not None
    assert controller.stage == 2
    for _ in range(100):
        assert controller.record_episode(True) is None


def test_curriculum_round_trip_restores_stage_and_ignores_start_stage(tmp_path):
    path = tmp_path / "curriculum_state.json"
    original = PushCurriculumController(
        PushCurriculumConfig(start_stage=1, window=2, min_stage_steps=0)
    )
    original.record_transition()
    original.record_episode(True)
    original.record_episode(True)
    original.save(path)
    restored = PushCurriculumController.load(
        path, PushCurriculumConfig(start_stage=0, window=2, min_stage_steps=0)
    )
    assert restored.stage == 2
    assert restored.transition_count == 1
    assert json.loads(path.read_text())["promotion_history"]


def test_no_curriculum_starts_in_full_stage():
    controller = PushCurriculumController(
        PushCurriculumConfig(enabled=False, start_stage=0)
    )
    assert controller.stage == 2


def test_fixed_layout_restores_authored_poses_exactly():
    env = _layout_env(0)
    env.rob._sim.positions[1] = [-3.0, 0.5, 0.025]
    env.rob._sim.positions[2] = [-3.0, 1.2, 0.005]
    env._randomize_push_layout()
    assert env.rob._sim.positions[1] == [-3.50, 0.80, 0.025]
    assert env.rob._sim.positions[2] == [-2.90, 0.80, 0.005]
    assert env._push_layout_mode == "fixed"
    assert not env._push_layout_randomized


def test_goal_jitter_keeps_block_fixed_and_goal_within_radius():
    env = _layout_env(1)
    env._randomize_push_layout()
    block = env.rob._sim.positions[1]
    goal = env.rob._sim.positions[2]
    assert block == [-3.50, 0.80, 0.025]
    assert math.dist(goal[:2], [-2.90, 0.80]) <= 0.20 + 1e-9
    assert env._valid_push_layout(tuple(block[:2]), tuple(goal[:2]))
    assert env._push_layout_mode == "goal_jitter"
