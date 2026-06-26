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
            3: [-3.125, 0.80, 0.05],
        }
        self.orientations = {3: [0.0, 0.0, 0.0]}

    def getObjectPosition(self, handle, _world):
        return list(self.positions[handle])

    def setObjectPosition(self, handle, position):
        self.positions[handle] = list(position)

    def setObjectOrientation(self, handle, orientation):
        self.orientations[handle] = list(orientation)

    def resetDynamicObject(self, _handle):
        pass

    def buildMatrix(self, pos, euler):
        a, b, g = euler
        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)
        cg, sg = math.cos(g), math.sin(g)
        return [
            cb * cg, -cb * sg, sb, 0.0,
            sa * sb * cg + ca * sg, -sa * sb * sg + ca * cg, -sa * cb, 0.0,
            -ca * sb * cg + sa * sg, ca * sb * sg + sa * cg, ca * cb, 0.0,
        ]

    def multiplyMatrices(self, m1, m2):
        result = [0.0] * 12
        for row in range(3):
            for col in range(4):
                result[row * 4 + col] = (
                    m1[row * 4 + 0] * m2[0 * 4 + col]
                    + m1[row * 4 + 1] * m2[1 * 4 + col]
                    + m1[row * 4 + 2] * m2[2 * 4 + col]
                )
        return result

    def getEulerAnglesFromMatrix(self, m):
        cb = math.sqrt(m[0] * m[0] + m[1] * m[1])
        a = math.atan2(-m[6], m[10])
        b = math.atan2(m[2], cb)
        g = math.atan2(-m[1], m[0])
        return [a, b, g]


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
    env.rob = type("Rob", (), {"_sim": _LayoutSim(), "_robobo": 3})()
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
    env._authored_robot_pose = (-3.125, 0.80, 0.05)
    env._authored_robot_orientation = (0.0, 0.0, 0.0)
    env._authored_robot_matrix = None
    env._authored_robot_heading = None
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
    assert controller.stage == 2


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


def test_curriculum_rejects_legacy_sparse_state(tmp_path):
    path = tmp_path / "curriculum_state.json"
    path.write_text(json.dumps({"version": 1, "stage": 0}))
    try:
        PushCurriculumController.load(path, PushCurriculumConfig())
    except ValueError as exc:
        assert "fresh run" in str(exc)
    else:
        raise AssertionError("legacy curriculum state should be rejected")


def test_no_curriculum_starts_in_full_stage():
    controller = PushCurriculumController(
        PushCurriculumConfig(enabled=False, start_stage=0)
    )
    assert controller.stage == 2


def test_approach_stage_restores_authored_object_poses():
    env = _layout_env(0)
    env.rob._sim.positions[1] = [-3.0, 0.5, 0.025]
    env.rob._sim.positions[2] = [-3.0, 1.2, 0.005]
    env._randomize_push_layout()
    assert env.rob._sim.positions[1] == [-3.50, 0.80, 0.025]
    assert env.rob._sim.positions[2] == [-2.90, 0.80, 0.005]
    assert env._push_layout_mode == "approach"
    assert not env._push_layout_randomized


def test_push_stage_jitters_goal_and_starts_robot_behind_block():
    env = _layout_env(1)
    env._randomize_push_layout()
    block = env.rob._sim.positions[1]
    goal = env.rob._sim.positions[2]
    assert block == [-3.50, 0.80, 0.025]
    assert math.dist(goal[:2], [-2.90, 0.80]) <= 0.20 + 1e-9
    assert env._valid_push_layout(tuple(block[:2]), tuple(goal[:2]))
    robot = env.rob._sim.positions[3]
    direction = (
        (goal[0] - block[0]) / math.dist(goal[:2], block[:2]),
        (goal[1] - block[1]) / math.dist(goal[:2], block[:2]),
    )
    expected_robot = [
        block[0] - env.config.push_standoff_distance * direction[0],
        block[1] - env.config.push_standoff_distance * direction[1],
    ]
    assert robot[:2] == expected_robot
    assert env._push_layout_mode == "push"


def test_sac_promotion_is_applied_before_vectorized_auto_reset(tmp_path):
    import numpy as np
    from tests.test_transfer_contract import _FakeRobobo
    from train_sac import RoboboSACEnv

    env = RoboboSACEnv(
        rob=_FakeRobobo(),
        max_episode_steps=1,
        no_record=True,
        domain_randomization=True,
        curriculum=True,
        curriculum_window=1,
        curriculum_min_stage_steps=0,
        curriculum_state_path=tmp_path / "curriculum_state.json",
    )
    env._base_env._push_geometry = lambda: (
        (-0.22, 0.0),
        (0.0, 0.0),
        (1.00, 0.0),
    )
    env._base_env._robot_block_contact = lambda: True
    try:
        env.reset()
        _, reward, terminated, truncated, info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        assert reward > 0.0
        assert terminated and not truncated
        assert info["curriculum_success"] == 1.0
        assert info["curriculum_promoted"] == 1.0
        assert env._config.push_curriculum_stage == 1
        assert env._domain_wrapper.enabled is False
        env.reset()
        assert env._config.push_curriculum_stage == 1
    finally:
        env.close()
