import json
import time
import zipfile
from types import SimpleNamespace

import gymnasium as gym
import cv2
import numpy as np
from gymnasium import spaces

from learning_machines.domain_randomization import (
    DomainRandomizationWrapper,
    RandomizationRanges,
)
from learning_machines.transfer import (
    ActionExecutor,
    CalibrationProfile,
    CheckpointManifest,
    IRSensorCalibration,
    ObservationAdapter,
    PreActionSafetyFilter,
    RewardConfig,
    SafetyConfig,
    SmoothingConfig,
    blob_progress_potential,
    transfer_reward,
)
from learning_machines.rl_robobo_compact_env import (
    RoboboCompactEnv,
    RoboboCompactEnvConfig,
)
from learning_machines.sac import StabilizedSAC
from robobo_interface.simulation import SimulationRobobo
from train_sac_push import (
    find_sac_resume_checkpoint,
    matching_sac_replay_buffer,
    promote_sac_checkpoint,
    validate_sac_manifest_recovery,
)


def test_calibration_polarity_clipping_and_serialization(tmp_path):
    sensors = (
        IRSensorCalibration(10, 110, 1, 1.0),
        IRSensorCalibration(110, 10, -1, 2.0),
        *(IRSensorCalibration(0, 100) for _ in range(6)),
    )
    profile = CalibrationProfile("test", tuple(sensors), source="unit")
    adapter = ObservationAdapter(profile)
    normalized = adapter.normalize_ir([60, 60, -10, 50, 100, 110, 0, 25])
    np.testing.assert_allclose(
        normalized, [0.5, 0.25, 0.0, 0.5, 1.0, 1.0, 0.0, 0.25]
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    assert CalibrationProfile.load(path) == profile


def test_equivalent_profiles_produce_identical_policy_vectors():
    profile = CalibrationProfile(
        "shared", tuple(IRSensorCalibration(5, 105) for _ in range(8))
    )
    sim = ObservationAdapter(profile)
    hardware = ObservationAdapter(profile)
    raw = np.arange(8, dtype=np.float32) * 10 + 5
    blob = np.array([0.2, 0.3, 0.1, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(
        sim.policy_vector(blob, sim.normalize_ir(raw)),
        hardware.policy_vector(blob, hardware.normalize_ir(raw)),
    )


def test_reward_is_elapsed_time_based_and_400ms_equivalent():
    cfg = RewardConfig(action_change_penalty=0.0)
    five_steps = sum(
        transfer_reward(0, 0.4, (i + 1) * 0.4, False, False, 0, cfg)[0]
        for i in range(5)
    )
    two_seconds = transfer_reward(0, 2.0, 2.0, False, False, 0, cfg)[0]
    assert five_steps == two_seconds == -1.0


def test_completion_bonus_uses_elapsed_seconds():
    cfg = RewardConfig(action_change_penalty=0.0, max_episode_seconds=60.0)
    reward, parts = transfer_reward(1, 0.4, 30.0, True, False, 0.0, cfg)
    assert parts["completion_bonus"] == 100.0
    assert reward == 199.8


def test_completion_bonus_is_bounded():
    cfg = RewardConfig(action_change_penalty=0.0, completion_bonus_max=200.0)
    _reward, parts = transfer_reward(1, 0.4, 0.4, True, False, 0.0, cfg)
    assert parts["completion_bonus"] == 200.0


def test_blob_potential_rewards_progress_but_not_stationary_visibility():
    centered_small = np.array([0.5, 0.5, 0.002, 1.0], dtype=np.float32)
    centered_large = np.array([0.5, 0.5, 0.02, 1.0], dtype=np.float32)
    off_center = np.array([0.0, 0.5, 0.002, 1.0], dtype=np.float32)
    assert blob_progress_potential(centered_large) > blob_progress_potential(
        centered_small
    )
    assert blob_progress_potential(centered_small) > blob_progress_potential(
        off_center
    )
    assert blob_progress_potential([0.5, 0.5, 0.1, 0.0]) == 0.0


def test_action_smoothing_rate_limit_and_pre_action_safety():
    executor = ActionExecutor(
        smoothing=SmoothingConfig(0.65, 0.35, 0.5),
        safety=PreActionSafetyFilter(SafetyConfig(
            warning_threshold=0.7, critical_threshold=0.9
        )),
    )
    executed, info = executor.execute(np.ones(2), np.zeros(8))
    np.testing.assert_allclose(executed, [0.35, 0.35])
    assert info["safety_override"] is None

    ir = np.zeros(8)
    ir[4] = 1.0
    emergency, info = executor.execute(np.ones(2), ir)
    assert info["safety_override"] == "emergency_reverse_turn"
    assert np.any(emergency < 0)


def test_safety_allows_slow_approach_to_centered_food():
    executor = ActionExecutor(
        smoothing=SmoothingConfig(0.0, 1.0, 1.0),
        safety=PreActionSafetyFilter(),
    )
    ir = np.zeros(8, dtype=np.float32)
    ir[4] = 1.0
    blob = np.array([0.5, 0.7, 0.02, 1.0], dtype=np.float32)

    executed, info = executor.execute(np.ones(2), ir, blob=blob)

    assert info["safety_override"] == "food_approach_speed_reduction"
    np.testing.assert_allclose(executed, [0.25, 0.25])


def test_safety_still_avoids_side_obstacle_with_visible_food():
    executor = ActionExecutor(
        smoothing=SmoothingConfig(0.0, 1.0, 1.0),
        safety=PreActionSafetyFilter(),
    )
    ir = np.zeros(8, dtype=np.float32)
    ir[4] = 0.95
    ir[2] = 1.0
    blob = np.array([0.5, 0.7, 0.02, 1.0], dtype=np.float32)

    executed, info = executor.execute(np.ones(2), ir, blob=blob)

    assert info["safety_override"] == "emergency_reverse_turn"
    assert np.any(executed < 0)


class _FakeEnv(gym.Env):
    observation_space = spaces.Dict({
        "ir": spaces.Box(0, 1, (8,), dtype=np.float32),
        "blob": spaces.Box(0, 1, (4,), dtype=np.float32),
        "image": spaces.Box(0, 255, (3, 8, 8), dtype=np.uint8),
    })
    action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {"input_action": np.asarray(action)}

    @staticmethod
    def _obs():
        return {
            "ir": np.full(8, 0.5, dtype=np.float32),
            "blob": np.array([0.5, 0.5, 0.1, 1.0], dtype=np.float32),
            "image": np.full((3, 8, 8), 128, dtype=np.uint8),
        }


def test_randomization_persistent_per_episode_changes_after_reset_and_augments_reset():
    ranges = RandomizationRanges(
        ir_noise_std=0.02,
        spike_probability=0.0,
        dropout_probability=0.0,
    )
    env = DomainRandomizationWrapper(_FakeEnv(), ranges=ranges, seed=7)
    reset_obs, reset_info = env.reset()
    first = json.dumps(reset_info["domain_randomization"], sort_keys=True)
    assert not np.array_equal(reset_obs["ir"], np.full(8, 0.5, dtype=np.float32))

    obs1, _, _, _, info1 = env.step(np.ones(2))
    obs2, _, _, _, info2 = env.step(np.ones(2))
    assert json.dumps(info1["domain_randomization"], sort_keys=True) == first
    assert json.dumps(info2["domain_randomization"], sort_keys=True) == first
    assert not np.array_equal(obs1["ir"], obs2["ir"])

    _, second_info = env.reset()
    second = json.dumps(second_info["domain_randomization"], sort_keys=True)
    assert second != first


def test_manifest_rejects_incompatible_checkpoint(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = CheckpointManifest("sac", "hardware", {"obs_dim": 12})
    manifest.save(path)
    loaded = CheckpointManifest.load(path)
    loaded.validate("sac", "hardware", 64)
    loaded.validate("sac", None, 64)
    try:
        loaded.validate("dreamerv3", "hardware", 64)
    except ValueError as exc:
        assert "algorithm" in str(exc)
    else:
        raise AssertionError("incompatible checkpoint was accepted")

    old_timing = CheckpointManifest(
        "sac",
        "hardware",
        {"obs_dim": 12},
        control_interval_seconds=0.2,
        reward_contract="robobo-reward-v3",
    )
    try:
        old_timing.validate("sac", "hardware", 64)
    except ValueError as exc:
        assert "control_interval_seconds" in str(exc)
        assert "reward_contract" in str(exc)
    else:
        raise AssertionError("200 ms checkpoint manifest was accepted")


class _FakeRobobo:
    def __init__(self):
        self._used_pids = set()
        self.commands = []
        self.tilt = 100

    def set_wheel_speeds(self, left, right, duration_s=0.4):
        self.commands.append((left, right, duration_s))

    def sleep(self, seconds):
        pass

    def read_irs(self):
        return [0.0] * 8

    def read_image_front(self):
        return np.zeros((16, 16, 3), dtype=np.uint8)

    def get_nr_food_collected(self):
        return 0

    def set_phone_tilt(self, tilt, speed):
        self.tilt = tilt

    def read_phone_tilt(self):
        return self.tilt


def test_compact_env_reports_executed_action_and_elapsed_metrics():
    rob = _FakeRobobo()
    env = RoboboCompactEnv(rob=rob, config=RoboboCompactEnvConfig(
        detect_blob_from_camera=False,
        randomize_food_positions=False,
        reset_settle_time=0.0,
    ))
    env.reset()
    _, reward, _, _, info = env.step(np.ones(2, dtype=np.float32))
    np.testing.assert_allclose(info["executed_action"], [0.35, 0.35])
    assert info["elapsed_seconds"] == 0.4
    assert info["food_per_minute"] == 0.0
    assert reward < -0.1
    assert env.max_episode_seconds == 60.0


def test_episode_step_limit_is_converted_to_seconds():
    env = RoboboCompactEnv(rob=_FakeRobobo(), config=RoboboCompactEnvConfig(
        max_episode_steps=10,
        detect_blob_from_camera=False,
        randomize_food_positions=False,
    ))
    assert env.max_episode_seconds == 4.0


def test_400ms_episode_truncates_on_exact_configured_step_count():
    env = RoboboCompactEnv(rob=_FakeRobobo(), config=RoboboCompactEnvConfig(
        max_episode_steps=10,
        detect_blob_from_camera=False,
        randomize_food_positions=False,
        reset_settle_time=0.0,
    ))
    env.reset()
    for step in range(10):
        _, _, _, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        assert truncated is (step == 9)
    assert np.isclose(info["elapsed_seconds"], 4.0)


def test_reset_points_camera_down_before_first_observation():
    rob = _FakeRobobo()
    env = RoboboCompactEnv(rob=rob, config=RoboboCompactEnvConfig(
        phone_tilt=105,
        reset_settle_time=0.0,
        randomize_food_positions=False,
    ))
    _, info = env.reset()
    assert rob.tilt == 105
    assert info["phone_tilt"] == 105


def test_hardware_reset_leaves_camera_untouched_when_tilt_is_disabled():
    rob = _FakeRobobo()
    rob.tilt = 73
    env = RoboboCompactEnv(rob=rob, config=RoboboCompactEnvConfig(
        phone_tilt=110,
        initialize_phone_tilt=False,
        reset_settle_time=0.0,
        randomize_food_positions=False,
    ))
    _, info = env.reset()

    assert rob.tilt == 73
    assert info["phone_tilt"] == 73


def test_hardware_reset_uses_bounded_async_camera_tilt_api():
    class AsyncTiltRobobo(_FakeRobobo):
        def __init__(self):
            super().__init__()
            self.async_tilts = []

        def set_phone_tilt(self, tilt, speed):
            self.async_tilts.append((tilt, speed))
            self.tilt = tilt
            self._used_pids.add(9)
            return 9

    rob = AsyncTiltRobobo()
    env = RoboboCompactEnv(rob=rob, config=RoboboCompactEnvConfig(
        phone_tilt=100,
        phone_tilt_speed=10,
        reset_settle_time=0.0,
        randomize_food_positions=False,
    ))
    env.reset()

    assert rob.async_tilts == [(100, 10)]
    assert not rob._used_pids


def test_physical_camera_tilt_accepts_official_limits():
    for tilt in (5, 110):
        RoboboCompactEnv(rob=_FakeRobobo(), config=RoboboCompactEnvConfig(
            phone_tilt=tilt,
            initialize_phone_tilt=False,
            detect_blob_from_camera=False,
            randomize_food_positions=False,
        ))


def test_invalid_hardware_tilt_feedback_is_treated_as_unavailable():
    class StaleTiltRobobo(_FakeRobobo):
        def set_phone_tilt(self, tilt, speed):
            self.commanded_tilt = tilt

        def read_phone_tilt(self):
            return 0

    rob = StaleTiltRobobo()
    env = RoboboCompactEnv(rob=rob, config=RoboboCompactEnvConfig(
        phone_tilt=100,
        reset_settle_time=0.0,
        randomize_food_positions=False,
    ))
    _, info = env.reset()
    assert rob.commanded_tilt == 100
    assert info["phone_tilt"] is None


def test_hardware_wheel_speed_cap_is_applied_by_environment():
    rob = _FakeRobobo()
    env = RoboboCompactEnv(rob=rob, config=RoboboCompactEnvConfig(
        max_wheel_speed=70,
        action_smoothing=False,
        detect_blob_from_camera=False,
        randomize_food_positions=False,
        reset_settle_time=0.0,
    ))
    env.reset()
    env.step(np.ones(2, dtype=np.float32))
    env.step(np.ones(2, dtype=np.float32))
    left, right, duration = rob.commands[-1]
    assert left == right == 70
    assert duration == 0.4


def test_hardware_uses_main_branch_blocking_move_api():
    class BlockingHardware(_FakeRobobo):
        def __init__(self):
            super().__init__()
            self.blocking_commands = []

        def move_blocking(self, left, right, millis):
            self.blocking_commands.append((left, right, millis))

    rob = BlockingHardware()
    env = RoboboCompactEnv(rob=rob, config=RoboboCompactEnvConfig(
        max_wheel_speed=70,
        action_smoothing=False,
        detect_blob_from_camera=False,
        randomize_food_positions=False,
        reset_settle_time=0.0,
    ))
    env.reset()
    env.step(np.ones(2, dtype=np.float32))
    env.step(np.ones(2, dtype=np.float32))

    assert rob.blocking_commands == [(35, 35, 400), (70, 70, 400)]
    assert rob.commands == [(0, 0, 0.4)]


def test_zero_reset_settle_does_not_advance_simulation():
    class SimRob(_FakeRobobo):
        def __init__(self):
            super().__init__()
            self._sim = object()
            self.steps = 0

        def step_simulation(self, steps):
            self.steps += steps

    rob = SimRob()
    env = RoboboCompactEnv.__new__(RoboboCompactEnv)
    env.rob = rob
    env._is_simulation = True
    env._settle(0.0)
    assert rob.steps == 0


def test_blob_tracker_keeps_target_instead_of_switching_to_largest():
    env = RoboboCompactEnv(
        rob=_FakeRobobo(),
        config=RoboboCompactEnvConfig(
            detect_blob_from_camera=True,
            randomize_food_positions=False,
            reset_settle_time=0.0,
        ),
    )
    env._reset_blob_tracker()
    first = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(first, (25, 60), 8, (0, 255, 0), -1)
    cv2.circle(first, (75, 60), 5, (0, 255, 0), -1)
    selected = env._detect_blob(first)
    assert selected[0] < 0.5

    second = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(second, (28, 60), 6, (0, 255, 0), -1)
    cv2.circle(second, (75, 60), 12, (0, 255, 0), -1)
    selected = env._detect_blob(second)
    assert selected[0] < 0.5
    assert env._blob_target_switches == 0


def test_push_red_green_blob_detection_uses_separate_masks():
    env = RoboboCompactEnv(
        rob=_FakeRobobo(),
        config=RoboboCompactEnvConfig(
            task="push",
            detect_blob_from_camera=True,
            randomize_food_positions=False,
            reset_settle_time=0.0,
        ),
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (25, 50), 10, (0, 0, 255), -1)
    cv2.circle(image, (75, 50), 12, (0, 255, 0), -1)

    red = env._detect_color_blob(
        image,
        (
            (env.config.push_red_hsv_low_1, env.config.push_red_hsv_high_1),
            (env.config.push_red_hsv_low_2, env.config.push_red_hsv_high_2),
        ),
    )
    green = env._detect_color_blob(
        image,
        ((env.config.push_green_hsv_low, env.config.push_green_hsv_high),),
    )

    assert red[3] == 1.0
    assert green[3] == 1.0
    assert red[0] < 0.5
    assert green[0] > 0.5


def _dense_push_env(geometry, **config_overrides):
    env = RoboboCompactEnv.__new__(RoboboCompactEnv)
    env.config = RoboboCompactEnvConfig(
        task="push",
        push_discount=0.9801,
        **config_overrides,
    )
    env._push_layout_randomized = False
    env._push_layout_mode = "full"
    env._contact_acquired = env.config.push_curriculum_stage == 1
    env._previous_block_goal_distance = env._geometry_block_goal_distance(geometry)
    env._previous_push_potential = env._push_potential(
        geometry, contact_acquired=env._contact_acquired
    )
    env._push_geometry = lambda: geometry
    env._robot_block_contact = lambda: False
    return env


def _visible_push_obs():
    return {
        "red_block": np.array([0.4, 0.5, 0.03, 1.0], dtype=np.float32),
        "green_goal": np.array([0.6, 0.5, 0.03, 1.0], dtype=np.float32),
    }


def test_push_contact_uses_respondable_robot_shapes():
    class ContactSim:
        handle_world = -1
        object_shape_type = 0
        shapeintparam_respondable = 1

        def getObjectPosition(self, handle, _world):
            return [float(handle), 0.0, 0.0]

        def getObjectsInTree(self, _root, _kind, _options):
            return [10, 11]

        def getObjectInt32Param(self, handle, _parameter):
            return int(handle == 11)

        def checkCollision(self, robot_shape, block):
            return int(robot_shape == 11 and block == 1)

    env = RoboboCompactEnv.__new__(RoboboCompactEnv)
    env.rob = type("Rob", (), {"_sim": ContactSim(), "_robobo": 3})()
    env._is_simulation = True
    env._red_block_handle = 1
    env._green_goal_handle = 2
    env._robot_respondable_handles = ()

    assert env._robot_block_contact()
    assert env._robot_respondable_handles == (11,)


def test_dense_push_rewards_approach_before_contact():
    start = ((-0.50, 0.0), (0.0, 0.0), (1.0, 0.0))
    approach = ((-0.30, 0.0), (0.0, 0.0), (1.0, 0.0))

    env = _dense_push_env(start, push_curriculum_stage=2)
    env._push_geometry = lambda: approach
    reward, done, info = env._push_reward(
        _visible_push_obs(), 0.4, False, 0.0
    )

    assert not done
    assert reward > 0.0
    assert info["push_phase"] == 0.0
    assert info["potential_shaping"] > info["time_cost"]
    assert info["goal_completion_bonus"] == 0.0


def test_dense_push_contact_latches_and_switches_to_goal_progress():
    start = ((-0.50, 0.0), (0.0, 0.0), (1.0, 0.0))
    contact = ((-0.20, 0.0), (0.0, 0.0), (1.0, 0.0))
    pushed = ((0.02, 0.0), (0.20, 0.0), (1.0, 0.0))
    env = _dense_push_env(start, push_curriculum_stage=2)
    env._push_geometry = lambda: contact
    env._robot_block_contact = lambda: True

    contact_reward, contact_done, contact_info = env._push_reward(
        _visible_push_obs(), 0.4, False, 0.0
    )
    env._push_geometry = lambda: pushed
    env._robot_block_contact = lambda: False
    push_reward, push_done, push_info = env._push_reward(
        _visible_push_obs(), 0.4, False, 0.0
    )

    assert not contact_done and not push_done
    assert contact_reward > 0.0
    assert contact_info["contact_bonus"] == 1.0
    assert contact_info["contact_acquired"] == 1.0
    assert push_info["contact_acquired"] == 1.0
    assert push_info["robot_block_contact"] == 0.0
    assert push_reward > 0.0


def test_approach_curriculum_terminates_on_first_contact():
    start = ((-0.50, 0.0), (0.0, 0.0), (1.0, 0.0))
    contact = ((-0.20, 0.0), (0.0, 0.0), (1.0, 0.0))
    env = _dense_push_env(start, push_curriculum_stage=0)
    env._push_geometry = lambda: contact
    env._robot_block_contact = lambda: True

    reward, terminated, info = env._push_reward(
        _visible_push_obs(), 0.4, False, 0.0
    )

    assert terminated
    assert reward > 0.0
    assert info["curriculum_success"] == 1.0
    assert info["push_success"] == 0.0
    assert info["approach_completion_bonus"] == 5.0


def test_dense_push_goal_success_has_dominant_terminal_bonus():
    start = ((0.58, 0.0), (0.80, 0.0), (1.0, 0.0))
    success = ((0.70, 0.0), (0.83, 0.0), (1.0, 0.0))
    env = _dense_push_env(
        start, push_success_distance=0.18, push_curriculum_stage=1
    )
    env._push_geometry = lambda: success

    reward, terminated, info = env._push_reward(
        _visible_push_obs(), 0.4, False, 0.0
    )

    assert terminated
    assert info["push_success"] == 1.0
    assert info["curriculum_success"] == 1.0
    assert info["goal_completion_bonus"] == 15.0
    assert reward > 8.0


def test_dense_push_tiny_progress_cannot_be_repeatedly_farmed():
    start = ((-0.20, 0.0), (0.0, 0.0), (1.0, 0.0))
    tiny_push = ((-0.19, 0.0), (0.01, 0.0), (1.0, 0.0))
    env = _dense_push_env(start, push_curriculum_stage=1)
    env._push_geometry = lambda: tiny_push
    first, _, _ = env._push_reward(_visible_push_obs(), 0.4, False, 0.0)
    second, _, _ = env._push_reward(_visible_push_obs(), 0.4, False, 0.0)

    assert first > second
    assert second < 0.0


def test_dense_push_faster_completion_scores_higher():
    terminal_reward = 10.0
    step_cost = 0.02
    gamma = 0.9801
    fast = -step_cost + gamma * terminal_reward
    slow = sum(gamma**step * -step_cost for step in range(10))
    slow += gamma**10 * terminal_reward
    assert fast > slow


def test_push_success_takes_precedence_on_final_allowed_step():
    assert not RoboboCompactEnv._resolve_push_truncation(True, True)
    assert RoboboCompactEnv._resolve_push_truncation(False, True)


def test_push_layout_randomization_moves_block_and_goal_with_constraints():
    class FakePushSim:
        handle_world = -1

        def __init__(self):
            self.positions = {
                1: [-3.2, 0.6, 0.025],
                2: [-2.8, 1.0, 0.005],
            }
            self.objects = {
                "/Food": 1,
                "Food": 1,
                "/Base": 2,
                "Base": 2,
            }
            self.reset_handles = []

        def getObject(self, name):
            return self.objects[name]

        def getObjectPosition(self, handle, _world):
            return list(self.positions[handle])

        def setObjectPosition(self, handle, position):
            self.positions[handle] = list(position)

        def resetDynamicObject(self, handle):
            self.reset_handles.append(handle)

    class FakePushRob:
        def __init__(self):
            self._sim = FakePushSim()

    env = RoboboCompactEnv.__new__(RoboboCompactEnv)
    env.config = RoboboCompactEnvConfig(
        task="push",
        randomize_push_layout=True,
        push_arena_radius=0.8,
        push_min_robot_distance=0.25,
        push_min_block_goal_distance=0.35,
        push_max_block_goal_distance=1.1,
        push_success_distance=0.12,
    )
    env.rob = FakePushRob()
    env._is_simulation = True
    env._red_block_handle = None
    env._green_goal_handle = None
    env._red_block_z = 0.0
    env._green_goal_z = 0.0
    env._arena_cx, env._arena_cy = env.config.arena_center
    env._initial_pos_x = env._arena_cx
    env._initial_pos_y = env._arena_cy

    env._ensure_push_handles()

    assert env._red_block_handle == 1
    assert env._green_goal_handle == 2
    assert env._red_block_z == 0.025
    assert env._green_goal_z == 0.005

    env._randomize_push_layout()

    block = env.rob._sim.positions[1]
    goal = env.rob._sim.positions[2]
    block_xy = (block[0], block[1])
    goal_xy = (goal[0], goal[1])
    assert env._push_layout_randomized is True
    assert block[2] == 0.025
    assert goal[2] == 0.005
    assert 0.25 <= env._xy_distance(block_xy, (env._initial_pos_x, env._initial_pos_y))
    assert 0.25 <= env._xy_distance(goal_xy, (env._initial_pos_x, env._initial_pos_y))
    block_goal_distance = env._xy_distance(block_xy, goal_xy)
    assert 0.35 <= block_goal_distance <= 1.1
    assert env.rob._sim.reset_handles == [1, 2]


def test_push_layout_randomization_can_be_disabled():
    class FakePushSim:
        handle_world = -1

        def __init__(self):
            self.positions = {
                1: [-3.2, 0.6, 0.025],
                2: [-2.8, 1.0, 0.005],
            }

        def getObjectPosition(self, handle, _world):
            return list(self.positions[handle])

        def setObjectPosition(self, handle, position):
            self.positions[handle] = list(position)

    class FakePushRob:
        def __init__(self):
            self._sim = FakePushSim()

    env = RoboboCompactEnv.__new__(RoboboCompactEnv)
    env.config = RoboboCompactEnvConfig(task="push", randomize_push_layout=False)
    env.rob = FakePushRob()
    env._is_simulation = True
    env._red_block_handle = 1
    env._green_goal_handle = 2
    before = dict(env.rob._sim.positions)

    env._randomize_push_layout()

    assert env._push_layout_randomized is False
    assert env.rob._sim.positions == before


def test_sac_push_observation_is_18_values_and_old_manifest_is_rejected(tmp_path):
    from train_sac_push import RoboboSACEnv
    from learning_machines.transfer import CheckpointManifest

    env = RoboboSACEnv(
        rob=_FakeRobobo(),
        no_record=True,
        randomize_food_positions=False,
        curriculum=False,
        calibration_path=None,
    )
    obs = env._flatten_obs({
        "red_block": np.zeros(4, dtype=np.float32),
        "green_goal": np.ones(4, dtype=np.float32),
        "ir": np.full(8, 0.5, dtype=np.float32),
    })
    assert env.observation_space.shape == (18,)
    assert obs.shape == (18,)

    path = tmp_path / "manifest.json"
    CheckpointManifest(
        "sac",
        "simulation",
        {"observation_dim": 14},
    ).save(path)
    loaded = CheckpointManifest.load(path)
    assert loaded.algorithm_config["observation_dim"] != 18


def test_sac_recording_metadata_uses_base_env_with_domain_randomization(tmp_path):
    from train_sac_push import RoboboSACEnv

    env = RoboboSACEnv(
        rob=_FakeRobobo(),
        max_episode_steps=1,
        domain_randomization=True,
        curriculum=False,
        randomize_food_positions=False,
        record_dir=tmp_path,
        no_record=False,
        image_size=16,
    )
    geometry = ((-0.22, 0.0), (0.0, 0.0), (1.0, 0.0))
    env._base_env._push_geometry = lambda: geometry
    try:
        env.reset()
        env.step(np.zeros(2, dtype=np.float32))
    finally:
        env.close()

    episodes = sorted((tmp_path / "episodes").glob("ep_*.npz"))
    assert len(episodes) == 1
    episode = np.load(episodes[0])
    assert episode["calibration_profile"].item() == "simulation"
    assert int(episode["phone_tilt"]) == env._base_env.config.phone_tilt
    assert episode["reward_contract"].item() == "robobo-push-phased-dense-v1"


def test_sac_resume_selects_highest_embedded_timestep(tmp_path):
    def write_checkpoint(name, steps):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("data", json.dumps({"num_timesteps": steps}))
        return path

    write_checkpoint("sac_latest.zip", 149)
    expected = write_checkpoint("sac_420000_steps.zip", 420000)
    replay = tmp_path / "sac_replay_buffer_420000_steps.pkl"
    replay.write_bytes(b"buffer")
    selected, steps = find_sac_resume_checkpoint(tmp_path)
    assert selected == expected
    assert steps == 420000
    assert matching_sac_replay_buffer(tmp_path, selected) == replay
    promote_sac_checkpoint(tmp_path, selected, replay)
    assert find_sac_resume_checkpoint(tmp_path)[1] == 420000
    assert (tmp_path / "replay_buffer.pkl").read_bytes() == b"buffer"


def _sac_recovery_args(**overrides):
    values = {
        "curriculum": False,
        "curriculum_success_threshold": 0.8,
        "curriculum_window": 100,
        "curriculum_min_stage_steps": 20_000,
        "curriculum_goal_jitter_radius": 0.2,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 1_000_000,
        "learning_starts": 2_000,
        "entropy_coefficient": 0.01,
        "max_grad_norm": 10.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_sac_missing_manifest_can_be_recovered_from_checkpoint_and_curriculum(tmp_path):
    checkpoint = tmp_path / "sac_220000_steps.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps({
            "num_timesteps": 220_000,
            "observation_space": {"_shape": [18]},
            "action_space": {"_shape": [2]},
            "gamma": 0.9801,
            "learning_rate": 3e-4,
            "batch_size": 256,
            "buffer_size": 1_000_000,
            "learning_starts": 2_000,
            "ent_coef": 0.01,
            "max_grad_norm": 10.0,
        }))
    state_path = tmp_path / "curriculum_state.json"
    state_path.write_text(json.dumps({
        "version": 2,
        "stage": 2,
        "stage_name": "full",
        "config": {
            "enabled": False,
            "start_stage": 0,
            "success_threshold": 0.8,
            "window": 100,
            "min_stage_steps": 20_000,
            "goal_jitter_radius": 0.2,
        },
    }))

    state = validate_sac_manifest_recovery(
        checkpoint, state_path, _sac_recovery_args()
    )

    assert state["stage"] == 2


def test_sac_manifest_recovery_rejects_different_curriculum_options(tmp_path):
    checkpoint = tmp_path / "sac_1_steps.zip"
    with zipfile.ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", json.dumps({}))
    state_path = tmp_path / "curriculum_state.json"
    state_path.write_text(json.dumps({
        "version": 2,
        "config": {
            "enabled": False,
            "success_threshold": 0.8,
            "window": 100,
            "min_stage_steps": 20_000,
            "goal_jitter_radius": 0.2,
        },
    }))

    with np.testing.assert_raises_regex(
        ValueError, "curriculum options differ"
    ):
        validate_sac_manifest_recovery(
            checkpoint,
            state_path,
            _sac_recovery_args(curriculum=True),
        )


def test_hardware_wall_clock_delay_is_penalized():
    env = RoboboCompactEnv(rob=_FakeRobobo(), config=RoboboCompactEnvConfig(
        detect_blob_from_camera=False,
        randomize_food_positions=False,
    ))
    env.reset()
    env._last_observation_wall_time = time.monotonic() - 1.0
    _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
    assert info["transition_seconds"] >= 1.0
    assert reward <= -0.5


def test_sac_replay_stores_policy_requested_action():
    class Buffer:
        def add(self, obs, next_obs, action, reward, dones, infos):
            self.action = action.copy()

    model = StabilizedSAC.__new__(StabilizedSAC)
    model._vec_normalize_env = None
    model._last_obs = np.zeros((1, 12), dtype=np.float32)
    model._last_original_obs = model._last_obs
    buffer = Buffer()
    model._store_transition(
        buffer,
        np.array([[1.0, 1.0]], dtype=np.float32),
        np.zeros((1, 12), dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        np.array([False]),
        [{"executed_action": np.array([0.35, 0.2], dtype=np.float32)}],
    )
    np.testing.assert_allclose(buffer.action, [[1.0, 1.0]])


def test_randomization_uses_one_smoothing_stage_and_randomizes_camera_pose():
    env = DomainRandomizationWrapper(
        RoboboCompactEnv(
            rob=_FakeRobobo(),
            config=RoboboCompactEnvConfig(
                reset_settle_time=0.0,
                randomize_food_positions=False,
            ),
        ),
        seed=13,
    )
    env.reset()
    previous_weight = env.episode_parameters["smoothing_previous_weight"]
    assert env.env.unwrapped.action_executor.smoothing.previous_weight == previous_weight
    assert "camera_tilt_offset" in env.episode_parameters
    assert "camera_shift_pixels" in env.episode_parameters


def test_randomization_ranges_can_be_derived_from_calibration_profiles():
    simulation = CalibrationProfile(
        "simulation",
        tuple(IRSensorCalibration(0, 100) for _ in range(8)),
    )
    hardware = CalibrationProfile(
        "hardware",
        tuple(IRSensorCalibration(5, 125) for _ in range(8)),
    )
    ranges = RandomizationRanges.from_calibration_profiles(simulation, hardware)
    assert ranges.ir_gain[0] < 1.0 < ranges.ir_gain[1]
    assert ranges.ir_bias[0] < 0.0 < ranges.ir_bias[1]


def test_simulation_and_dynamics_timesteps_are_configured_separately():
    class FakeSim:
        simulation_stopped = 0
        floatparam_simulation_time_step = 4
        floatparam_physicstimestep = 5
        boolparam_realtime_simulation = 6
        intparam_idle_fps = 7

        def __init__(self):
            self.values = {}
            self.bool_values = {}
            self.int_values = {}

        def getSimulationState(self):
            return self.simulation_stopped

        def setFloatParam(self, parameter, value):
            self.values[parameter] = value

        def setBoolParam(self, parameter, value):
            self.bool_values[parameter] = value

        def setInt32Param(self, parameter, value):
            self.int_values[parameter] = value

        def getSimulationTimeStep(self):
            return self.values[self.floatparam_simulation_time_step]

        def getFloatParam(self, parameter):
            return self.values[parameter]

    class FakeClient:
        def __init__(self):
            self.stepping = None

        def setStepping(self, enabled):
            self.stepping = enabled

    rob = SimulationRobobo.__new__(SimulationRobobo)
    rob._sim = FakeSim()
    rob._client = FakeClient()
    rob._logger = lambda _message: None
    rob.configure_simulation_timing()

    assert rob._sim.values[rob._sim.floatparam_simulation_time_step] == 0.4
    assert rob._sim.values[rob._sim.floatparam_physicstimestep] == 0.005
    assert rob._sim.bool_values[rob._sim.boolparam_realtime_simulation] is False
    assert rob._sim.int_values[rob._sim.intparam_idle_fps] == 0
    assert rob._client.stepping is True
