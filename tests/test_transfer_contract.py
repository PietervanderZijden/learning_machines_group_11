import json
import time
import zipfile

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
from train_sac import (
    find_sac_resume_checkpoint,
    matching_sac_replay_buffer,
    promote_sac_checkpoint,
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
    # Runtime hardware calibration may differ from the calibration used to
    # train the checkpoint. All other deployment contracts remain strict.
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
    assert reward < -0.1  # elapsed-time cost plus command-change cost
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

        def __init__(self):
            self.values = {}

        def getSimulationState(self):
            return self.simulation_stopped

        def setFloatParam(self, parameter, value):
            self.values[parameter] = value

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
    rob.configure_simulation_timing()

    assert rob._sim.values[rob._sim.floatparam_simulation_time_step] == 0.4
    assert rob._sim.values[rob._sim.floatparam_physicstimestep] == 0.005
    assert rob._client.stepping is True
