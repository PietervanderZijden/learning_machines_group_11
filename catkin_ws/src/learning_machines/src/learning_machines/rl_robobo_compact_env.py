from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from learning_machines.transfer import (
    ActionExecutor,
    CalibrationProfile,
    CONTROL_INTERVAL_SECONDS,
    ObservationAdapter,
    RewardConfig,
    SmoothingConfig,
    default_calibration_profile,
    transfer_reward,
)


@dataclass
class RoboboCompactEnvConfig:
    task: str = "food_collection"
    image_size: tuple[int, int] = (64, 64)
    max_wheel_speed: int = 100
    step_millis: int = 400
    phone_tilt: int = 100
    phone_tilt_speed: int = 100
    phone_tilt_tolerance: int = 20
    phone_tilt_timeout: float = 20.0
    initialize_phone_tilt: bool = True
    max_episode_steps: int = 150
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
    return_image: bool = False
    image_obs_size: tuple[int, int] = (64, 64)
    include_position_info: bool = False
    detect_blob_from_camera: bool = True
    calibration_path: str | None = None
    calibration_profile: CalibrationProfile | None = None
    max_episode_seconds: float | None = None
    time_penalty_per_second: float = 0.5
    action_change_penalty: float = 0.02
    action_smoothing: bool = True
    pre_action_safety: bool = True
    smoothing_previous_weight: float = 0.65
    smoothing_requested_weight: float = 0.35
    max_action_delta: float = 0.5
    blob_track_max_distance: float = 0.30
    blob_track_max_missed: int = 6
    active_food_count: int | None = None
    randomize_push_layout: bool = True
    push_curriculum_stage: int = 2
    push_goal_jitter_radius: float = 0.20
    push_arena_radius: float = 0.90
    push_min_robot_distance: float = 0.35
    push_min_block_goal_distance: float = 0.35
    push_max_block_goal_distance: float = 1.20
    push_success_distance: float = 0.18
    push_discount: float = 0.997
    push_approach_potential_scale: float = 2.0
    push_goal_potential_offset: float = 2.0
    push_goal_potential_scale: float = 4.0
    push_contact_bonus: float = 1.0
    push_approach_completion_bonus: float = 5.0
    push_goal_completion_bonus: float = 15.0
    push_standoff_distance: float = 0.35
    push_time_penalty_per_second: float = 0.05
    push_action_change_penalty: float = 0.0
    push_red_hsv_low_1: tuple[int, int, int] = (0, 80, 60)
    push_red_hsv_high_1: tuple[int, int, int] = (12, 255, 255)
    push_red_hsv_low_2: tuple[int, int, int] = (168, 80, 60)
    push_red_hsv_high_2: tuple[int, int, int] = (180, 255, 255)
    push_green_hsv_low: tuple[int, int, int] = (35, 70, 60)
    push_green_hsv_high: tuple[int, int, int] = (90, 255, 255)
    hardware_inference_only: bool = False
    hardware_wheel_command: Callable[[int, int, int], None] | None = None


@dataclass
class _ObsContext:
    raw_irs: list[float]
    irs: np.ndarray
    timing: dict[str, float]


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
        self._is_simulation = hasattr(rob, "_sim")

        if self.config.task not in {"food_collection", "push"}:
            raise ValueError("task must be 'food_collection' or 'push'")
        if self.config.hardware_inference_only and self._is_simulation:
            raise ValueError("hardware_inference_only requires physical hardware")

        obs_spaces = {
            "ir": spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
        }
        if self.config.task == "push":
            obs_spaces["red_block"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )
            obs_spaces["green_goal"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )
        else:
            obs_spaces["blob"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )
        if self.config.return_image:
            h, w = self.config.image_obs_size
            obs_spaces["image"] = spaces.Box(low=0, high=255, shape=(3, h, w), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        self._step_count = 0
        self._prev_food_count = 0
        self._food_handles: list[int] = []
        self._arena_cx, self._arena_cy = self.config.arena_center
        self._num_food = 7
        self._initial_pos_x = self._arena_cx
        self._initial_pos_y = self._arena_cy
        self._last_step_timing: dict[str, float] = {}
        if not 5 <= self.config.phone_tilt <= 110:
            raise ValueError("phone_tilt must be in the Robobo range [5, 110]")
        if not 0 < self.config.phone_tilt_speed <= 100:
            raise ValueError("phone_tilt_speed must be in the range [1, 100]")
        profile = self.config.calibration_profile
        if profile is None and self.config.calibration_path:
            profile = CalibrationProfile.load(self.config.calibration_path)
        self.observation_adapter = ObservationAdapter(profile or default_calibration_profile())
        smoothing = SmoothingConfig(
            previous_weight=self.config.smoothing_previous_weight if self.config.action_smoothing else 0.0,
            requested_weight=self.config.smoothing_requested_weight if self.config.action_smoothing else 1.0,
            max_delta=self.config.max_action_delta,
        )
        self.action_executor = ActionExecutor(
            smoothing=smoothing,
            safety_enabled=self.config.pre_action_safety,
        )
        self._latest_ir = np.zeros(8, dtype=np.float32)
        self._latest_blob = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        self._latest_red_block = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        self._latest_green_goal = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        self._elapsed_seconds = 0.0
        self._collision_count = 0
        self._safety_override_count = 0
        self._action_change_total = 0.0
        self._saturation_total = 0.0
        self._last_observation_wall_time = time.monotonic()
        self._observation_sim_seconds = 0.0
        self._tracked_blob: np.ndarray | None = None
        self._red_block_handle: int | None = None
        self._green_goal_handle: int | None = None
        self._red_block_z = 0.025
        self._green_goal_z = 0.005
        self._push_layout_randomized = False
        self._push_layout_mode = "full"
        self._authored_red_block_pose: tuple[float, float, float] | None = None
        self._authored_green_goal_pose: tuple[float, float, float] | None = None
        self._authored_robot_pose: tuple[float, float, float] | None = None
        self._authored_robot_orientation: tuple[float, float, float] | None = None
        self._robot_respondable_handles: tuple[int, ...] = ()
        self._previous_block_goal_distance: float | None = None
        self._previous_push_potential: float | None = None
        self._contact_acquired = False
        self._blob_track_missed = 0
        self._blob_target_switches = 0
        self._blob_target_confidence = 0.0
        self.max_episode_seconds = (
            float(self.config.max_episode_seconds)
            if self.config.max_episode_seconds is not None
            else self.config.max_episode_steps * CONTROL_INTERVAL_SECONDS
        )
        self._reward_config = RewardConfig(
            food_reward=self.config.collect_reward,
            time_penalty_per_second=self.config.time_penalty_per_second,
            completion_bonus_scale=self.config.speed_bonus_scale,
            collision_penalty=self.config.collision_penalty,
            action_change_penalty=self.config.action_change_penalty,
            max_episode_seconds=self.max_episode_seconds,
        )

        self._blob_low = np.array([35, 80, 80], dtype=np.uint8)
        self._blob_high = np.array([85, 255, 255], dtype=np.uint8)
        self._morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        if self._is_simulation and self.rob.is_stopped():
            self.rob.play_simulation()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_simulation()
        self._reset_blob_tracker()
        self._initialize_camera_pose()
        if self._is_simulation:
            self._fix_lifted_food()
        reset_food_count = self.rob.get_nr_food_collected() if self._is_simulation else 0
        if reset_food_count != 0:
            raise RuntimeError(
                f"episode reset started with {reset_food_count} collected food items"
            )
        self._step_count = 0
        self._prev_food_count = 0
        self._elapsed_seconds = 0.0
        self._collision_count = 0
        self._safety_override_count = 0
        self._action_change_total = 0.0
        self._saturation_total = 0.0
        self.action_executor.reset()
        self.rob._used_pids.clear()
        observation_sim_start = self.rob.get_sim_time() if self._is_simulation else None
        obs, obs_context = self._get_obs_with_context()
        if observation_sim_start is not None:
            self._observation_sim_seconds = max(
                0.0, self.rob.get_sim_time() - observation_sim_start
            )
        if self.config.task == "push":
            geometry = self._push_geometry()
            if geometry is None and not self.config.hardware_inference_only:
                raise RuntimeError(
                    "push reward requires CoppeliaSim world positions for the "
                    "robot, red block, and green goal"
                )
            if geometry is not None:
                self._previous_block_goal_distance = self._geometry_block_goal_distance(
                    geometry
                )
                self._contact_acquired = self.config.push_curriculum_stage == 1
                self._previous_push_potential = self._push_potential(
                    geometry, contact_acquired=self._contact_acquired
                )
        self._last_observation_wall_time = time.monotonic()
        info = self._get_info(obs_context)
        if self.config.task == "push":
            info.update(self._push_observation_info(obs, progress=0.0))
        info["phone_tilt"] = self._read_phone_tilt()
        return obs, info

    def step(self, action):
        self._step_count += 1
        requested_action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        action, action_info = self.action_executor.execute(
            requested_action, self._latest_ir, blob=self._latest_blob
        )

        left_speed = float(action[0] * self.config.max_wheel_speed)
        right_speed = float(action[1] * self.config.max_wheel_speed)
        duration_s = self.config.step_millis / 1000.0
        if not np.isclose(duration_s, CONTROL_INTERVAL_SECONDS):
            raise ValueError("the transfer contract requires a fixed 400 ms control interval")

        try:
            wheel_start = time.perf_counter()
            if (
                not self._is_simulation
                and self.config.hardware_wheel_command is not None
            ):
                self.config.hardware_wheel_command(
                    int(np.clip(round(left_speed), -100, 100)),
                    int(np.clip(round(right_speed), -100, 100)),
                    int(round(duration_s * 1000.0)),
                )
                explicit_step_seconds = 0.0
            elif not self._is_simulation and hasattr(self.rob, "move_blocking"):
                self.rob.move_blocking(
                    int(np.clip(round(left_speed), -100, 100)),
                    int(np.clip(round(right_speed), -100, 100)),
                    int(round(duration_s * 1000.0)),
                )
                explicit_step_seconds = 0.0
            else:
                self.rob.set_wheel_speeds(
                    left_speed, right_speed, duration_s=duration_s
                )
                explicit_step_seconds = duration_s
            if self._is_simulation:
                explicit_step_seconds = max(
                    0.0, duration_s - self._observation_sim_seconds
                )
            if explicit_step_seconds > 0:
                self.rob.sleep(explicit_step_seconds)
            wheel_time = time.perf_counter() - wheel_start
        except RuntimeError:
            if not self.rob.is_running():
                obs, obs_context = self._get_obs_with_context()
                info = self._get_info(obs_context)
                info["simulation_stopped"] = True
                return obs, 0.0, False, True, info
            raise

        obs_start = time.perf_counter()
        observation_sim_start = self.rob.get_sim_time() if self._is_simulation else None
        obs, obs_context = self._get_obs_with_context()
        if observation_sim_start is not None:
            self._observation_sim_seconds = max(
                0.0, self.rob.get_sim_time() - observation_sim_start
            )
        obs_time = time.perf_counter() - obs_start
        info_start = time.perf_counter()
        info = self._get_info(obs_context)
        info_time = time.perf_counter() - info_start
        self._last_step_timing = {
            "wheel_step": wheel_time,
            "observation": obs_time,
            "info": info_time,
            **obs_context.timing,
        }

        observation_wall_time = time.monotonic()
        elapsed_delta_seconds = duration_s
        if not self._is_simulation:
            elapsed_delta_seconds = max(
                duration_s, observation_wall_time - self._last_observation_wall_time
            )
        self._last_observation_wall_time = observation_wall_time
        self._elapsed_seconds += elapsed_delta_seconds
        truncated = self._elapsed_seconds >= self.max_episode_seconds - 1e-9
        if self.config.task == "push":
            if self.config.hardware_inference_only:
                self._collision_count += int(bool(info["collision"]))
                self._safety_override_count += int(
                    action_info["safety_override"] is not None
                )
                self._action_change_total += action_info["action_change"]
                self._saturation_total += action_info["action_saturation"]
                info["left_speed"] = left_speed
                info["right_speed"] = right_speed
                info.update(action_info)
                info.update(self._push_observation_info(obs, progress=0.0))
                info["elapsed_seconds"] = self._elapsed_seconds
                info["transition_seconds"] = elapsed_delta_seconds
                info["collisions"] = self._collision_count
                info["safety_overrides"] = self._safety_override_count
                info["mean_action_change"] = (
                    self._action_change_total / self._step_count
                )
                info["action_saturation_rate"] = (
                    self._saturation_total / self._step_count
                )
                return obs, 0.0, False, truncated, info
            reward, terminated, reward_info = self._push_reward(
                obs=obs,
                elapsed_delta_seconds=elapsed_delta_seconds,
                collision=bool(info["collision"]),
                action_change=action_info["action_change"],
            )
            truncated = self._resolve_push_truncation(terminated, truncated)
            self._collision_count += int(bool(info["collision"]))
            self._safety_override_count += int(action_info["safety_override"] is not None)
            self._action_change_total += action_info["action_change"]
            self._saturation_total += action_info["action_saturation"]
            info["left_speed"] = left_speed
            info["right_speed"] = right_speed
            info.update(reward_info)
            info.update(action_info)
            info["elapsed_seconds"] = self._elapsed_seconds
            info["transition_seconds"] = elapsed_delta_seconds
            info["collisions"] = self._collision_count
            info["safety_overrides"] = self._safety_override_count
            info["mean_action_change"] = self._action_change_total / self._step_count
            info["action_saturation_rate"] = self._saturation_total / self._step_count
            info["red_block_visible"] = float(obs["red_block"][3] > 0.5)
            info["green_goal_visible"] = float(obs["green_goal"][3] > 0.5)
            info["safety_with_visible_block"] = float(
                action_info["safety_override"] is not None
                and obs["red_block"][3] > 0.5
            )
            return obs, reward, terminated, truncated, info

        new_food = info["food_collected"]
        newly_collected = max(0, new_food - self._prev_food_count)
        self._prev_food_count = new_food

        terminated = new_food >= self._num_food
        reward, reward_info = transfer_reward(
            newly_collected=newly_collected,
            elapsed_delta_seconds=elapsed_delta_seconds,
            elapsed_seconds=self._elapsed_seconds,
            completed=terminated,
            collision=bool(info["collision"]),
            action_change=action_info["action_change"],
            config=self._reward_config,
        )
        self._collision_count += int(bool(info["collision"]))
        self._safety_override_count += int(action_info["safety_override"] is not None)
        self._action_change_total += action_info["action_change"]
        self._saturation_total += action_info["action_saturation"]

        info["newly_collected"] = newly_collected
        info["left_speed"] = left_speed
        info["right_speed"] = right_speed
        info.update(reward_info)
        info.update(action_info)
        info["speed_bonus"] = reward_info["completion_bonus"]
        info["elapsed_seconds"] = self._elapsed_seconds
        info["transition_seconds"] = elapsed_delta_seconds
        info["food_per_minute"] = 60.0 * new_food / max(duration_s, self._elapsed_seconds)
        info["completion_time"] = self._elapsed_seconds if terminated else None
        info["collisions"] = self._collision_count
        info["safety_overrides"] = self._safety_override_count
        info["mean_action_change"] = self._action_change_total / self._step_count
        info["action_saturation_rate"] = self._saturation_total / self._step_count
        info["blob_visible"] = float(obs["blob"][3] > 0.5)
        info["safety_with_visible_food"] = float(
            action_info["safety_override"] is not None
            and obs["blob"][3] > 0.5
        )

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            if not self.config.hardware_inference_only:
                self.rob.set_wheel_speeds(0, 0)
            if self._is_simulation and not self.rob.is_stopped():
                self.rob.stop_simulation()
        except Exception:
            pass

    def _reset_simulation(self) -> None:
        self.rob.set_wheel_speeds(0, 0)
        if not self._is_simulation:
            time.sleep(self.config.settle_sleep)
            return

        if not self.rob.is_stopped():
            self.rob.stop_simulation()
        time.sleep(0.1)
        self.rob.configure_simulation_timing()

        self._ensure_food_handles()
        if self.config.task == "push":
            self._ensure_push_handles()
        self._cache_initial_position()
        if self.config.task == "push":
            self._randomize_push_layout()
        elif self.config.randomize_food_positions:
            self._randomize_food_positions()
        elif self.config.active_food_count is not None:
            self._apply_food_curriculum()

        self.rob.play_simulation()
        self._fix_lifted_food()

    def _read_phone_tilt(self) -> int | None:
        try:
            value = int(self.rob.read_phone_tilt())
            # Some Robobo hardware installations actuate tilt correctly but
            # never publish /robot/tilt, leaving the subscriber's startup
            # value at zero. Treat impossible hardware values as unavailable;
            # simulation retains strict feedback checking.
            if not self._is_simulation and not 26 <= value <= 109:
                return None
            return value
        except Exception:
            return None

    def _settle(self, seconds: float) -> None:
        """Advance the simulation to let physics/scripts settle.

        In stepping mode this is equivalent to the requested amount of
        simulation time without relying on wall-clock sleep.
        """
        if self._is_simulation:
            if seconds <= 0:
                return
            steps = max(1, math.ceil(seconds / CONTROL_INTERVAL_SECONDS))
            self.rob.step_simulation(steps)
        else:
            if seconds > 0:
                self.rob.sleep(seconds)

    def _wait_for_tilt_script_ready(self, timeout: float) -> int | None:
        """Wait until the tilt motor script has initialized after a restart.

        CoppeliaSim child scripts initialize on the first simulation step, so
        this helper steps the simulation until read_phone_tilt() succeeds.
        """
        if not self._is_simulation:
            return self._read_phone_tilt()

        max_steps = max(1, math.ceil(timeout / CONTROL_INTERVAL_SECONDS))
        for _ in range(max_steps + 1):
            try:
                return self.rob.read_phone_tilt()
            except Exception as exc:
                msg = str(exc)
                if "script is not initialized" in msg or "has already ended" in msg:
                    self.rob.step_simulation(1)
                    continue
                raise
        raise RuntimeError(
            "Tilt motor script did not initialize after simulation restart"
        )

    def _initialize_camera_pose(self) -> None:
        """Point the camera at the arena before the first observation."""
        if (
            not self.config.initialize_phone_tilt
            or not (self.config.return_image or self.config.detect_blob_from_camera)
        ):
            self._settle(self.config.reset_settle_time)
            return

        actual = self._wait_for_tilt_script_ready(self.config.phone_tilt_timeout)
        try:
            if not self._is_simulation:
                blockid = self.rob.set_phone_tilt(
                    self.config.phone_tilt,
                    self.config.phone_tilt_speed,
                )
                try:
                    if actual is not None:
                        deadline = time.monotonic() + self.config.phone_tilt_timeout
                        while time.monotonic() < deadline:
                            actual = self._read_phone_tilt()
                            if (
                                actual is not None
                                and abs(actual - self.config.phone_tilt)
                                <= self.config.phone_tilt_tolerance
                            ):
                                break
                            self.rob.sleep(0.1)
                        else:
                            raise RuntimeError(
                                "phone tilt did not reach ground-facing target "
                                f"{self.config.phone_tilt}; actual={actual}"
                            )
                finally:
                    if hasattr(self.rob, "_used_pids"):
                        self.rob._used_pids.discard(blockid)
                self._settle(self.config.reset_settle_time)
                return
            self.rob.set_phone_tilt(
                self.config.phone_tilt, self.config.phone_tilt_speed
            )
        except Exception as exc:
            raise RuntimeError("failed to command the Robobo phone tilt") from exc

        max_steps = max(
            1,
            math.ceil(self.config.phone_tilt_timeout / CONTROL_INTERVAL_SECONDS),
        )
        steps = 0
        while (
            actual is not None
            and abs(actual - self.config.phone_tilt) > self.config.phone_tilt_tolerance
            and steps < max_steps
        ):
            if self._is_simulation:
                self.rob.step_simulation(1)
            else:
                self.rob.sleep(0.1)
            steps += 1
            actual = self._read_phone_tilt()

        if actual is not None and abs(actual - self.config.phone_tilt) > self.config.phone_tilt_tolerance:
            print(
                f"Warning: phone tilt reached {actual}/{self.config.phone_tilt} "
                f"after {steps} steps; continuing — motor will catch up during "
                "the episode"
            )
        self._settle(self.config.reset_settle_time)

    def _reset_blob_tracker(self) -> None:
        self._tracked_blob = None
        self._blob_track_missed = 0
        self._blob_target_switches = 0
        self._blob_target_confidence = 0.0
        self._previous_block_goal_distance = None
        self._previous_push_potential = None
        self._contact_acquired = False

    def _cache_initial_position(self) -> None:
        try:
            pos = self.rob.get_position()
            self._initial_pos_x = pos.x
            self._initial_pos_y = pos.y
            if self._authored_robot_pose is None:
                self._authored_robot_pose = (float(pos.x), float(pos.y), float(pos.z))
                try:
                    orientation = self.rob.get_orientation()
                    self._authored_robot_orientation = (
                        float(orientation.yaw),
                        float(orientation.pitch),
                        float(orientation.roll),
                    )
                except Exception:
                    self._authored_robot_orientation = (0.0, 0.0, 0.0)
        except Exception:
            self._initial_pos_x = self._arena_cx
            self._initial_pos_y = self._arena_cy

    def _ensure_food_handles(self) -> None:
        if not self._food_handles or not self._food_handles_are_valid():
            self._discover_food_handles()

    def _food_handles_are_valid(self) -> bool:
        if not self._food_handles:
            return False
        try:
            self.rob._sim.getObjectPosition(self._food_handles[0], self.rob._sim.handle_world)
            return True
        except Exception:
            return False

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

    def _ensure_push_handles(self) -> None:
        if self._push_handles_are_valid():
            self._cache_robot_respondable_handles()
            return
        self._red_block_handle = self._find_sim_object((
            "/red_block",
            "/RedBlock",
            "/Red_Block",
            "/Block",
            "/Food",
            "/push_block",
            "/PushBlock",
            "red_block",
            "RedBlock",
            "Food",
        ))
        self._green_goal_handle = self._find_sim_object((
            "/green_goal",
            "/GreenGoal",
            "/Green_Goal",
            "/Goal",
            "/goal",
            "/Base",
            "/push_goal",
            "/PushGoal",
            "green_goal",
            "GreenGoal",
            "Base",
        ))
        self._cache_push_object_heights()
        self._cache_robot_respondable_handles()

    def _cache_robot_respondable_handles(self) -> None:
        if not self._is_simulation or getattr(
            self, "_robot_respondable_handles", ()
        ):
            return
        sim = self.rob._sim
        robot_root = getattr(self.rob, "_robobo", None)
        if robot_root is None:
            return
        handles: list[int] = []
        try:
            candidates = sim.getObjectsInTree(
                robot_root, sim.object_shape_type, 0
            )
        except Exception:
            candidates = ()
        for handle in candidates:
            try:
                respondable = sim.getObjectInt32Param(
                    handle, sim.shapeintparam_respondable
                )
            except Exception:
                continue
            if respondable:
                handles.append(int(handle))
        self._robot_respondable_handles = tuple(handles)

    def _find_sim_object(self, names: tuple[str, ...]) -> int | None:
        if not self._is_simulation:
            return None
        sim = self.rob._sim
        for name in names:
            try:
                handle = sim.getObject(name)
                if handle >= 0:
                    return int(handle)
            except Exception:
                pass
        return None

    def _push_handles_are_valid(self) -> bool:
        if not self._is_simulation:
            return False
        if self._red_block_handle is None or self._green_goal_handle is None:
            return False
        try:
            sim = self.rob._sim
            sim.getObjectPosition(self._red_block_handle, sim.handle_world)
            sim.getObjectPosition(self._green_goal_handle, sim.handle_world)
            return True
        except Exception:
            return False

    def _cache_push_object_heights(self) -> None:
        if not self._push_handles_are_valid():
            return
        sim = self.rob._sim
        try:
            block = sim.getObjectPosition(self._red_block_handle, sim.handle_world)
            self._red_block_z = float(block[2])
        except Exception:
            pass
        try:
            goal = sim.getObjectPosition(self._green_goal_handle, sim.handle_world)
            self._green_goal_z = float(goal[2])
        except Exception:
            pass
        if getattr(self, "_authored_red_block_pose", None) is None:
            block = sim.getObjectPosition(self._red_block_handle, sim.handle_world)
            goal = sim.getObjectPosition(self._green_goal_handle, sim.handle_world)
            self._authored_red_block_pose = tuple(float(value) for value in block)
            self._authored_green_goal_pose = tuple(float(value) for value in goal)

    def _sample_push_point(self) -> tuple[float, float]:
        angle = random.uniform(0.0, 2.0 * math.pi)
        radius = random.uniform(
            self.config.push_min_robot_distance,
            self.config.push_arena_radius,
        )
        return (
            self._arena_cx + radius * math.cos(angle),
            self._arena_cy + radius * math.sin(angle),
        )

    @staticmethod
    def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    def _valid_push_layout(
        self,
        block_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> bool:
        robot_xy = (self._initial_pos_x, self._initial_pos_y)
        for point in (block_xy, goal_xy):
            if self._xy_distance(point, (self._arena_cx, self._arena_cy)) > self.config.push_arena_radius:
                return False
        if self._xy_distance(block_xy, robot_xy) < self.config.push_min_robot_distance:
            return False
        if self._xy_distance(goal_xy, robot_xy) < self.config.push_min_robot_distance:
            return False
        distance = self._xy_distance(block_xy, goal_xy)
        min_distance = max(
            self.config.push_min_block_goal_distance,
            self.config.push_success_distance * 2.0,
        )
        return min_distance <= distance <= self.config.push_max_block_goal_distance

    def _set_push_object_pose(self, handle: int | None, xy: tuple[float, float], z: float) -> None:
        if handle is None:
            return
        sim = self.rob._sim
        sim.setObjectPosition(handle, [xy[0], xy[1], z])
        try:
            sim.resetDynamicObject(handle)
        except Exception:
            pass

    def _set_robot_push_pose(
        self,
        block_xy: tuple[float, float],
        goal_xy: tuple[float, float],
    ) -> None:
        if self._authored_robot_pose is None:
            return
        distance = self._xy_distance(block_xy, goal_xy)
        if distance <= 1e-9:
            return
        direction = (
            (goal_xy[0] - block_xy[0]) / distance,
            (goal_xy[1] - block_xy[1]) / distance,
        )
        robot_xy = (
            block_xy[0] - self.config.push_standoff_distance * direction[0],
            block_xy[1] - self.config.push_standoff_distance * direction[1],
        )
        sim = self.rob._sim
        robot_handle = getattr(self.rob, "_robobo", None)
        if robot_handle is None:
            return
        sim.setObjectPosition(
            robot_handle,
            [robot_xy[0], robot_xy[1], self._authored_robot_pose[2]],
        )
        orientation = self._authored_robot_orientation or (0.0, 0.0, 0.0)
        authored_block = self._authored_red_block_pose
        if authored_block is not None and self._authored_robot_pose is not None:
            to_block = (
                authored_block[0] - self._authored_robot_pose[0],
                authored_block[1] - self._authored_robot_pose[1],
            )
            forward_offset = orientation[2] - math.atan2(to_block[1], to_block[0])
        else:
            forward_offset = 0.0
        heading = math.atan2(direction[1], direction[0]) + forward_offset
        sim.setObjectOrientation(
            robot_handle,
            [orientation[0], orientation[1], heading],
        )

    def _randomize_push_layout(self) -> None:
        self._push_layout_randomized = False
        if not self._push_handles_are_valid():
            return
        stage = int(self.config.push_curriculum_stage)
        if stage not in (0, 1, 2):
            raise ValueError("push curriculum stage must be 0, 1, or 2")
        if not self.config.randomize_push_layout:
            stage = 0
        if (
            getattr(self, "_authored_red_block_pose", None) is None
            or getattr(self, "_authored_green_goal_pose", None) is None
        ):
            self._cache_push_object_heights()
        authored_block = self._authored_red_block_pose
        authored_goal = self._authored_green_goal_pose
        if authored_block is None or authored_goal is None:
            return
        block_xy = (authored_block[0], authored_block[1])
        goal_xy = (authored_goal[0], authored_goal[1])
        if stage == 0:
            self._push_layout_mode = "approach"
            self._set_push_object_pose(self._red_block_handle, block_xy, authored_block[2])
            self._set_push_object_pose(self._green_goal_handle, goal_xy, authored_goal[2])
            return
        if stage == 1:
            self._push_layout_mode = "push"
            for _ in range(100):
                angle = random.uniform(0.0, 2.0 * math.pi)
                radius = self.config.push_goal_jitter_radius * math.sqrt(random.random())
                candidate_goal = (
                    goal_xy[0] + radius * math.cos(angle),
                    goal_xy[1] + radius * math.sin(angle),
                )
                if self._valid_push_layout(block_xy, candidate_goal):
                    goal_xy = candidate_goal
                    break
            self._set_push_object_pose(self._red_block_handle, block_xy, authored_block[2])
            self._set_push_object_pose(self._green_goal_handle, goal_xy, authored_goal[2])
            self._set_robot_push_pose(block_xy, goal_xy)
            self._push_layout_randomized = True
            return

        self._push_layout_mode = "full"
        block_xy = goal_xy = None
        for _ in range(100):
            candidate_goal = self._sample_push_point()
            candidate_block = self._sample_push_point()
            if self._valid_push_layout(candidate_block, candidate_goal):
                block_xy = candidate_block
                goal_xy = candidate_goal
                break

        if block_xy is None or goal_xy is None:
            radii = np.linspace(
                self.config.push_min_robot_distance,
                self.config.push_arena_radius,
                12,
            )
            for radius in radii:
                candidate_block = (self._arena_cx - radius, self._arena_cy)
                candidate_goal = (self._arena_cx + radius, self._arena_cy)
                if self._valid_push_layout(candidate_block, candidate_goal):
                    block_xy, goal_xy = candidate_block, candidate_goal
                    break
        if block_xy is None or goal_xy is None:
            raise RuntimeError("push layout constraints have no valid full-stage layout")

        self._set_push_object_pose(self._red_block_handle, block_xy, self._red_block_z)
        self._set_push_object_pose(self._green_goal_handle, goal_xy, self._green_goal_z)
        self._push_layout_randomized = True

    def _randomize_food_positions(self) -> None:
        sim = self.rob._sim
        active_count = self._active_food_count()
        for index, h in enumerate(self._food_handles):
            if index >= active_count:
                try:
                    sim.setObjectPosition(h, [self._arena_cx, self._arena_cy, -5.0])
                except Exception:
                    pass
                continue
            placed = False
            for _ in range(20):
                angle = random.uniform(0, 2 * math.pi)
                r = random.uniform(self.config.food_min_radius, self.config.food_arena_radius)
                fx = self._arena_cx + r * math.cos(angle)
                fy = self._arena_cy + r * math.sin(angle)
                dx = fx - self._initial_pos_x
                dy = fy - self._initial_pos_y
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
        self._num_food = active_count

    def _active_food_count(self) -> int:
        if self.config.active_food_count is None:
            return max(1, len(self._food_handles))
        return max(
            1,
            min(int(self.config.active_food_count), len(self._food_handles)),
        )

    def _apply_food_curriculum(self) -> None:
        active_count = self._active_food_count()
        for index, handle in enumerate(self._food_handles):
            if index >= active_count:
                try:
                    self.rob._sim.setObjectPosition(
                        handle, [self._arena_cx, self._arena_cy, -5.0]
                    )
                except Exception:
                    pass
        self._num_food = active_count

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
        obs, _ = self._get_obs_with_context()
        return obs

    def _get_obs_with_context(self) -> tuple[dict, _ObsContext]:
        timing: dict[str, float] = {}
        ir_start = time.perf_counter()
        raw_irs = self._safe_read_irs()
        irs = self.observation_adapter.normalize_ir(raw_irs)
        self._latest_ir = irs.copy()
        timing["ir_read"] = time.perf_counter() - ir_start

        image_bgr = None
        needs_image = self.config.return_image or self.config.detect_blob_from_camera
        if needs_image:
            image_start = time.perf_counter()
            try:
                image_bgr = self.rob.read_image_front()
            except Exception:
                image_bgr = None
            timing["image_read"] = time.perf_counter() - image_start
        else:
            timing["image_read"] = 0.0

        if (
            self.config.task == "food_collection"
            and self.config.detect_blob_from_camera
            and image_bgr is not None
        ):
            blob_start = time.perf_counter()
            blob = self._detect_blob(image_bgr)
            timing["blob_detection"] = time.perf_counter() - blob_start
        else:
            blob = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
            timing["blob_detection"] = 0.0

        if self.config.task == "push":
            if self.config.detect_blob_from_camera and image_bgr is not None:
                push_start = time.perf_counter()
                red_block = self._detect_color_blob(
                    image_bgr,
                    (
                        (self.config.push_red_hsv_low_1, self.config.push_red_hsv_high_1),
                        (self.config.push_red_hsv_low_2, self.config.push_red_hsv_high_2),
                    ),
                )
                green_goal = self._detect_color_blob(
                    image_bgr,
                    ((self.config.push_green_hsv_low, self.config.push_green_hsv_high),),
                )
                timing["push_blob_detection"] = time.perf_counter() - push_start
            else:
                red_block = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
                green_goal = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
                timing["push_blob_detection"] = 0.0
            obs = {
                "red_block": red_block,
                "green_goal": green_goal,
                "ir": irs,
            }
            self._latest_red_block = red_block.copy()
            self._latest_green_goal = green_goal.copy()
            self._latest_blob = red_block.copy()
        else:
            obs = {
                "blob": blob,
                "ir": irs,
            }
            self._latest_blob = blob.copy()

        if self.config.return_image:
            if image_bgr is not None:
                h, w = self.config.image_obs_size
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image_rgb, (w, h))
                # CHW format for PyTorch
                obs["image"] = np.transpose(image_resized, (2, 0, 1)).copy()
            else:
                h, w = self.config.image_obs_size
                obs["image"] = np.zeros((3, h, w), dtype=np.uint8)

        return obs, _ObsContext(raw_irs=raw_irs, irs=irs, timing=timing)

    def _detect_color_blob(
        self,
        image_bgr: np.ndarray,
        hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...],
    ) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros((h, w), dtype=np.uint8)
        for low, high in hsv_ranges:
            mask = cv2.bitwise_or(
                mask,
                cv2.inRange(
                    hsv,
                    np.array(low, dtype=np.uint8),
                    np.array(high, dtype=np.uint8),
                ),
            )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / (h * w)
            if area_ratio < 0.001 or area_ratio > 0.5:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            candidates.append(np.array([
                moments["m10"] / moments["m00"] / w,
                moments["m01"] / moments["m00"] / h,
                area_ratio,
                1.0,
            ], dtype=np.float32))
        if not candidates:
            return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        return max(candidates, key=lambda candidate: float(candidate[2])).copy()

    def _detect_blob(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._blob_low, self._blob_high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._morph_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / (h * w)
            if area_ratio < 0.001 or area_ratio > 0.5:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            candidates.append(np.array([
                moments["m10"] / moments["m00"] / w,
                moments["m01"] / moments["m00"] / h,
                area_ratio,
                1.0,
            ], dtype=np.float32))

        if not candidates:
            self._blob_track_missed += 1
            self._blob_target_confidence = max(
                0.0, 1.0 - self._blob_track_missed / (self.config.blob_track_max_missed + 1)
            )
            if self._blob_track_missed > self.config.blob_track_max_missed:
                self._tracked_blob = None
            return np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)

        previous = self._tracked_blob
        if previous is None:
            selected = max(candidates, key=lambda candidate: float(candidate[2]))
        else:
            distances = [
                float(np.linalg.norm(candidate[:2] - previous[:2]))
                + 0.25 * abs(float(np.log((candidate[2] + 1e-6) / (previous[2] + 1e-6))))
                for candidate in candidates
            ]
            best_index = int(np.argmin(distances))
            if distances[best_index] <= self.config.blob_track_max_distance:
                selected = candidates[best_index]
            else:
                selected = max(candidates, key=lambda candidate: float(candidate[2]))
                self._blob_target_switches += 1
        self._tracked_blob = selected.copy()
        self._blob_track_missed = 0
        self._blob_target_confidence = 1.0
        return selected

    def _sim_xy_distance(self) -> float | None:
        geometry = self._push_geometry()
        if geometry is None:
            return None
        return self._geometry_block_goal_distance(geometry)

    def _push_geometry(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None:
        if not self._push_handles_are_valid():
            return None
        try:
            sim = self.rob._sim
            robot = self.rob.get_position()
            block = sim.getObjectPosition(self._red_block_handle, sim.handle_world)
            goal = sim.getObjectPosition(self._green_goal_handle, sim.handle_world)
            return (
                (float(robot.x), float(robot.y)),
                (float(block[0]), float(block[1])),
                (float(goal[0]), float(goal[1])),
            )
        except Exception:
            return None

    @staticmethod
    def _geometry_block_goal_distance(
        geometry: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> float:
        _robot, block, goal = geometry
        return math.hypot(block[0] - goal[0], block[1] - goal[1])

    @staticmethod
    def _geometry_robot_block_distance(
        geometry: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
    ) -> float:
        robot, block, _goal = geometry
        return math.hypot(robot[0] - block[0], robot[1] - block[1])

    def _push_potential(
        self,
        geometry: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
        contact_acquired: bool | None = None,
    ) -> float:
        if contact_acquired is None:
            contact_acquired = self._contact_acquired
        if not contact_acquired:
            robot_block_distance = self._geometry_robot_block_distance(geometry)
            normalized = min(
                1.0,
                robot_block_distance / (2.0 * self.config.push_arena_radius),
            )
            return self.config.push_approach_potential_scale * (1.0 - normalized)

        block_goal_distance = self._geometry_block_goal_distance(geometry)
        distance_range = max(
            1e-9,
            self.config.push_max_block_goal_distance
            - self.config.push_success_distance,
        )
        normalized = min(
            1.0,
            max(0.0, block_goal_distance - self.config.push_success_distance)
            / distance_range,
        )
        return (
            self.config.push_goal_potential_offset
            + self.config.push_goal_potential_scale * (1.0 - normalized)
        )

    def _robot_block_contact(self) -> bool:
        if not self._push_handles_are_valid():
            return False
        self._cache_robot_respondable_handles()
        sim = self.rob._sim
        for handle in self._robot_respondable_handles:
            try:
                result = sim.checkCollision(handle, self._red_block_handle)
                value = result[0] if isinstance(result, (tuple, list)) else result
                if int(value) > 0:
                    return True
            except Exception:
                continue
        return False

    def _camera_block_goal_distance(self, obs: dict) -> float | None:
        red = np.asarray(obs.get("red_block", np.zeros(4)), dtype=np.float32)
        green = np.asarray(obs.get("green_goal", np.zeros(4)), dtype=np.float32)
        if red.shape != (4,) or green.shape != (4,):
            return None
        if red[3] <= 0.5 or green[3] <= 0.5:
            return None
        return float(np.linalg.norm(red[:2] - green[:2]))

    def _block_goal_distance(self, obs: dict) -> float | None:
        sim_distance = self._sim_xy_distance()
        if sim_distance is not None:
            return sim_distance
        return self._camera_block_goal_distance(obs)

    def _push_success(self, obs: dict, distance: float | None) -> bool:
        if distance is None:
            return False
        return distance <= self.config.push_success_distance

    @staticmethod
    def _resolve_push_truncation(terminated: bool, time_limit_reached: bool) -> bool:
        return bool(time_limit_reached and not terminated)

    def _push_observation_info(self, obs: dict, progress: float) -> dict[str, float]:
        distance = self._block_goal_distance(obs)
        red_visible = float(obs["red_block"][3] > 0.5)
        green_visible = float(obs["green_goal"][3] > 0.5)
        success = self._push_success(obs, distance)
        return {
            "block_goal_distance": float(distance) if distance is not None else float("nan"),
            "block_goal_progress": float(progress),
            "push_success": float(success),
            "red_block_visible": red_visible,
            "green_goal_visible": green_visible,
            "push_layout_randomized": float(self._push_layout_randomized),
            "push_layout_mode": getattr(self, "_push_layout_mode", "full"),
            "curriculum_stage": float(self.config.push_curriculum_stage),
            "push_phase": float(self._contact_acquired),
        }

    def _push_reward(
        self,
        obs: dict,
        elapsed_delta_seconds: float,
        collision: bool,
        action_change: float,
    ) -> tuple[float, bool, dict[str, float]]:
        geometry = self._push_geometry()
        if geometry is None:
            raise RuntimeError(
                "push reward lost CoppeliaSim world positions for the robot, "
                "red block, or green goal"
            )
        distance = self._geometry_block_goal_distance(geometry)
        robot_block_distance = self._geometry_robot_block_distance(geometry)
        progress = 0.0
        if self._previous_block_goal_distance is not None:
            progress = self._previous_block_goal_distance - distance
        self._previous_block_goal_distance = distance
        goal_success = self._push_success(obs, distance)
        contact_now = self._robot_block_contact()
        contact_just_acquired = contact_now and not self._contact_acquired
        if contact_now:
            self._contact_acquired = True
        stage = int(self.config.push_curriculum_stage)
        curriculum_success = self._contact_acquired if stage == 0 else goal_success
        terminated = bool(
            (stage == 0 and contact_just_acquired)
            or (stage != 0 and goal_success)
        )
        potential = self._push_potential(
            geometry, contact_acquired=self._contact_acquired
        )
        previous_potential = self._previous_push_potential
        if previous_potential is None:
            previous_potential = potential
        next_potential = 0.0 if terminated else potential
        potential_shaping = (
            self.config.push_discount * next_potential - previous_potential
        )
        self._previous_push_potential = potential
        time_cost = (
            max(0.0, elapsed_delta_seconds)
            * self.config.push_time_penalty_per_second
        )
        contact_bonus = (
            self.config.push_contact_bonus
            if contact_just_acquired and stage == 2
            else 0.0
        )
        approach_completion_bonus = (
            self.config.push_approach_completion_bonus
            if contact_just_acquired and stage == 0
            else 0.0
        )
        goal_completion_bonus = (
            self.config.push_goal_completion_bonus
            if goal_success and stage != 0
            else 0.0
        )
        reward = (
            potential_shaping
            + contact_bonus
            + approach_completion_bonus
            + goal_completion_bonus
            - time_cost
        )
        info = self._push_observation_info(obs, progress=progress)
        info.update({
            "push_potential": float(potential),
            "potential_shaping": float(potential_shaping),
            "robot_block_distance": float(robot_block_distance),
            "robot_block_contact": float(contact_now),
            "contact_acquired": float(self._contact_acquired),
            "contact_bonus": float(contact_bonus),
            "approach_completion_bonus": float(approach_completion_bonus),
            "goal_completion_bonus": float(goal_completion_bonus),
            "time_cost": float(time_cost),
            "collision_penalty": 0.0,
            "action_change_penalty": 0.0,
            "push_success": float(goal_success),
            "curriculum_success": float(curriculum_success),
        })
        return float(reward), terminated, info

    def _get_info(self, obs_context: _ObsContext | None = None) -> dict:
        try:
            food_collected = self.rob.get_nr_food_collected()
        except Exception:
            food_collected = self._prev_food_count

        if obs_context is None:
            raw_irs = self._safe_read_irs()
            ir_arr = self.observation_adapter.normalize_ir(raw_irs)
        else:
            ir_arr = obs_context.irs
        front_ir = [ir_arr[i] for i in [2, 3, 4, 5, 7]]
        collision = bool(max(front_ir) >= self.config.collision_ir_threshold)

        info = {
            "food_collected": food_collected,
            "collision": collision,
            "step_count": self._step_count,
            "raw_ir": list(obs_context.raw_irs) if obs_context is not None else list(raw_irs),
            "normalized_ir": ir_arr.copy(),
            "calibration_profile": self.observation_adapter.profile.name,
            "blob_target_confidence": self._blob_target_confidence,
            "blob_target_switches": self._blob_target_switches,
            "phone_tilt": self._read_phone_tilt(),
        }
        if self.config.include_position_info:
            pos = self.rob.get_position()
            info.update({
                "x": pos.x,
                "y": pos.y,
                "z": pos.z,
            })
        return info
