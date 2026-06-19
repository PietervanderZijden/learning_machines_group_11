from __future__ import annotations

import math
import random
import time
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
    image_size: tuple[int, int] = (64, 64)
    max_wheel_speed: int = 100
    step_millis: int = 400
    phone_tilt: int = 100
    phone_tilt_speed: int = 100
    phone_tilt_tolerance: int = 5
    phone_tilt_timeout: float = 6.0
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
    smoothing_previous_weight: float = 0.65
    smoothing_requested_weight: float = 0.35
    max_action_delta: float = 0.5
    blob_track_max_distance: float = 0.30
    blob_track_max_missed: int = 6
    active_food_count: int | None = None


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

        obs_spaces = {
            "blob": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
            "ir": spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
        }
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
        self.action_executor = ActionExecutor(smoothing=smoothing)
        self._latest_ir = np.zeros(8, dtype=np.float32)
        self._latest_blob = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        self._elapsed_seconds = 0.0
        self._collision_count = 0
        self._safety_override_count = 0
        self._action_change_total = 0.0
        self._saturation_total = 0.0
        self._last_observation_wall_time = time.monotonic()
        self._observation_sim_seconds = 0.0
        self._tracked_blob: np.ndarray | None = None
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
        self._last_observation_wall_time = time.monotonic()
        info = self._get_info(obs_context)
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
            if not self._is_simulation and hasattr(self.rob, "move_blocking"):
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

        new_food = info["food_collected"]
        newly_collected = max(0, new_food - self._prev_food_count)
        self._prev_food_count = new_food

        terminated = new_food >= self._num_food
        observation_wall_time = time.monotonic()
        elapsed_delta_seconds = duration_s
        if not self._is_simulation:
            elapsed_delta_seconds = max(
                duration_s, observation_wall_time - self._last_observation_wall_time
            )
        self._last_observation_wall_time = observation_wall_time
        self._elapsed_seconds += elapsed_delta_seconds
        truncated = self._elapsed_seconds >= self.max_episode_seconds - 1e-9
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
        self._cache_initial_position()
        if self.config.randomize_food_positions:
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
            raise RuntimeError(
                f"phone tilt did not reach ground-facing target {self.config.phone_tilt}; "
                f"actual={actual}"
            )
        self._settle(self.config.reset_settle_time)

    def _reset_blob_tracker(self) -> None:
        self._tracked_blob = None
        self._blob_track_missed = 0
        self._blob_target_switches = 0
        self._blob_target_confidence = 0.0

    def _cache_initial_position(self) -> None:
        try:
            pos = self.rob.get_position()
            self._initial_pos_x = pos.x
            self._initial_pos_y = pos.y
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

        if self.config.detect_blob_from_camera and image_bgr is not None:
            blob_start = time.perf_counter()
            blob = self._detect_blob(image_bgr)
            timing["blob_detection"] = time.perf_counter() - blob_start
        else:
            blob = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
            timing["blob_detection"] = 0.0

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
