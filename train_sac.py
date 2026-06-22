"""
SAC training for the Robobo red-block push task.

Uses stable-baselines3 SAC with the deployable 18-value observation:
red_block[4] + green_goal[4] + IR[8] + previous_executed_action[2].

Usage:
    python train_sac.py
    python train_sac.py --total-timesteps 500000
    python train_sac.py --resume
    python train_sac.py --no-wandb
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import datetime
import zipfile
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _save_episode_atomic(path: Path, data: dict[str, np.ndarray]) -> None:
    """Write a complete episode without exposing a partial NPZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(path)


def _checkpoint_timesteps(path: Path) -> int:
    """Read SB3's persisted timestep counter without loading the model."""
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
        return int(data.get("num_timesteps", -1))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
        return -1


def find_sac_resume_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int]:
    """Return the valid SAC checkpoint with the greatest saved timestep."""
    candidates = list(checkpoint_dir.glob("sac_*_steps*.zip"))
    candidates.extend(
        path
        for path in (
            checkpoint_dir / "sac_latest.zip",
            checkpoint_dir / "sac_latest",
        )
        if path.exists()
    )
    ranked = [
        (steps, path)
        for path in candidates
        if (steps := _checkpoint_timesteps(path)) >= 0
    ]
    if not ranked:
        return None, -1
    steps, path = max(ranked, key=lambda item: (item[0], item[1].stat().st_mtime))
    return path, steps


def matching_sac_replay_buffer(checkpoint_dir: Path, checkpoint: Path) -> Path | None:
    if checkpoint.name.startswith("sac_") and "_steps" in checkpoint.name:
        numbered = checkpoint_dir / checkpoint.name.replace(
            "sac_", "sac_replay_buffer_", 1
        ).replace(".zip", ".pkl")
        if numbered.exists():
            return numbered
    standalone = checkpoint_dir / "replay_buffer.pkl"
    return standalone if standalone.exists() else None


def promote_sac_checkpoint(
    checkpoint_dir: Path,
    checkpoint: Path,
    replay_buffer: Path | None,
) -> None:
    """Repair the conventional latest files from a verified checkpoint pair."""
    latest = checkpoint_dir / "sac_latest.zip"
    if checkpoint.resolve() != latest.resolve():
        shutil.copy2(checkpoint, latest)
    if replay_buffer is not None:
        latest_buffer = checkpoint_dir / "replay_buffer.pkl"
        if replay_buffer.resolve() != latest_buffer.resolve():
            shutil.copy2(replay_buffer, latest_buffer)


class RoboboSACEnv(gym.Env):
    """Plain deployable SAC vector environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        rob=None,
        max_episode_steps=150,
        randomize_food_positions=False,
        domain_randomization=False,
        randomization_ranges=None,
        ir_noise_std=0.02,
        ir_noise_prob=0.5,
        pose_jitter=0.02,
        wheel_noise_std=0.03,
        wheel_noise_prob=0.3,
        time_penalty_per_second=0.5,
        collision_penalty=0.0,
        action_change_penalty=0.02,
        reward_scale=0.01,
        shaping_scale=5.0,
        emergency_override_penalty=1.0,
        calibration_path=None,
        curriculum=True,
        curriculum_one_food_steps=100_000,
        curriculum_three_food_steps=250_000,
        randomization_start_steps=300_000,
        image_size=64,
        record_dir="recorded_episodes",
        no_record=False,
    ):
        super().__init__()
        from learning_machines.rl_robobo_compact_env import (
            RoboboCompactEnv,
            RoboboCompactEnvConfig,
        )

        self._config = RoboboCompactEnvConfig(
            task="push",
            max_episode_steps=max_episode_steps,
            randomize_food_positions=randomize_food_positions,
            time_penalty_per_second=time_penalty_per_second,
            push_time_penalty_per_second=time_penalty_per_second,
            collision_penalty=collision_penalty,
            action_change_penalty=action_change_penalty,
            push_action_change_penalty=action_change_penalty,
            calibration_path=calibration_path,
            return_image=not no_record,
            image_obs_size=(image_size, image_size),
        )
        inner = RoboboCompactEnv(rob=rob, config=self._config)
        self._domain_wrapper = None
        if domain_randomization:
            from learning_machines.domain_randomization import DomainRandomizationWrapper
            self._domain_wrapper = DomainRandomizationWrapper(
                inner, enabled=not curriculum, ranges=randomization_ranges
            )
            inner = self._domain_wrapper
        self._inner = inner
        self.curriculum = bool(curriculum)
        self.curriculum_one_food_steps = int(curriculum_one_food_steps)
        self.curriculum_three_food_steps = int(curriculum_three_food_steps)
        self.randomization_start_steps = int(randomization_start_steps)
        self._previous_executed_action = np.zeros(2, dtype=np.float32)
        self.reward_scale = float(reward_scale)
        self.emergency_override_penalty = float(emergency_override_penalty)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(18,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        self._record = not no_record
        self._record_dir = Path(record_dir) if record_dir else None
        self._episode_count = 0
        if self._record and self._record_dir is not None:
            existing = sorted(
                self._record_dir.glob("episodes/ep_*.npz"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if existing:
                self._episode_count = int(existing[-1].stem.split("_")[-1]) + 1
        self._episode_images: list[np.ndarray] = []
        self._episode_actions: list[np.ndarray] = []
        self._episode_rewards: list[float] = []
        self._episode_dones: list[bool] = []
        self._episode_irs: list[np.ndarray] = []

    def _flatten_obs(self, obs_dict):
        red_block = obs_dict["red_block"]
        green_goal = obs_dict["green_goal"]
        ir = obs_dict["ir"]
        return np.concatenate(
            [red_block, green_goal, ir, self._previous_executed_action]
        ).astype(np.float32)

    def set_curriculum_step(self, step: int) -> dict[str, float]:
        if not self.curriculum:
            return {"push_task": 1.0, "randomization_enabled": 1.0}
        randomization_enabled = step >= self.randomization_start_steps
        if self._domain_wrapper is not None:
            self._domain_wrapper.enabled = randomization_enabled
        return {
            "push_task": 1.0,
            "randomization_enabled": float(randomization_enabled),
        }

    def _get_push_success(self, info):
        return float(info.get("push_success", 0.0))

    def _save_episode(self):
        if not self._record or self._record_dir is None or len(self._episode_images) < 2:
            return
        ep_data = {
            "images": np.array(self._episode_images, dtype=np.uint8),
            "actions": np.array(self._episode_actions, dtype=np.float32),
            "rewards": np.array(self._episode_rewards, dtype=np.float32),
            "dones": np.array(self._episode_dones, dtype=bool),
            "observation_contract": np.array("robobo-push-obs-v1"),
            "reward_contract": np.array("robobo-push-reward-v1"),
            "control_interval_seconds": np.array(0.4),
            "calibration_profile": np.array(
                self._inner.observation_adapter.profile.name
            ),
            "phone_tilt": np.array(self._inner.config.phone_tilt),
        }
        if self._episode_irs:
            ep_data["irs"] = np.array(self._episode_irs, dtype=np.float32)
        ep_path = self._record_dir / "episodes" / f"ep_{self._episode_count:06d}.npz"
        _save_episode_atomic(ep_path, ep_data)
        self._episode_count += 1

    def reset(self, *, seed=None, options=None):
        obs_dict, info = self._inner.reset(seed=seed, options=options)
        self._previous_executed_action.fill(0.0)
        if self._record:
            self._episode_images = []
            self._episode_actions = []
            self._episode_rewards = []
            self._episode_dones = []
            self._episode_irs = []
            if "image" in obs_dict:
                self._episode_images.append(obs_dict["image"].copy())
                self._episode_irs.append(obs_dict["ir"].astype(np.float32).copy())
        return self._flatten_obs(obs_dict), info

    def step(self, action):
        obs_dict, raw_reward, terminated, truncated, info = self._inner.step(action)
        executed_action = np.asarray(
            info.get("executed_action", action), dtype=np.float32
        ).copy()
        self._previous_executed_action = executed_action
        push_success = self._get_push_success(info)
        obs = self._flatten_obs(obs_dict)
        info = dict(info)
        info["raw_reward"] = float(raw_reward)
        info["push_success"] = push_success
        info.setdefault("success_reward", 0.0)
        info.setdefault("progress_reward", 0.0)
        info.setdefault(
            "time_penalty",
            self._config.time_penalty_per_second * self._config.step_millis / 1000.0,
        )
        info.setdefault("collision_penalty", 0.0)
        emergency_cost = (
            self.emergency_override_penalty
            if info.get("safety_override") == "emergency_reverse_turn"
            else 0.0
        )
        shaping_reward = 0.0
        unscaled_training_reward = float(raw_reward) - emergency_cost
        training_reward = self.reward_scale * unscaled_training_reward
        if self._record:
            self._episode_actions.append(executed_action)
            self._episode_rewards.append(float(raw_reward))
            self._episode_dones.append(bool(terminated or truncated))
            if "image" in obs_dict:
                self._episode_images.append(obs_dict["image"].copy())
                self._episode_irs.append(obs_dict["ir"].astype(np.float32).copy())
            if terminated or truncated:
                self._save_episode()
        info["shaping_reward"] = shaping_reward
        info["safety_override_penalty"] = emergency_cost
        info["unscaled_training_reward"] = unscaled_training_reward
        info["training_reward"] = training_reward
        return obs, training_reward, terminated, truncated, info

    def close(self):
        self._inner.close()


def main():
    parser = argparse.ArgumentParser(description="SAC for Robobo red-block pushing")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument(
        "--host",
        default=os.environ.get("COPPELIA_SIM_IP", "127.0.0.1"),
    )
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--image-size", type=int, default=64,
                        help="Image size for recorded episodes (default: 64x64)")
    parser.add_argument("--record-dir", type=str, default="recorded_episodes",
                        help="Directory to save image episodes for offline training")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable episode recording")
    parser.add_argument("--checkpoint-dir", type=str, default="sac_models")
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument(
        "--time-penalty-per-second",
        type=float,
        default=0.5,
        help="Elapsed-time penalty; 0.5/s equals 0.2 per 400 ms transition",
    )
    parser.add_argument("--collision-penalty", type=float, default=0.0,
                        help="Additional dense reward penalty when front IR indicates collision")
    parser.add_argument("--action-change-penalty", type=float, default=0.02)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--shaping-scale", type=float, default=5.0)
    parser.add_argument("--emergency-override-penalty", type=float, default=1.0)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--calibration", default="config/calibration/simulation.json")
    parser.add_argument(
        "--hardware-calibration",
        default=None,
        help="Optional measured hardware profile used to derive training randomization ranges",
    )
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable persistent sensor/actuator and per-step randomization",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--curriculum-one-food-steps", type=int, default=100_000)
    parser.add_argument("--curriculum-three-food-steps", type=int, default=250_000)
    parser.add_argument("--randomization-start-steps", type=int, default=300_000)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "learning_machines" / "src"))
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))

    os.environ["COPPELIA_SIM_PORT"] = str(args.port)
    os.environ["COPPELIA_SIM_IP"] = args.host
    print(
        f"SAC effective configuration: simulator={args.host}:{args.port}",
        flush=True,
    )
    from learning_machines.coppelia_startup import check_coppelia_service
    try:
        check_coppelia_service(args.host, args.port)
    except ConnectionError as exc:
        raise SystemExit(f"CoppeliaSim preflight failed: {exc}") from None

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_manifest_path = checkpoint_dir / "manifest.json"
    if existing_manifest_path.exists():
        from learning_machines.transfer import CheckpointManifest

        existing_manifest = CheckpointManifest.load(existing_manifest_path)
        existing_dim = existing_manifest.algorithm_config.get("observation_dim")
        if existing_dim != 18:
            raise ValueError(
                f"{checkpoint_dir} contains a legacy {existing_dim}-value SAC "
                "run. Use a fresh checkpoint directory for the 18-value "
                "push observation contract."
            )
        if not args.resume and any(checkpoint_dir.glob("sac*.zip")):
            raise ValueError(
                f"{checkpoint_dir} already contains SAC checkpoints. "
                "Pass --resume or choose a fresh checkpoint directory."
            )

    log_dir = checkpoint_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.wandb_run_name or f"sac-transfer-{datetime.datetime.now().strftime('%m%d-%H%M%S')}"

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_id_path = checkpoint_dir / "wandb_run_id.txt"
        wandb_id = wandb_id_path.read_text().strip() if args.resume and wandb_id_path.exists() else None
        if wandb_id:
            print(f"Resuming wandb run: {wandb_id}")
            wandb_run = wandb.init(
                project="learning-machines",
                entity="Learningmachine",
                id=wandb_id,
                resume="must",
                config=vars(args),
                sync_tensorboard=False,
            )
        else:
            wandb_run = wandb.init(
                project="learning-machines",
                entity="Learningmachine",
                name=run_name,
                config=vars(args),
                sync_tensorboard=False,
                resume="allow",
            )
            wandb_id_path.write_text(wandb_run.id)
            print(f"New wandb run: {wandb_run.id}")
        wandb_run.config.update({
            "observation_contract": "robobo-push-obs-v1",
            "reward_contract": "robobo-push-reward-v1",
            "control_interval_seconds": 0.4,
            "phone_tilt": 100,
            "task": "push",
        }, allow_val_change=True)

    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from learning_machines.sac import StabilizedSAC
    from learning_machines.domain_randomization import RandomizationRanges
    from learning_machines.transfer import CalibrationProfile, CheckpointManifest

    randomization_ranges = None
    if args.hardware_calibration:
        randomization_ranges = RandomizationRanges.from_calibration_profiles(
            CalibrationProfile.load(args.calibration),
            CalibrationProfile.load(args.hardware_calibration),
        )
        if wandb_run is not None:
            wandb_run.config.update({
                "derived_randomization_ranges": randomization_ranges.__dict__,
            }, allow_val_change=True)

    class TransferMetricsCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_food = deque(maxlen=100)
            self.best_food_mean = float("-inf")
            best_path = checkpoint_dir / "best_metrics.json"
            if best_path.exists():
                try:
                    data = json.loads(best_path.read_text())
                    self.best_food_mean = float(
                        data.get("rolling_success_rate", data.get("rolling_food_mean"))
                    )
                except (
                    OSError,
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    pass

        def _on_step(self) -> bool:
            curriculum_metrics = {}
            base_env = self.training_env.envs[0].unwrapped
            if hasattr(base_env, "set_curriculum_step"):
                curriculum_metrics = base_env.set_curriculum_step(
                    self.num_timesteps
                )
            infos = self.locals.get("infos", [])
            if not infos:
                return True
            info = infos[0]
            dones = self.locals.get("dones", [])
            if len(dones) and bool(dones[0]):
                self.episode_food.append(float(info.get("push_success", 0.0)))
                if len(self.episode_food) >= 20:
                    rolling_success = float(np.mean(self.episode_food))
                    if rolling_success > self.best_food_mean:
                        self.best_food_mean = rolling_success
                        self.model.save(str(checkpoint_dir / "sac_best"))
                        (checkpoint_dir / "best_metrics.json").write_text(
                            json.dumps(
                                {
                                    "global_step": self.num_timesteps,
                                    "rolling_episodes": len(self.episode_food),
                                    "rolling_success_rate": rolling_success,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
            metrics = {
                "rollout/elapsed_seconds": info.get("elapsed_seconds"),
                "rollout/push_success": info.get("push_success"),
                "rollout/block_goal_distance": info.get("block_goal_distance"),
                "rollout/block_goal_progress": info.get("block_goal_progress"),
                "rollout/red_block_visible": info.get("red_block_visible"),
                "rollout/green_goal_visible": info.get("green_goal_visible"),
                "rollout/collisions": info.get("collisions"),
                "rollout/safety_overrides": info.get("safety_overrides"),
                "rollout/action_change": info.get("action_change"),
                "rollout/action_saturation": info.get("action_saturation"),
                "rollout/success_reward": info.get("success_reward"),
                "rollout/progress_reward": info.get("progress_reward"),
                "rollout/time_penalty": info.get("time_penalty"),
                "rollout/collision_penalty": info.get("collision_penalty"),
                "rollout/action_change_penalty": info.get("action_change_penalty"),
                "rollout/shaping_reward": info.get("shaping_reward"),
                "rollout/safety_override_penalty": info.get("safety_override_penalty"),
                "rollout/unscaled_training_reward": info.get("unscaled_training_reward"),
                "rollout/training_reward": info.get("training_reward"),
                "rollout/safety_with_visible_block": info.get(
                    "safety_with_visible_block"
                ),
                "curriculum/push_task": curriculum_metrics.get("push_task"),
                "curriculum/randomization_enabled": curriculum_metrics.get(
                    "randomization_enabled"
                ),
                "rollout/success_rate_100": (
                    float(np.mean(self.episode_food))
                    if self.episode_food else None
                ),
            }
            metrics = {key: float(value) for key, value in metrics.items() if value is not None}
            for key, value in metrics.items():
                self.logger.record(key, value)
            if wandb_run is not None and self.num_timesteps % 20 == 0:
                payload = dict(metrics)
                payload["global_step"] = self.num_timesteps
                for key, value in self.model.logger.name_to_value.items():
                    if key.startswith("train/") and isinstance(value, (int, float, np.number)):
                        payload[key] = float(value)
                executed = info.get("executed_action")
                requested = info.get("policy_requested_action", info.get("requested_action"))
                if executed is not None:
                    payload["actions/executed_left"] = float(executed[0])
                    payload["actions/executed_right"] = float(executed[1])
                if requested is not None:
                    payload["actions/requested_left"] = float(requested[0])
                    payload["actions/requested_right"] = float(requested[1])
                observation = self.locals.get("new_obs")
                if observation is not None:
                    vector = np.asarray(observation)[0]
                    payload.update({
                        "observations/red_block_x": float(vector[0]),
                        "observations/red_block_y": float(vector[1]),
                        "observations/red_block_area": float(vector[2]),
                        "observations/red_block_found": float(vector[3]),
                        "observations/green_goal_x": float(vector[4]),
                        "observations/green_goal_y": float(vector[5]),
                        "observations/green_goal_area": float(vector[6]),
                        "observations/green_goal_found": float(vector[7]),
                        "observations/previous_executed_left": float(vector[16]),
                        "observations/previous_executed_right": float(vector[17]),
                    })
                    if self.num_timesteps % 100 == 0:
                        import wandb
                        payload["observations/ir_histogram"] = wandb.Histogram(
                            vector[8:16]
                        )
                        if "raw_ir" in info:
                            payload["observations/raw_ir_histogram"] = wandb.Histogram(
                                info["raw_ir"]
                            )
                wandb_run.log(payload)
            return True

    model_path = checkpoint_dir / "sac_latest"
    buffer_path = checkpoint_dir / "replay_buffer.pkl"

    env = Monitor(
        RoboboSACEnv(
            max_episode_steps=args.max_episode_steps,
            randomize_food_positions=False,
            domain_randomization=args.domain_randomization,
            randomization_ranges=randomization_ranges,
            time_penalty_per_second=args.time_penalty_per_second,
            collision_penalty=args.collision_penalty,
            action_change_penalty=args.action_change_penalty,
            reward_scale=args.reward_scale,
            shaping_scale=args.shaping_scale,
            emergency_override_penalty=args.emergency_override_penalty,
            calibration_path=args.calibration,
            curriculum=args.curriculum,
            curriculum_one_food_steps=args.curriculum_one_food_steps,
            curriculum_three_food_steps=args.curriculum_three_food_steps,
            randomization_start_steps=args.randomization_start_steps,
            image_size=args.image_size,
            record_dir=args.record_dir,
            no_record=args.no_record,
        ),
        filename=str(log_dir / "monitor.csv"),
        info_keywords=(
            "raw_reward",
            "push_success",
            "block_goal_distance",
            "block_goal_progress",
            "success_reward",
            "progress_reward",
            "red_block_visible",
            "green_goal_visible",
            "time_penalty",
            "collision_penalty",
            "action_change_penalty",
            "shaping_reward",
            "safety_override_penalty",
            "unscaled_training_reward",
            "training_reward",
            "elapsed_seconds",
            "collisions",
            "safety_overrides",
            "mean_action_change",
            "action_saturation_rate",
            "safety_with_visible_block",
        ),
    )

    callbacks = []
    callbacks.append(TransferMetricsCallback())
    if wandb_run is not None:
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("*", step_metric="global_step")

    resume_path, resume_steps = (
        find_sac_resume_checkpoint(checkpoint_dir)
        if args.resume
        else (None, -1)
    )
    if resume_path is not None:
        manifest_path = checkpoint_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(
                "refusing to resume a SAC checkpoint without a versioned manifest"
            )
        manifest = CheckpointManifest.load(manifest_path)
        calibration_name = CalibrationProfile.load(args.calibration).name
        manifest_mismatches = {
            "algorithm": (manifest.algorithm, "sac"),
            "calibration_profile": (manifest.calibration_profile, calibration_name),
            "image_size": (manifest.image_size, 64),
            "phone_tilt": (manifest.phone_tilt, 100),
            "observation_contract": (
                manifest.observation_contract,
                "robobo-push-obs-v1",
            ),
            "reward_contract": (
                manifest.reward_contract,
                "robobo-push-reward-v1",
            ),
        }
        manifest_mismatches = {
            key: values
            for key, values in manifest_mismatches.items()
            if values[0] != values[1]
        }
        if manifest_mismatches:
            raise ValueError(
                f"checkpoint SAC push manifest is incompatible: {manifest_mismatches}"
            )
        required = {
            "observation_dim": 18,
            "task": "push",
            "reward_scale": args.reward_scale,
            "shaping_scale": args.shaping_scale,
            "emergency_override_penalty": args.emergency_override_penalty,
            "entropy_coefficient": args.entropy_coefficient,
            "max_grad_norm": args.max_grad_norm,
            "curriculum": args.curriculum,
        }
        if manifest.reward_contract != "robobo-push-reward-v1":
            raise ValueError(
                f"checkpoint reward contract is {manifest.reward_contract}, "
                "expected robobo-push-reward-v1; start a fresh run"
            )
        mismatches = {
            key: (manifest.algorithm_config.get(key), expected)
            for key, expected in required.items()
            if manifest.algorithm_config.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"checkpoint SAC stability configuration is incompatible: {mismatches}"
            )
        print(f"Resuming from {resume_path} at {resume_steps:,} timesteps")
        model = StabilizedSAC.load(str(resume_path), env=env, device="auto")
        replay_path = matching_sac_replay_buffer(checkpoint_dir, resume_path)
        if replay_path is not None:
            print(f"Loading replay buffer from {replay_path}")
            model.load_replay_buffer(str(replay_path))
            print(f"Buffer size: {model.replay_buffer.size():,}")
        else:
            print("No replay buffer found, starting with empty buffer")
        latest_steps = _checkpoint_timesteps(checkpoint_dir / "sac_latest.zip")
        if resume_steps > latest_steps:
            promote_sac_checkpoint(checkpoint_dir, resume_path, replay_path)
            print(
                f"Promoted {resume_steps:,}-step checkpoint to sac_latest"
            )
    else:
        print("Starting fresh training run")
        model = StabilizedSAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.9801,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef=args.entropy_coefficient,
            target_update_interval=1,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], qf=[256, 256]),
            ),
            verbose=1,
            tensorboard_log=str(log_dir),
            device="auto",
        )

    remaining = args.total_timesteps - model.num_timesteps
    if remaining <= 0:
        print(f"Already trained {model.num_timesteps:,} steps. Nothing to do.")
        env.close()
        return

    print(f"Training for {remaining:,} more steps ({model.num_timesteps:,} done)")

    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(checkpoint_dir),
        name_prefix="sac",
        save_replay_buffer=True,
    )
    callbacks.append(checkpoint_cb)

    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving...")
    finally:
        _, existing_steps = find_sac_resume_checkpoint(checkpoint_dir)
        if model.num_timesteps >= existing_steps:
            model.save(str(model_path))
            model.save_replay_buffer(str(buffer_path))
            print(f"Saved model to {model_path}")
            print(f"Saved buffer to {buffer_path}")
        else:
            recovery_path = checkpoint_dir / f"sac_recovery_{model.num_timesteps}_steps"
            recovery_buffer = (
                checkpoint_dir
                / f"sac_recovery_replay_buffer_{model.num_timesteps}_steps.pkl"
            )
            model.save(str(recovery_path))
            model.save_replay_buffer(str(recovery_buffer))
            print(
                f"Preserved newer {existing_steps:,}-step checkpoint; "
                f"saved this run to {recovery_path}"
            )
        calibration_name = CalibrationProfile.load(args.calibration).name
        CheckpointManifest(
            algorithm="sac",
            calibration_profile=calibration_name,
            image_size=64,
            observation_contract="robobo-push-obs-v1",
            reward_contract="robobo-push-reward-v1",
            algorithm_config={
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "buffer_size": args.buffer_size,
                "observation_dim": 18,
                "task": "push",
                "domain_randomization": args.domain_randomization,
                "hardware_calibration": args.hardware_calibration,
                "time_penalty_per_second": args.time_penalty_per_second,
                "collision_penalty": args.collision_penalty,
                "action_change_penalty": args.action_change_penalty,
                "reward_scale": args.reward_scale,
                "shaping_scale": args.shaping_scale,
                "emergency_override_penalty": args.emergency_override_penalty,
                "entropy_coefficient": args.entropy_coefficient,
                "max_grad_norm": args.max_grad_norm,
                "gamma": 0.9801,
                "curriculum": args.curriculum,
                "curriculum_one_food_steps": args.curriculum_one_food_steps,
                "curriculum_three_food_steps": args.curriculum_three_food_steps,
                "randomization_start_steps": args.randomization_start_steps,
            },
        ).save(checkpoint_dir / "manifest.json")
        if wandb_run is not None:
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    main()
