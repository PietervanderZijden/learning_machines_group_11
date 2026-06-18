"""
SAC training for Robobo food collection.

Uses stable-baselines3 SAC with the deployable 12-value blob+IR observation.
Food count is reward/evaluation metadata and is never a policy input.

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
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


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
        randomize_food_positions=True,
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
    ):
        super().__init__()
        from learning_machines.rl_robobo_compact_env import (
            RoboboCompactEnv,
            RoboboCompactEnvConfig,
        )

        self._config = RoboboCompactEnvConfig(
            max_episode_steps=max_episode_steps,
            randomize_food_positions=randomize_food_positions,
            time_penalty_per_second=time_penalty_per_second,
            collision_penalty=collision_penalty,
            action_change_penalty=action_change_penalty,
            calibration_path=calibration_path,
        )
        inner = RoboboCompactEnv(rob=rob, config=self._config)
        if domain_randomization:
            from learning_machines.domain_randomization import DomainRandomizationWrapper
            inner = DomainRandomizationWrapper(
                inner, enabled=True, ranges=randomization_ranges
            )
        self._inner = inner
        self.reward_scale = float(reward_scale)
        self.shaping_scale = float(shaping_scale)
        self.emergency_override_penalty = float(emergency_override_penalty)
        self._previous_potential = 0.0
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

    def _flatten_obs(self, obs_dict):
        blob = obs_dict["blob"]
        ir = obs_dict["ir"]
        return np.concatenate([blob, ir]).astype(np.float32)

    def _get_food_count(self, info):
        return float(info.get("food_collected", 0))

    def reset(self, *, seed=None, options=None):
        obs_dict, info = self._inner.reset(seed=seed, options=options)
        from learning_machines.transfer import blob_progress_potential
        self._previous_potential = blob_progress_potential(obs_dict["blob"])
        return self._flatten_obs(obs_dict), info

    def step(self, action):
        from learning_machines.transfer import blob_progress_potential
        obs_dict, raw_reward, terminated, truncated, info = self._inner.step(action)
        food = self._get_food_count(info)
        obs = self._flatten_obs(obs_dict)
        info = dict(info)
        info["raw_reward"] = float(raw_reward)
        info["food_collected"] = food
        info.setdefault("newly_collected", 0)
        info.setdefault("collect_reward", 0.0)
        info.setdefault("speed_bonus", 0.0)
        info.setdefault(
            "time_penalty",
            self._config.time_penalty_per_second * self._config.step_millis / 1000.0,
        )
        info.setdefault("collision_penalty", 0.0)
        next_potential = blob_progress_potential(obs_dict["blob"])
        if info["newly_collected"] > 0:
            shaping_reward = 0.0
        else:
            shaping_reward = self.shaping_scale * (
                0.9801 * next_potential - self._previous_potential
            )
        self._previous_potential = next_potential
        emergency_cost = (
            self.emergency_override_penalty
            if info.get("safety_override") == "emergency_reverse_turn"
            else 0.0
        )
        unscaled_training_reward = (
            float(raw_reward) + shaping_reward - emergency_cost
        )
        training_reward = self.reward_scale * unscaled_training_reward
        info["shaping_reward"] = shaping_reward
        info["safety_override_penalty"] = emergency_cost
        info["unscaled_training_reward"] = unscaled_training_reward
        info["training_reward"] = training_reward
        return obs, training_reward, terminated, truncated, info

    def close(self):
        self._inner.close()


def main():
    parser = argparse.ArgumentParser(description="SAC for Robobo food collection")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--max-episode-steps", type=int, default=150)
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
    parser.add_argument("--entropy-coefficient", type=float, default=0.1)
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
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "learning_machines" / "src"))
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))

    os.environ["COPPELIA_SIM_PORT"] = str(args.port)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
            "observation_contract": "robobo-obs-v2",
            "reward_contract": "robobo-reward-v4",
            "control_interval_seconds": 0.4,
            "phone_tilt": 100,
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
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if not infos:
                return True
            info = infos[0]
            metrics = {
                "rollout/elapsed_seconds": info.get("elapsed_seconds"),
                "rollout/food_collected": info.get("food_collected"),
                "rollout/food_per_minute": info.get("food_per_minute"),
                "rollout/collisions": info.get("collisions"),
                "rollout/safety_overrides": info.get("safety_overrides"),
                "rollout/action_change": info.get("action_change"),
                "rollout/action_saturation": info.get("action_saturation"),
                "rollout/time_penalty": info.get("time_penalty"),
                "rollout/collision_penalty": info.get("collision_penalty"),
                "rollout/action_change_penalty": info.get("action_change_penalty"),
                "rollout/shaping_reward": info.get("shaping_reward"),
                "rollout/safety_override_penalty": info.get("safety_override_penalty"),
                "rollout/unscaled_training_reward": info.get("unscaled_training_reward"),
                "rollout/training_reward": info.get("training_reward"),
                "rollout/blob_target_confidence": info.get("blob_target_confidence"),
                "rollout/blob_target_switches": info.get("blob_target_switches"),
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
                        "observations/blob_x": float(vector[0]),
                        "observations/blob_y": float(vector[1]),
                        "observations/blob_area": float(vector[2]),
                        "observations/blob_found": float(vector[3]),
                    })
                    if self.num_timesteps % 100 == 0:
                        import wandb
                        payload["observations/ir_histogram"] = wandb.Histogram(vector[4:])
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
            randomize_food_positions=True,
            domain_randomization=args.domain_randomization,
            randomization_ranges=randomization_ranges,
            time_penalty_per_second=args.time_penalty_per_second,
            collision_penalty=args.collision_penalty,
            action_change_penalty=args.action_change_penalty,
            reward_scale=args.reward_scale,
            shaping_scale=args.shaping_scale,
            emergency_override_penalty=args.emergency_override_penalty,
            calibration_path=args.calibration,
        ),
        filename=str(log_dir / "monitor.csv"),
        info_keywords=(
            "food_collected",
            "raw_reward",
            "newly_collected",
            "collect_reward",
            "speed_bonus",
            "time_penalty",
            "collision_penalty",
            "action_change_penalty",
            "shaping_reward",
            "safety_override_penalty",
            "unscaled_training_reward",
            "training_reward",
            "elapsed_seconds",
            "food_per_minute",
            "collisions",
            "safety_overrides",
            "mean_action_change",
            "action_saturation_rate",
            "blob_target_confidence",
            "blob_target_switches",
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
        manifest.validate(
            "sac",
            CalibrationProfile.load(args.calibration).name,
            64,
            100,
        )
        required = {
            "reward_scale": args.reward_scale,
            "shaping_scale": args.shaping_scale,
            "emergency_override_penalty": args.emergency_override_penalty,
            "entropy_coefficient": args.entropy_coefficient,
            "max_grad_norm": args.max_grad_norm,
        }
        if manifest.reward_contract != "robobo-reward-v4":
            raise ValueError(
                f"checkpoint reward contract is {manifest.reward_contract}, "
                "expected robobo-reward-v4; start a fresh run"
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
            algorithm_config={
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "buffer_size": args.buffer_size,
                "observation_dim": 12,
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
            },
        ).save(checkpoint_dir / "manifest.json")
        if wandb_run is not None:
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    main()
