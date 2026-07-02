'SAC training for the Robobo red-block push task.'
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

SAC_GAMMA = 0.9801
PUSH_REWARD_CONTRACT = "robobo-push-phased-dense-v1"
PUSH_TIME_PENALTY_PER_SECOND = 0.05
PUSH_APPROACH_POTENTIAL_SCALE = 2.0
PUSH_GOAL_POTENTIAL_OFFSET = 2.0
PUSH_GOAL_POTENTIAL_SCALE = 4.0
PUSH_CONTACT_BONUS = 1.0
PUSH_APPROACH_COMPLETION_BONUS = 5.0
PUSH_GOAL_COMPLETION_BONUS = 15.0


def _save_episode_atomic(path: Path, data: dict[str, np.ndarray]) -> None:
    'Write a complete episode without exposing a partial NPZ file.'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(path)


def _checkpoint_timesteps(path: Path) -> int:
    "Read SB3's persisted timestep counter without loading the model."
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
        return int(data.get("num_timesteps", -1))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
        return -1


def _sac_checkpoint_data(path: Path) -> dict:
    'Read SB3 metadata used to validate manifest recovery.'
    try:
        with zipfile.ZipFile(path) as archive:
            return json.loads(archive.read("data"))
    except (
        OSError,
        KeyError,
        ValueError,
        zipfile.BadZipFile,
        json.JSONDecodeError,
    ) as exc:
        raise ValueError(f"cannot inspect SAC checkpoint {path}: {exc}") from exc


def validate_sac_manifest_recovery(
    checkpoint: Path,
    curriculum_state_path: Path,
    args,
) -> dict:
    'Validate a phased-dense SAC run before reconstructing a manifest.'
    if not curriculum_state_path.exists():
        raise ValueError(
            "refusing to recover a SAC manifest without curriculum_state.json; "
            "the phased dense reward contract cannot be established safely"
        )
    curriculum_state = json.loads(curriculum_state_path.read_text())
    if int(curriculum_state.get("version", 1)) != 2:
        raise ValueError(
            "cannot recover SAC manifest from an incompatible curriculum "
            "state; start a fresh run"
        )
    curriculum_config = curriculum_state.get("config", {})
    expected_curriculum = {
        "enabled": args.curriculum,
        "success_threshold": args.curriculum_success_threshold,
        "window": args.curriculum_window,
        "min_stage_steps": args.curriculum_min_stage_steps,
        "goal_jitter_radius": args.curriculum_goal_jitter_radius,
    }
    curriculum_mismatches = {
        key: (curriculum_config.get(key), expected)
        for key, expected in expected_curriculum.items()
        if curriculum_config.get(key) != expected
    }
    if curriculum_mismatches:
        raise ValueError(
            "cannot recover SAC manifest because curriculum options differ "
            f"from persisted state: {curriculum_mismatches}"
        )

    data = _sac_checkpoint_data(checkpoint)
    observation_shape = tuple(data.get("observation_space", {}).get("_shape", ()))
    action_shape = tuple(data.get("action_space", {}).get("_shape", ()))
    expected = {
        "observation_shape": (18,),
        "action_shape": (2,),
        "gamma": SAC_GAMMA,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "ent_coef": args.entropy_coefficient,
        "max_grad_norm": args.max_grad_norm,
    }
    actual = {
        "observation_shape": observation_shape,
        "action_shape": action_shape,
        "gamma": data.get("gamma"),
        "learning_rate": data.get("learning_rate"),
        "batch_size": data.get("batch_size"),
        "buffer_size": data.get("buffer_size"),
        "learning_starts": data.get("learning_starts"),
        "ent_coef": data.get("ent_coef"),
        "max_grad_norm": data.get("max_grad_norm"),
    }
    mismatches = {
        key: (actual[key], expected_value)
        for key, expected_value in expected.items()
        if actual[key] != expected_value
    }
    if mismatches:
        raise ValueError(
            "cannot recover SAC manifest because checkpoint configuration is "
            f"incompatible: {mismatches}"
        )
    return curriculum_state


def find_sac_resume_checkpoint(checkpoint_dir: Path) -> tuple[Path | None, int]:
    'Return the valid SAC checkpoint with the greatest saved timestep.'
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
    'Repair the conventional latest files from a verified checkpoint pair.'
    latest = checkpoint_dir / "sac_latest.zip"
    if checkpoint.resolve() != latest.resolve():
        shutil.copy2(checkpoint, latest)
    if replay_buffer is not None:
        latest_buffer = checkpoint_dir / "replay_buffer.pkl"
        if replay_buffer.resolve() != latest_buffer.resolve():
            shutil.copy2(replay_buffer, latest_buffer)


def build_sac_manifest(
    manifest_class,
    calibration_name: str,
    args,
    curriculum_stage: int,
    curriculum_stage_name: str,
):
    'Create the canonical manifest for fresh, resumed, and recovered runs.'
    return manifest_class(
        algorithm="sac",
        calibration_profile=calibration_name,
        image_size=64,
        observation_contract="robobo-push-obs-v1",
        reward_contract=PUSH_REWARD_CONTRACT,
        algorithm_config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "buffer_size": args.buffer_size,
            "observation_dim": 18,
            "task": "push",
            "domain_randomization": args.domain_randomization,
            "hardware_calibration": args.hardware_calibration,
            "entropy_coefficient": args.entropy_coefficient,
            "max_grad_norm": args.max_grad_norm,
            "gamma": SAC_GAMMA,
            "max_episode_steps": args.max_episode_steps,
            "curriculum": args.curriculum,
            "curriculum_stage": curriculum_stage,
            "curriculum_stage_name": curriculum_stage_name,
            "curriculum_success_threshold": args.curriculum_success_threshold,
            "curriculum_window": args.curriculum_window,
            "curriculum_min_stage_steps": args.curriculum_min_stage_steps,
            "curriculum_goal_jitter_radius": args.curriculum_goal_jitter_radius,
            "reward_contract": PUSH_REWARD_CONTRACT,
            "push_time_penalty_per_second": PUSH_TIME_PENALTY_PER_SECOND,
            "push_approach_potential_scale": PUSH_APPROACH_POTENTIAL_SCALE,
            "push_goal_potential_offset": PUSH_GOAL_POTENTIAL_OFFSET,
            "push_goal_potential_scale": PUSH_GOAL_POTENTIAL_SCALE,
            "push_contact_bonus": PUSH_CONTACT_BONUS,
            "push_approach_completion_bonus": PUSH_APPROACH_COMPLETION_BONUS,
            "push_goal_completion_bonus": PUSH_GOAL_COMPLETION_BONUS,
        },
    )


class RoboboSACEnv(gym.Env):
    'Plain deployable SAC vector environment.'

    metadata = {"render_modes": []}

    def __init__(
        self,
        rob=None,
        max_episode_steps=200,
        randomize_food_positions=False,
        randomize_push_layout=True,
        domain_randomization=False,
        randomization_ranges=None,
        ir_noise_std=0.02,
        ir_noise_prob=0.5,
        pose_jitter=0.02,
        wheel_noise_std=0.03,
        wheel_noise_prob=0.3,
        time_penalty_per_second=PUSH_TIME_PENALTY_PER_SECOND,
        collision_penalty=0.0,
        action_change_penalty=0.0,
        emergency_override_penalty=0.0,
        calibration_path=None,
        curriculum=True,
        curriculum_start_stage=0,
        curriculum_success_threshold=0.80,
        curriculum_window=100,
        curriculum_min_stage_steps=20_000,
        curriculum_goal_jitter_radius=0.20,
        curriculum_state_path=None,
        image_size=64,
        record_dir="recorded_episodes",
        no_record=False,
    ):
        super().__init__()
        from learning_machines.rl_robobo_compact_env import (
            RoboboCompactEnv,
            RoboboCompactEnvConfig,
        )
        from learning_machines.push_curriculum import (
            PushCurriculumConfig,
            PushCurriculumController,
        )

        curriculum_config = PushCurriculumConfig(
            enabled=curriculum,
            start_stage=curriculum_start_stage,
            success_threshold=curriculum_success_threshold,
            window=curriculum_window,
            min_stage_steps=curriculum_min_stage_steps,
            goal_jitter_radius=curriculum_goal_jitter_radius,
        )
        self._curriculum_state_path = (
            Path(curriculum_state_path) if curriculum_state_path else None
        )
        if self._curriculum_state_path is not None and self._curriculum_state_path.exists():
            self.curriculum_controller = PushCurriculumController.load(
                self._curriculum_state_path, curriculum_config
            )
        else:
            self.curriculum_controller = PushCurriculumController(curriculum_config)

        self._config = RoboboCompactEnvConfig(
            task="push",
            max_episode_steps=max_episode_steps,
            randomize_food_positions=randomize_food_positions,
            randomize_push_layout=randomize_push_layout,
            push_curriculum_stage=self.curriculum_controller.stage,
            push_goal_jitter_radius=curriculum_goal_jitter_radius,
            push_discount=SAC_GAMMA,
            push_time_penalty_per_second=time_penalty_per_second,
            push_approach_potential_scale=PUSH_APPROACH_POTENTIAL_SCALE,
            push_goal_potential_offset=PUSH_GOAL_POTENTIAL_OFFSET,
            push_goal_potential_scale=PUSH_GOAL_POTENTIAL_SCALE,
            push_contact_bonus=PUSH_CONTACT_BONUS,
            push_approach_completion_bonus=PUSH_APPROACH_COMPLETION_BONUS,
            push_goal_completion_bonus=PUSH_GOAL_COMPLETION_BONUS,
            calibration_path=calibration_path,
            return_image=not no_record,
            image_obs_size=(image_size, image_size),
        )
        inner = RoboboCompactEnv(rob=rob, config=self._config)
        self._base_env = inner
        self._domain_wrapper = None
        if domain_randomization:
            from learning_machines.domain_randomization import DomainRandomizationWrapper
            self._domain_wrapper = DomainRandomizationWrapper(
                inner,
                enabled=self.curriculum_controller.stage == 2,
                ranges=randomization_ranges,
            )
            inner = self._domain_wrapper
        self._inner = inner
        self._domain_randomization_configured = bool(domain_randomization)
        self._previous_executed_action = np.zeros(2, dtype=np.float32)
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

    def _apply_curriculum(self) -> dict:
        stage = self.curriculum_controller.stage
        self._config.push_curriculum_stage = stage
        randomization_enabled = self._domain_randomization_configured and stage == 2
        if self._domain_wrapper is not None:
            self._domain_wrapper.enabled = randomization_enabled
        metrics = self.curriculum_controller.metrics()
        metrics["randomization_enabled"] = float(randomization_enabled)
        return metrics

    def _get_curriculum_success(self, info):
        return float(info.get("curriculum_success", 0.0))

    def _save_episode(self):
        if not self._record or self._record_dir is None or len(self._episode_images) < 2:
            return
        ep_data = {
            "images": np.array(self._episode_images, dtype=np.uint8),
            "actions": np.array(self._episode_actions, dtype=np.float32),
            "rewards": np.array(self._episode_rewards, dtype=np.float32),
            "dones": np.array(self._episode_dones, dtype=bool),
            "observation_contract": np.array("robobo-push-obs-v1"),
            "reward_contract": np.array(PUSH_REWARD_CONTRACT),
            "curriculum_stage": np.array(self._episode_curriculum_stage),
            "curriculum_stage_name": np.array(
                ("approach", "push", "full")[self._episode_curriculum_stage]
            ),
            "push_layout_mode": np.array(
                getattr(self._base_env, "_push_layout_mode", "full")
            ),
            "control_interval_seconds": np.array(0.4),
            "calibration_profile": np.array(
                self._base_env.observation_adapter.profile.name
            ),
            "phone_tilt": np.array(self._base_env.config.phone_tilt),
        }
        if self._episode_irs:
            ep_data["irs"] = np.array(self._episode_irs, dtype=np.float32)
        ep_path = self._record_dir / "episodes" / f"ep_{self._episode_count:06d}.npz"
        _save_episode_atomic(ep_path, ep_data)
        self._episode_count += 1

    def reset(self, *, seed=None, options=None):
        self._apply_curriculum()
        self._episode_curriculum_stage = self.curriculum_controller.stage
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
        curriculum_success = self._get_curriculum_success(info)
        obs = self._flatten_obs(obs_dict)
        info = dict(info)
        info["raw_reward"] = float(raw_reward)
        info["curriculum_success"] = curriculum_success
        info.setdefault("push_potential", 0.0)
        info.setdefault("potential_shaping", 0.0)
        info.setdefault("robot_block_distance", float("nan"))
        info.setdefault(
            "time_cost",
            0.0,
        )
        info.setdefault("collision_penalty", 0.0)
        training_reward = float(raw_reward)
        self.curriculum_controller.record_transition()
        promotion = None
        episode_stage = self._episode_curriculum_stage
        if terminated or truncated:
            promotion = self.curriculum_controller.record_episode(
                bool(curriculum_success)
            )
            if promotion is not None:
                stage_steps = float(promotion["stage_steps"])
                stage_episodes = float(promotion["episodes"])
                stage_success = float(promotion["success_rate"])
            else:
                stage_steps = float(self.curriculum_controller.stage_steps)
                stage_episodes = float(len(self.curriculum_controller.recent_outcomes))
                stage_success = self.curriculum_controller.rolling_success
            if promotion is not None:
                self._apply_curriculum()
            if self._curriculum_state_path is not None:
                self.curriculum_controller.save(self._curriculum_state_path)
        if self._record:
            self._episode_actions.append(executed_action)
            self._episode_rewards.append(float(raw_reward))
            self._episode_dones.append(bool(terminated or truncated))
            if "image" in obs_dict:
                self._episode_images.append(obs_dict["image"].copy())
                self._episode_irs.append(obs_dict["ir"].astype(np.float32).copy())
            if terminated or truncated:
                self._save_episode()
        info["training_reward"] = training_reward
        info.update({
            "curriculum_stage": float(self.curriculum_controller.stage),
            "curriculum_stage_name": self.curriculum_controller.stage_name,
            "episode_curriculum_stage": float(episode_stage),
            "curriculum_stage_steps": (
                stage_steps if terminated or truncated
                else float(self.curriculum_controller.stage_steps)
            ),
            "curriculum_stage_episodes": (
                stage_episodes if terminated or truncated
                else float(len(self.curriculum_controller.recent_outcomes))
            ),
            "curriculum_rolling_success": (
                stage_success if terminated or truncated
                else self.curriculum_controller.rolling_success
            ),
            "domain_randomization_enabled": float(
                self._domain_wrapper is not None and self._domain_wrapper.enabled
            ),
            "curriculum_promoted": float(promotion is not None),
        })
        if promotion is not None:
            info["curriculum_promotion"] = promotion
        return obs, training_reward, terminated, truncated, info

    def close(self):
        if self._curriculum_state_path is not None:
            self.curriculum_controller.save(self._curriculum_state_path)
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
    parser.add_argument("--max-episode-steps", type=int, default=200)
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
    parser.add_argument("--curriculum-start-stage", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--curriculum-success-threshold", type=float, default=0.80)
    parser.add_argument("--curriculum-window", type=int, default=100)
    parser.add_argument("--curriculum-min-stage-steps", type=int, default=20_000)
    parser.add_argument("--curriculum-goal-jitter-radius", type=float, default=0.20)
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
        if existing_manifest.reward_contract != PUSH_REWARD_CONTRACT:
            raise ValueError(
                f"{checkpoint_dir} uses reward contract "
                f"{existing_manifest.reward_contract}; phased dense push training requires "
                "a fresh checkpoint directory"
            )
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
    curriculum_state_path = checkpoint_dir / "curriculum_state.json"
    if curriculum_state_path.exists() and not args.resume:
        raise ValueError(
            f"{checkpoint_dir} already contains curriculum state. "
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
            "reward_contract": PUSH_REWARD_CONTRACT,
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

    calibration_name = CalibrationProfile.load(args.calibration).name

    def save_manifest(stage: int, stage_name: str) -> None:
        build_sac_manifest(
            CheckpointManifest,
            calibration_name,
            args,
            stage,
            stage_name,
        ).save(checkpoint_dir / "manifest.json")

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
            self.best_stage = -1
            self.best_success_rate = float("-inf")
            best_path = checkpoint_dir / "best_metrics.json"
            if best_path.exists():
                try:
                    data = json.loads(best_path.read_text())
                    self.best_stage = int(data.get("curriculum_stage", -1))
                    self.best_success_rate = float(data["rolling_success_rate"])
                except (
                    OSError,
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
                    pass

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", [])
            if not infos:
                return True
            info = infos[0]
            dones = self.locals.get("dones", [])
            if len(dones) and bool(dones[0]):
                stage = int(
                    info.get(
                        "episode_curriculum_stage",
                        info.get("curriculum_stage", 2),
                    )
                )
                rolling_success = float(info.get("curriculum_rolling_success", 0.0))
                stage_episodes = int(info.get("curriculum_stage_episodes", 0))
                if stage_episodes:
                    stage_metrics_path = checkpoint_dir / f"best_metrics_stage_{stage}.json"
                    previous_stage_rate = float("-inf")
                    if stage_metrics_path.exists():
                        previous_stage_rate = float(
                            json.loads(stage_metrics_path.read_text())["rolling_success_rate"]
                        )
                    if rolling_success > previous_stage_rate:
                        self.model.save(str(checkpoint_dir / f"sac_best_stage_{stage}"))
                        stage_metrics_path.write_text(json.dumps({
                            "global_step": self.num_timesteps,
                            "curriculum_stage": stage,
                            "rolling_episodes": stage_episodes,
                            "rolling_success_rate": rolling_success,
                        }, indent=2) + "\n")
                if (stage, rolling_success) > (self.best_stage, self.best_success_rate):
                    self.best_stage = stage
                    self.best_success_rate = rolling_success
                    self.model.save(str(checkpoint_dir / "sac_best"))
                    (checkpoint_dir / "best_metrics.json").write_text(
                        json.dumps(
                            {
                                "global_step": self.num_timesteps,
                                "curriculum_stage": stage,
                                "rolling_episodes": stage_episodes,
                                "rolling_success_rate": rolling_success,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                if info.get("curriculum_promoted"):
                    promoted_stage = int(info["curriculum_stage"])
                    save_manifest(
                        promoted_stage,
                        str(info.get("curriculum_stage_name", "unknown")),
                    )
                    self.model.save(
                        str(checkpoint_dir / f"sac_promotion_stage_{promoted_stage}")
                    )
                    self.model.save_replay_buffer(
                        str(checkpoint_dir / f"sac_promotion_replay_stage_{promoted_stage}.pkl")
                    )
                    print(
                        f"Promoted push curriculum to stage {promoted_stage} "
                        f"({info.get('curriculum_stage_name')}); checkpoint saved"
                    )
            metrics = {
                "rollout/elapsed_seconds": info.get("elapsed_seconds"),
                "rollout/push_success": info.get("push_success"),
                "rollout/curriculum_success": info.get("curriculum_success"),
                "rollout/block_goal_distance": info.get("block_goal_distance"),
                "rollout/block_goal_progress": info.get("block_goal_progress"),
                "rollout/red_block_visible": info.get("red_block_visible"),
                "rollout/green_goal_visible": info.get("green_goal_visible"),
                "rollout/push_layout_randomized": info.get("push_layout_randomized"),
                "rollout/collisions": info.get("collisions"),
                "rollout/safety_overrides": info.get("safety_overrides"),
                "rollout/action_change": info.get("action_change"),
                "rollout/action_saturation": info.get("action_saturation"),
                "rollout/push_potential": info.get("push_potential"),
                "rollout/potential_shaping": info.get("potential_shaping"),
                "rollout/robot_block_distance": info.get("robot_block_distance"),
                "rollout/robot_block_contact": info.get("robot_block_contact"),
                "rollout/contact_acquired": info.get("contact_acquired"),
                "rollout/contact_bonus": info.get("contact_bonus"),
                "rollout/approach_completion_bonus": info.get(
                    "approach_completion_bonus"
                ),
                "rollout/goal_completion_bonus": info.get(
                    "goal_completion_bonus"
                ),
                "rollout/time_cost": info.get("time_cost"),
                "rollout/collision_penalty": info.get("collision_penalty"),
                "rollout/action_change_penalty": info.get("action_change_penalty"),
                "rollout/training_reward": info.get("training_reward"),
                "rollout/safety_with_visible_block": info.get(
                    "safety_with_visible_block"
                ),
                "curriculum/stage": info.get("curriculum_stage"),
                "curriculum/stage_steps": info.get("curriculum_stage_steps"),
                "curriculum/stage_episodes": info.get("curriculum_stage_episodes"),
                "curriculum/rolling_success": info.get("curriculum_rolling_success"),
                "curriculum/randomization_enabled": info.get("domain_randomization_enabled"),
            }
            metrics = {key: float(value) for key, value in metrics.items() if value is not None}
            for key, value in metrics.items():
                self.logger.record(key, value)
            if wandb_run is not None and self.num_timesteps % 20 == 0:
                payload = dict(metrics)
                payload["global_step"] = self.num_timesteps
                payload["curriculum/stage_name"] = info.get(
                    "curriculum_stage_name"
                )
                payload["curriculum/object_randomization_mode"] = info.get(
                    "push_layout_mode"
                )
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
            randomize_push_layout=True,
            domain_randomization=args.domain_randomization,
            randomization_ranges=randomization_ranges,
            calibration_path=args.calibration,
            curriculum=args.curriculum,
            curriculum_start_stage=args.curriculum_start_stage,
            curriculum_success_threshold=args.curriculum_success_threshold,
            curriculum_window=args.curriculum_window,
            curriculum_min_stage_steps=args.curriculum_min_stage_steps,
            curriculum_goal_jitter_radius=args.curriculum_goal_jitter_radius,
            curriculum_state_path=curriculum_state_path,
            image_size=args.image_size,
            record_dir=args.record_dir,
            no_record=args.no_record,
        ),
        filename=str(log_dir / "monitor.csv"),
        info_keywords=(
            "raw_reward",
            "push_success",
            "curriculum_success",
            "block_goal_distance",
            "block_goal_progress",
            "push_potential",
            "potential_shaping",
            "robot_block_distance",
            "robot_block_contact",
            "contact_acquired",
            "contact_bonus",
            "approach_completion_bonus",
            "goal_completion_bonus",
            "red_block_visible",
            "green_goal_visible",
            "push_layout_randomized",
            "time_cost",
            "collision_penalty",
            "action_change_penalty",
            "training_reward",
            "curriculum_stage",
            "episode_curriculum_stage",
            "curriculum_stage_steps",
            "curriculum_stage_episodes",
            "curriculum_rolling_success",
            "domain_randomization_enabled",
            "elapsed_seconds",
            "collisions",
            "safety_overrides",
            "mean_action_change",
            "action_saturation_rate",
            "safety_with_visible_block",
        ),
    )
    if not args.resume or existing_manifest_path.exists():
        save_manifest(
            env.unwrapped.curriculum_controller.stage,
            env.unwrapped.curriculum_controller.stage_name,
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
            recovered_state = validate_sac_manifest_recovery(
                resume_path,
                curriculum_state_path,
                args,
            )
            recovered_stage = int(recovered_state["stage"])
            recovered_stage_name = str(recovered_state["stage_name"])
            save_manifest(recovered_stage, recovered_stage_name)
            print(
                "Recovered missing manifest.json from verified SB3 checkpoint "
                f"metadata and curriculum state (stage {recovered_stage}: "
                f"{recovered_stage_name})."
            )
        manifest = CheckpointManifest.load(manifest_path)
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
                PUSH_REWARD_CONTRACT,
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
            "entropy_coefficient": args.entropy_coefficient,
            "max_grad_norm": args.max_grad_norm,
            "curriculum": args.curriculum,
            "max_episode_steps": args.max_episode_steps,
            "gamma": SAC_GAMMA,
            "curriculum_success_threshold": args.curriculum_success_threshold,
            "curriculum_window": args.curriculum_window,
            "curriculum_min_stage_steps": args.curriculum_min_stage_steps,
            "curriculum_goal_jitter_radius": args.curriculum_goal_jitter_radius,
            "reward_contract": PUSH_REWARD_CONTRACT,
            "push_time_penalty_per_second": PUSH_TIME_PENALTY_PER_SECOND,
            "push_approach_potential_scale": PUSH_APPROACH_POTENTIAL_SCALE,
            "push_goal_potential_offset": PUSH_GOAL_POTENTIAL_OFFSET,
            "push_goal_potential_scale": PUSH_GOAL_POTENTIAL_SCALE,
            "push_contact_bonus": PUSH_CONTACT_BONUS,
            "push_approach_completion_bonus": PUSH_APPROACH_COMPLETION_BONUS,
            "push_goal_completion_bonus": PUSH_GOAL_COMPLETION_BONUS,
        }
        if manifest.reward_contract != PUSH_REWARD_CONTRACT:
            raise ValueError(
                f"checkpoint reward contract is {manifest.reward_contract}, "
                f"expected {PUSH_REWARD_CONTRACT}; start a fresh run"
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
            gamma=SAC_GAMMA,
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
        save_manifest(
            env.unwrapped.curriculum_controller.stage,
            env.unwrapped.curriculum_controller.stage_name,
        )
        if wandb_run is not None:
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    main()
