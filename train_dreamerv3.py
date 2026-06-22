"""
DreamerV3 training for the Robobo red-block push task.

Trains a world model from experience, then uses imagined rollouts
to train an actor-critic policy.

Usage:
    python train_dreamerv3.py
    python train_dreamerv3.py --total-steps 500000
    python train_dreamerv3.py --resume
    python train_dreamerv3.py --no-wandb
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from tqdm import tqdm


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _save_episode_atomic(path: Path, data: dict[str, np.ndarray]) -> None:
    """Write a complete episode without exposing a partial NPZ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **data)
    temporary.replace(path)


def _as_float(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return None


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Scale a float [0, 1] HWC image to uint8 [0, 255] for wandb."""
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(255.0 * image, 0, 255).astype(np.uint8)
    return image


def dreamer_updates_per_env_step(
    replay_ratio: float, batch_size: int, sequence_length: int
) -> float:
    if replay_ratio < 0:
        raise ValueError("replay ratio must be non-negative")
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch size and sequence length must be positive")
    return replay_ratio / (batch_size * sequence_length)


THROUGHPUT_PRESETS = {
    "paper": {},
    "balanced": {
        "train_ratio": 128.0,
        "batch_size": 8,
        "sequence_length": 48,
        "imagination_starts": 8,
        "prefill_steps": 2000,
        "image_size": 64,
    },
    "fast": {
        "train_ratio": 64.0,
        "batch_size": 8,
        "sequence_length": 32,
        "imagination_starts": 4,
        "prefill_steps": 1000,
        "image_size": 48,
    },
}


def _explicit_cli_options(argv: list[str]) -> set[str]:
    options: set[str] = set()
    for arg in argv:
        if not arg.startswith("--"):
            continue
        options.add(arg.split("=", 1)[0])
    return options


def _fmt_value(value) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    value_float = _as_float(value)
    if value_float is None:
        return str(value)
    if abs(value_float) >= 1000:
        return f"{value_float:,.1f}"
    if abs(value_float) >= 10:
        return f"{value_float:.2f}"
    return f"{value_float:.4f}"


class TrainingLogger:
    """SB3-style terminal logger with progress bar and periodic metric tables."""

    def __init__(
        self,
        total_timesteps: int,
        log_interval: int = 2048,
        initial_step: int = 0,
        table_log_path: Path | None = None,
    ):
        self.total = total_timesteps
        self.log_interval = log_interval  # log every N timesteps
        self.table_log_path = table_log_path

        self.pbar = tqdm(
            total=total_timesteps,
            initial=min(initial_step, total_timesteps),
            desc="Training",
            dynamic_ncols=True,
            smoothing=0.05,
            bar_format="{desc:<10} {percentage:3.0f}%|{bar:28}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_successes = deque(maxlen=100)
        self.episodes_done = 0

        self._last_log_step = 0
        self._start_time = time.time()
        self._last_train_losses = {}

    def update(self, steps: int):
        self.pbar.update(steps)

    def record_episode(self, reward: float, length: int, success: float):
        self.episode_returns.append(reward)
        self.episode_lengths.append(length)
        self.episode_successes.append(success)
        self.episodes_done += 1

    def record_train(self, losses: dict):
        self._last_train_losses = {
            k: v for k, v in losses.items() if _as_float(v) is not None
        }

    def _set_postfix(self, current_step: int):
        elapsed = time.time() - self._start_time
        fps = current_step / max(1, elapsed)
        postfix = [f"fps={fps:.1f}"]
        if self.episode_returns:
            postfix.append(f"ret100={np.mean(self.episode_returns):.1f}")
            postfix.append(f"succ100={np.mean(self.episode_successes):.2f}")
        for key, label in (
            ("wm_loss", "wm"),
            ("actor_loss", "actor"),
            ("critic_loss", "critic"),
        ):
            if key in self._last_train_losses:
                postfix.append(f"{label}={_fmt_value(self._last_train_losses[key])}")
        self.pbar.set_postfix_str(" ".join(postfix), refresh=False)

    def _add_section(self, lines: list[str], title: str, rows: list[tuple[str, object]]):
        if not rows:
            return
        lines.append(f"[{title}]")
        name_width = 30
        value_width = 12
        for name, value in rows:
            lines.append(
                f"|    {name:<{name_width}} | {_fmt_value(value):>{value_width}} |"
            )

    def maybe_log(self, current_step: int):
        self._set_postfix(current_step)
        if current_step - self._last_log_step < self.log_interval:
            return
        self._last_log_step = current_step

        elapsed = time.time() - self._start_time
        fps = current_step / max(1, elapsed)
        remaining = (self.total - current_step) / max(1, fps)

        lines = []
        lines.append("")
        lines.append(f"DreamerV3 metrics @ step {current_step:,}")
        lines.append("-" * 52)

        self._add_section(
            lines,
            "rollout",
            [
                ("ep_rew_mean", np.mean(self.episode_returns)),
                ("ep_len_mean", np.mean(self.episode_lengths)),
                ("ep_success_mean", np.mean(self.episode_successes)),
            ] if self.episode_returns else [],
        )
        self._add_section(
            lines,
            "time",
            [
                ("episodes", self.episodes_done),
                ("fps", fps),
                ("elapsed_s", int(elapsed)),
                ("remaining_s", int(remaining)),
            ],
        )
        if self._last_train_losses:
            loss_keys = [
                "wm_loss", "recon_loss", "image_recon_loss", "ir_recon_loss",
                "food_recon_loss", "reward_loss", "continue_loss",
                "kl_dyn", "kl_rep", "actor_loss", "critic_loss",
                "critic_imagination_loss", "critic_replay_loss",
                "critic_replay_value_loss", "critic_slow_regularization",
            ]
            self._add_section(
                lines,
                "losses",
                [(key, self._last_train_losses[key]) for key in loss_keys if key in self._last_train_losses],
            )
            diagnostic_keys = [
                "actor_entropy", "imag_returns", "return_range", "actor_std",
                "actor_mean_abs",
                "imagination_starts",
                "action_saturation", "advantage_p05", "advantage_p50",
                "advantage_p95", "world_grad_norm", "actor_grad_norm",
                "critic_grad_norm", "raw_kl_mean", "raw_kl_median",
                "raw_kl_p95", "raw_kl_fraction_above_one",
            ]
            self._add_section(
                lines,
                "diagnostics",
                [
                    (key, self._last_train_losses[key])
                    for key in diagnostic_keys
                    if key in self._last_train_losses
                ],
            )
        lines.append("-" * 52)
        table_str = "\n".join(lines)
        tqdm.write(table_str)
        if self.table_log_path is not None:
            self.table_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.table_log_path.open("a") as f:
                f.write(table_str + "\n")

    def close(self):
        self.pbar.close()


def main():
    parser = argparse.ArgumentParser(description="DreamerV3 for Robobo red-block pushing")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument(
        "--host",
        default=os.environ.get("COPPELIA_SIM_IP", "127.0.0.1"),
    )
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument(
        "--push-layout-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize red block and green goal positions on each simulator reset.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="dreamerv3_models")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Also write local TensorBoard events; W&B logging is independent.",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--throughput-preset",
        choices=tuple(THROUGHPUT_PRESETS),
        default="paper",
        help=(
            "Preset for live training speed. paper keeps paper-like defaults; "
            "balanced/fast reduce update frequency, sequence length, and imagination starts."
        ),
    )
    parser.add_argument("--prefill-steps", type=int, default=5000)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=512.0,
        help=(
            "Replay transitions trained per environment transition. "
            "Updates/step = train_ratio / (batch_size * sequence_length)."
        ),
    )
    parser.add_argument("--model-size", choices=("12m", "25m", "50m"), default="12m")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument(
        "--reward-event-fraction",
        type=float,
        default=0.0,
        help="Opt-in sparse-reward oversampling fraction; 0.0 is uniform replay.",
    )
    parser.add_argument(
        "--reward-event-threshold",
        type=float,
        default=1.0,
        help="Raw reward at which a replay transition is treated as a reward event.",
    )
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument(
        "--world-learning-rate",
        type=float,
        default=None,
        help="Deprecated world-model-only override; defaults to --learning-rate.",
    )
    parser.add_argument("--grad-clip", type=float, default=100.0)
    parser.add_argument("--agc", type=float, default=0.3)
    parser.add_argument("--actor-std-min", type=float, default=0.1)
    parser.add_argument("--actor-std-max", type=float, default=1.0)
    parser.add_argument("--actor-mean-limit", type=float, default=2.5)
    parser.add_argument(
        "--imagination-starts",
        type=int,
        default=0,
        help="Replay states per sequence used to start dreams; 0 uses all states.",
    )
    parser.add_argument("--replay-value-weight", type=float, default=0.3)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument(
        "--time-penalty-per-second",
        type=float,
        default=0.1,
        help="Push-task elapsed-time penalty.",
    )
    parser.add_argument(
        "--action-change-penalty",
        type=float,
        default=0.01,
        help="Push-task penalty for large action changes.",
    )
    parser.add_argument(
        "--object-recon-weight",
        type=float,
        default=0.0,
        help="Optional red/green reconstruction saliency weight; 0 is paper-aligned.",
    )
    parser.add_argument(
        "--checkpoint-history",
        type=int,
        default=5,
        help="Number of step-numbered checkpoints (dreamerv3_step_<step>.pt) to retain. "
             "0 disables step-numbered checkpoints.",
    )
    parser.add_argument("--log-interval", type=int, default=2048)
    parser.add_argument("--record-dir", type=str, default="recorded_episodes",
                        help="Directory to save image episodes for offline training")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable episode recording")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Image size for recording (default: 64x64)")
    parser.add_argument("--calibration", default="config/calibration/simulation.json")
    parser.add_argument("--hardware-calibration", default=None)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--domain-randomization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable domain randomization wrapper. Use --no-domain-randomization to disable.",
    )
    parser.add_argument("--curriculum-one-food-steps", type=int, default=100_000)
    parser.add_argument("--curriculum-three-food-steps", type=int, default=250_000)
    parser.add_argument("--randomization-start-steps", type=int, default=300_000)
    explicit_options = _explicit_cli_options(sys.argv[1:])
    args = parser.parse_args()
    preset_overrides = THROUGHPUT_PRESETS[args.throughput_preset]
    for key, value in preset_overrides.items():
        option = "--" + key.replace("_", "-")
        if option not in explicit_options:
            setattr(args, key, value)
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.world_learning_rate is not None and args.world_learning_rate <= 0:
        parser.error("--world-learning-rate must be positive")
    if not 0 < args.actor_std_min <= args.actor_std_max:
        parser.error("actor std bounds must satisfy 0 < min <= max")
    if args.imagination_starts < 0:
        parser.error("--imagination-starts must be non-negative")
    if args.replay_value_weight < 0:
        parser.error("--replay-value-weight must be non-negative")
    if args.agc < 0:
        parser.error("--agc must be non-negative")

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "learning_machines" / "src"))
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))

    os.environ["COPPELIA_SIM_PORT"] = str(args.port)
    os.environ["COPPELIA_SIM_IP"] = args.host
    print(
        f"DreamerV3 effective configuration: "
        f"simulator={args.host}:{args.port}, "
        f"throughput_preset={args.throughput_preset}, "
        f"train_ratio={args.train_ratio}, "
        f"batch_size={args.batch_size}, "
        f"sequence_length={args.sequence_length}, "
        f"imagination_starts={args.imagination_starts}, "
        f"image_size={args.image_size}, "
        f"target_updates_per_env_step="
        f"{dreamer_updates_per_env_step(args.train_ratio, args.batch_size, args.sequence_length):.4f}",
        flush=True,
    )
    from learning_machines.coppelia_startup import check_coppelia_service
    try:
        check_coppelia_service(args.host, args.port)
    except ConnectionError as exc:
        raise SystemExit(f"CoppeliaSim preflight failed: {exc}") from None

    # Wandb
    wandb_run = None
    if not args.no_wandb:
        import wandb
        run_name = args.wandb_run_name or f"dreamerv3-{datetime.datetime.now().strftime('%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project="learning-machines",
            entity="Learningmachine",
            name=run_name,
            config=vars(args),
            sync_tensorboard=False,
            resume="allow",
        )
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("*", step_metric="global_step")

    import torch
    from learning_machines.dreamerv3 import DreamerV3
    from learning_machines.dreamerv3.config import DreamerV3Config, apply_model_size_preset
    from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
    from learning_machines.domain_randomization import DomainRandomizationWrapper
    from learning_machines.domain_randomization import RandomizationRanges
    from learning_machines.transfer import CalibrationProfile, CheckpointManifest

    # Config - use smaller buffer and a bounded number of imagination starts
    # for the image-based Robobo workload.
    world_learning_rate = (
        args.world_learning_rate
        if args.world_learning_rate is not None
        else args.learning_rate
    )
    cfg = DreamerV3Config(
        obs_dim=12,
        action_dim=2,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        world_lr=world_learning_rate,
        actor_lr=args.learning_rate,
        critic_lr=args.learning_rate,
        buffer_capacity=args.buffer_capacity,
        grad_clip=args.grad_clip,
        agc=args.agc,
        actor_std_min=args.actor_std_min,
        actor_std_max=args.actor_std_max,
        actor_mean_limit=args.actor_mean_limit,
        imagination_starts=args.imagination_starts,
        replay_value_weight=args.replay_value_weight,
        reward_event_fraction=args.reward_event_fraction,
        reward_event_threshold=args.reward_event_threshold,
        food_recon_weight=args.object_recon_weight,
        use_images=True,
        use_multimodal=True,
        image_size=args.image_size,
        ir_dim=8,
    )
    apply_model_size_preset(cfg, args.model_size)

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = DreamerV3(cfg, device="auto")
    print(f"Device: {agent.device}")
    if wandb_run is not None:
        wandb_run.config.update({
            "observation_contract": "robobo-push-obs-v1",
            "reward_contract": "robobo-push-reward-v1",
            "control_interval_seconds": 0.4,
            "phone_tilt": 100,
            "task": "push",
            "push_layout_randomization": args.push_layout_randomization,
            "model_size": cfg.model_size,
            "world_lr": cfg.world_lr,
            "actor_lr": cfg.actor_lr,
            "critic_lr": cfg.critic_lr,
            "optimizer": cfg.optimizer,
            "agc": cfg.agc,
            "imagination_starts": cfg.imagination_starts,
            "replay_value_weight": cfg.replay_value_weight,
        }, allow_val_change=True)

    # Environment
    env_config = RoboboCompactEnvConfig(
        task="push",
        max_episode_steps=args.max_episode_steps,
        return_image=True,
        image_obs_size=(args.image_size, args.image_size),
        calibration_path=args.calibration,
        randomize_push_layout=args.push_layout_randomization,
        time_penalty_per_second=args.time_penalty_per_second,
        push_time_penalty_per_second=args.time_penalty_per_second,
        action_change_penalty=args.action_change_penalty,
        push_action_change_penalty=args.action_change_penalty,
    )
    rob_env = RoboboCompactEnv(config=env_config)
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
    env = DomainRandomizationWrapper(
        rob_env,
        enabled=(args.domain_randomization and not args.curriculum),
        ranges=randomization_ranges,
    )

    def apply_curriculum(step: int) -> tuple[int, bool]:
        active_food = 0
        env_config.active_food_count = active_food
        randomization_enabled = (
            args.domain_randomization
            and (not args.curriculum or step >= args.randomization_start_steps)
        )
        if env.enabled is not None:
            env.enabled = randomization_enabled
        return active_food, randomization_enabled

    apply_curriculum(agent.global_step)

    # Image episode recorder
    record_dir = Path(args.record_dir)
    if not args.no_record:
        record_dir.mkdir(parents=True, exist_ok=True)
        (record_dir / "episodes").mkdir(parents=True, exist_ok=True)
        print(f"[Recorder] Saving image episodes to {record_dir}")
        existing_episode_ids = [
            int(path.stem.split("_")[-1])
            for path in (record_dir / "episodes").glob("ep_*.npz")
            if path.stem.split("_")[-1].isdigit()
        ]
        next_episode_id = max(existing_episode_ids, default=-1) + 1
    else:
        next_episode_id = 0

    # Checkpointing
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "dreamerv3_latest.pt"
    best_model_path = checkpoint_dir / "dreamerv3_best.pt"

    # Tensorboard
    log_dir = checkpoint_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir))
    else:
        writer = _NullSummaryWriter()

    # Resume
    if args.resume and model_path.exists():
        manifest_path = checkpoint_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(
                "refusing to resume a DreamerV3 checkpoint without a manifest"
            )
        manifest = CheckpointManifest.load(manifest_path)
        calibration_name = CalibrationProfile.load(args.calibration).name
        manifest_mismatches = {
            "algorithm": (manifest.algorithm, "dreamerv3"),
            "calibration_profile": (manifest.calibration_profile, calibration_name),
            "image_size": (manifest.image_size, args.image_size),
            "phone_tilt": (manifest.phone_tilt, 100),
            "observation_contract": (
                manifest.observation_contract,
                "robobo-push-obs-v1",
            ),
            "reward_contract": (
                manifest.reward_contract,
                "robobo-push-reward-v1",
            ),
            "push_layout_randomization": (
                manifest.algorithm_config.get("push_layout_randomization"),
                args.push_layout_randomization,
            ),
        }
        manifest_mismatches = {
            key: values
            for key, values in manifest_mismatches.items()
            if values[0] != values[1]
        }
        if manifest_mismatches:
            raise ValueError(
                f"checkpoint DreamerV3 push manifest is incompatible: {manifest_mismatches}"
            )
        agent.load(model_path)
        print(f"Resumed from {model_path}, step={agent.global_step}")
        restored = agent.buffer.restore_recorded_episodes(args.record_dir)
        print(
            f"Restored {restored:,} recent transitions from recorded episodes "
            f"into Dreamer replay"
        )
    else:
        print("Starting fresh training")

    # Prefill buffer
    if agent.buffer.size < args.prefill_steps:
        prefill_needed = args.prefill_steps - agent.buffer.size
        print(f"Prefilling buffer ({prefill_needed} additional steps)...")
        obs_dict, info = env.reset()
        # Get observation based on mode
        if cfg.use_multimodal and "image" in obs_dict:
            obs = obs_dict["image"].astype(np.float32) / 255.0
            ir_obs = obs_dict["ir"].astype(np.float32)
        elif cfg.use_images and "image" in obs_dict:
            obs = obs_dict["image"].astype(np.float32) / 255.0
            ir_obs = None
        else:
            obs = np.concatenate([obs_dict["blob"], obs_dict["ir"]]).astype(np.float32)
            ir_obs = None
        steps_prefilled = 0

        while steps_prefilled < prefill_needed:
            action = env.action_space.sample()
            obs_dict, reward, terminated, truncated, info = env.step(action)
            executed_action = np.asarray(info.get("executed_action", action), dtype=np.float32)
            agent.set_executed_action(executed_action)
            # Get next observation based on mode
            if cfg.use_multimodal and "image" in obs_dict:
                next_obs = obs_dict["image"].astype(np.float32) / 255.0
                next_ir = obs_dict["ir"].astype(np.float32)
            elif cfg.use_images and "image" in obs_dict:
                next_obs = obs_dict["image"].astype(np.float32) / 255.0
                next_ir = None
            else:
                next_obs = np.concatenate([obs_dict["blob"], obs_dict["ir"]]).astype(np.float32)
                next_ir = None
            done = terminated or truncated

            # Add to buffer (handle multi-modal)
            if cfg.use_multimodal and ir_obs is not None:
                agent.buffer.add(
                    obs, executed_action, reward, done, ir=ir_obs,
                    next_obs=next_obs if done else None,
                    next_ir=next_ir if done else None,
                )
            else:
                agent.buffer.add(
                    obs, executed_action, reward, done,
                    next_obs=next_obs if done else None,
                )
            obs = next_obs
            ir_obs = next_ir
            steps_prefilled += 1
            agent.global_step += 1

            if done:
                obs_dict, info = env.reset()
                if cfg.use_multimodal and "image" in obs_dict:
                    obs = obs_dict["image"].astype(np.float32) / 255.0
                    ir_obs = obs_dict["ir"].astype(np.float32)
                elif cfg.use_images and "image" in obs_dict:
                    obs = obs_dict["image"].astype(np.float32) / 255.0
                    ir_obs = None
                else:
                    obs = np.concatenate([obs_dict["blob"], obs_dict["ir"]]).astype(np.float32)
                    ir_obs = None

            if steps_prefilled % 1000 == 0:
                print(f"  Prefilled {steps_prefilled}/{prefill_needed}")

        print(f"Buffer filled: {agent.buffer.size} transitions\n")

    # Training loop
    logger = TrainingLogger(
        args.total_steps,
        log_interval=args.log_interval,
        initial_step=agent.global_step,
        table_log_path=log_dir / "dreamerv3_tables.log",
    )

    obs_dict, info = env.reset()
    # Get observation based on mode
    if cfg.use_multimodal and "image" in obs_dict:
        obs = obs_dict["image"].astype(np.float32) / 255.0
        ir_obs = obs_dict["ir"].astype(np.float32)
    elif cfg.use_images and "image" in obs_dict:
        obs = obs_dict["image"].astype(np.float32) / 255.0
        ir_obs = None
    else:
        obs = np.concatenate([obs_dict["blob"], obs_dict["ir"]]).astype(np.float32)
        ir_obs = None
    episode_reward = 0
    episode_length = 0
    episode_images = []
    episode_actions = []
    episode_rewards = []
    episode_dones = []
    episode_irs = []
    episode_count = next_episode_id
    episode_collisions = 0
    episode_safety_overrides = 0
    episode_action_change = 0.0
    episode_saturation = 0.0
    episode_safety_with_visible_block = 0
    update_budget = 0.0
    last_recon_step = -args.log_interval
    best_success_rate = float("-inf")
    best_metrics_path = checkpoint_dir / "best_metrics.json"
    if best_metrics_path.exists():
        try:
            data = json.loads(best_metrics_path.read_text())
            best_success_rate = float(
                data.get("rolling_success_rate", data.get("rolling_food_mean"))
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            best_success_rate = float("-inf")
    policy_state = None
    agent.reset_policy_state()

    try:
        while agent.global_step < args.total_steps:
            # === Collect experience ===
            with torch.no_grad():
                if cfg.use_multimodal and ir_obs is not None:
                    action, policy_state = agent.select_action(obs, state=policy_state, ir=ir_obs)
                else:
                    action, policy_state = agent.select_action(obs, state=policy_state)

            # Record image before action
            if not args.no_record and "image" in obs_dict:
                episode_images.append(obs_dict["image"].copy())
                if "ir" in obs_dict:
                    episode_irs.append(obs_dict["ir"].astype(np.float32).copy())

            obs_dict, reward, terminated, truncated, info = env.step(action)
            executed_action = np.asarray(info.get("executed_action", action), dtype=np.float32)
            agent.set_executed_action(executed_action)
            if not args.no_record:
                episode_actions.append(executed_action.copy())
            # Get next observation based on mode
            if cfg.use_multimodal and "image" in obs_dict:
                next_obs = obs_dict["image"].astype(np.float32) / 255.0
                next_ir = obs_dict["ir"].astype(np.float32)
            elif cfg.use_images and "image" in obs_dict:
                next_obs = obs_dict["image"].astype(np.float32) / 255.0
                next_ir = None
            else:
                next_obs = np.concatenate([obs_dict["blob"], obs_dict["ir"]]).astype(np.float32)
                next_ir = None
            done = terminated or truncated

            # Record reward and done
            if not args.no_record:
                episode_rewards.append(reward)
                episode_dones.append(done)

            # Add to buffer (handle multi-modal)
            if cfg.use_multimodal and ir_obs is not None:
                agent.buffer.add(
                    obs, executed_action, reward, done, ir=ir_obs,
                    next_obs=next_obs if done else None,
                    next_ir=next_ir if done else None,
                )
            else:
                agent.buffer.add(
                    obs, executed_action, reward, done,
                    next_obs=next_obs if done else None,
                )
            obs = next_obs
            ir_obs = next_ir
            episode_reward += reward
            episode_length += 1
            episode_collisions += int(bool(info.get("collision", False)))
            episode_safety_overrides += int(info.get("safety_override") is not None)
            episode_action_change += float(info.get("action_change", 0.0))
            episode_saturation += float(info.get("action_saturation", 0.0))
            episode_safety_with_visible_block += int(
                bool(info.get("safety_with_visible_block", False))
            )
            agent.global_step += 1

            logger.update(1)

            if done:
                success = float(info.get("push_success", 0.0))
                logger.record_episode(episode_reward, episode_length, success)
                if len(logger.episode_successes) >= 20:
                    rolling_success_mean = float(np.mean(logger.episode_successes))
                    if rolling_success_mean > best_success_rate:
                        best_success_rate = rolling_success_mean
                        agent.save(best_model_path)
                        best_metrics_path.write_text(
                            json.dumps(
                                {
                                    "global_step": agent.global_step,
                                    "rolling_episodes": len(logger.episode_successes),
                                    "rolling_success_rate": best_success_rate,
                                },
                                indent=2,
                            )
                            + "\n"
                        )
                        tqdm.write(
                            f"[Checkpoint] New best rolling success rate "
                            f"{best_success_rate:.3f} at step {agent.global_step:,}"
                        )
                episode_metrics = {
                    "episode/return": float(episode_reward),
                    "episode/length": int(episode_length),
                    "episode/elapsed_seconds": float(info.get("elapsed_seconds", episode_length * 0.4)),
                    "episode/push_success": success,
                    "episode/block_goal_distance": float(info.get("block_goal_distance", np.nan)),
                    "episode/block_goal_progress": float(info.get("block_goal_progress", 0.0)),
                    "episode/red_block_visible": float(info.get("red_block_visible", 0.0)),
                    "episode/green_goal_visible": float(info.get("green_goal_visible", 0.0)),
                    "episode/push_layout_randomized": float(info.get("push_layout_randomized", 0.0)),
                    "episode/collisions": int(episode_collisions),
                    "episode/safety_overrides": int(episode_safety_overrides),
                    "episode/mean_action_change": episode_action_change / max(1, episode_length),
                    "episode/action_saturation_rate": episode_saturation / max(1, episode_length),
                    "episode/safety_with_visible_block": int(
                        episode_safety_with_visible_block
                    ),
                    "episode/success_rate_100": float(
                        np.mean(logger.episode_successes)
                    ),
                    "episode/best_success_rate_100": float(
                        best_success_rate
                        if np.isfinite(best_success_rate)
                        else np.mean(logger.episode_successes)
                    ),
                }
                active_food, randomization_enabled = apply_curriculum(
                    agent.global_step
                )
                episode_metrics.update(
                    {
                        "curriculum/push_task": 1.0,
                        "curriculum/randomization_enabled": float(
                            randomization_enabled
                        ),
                    }
                )
                for key, value in episode_metrics.items():
                    writer.add_scalar(key, value, agent.global_step)
                if wandb_run is not None:
                    wandb_payload = dict(episode_metrics)
                    if "image" in obs_dict and episode_count % 10 == 0:
                        import wandb
                        wandb_payload["diagnostics/camera_frame"] = wandb.Image(
                            np.transpose(obs_dict["image"], (1, 2, 0)),
                            caption=f"step={agent.global_step} tilt={info.get('phone_tilt')}",
                        )
                        wandb_payload["diagnostics/ir_histogram"] = wandb.Histogram(
                            obs_dict["ir"]
                        )
                        if "raw_ir" in info:
                            wandb_payload["diagnostics/raw_ir_histogram"] = wandb.Histogram(
                                info["raw_ir"]
                            )
                    wandb_payload["global_step"] = agent.global_step
                    wandb_run.log(wandb_payload)

                # Save episode with images
                if not args.no_record and "image" in obs_dict:
                    episode_images.append(obs_dict["image"].copy())
                    if "ir" in obs_dict:
                        episode_irs.append(obs_dict["ir"].astype(np.float32).copy())
                if not args.no_record and len(episode_images) > 1:
                    ep_data = {
                        "images": np.array(episode_images, dtype=np.uint8),
                        "actions": np.array(episode_actions, dtype=np.float32),
                        "rewards": np.array(episode_rewards, dtype=np.float32),
                        "dones": np.array(episode_dones, dtype=bool),
                        "observation_contract": np.array("robobo-push-obs-v1"),
                        "reward_contract": np.array("robobo-push-reward-v1"),
                        "control_interval_seconds": np.array(0.4),
                        "calibration_profile": np.array(
                            env.unwrapped.observation_adapter.profile.name
                        ),
                        "phone_tilt": np.array(env.unwrapped.config.phone_tilt),
                    }
                    if episode_irs:
                        ep_data["irs"] = np.array(episode_irs, dtype=np.float32)
                    ep_path = record_dir / "episodes" / f"ep_{episode_count:06d}.npz"
                    _save_episode_atomic(ep_path, ep_data)
                    episode_count += 1
                    if episode_count % 10 == 0:
                        tqdm.write(f"[Recorder] Saved {episode_count} episodes")

                episode_images.clear()
                episode_actions.clear()
                episode_rewards.clear()
                episode_dones.clear()
                episode_irs.clear()
                episode_reward = 0
                episode_length = 0
                episode_collisions = 0
                episode_safety_overrides = 0
                episode_action_change = 0.0
                episode_saturation = 0.0
                episode_safety_with_visible_block = 0
                policy_state = None
                agent.reset_policy_state()
                obs_dict, info = env.reset()
                if cfg.use_multimodal and "image" in obs_dict:
                    obs = obs_dict["image"].astype(np.float32) / 255.0
                    ir_obs = obs_dict["ir"].astype(np.float32)
                elif cfg.use_images and "image" in obs_dict:
                    obs = obs_dict["image"].astype(np.float32) / 255.0
                    ir_obs = None
                else:
                    obs = np.concatenate([obs_dict["blob"], obs_dict["ir"]]).astype(np.float32)
                    ir_obs = None

            # === Train world model + actor-critic ===
            if agent.global_step >= args.prefill_steps:
                target_updates = dreamer_updates_per_env_step(
                    args.train_ratio, cfg.batch_size, cfg.sequence_length
                )
                update_budget += target_updates
                updates_this_step = 0
                while (
                    update_budget >= 1.0
                    and agent.buffer.size >= cfg.sequence_length
                    and updates_this_step < 8
                ):
                    update_budget -= 1.0
                    updates_this_step += 1
                    batch = agent.buffer.sample(cfg.batch_size, agent.device)
                    losses = agent.train_step(batch)
                    if not all(np.isfinite(float(value)) for value in losses.values()):
                        raise RuntimeError(
                            f"non-finite DreamerV3 training metrics: {losses}"
                        )
                    losses["reward_event_sample_fraction"] = float(
                        batch["reward_event_sample_fraction"].item()
                    )
                    losses["updates_per_env_step_target"] = target_updates

                    for k, v in losses.items():
                        writer.add_scalar(f"train/{k}", v, agent.global_step)
                    if wandb_run is not None:
                        payload = {f"train/{k}": float(v) for k, v in losses.items()}
                        payload["global_step"] = agent.global_step
                        if agent.global_step - last_recon_step >= args.log_interval:
                            import wandb
                            from learning_machines.distributional import logits_to_value
                            with torch.no_grad():
                                diagnostic = agent.world_model.observe_sequence(
                                    batch["obs"],
                                    batch["action"],
                                    ir_seq=batch.get("ir"),
                                )
                            predicted_reward = logits_to_value(
                                diagnostic["reward_logits"]
                            ).detach().cpu().numpy()
                            payload["diagnostics/predicted_reward_histogram"] = wandb.Histogram(
                                predicted_reward
                            )
                            payload["diagnostics/actual_reward_histogram"] = wandb.Histogram(
                                batch["reward"].detach().cpu().numpy()
                            )
                            if cfg.use_images or cfg.use_multimodal:
                                original = batch["obs"][0, 1].detach().cpu()
                                reconstruction = diagnostic["obs_pred"][0, 0].detach().cpu()
                                C_img, H_img, W_img = original.shape
                                comparison = torch.zeros(C_img, H_img, W_img * 2)
                                comparison[:, :, :W_img] = original
                                comparison[:, :, W_img:] = reconstruction
                                payload["recon/target"] = wandb.Image(
                                    _to_uint8_image(original.permute(1, 2, 0).numpy())
                                )
                                payload["recon/predicted"] = wandb.Image(
                                    _to_uint8_image(reconstruction.permute(1, 2, 0).numpy())
                                )
                                payload["recon/comparison"] = wandb.Image(
                                    _to_uint8_image(comparison.permute(1, 2, 0).numpy()),
                                    caption="left=target  right=prediction",
                                )
                                with torch.no_grad():
                                    video_pred = agent.world_model.video_pred(batch)
                                target_vid = batch["obs"][0, 1:].detach().cpu()
                                video_pred_cpu = video_pred[0].detach().cpu()
                                T, C, H, W = video_pred_cpu.shape
                                vid_stacked = torch.zeros(T, C, H, W * 2)
                                vid_stacked[:, :, :, :W] = target_vid
                                vid_stacked[:, :, :, W:] = video_pred_cpu
                                vid_stacked = (vid_stacked.permute(0, 2, 3, 1) * 255.0).clamp(0, 255)
                                payload["recon/open_loop"] = wandb.Video(
                                    vid_stacked.numpy().astype(np.uint8),
                                    fps=4, format="gif",
                                    caption="left=target  right=prediction",
                                )
                            last_recon_step = agent.global_step
                        wandb_run.log(payload)
                    logger.record_train(losses)

            # === Periodic log ===
            logger.maybe_log(agent.global_step)

            # === Save checkpoint ===
            if agent.global_step % args.checkpoint_every == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                agent.save(model_path)
                if args.checkpoint_history > 0:
                    step_path = checkpoint_dir / f"dreamerv3_step_{agent.global_step}.pt"
                    agent.save(step_path)
                    step_checkpoints = sorted(
                        checkpoint_dir.glob("dreamerv3_step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]),
                    )
                    for old_path in step_checkpoints[:-args.checkpoint_history]:
                        old_path.unlink()

    except KeyboardInterrupt:
        print("\nInterrupted — saving...")
    finally:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        agent.save(model_path)
        calibration_name = CalibrationProfile.load(args.calibration).name
        CheckpointManifest(
            algorithm="dreamerv3",
            calibration_profile=calibration_name,
            image_size=args.image_size,
            observation_contract="robobo-push-obs-v1",
            reward_contract="robobo-push-reward-v1",
            algorithm_config={
                "world_lr": cfg.world_lr,
                "actor_lr": cfg.actor_lr,
                "critic_lr": cfg.critic_lr,
                "use_multimodal": cfg.use_multimodal,
                "task": "push",
                "push_layout_randomization": args.push_layout_randomization,
                "model_size": cfg.model_size,
                "cnn_base_channels": cfg.cnn_base_channels,
                "block_gru_blocks": cfg.block_gru_blocks,
                "ir_dim": cfg.ir_dim,
                "domain_randomization": True,
                "hardware_calibration": args.hardware_calibration,
                "sequence_length": cfg.sequence_length,
                "imagination_horizon": cfg.imagination_horizon,
                "gamma": cfg.gamma,
                "reward_event_fraction": cfg.reward_event_fraction,
                "reward_event_threshold": cfg.reward_event_threshold,
                "train_ratio": args.train_ratio,
                "buffer_capacity": cfg.buffer_capacity,
                "grad_clip": cfg.grad_clip,
                "actor_std_min": cfg.actor_std_min,
                "actor_std_max": cfg.actor_std_max,
                "actor_mean_limit": cfg.actor_mean_limit,
                "optimizer": cfg.optimizer,
                "agc": cfg.agc,
                "imagination_starts": cfg.imagination_starts,
                "replay_value_weight": cfg.replay_value_weight,
                "food_recon_weight": cfg.food_recon_weight,
                "curriculum": args.curriculum,
                "curriculum_one_food_steps": args.curriculum_one_food_steps,
                "curriculum_three_food_steps": args.curriculum_three_food_steps,
                "randomization_start_steps": args.randomization_start_steps,
            },
        ).save(checkpoint_dir / "manifest.json")
        logger.close()
        print(f"Saved model to {model_path}")
        if wandb_run is not None:
            wandb_run.finish()
        writer.close()
        env.close()


if __name__ == "__main__":
    main()
