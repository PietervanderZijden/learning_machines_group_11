"""
DreamerV3 training for Robobo food collection.

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
        self.episode_foods = deque(maxlen=100)
        self.episodes_done = 0

        self._last_log_step = 0
        self._start_time = time.time()
        self._last_train_losses = {}

    def update(self, steps: int):
        self.pbar.update(steps)

    def record_episode(self, reward: float, length: int, food: int):
        self.episode_returns.append(reward)
        self.episode_lengths.append(length)
        self.episode_foods.append(food)
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
            postfix.append(f"food100={np.mean(self.episode_foods):.1f}")
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
                ("ep_food_mean", np.mean(self.episode_foods)),
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
                "reward_loss", "continue_loss",
                "kl_dyn", "kl_rep", "actor_loss", "critic_loss",
            ]
            self._add_section(
                lines,
                "losses",
                [(key, self._last_train_losses[key]) for key in loss_keys if key in self._last_train_losses],
            )
            diagnostic_keys = [
                "actor_entropy", "imag_returns", "return_range", "actor_std",
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
    parser = argparse.ArgumentParser(description="DreamerV3 for Robobo food collection")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--checkpoint-dir", type=str, default="dreamerv3_models")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Also write local TensorBoard events; W&B logging is independent.",
    )
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill-steps", type=int, default=5000)
    parser.add_argument("--train-ratio", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=50)
    parser.add_argument(
        "--reward-event-fraction",
        type=float,
        default=0.25,
        help="Target fraction of replay sequences containing a food reward.",
    )
    parser.add_argument(
        "--reward-event-threshold",
        type=float,
        default=1.0,
        help="Raw reward at which a replay transition is treated as a reward event.",
    )
    parser.add_argument("--world-learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=2048)
    parser.add_argument("--record-dir", type=str, default="recorded_episodes",
                        help="Directory to save image episodes for offline training")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable episode recording")
    parser.add_argument("--image-size", type=int, default=64,
                        help="Image size for recording (default: 64x64)")
    parser.add_argument("--calibration", default="config/calibration/simulation.json")
    parser.add_argument("--hardware-calibration", default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "learning_machines" / "src"))
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))

    os.environ["COPPELIA_SIM_PORT"] = str(args.port)
    print(f"DreamerV3 effective configuration: port={args.port}")

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
    from learning_machines.dreamerv3.config import DreamerV3Config
    from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
    from learning_machines.domain_randomization import DomainRandomizationWrapper
    from learning_machines.domain_randomization import RandomizationRanges
    from learning_machines.transfer import CalibrationProfile, CheckpointManifest

    # Config - use smaller buffer for images to avoid memory issues
    cfg = DreamerV3Config(
        obs_dim=12,
        action_dim=2,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        world_lr=args.world_learning_rate,
        actor_lr=1e-4,
        critic_lr=1e-4,
        buffer_capacity=10_000,  # Smaller for images
        reward_event_fraction=args.reward_event_fraction,
        reward_event_threshold=args.reward_event_threshold,
        use_images=True,
        use_multimodal=True,
        image_size=args.image_size,
        ir_dim=8,
    )

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    agent = DreamerV3(cfg, device="auto")
    print(f"Device: {agent.device}")
    if wandb_run is not None:
        wandb_run.config.update({
            "observation_contract": "robobo-obs-v2",
            "reward_contract": "robobo-reward-v4",
            "control_interval_seconds": 0.4,
            "phone_tilt": 100,
            "world_lr": cfg.world_lr,
            "actor_lr": cfg.actor_lr,
            "critic_lr": cfg.critic_lr,
        }, allow_val_change=True)

    # Environment
    env_config = RoboboCompactEnvConfig(
        max_episode_steps=args.max_episode_steps,
        return_image=True,
        image_obs_size=(args.image_size, args.image_size),
        calibration_path=args.calibration,
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
    env = DomainRandomizationWrapper(rob_env, ranges=randomization_ranges)

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
        CheckpointManifest.load(manifest_path).validate(
            "dreamerv3",
            CalibrationProfile.load(args.calibration).name,
            args.image_size,
            100,
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
                if ir_obs is not None:
                    episode_irs.append(ir_obs.copy())

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
            agent.global_step += 1

            logger.update(1)

            if done:
                food = info.get("food_collected", 0)
                logger.record_episode(episode_reward, episode_length, food)
                episode_metrics = {
                    "episode/return": float(episode_reward),
                    "episode/length": int(episode_length),
                    "episode/elapsed_seconds": float(info.get("elapsed_seconds", episode_length * 0.4)),
                    "episode/food_collected": int(food),
                    "episode/food_per_minute": float(info.get("food_per_minute", 0.0)),
                    "episode/collisions": int(episode_collisions),
                    "episode/safety_overrides": int(episode_safety_overrides),
                    "episode/mean_action_change": episode_action_change / max(1, episode_length),
                    "episode/action_saturation_rate": episode_saturation / max(1, episode_length),
                }
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
                    if cfg.use_multimodal and "ir" in obs_dict:
                        episode_irs.append(obs_dict["ir"].astype(np.float32).copy())
                if not args.no_record and len(episode_images) > 1:
                    ep_data = {
                        "images": np.array(episode_images, dtype=np.uint8),
                        "actions": np.array(episode_actions, dtype=np.float32),
                        "rewards": np.array(episode_rewards, dtype=np.float32),
                        "dones": np.array(episode_dones, dtype=bool),
                        "observation_contract": np.array("robobo-obs-v2"),
                        "reward_contract": np.array("robobo-reward-v4"),
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
                steps_since_prefill = agent.global_step - args.prefill_steps
                if steps_since_prefill % args.train_ratio == 0:
                    if agent.buffer.size >= cfg.sequence_length:
                        batch = agent.buffer.sample(cfg.batch_size, agent.device)
                        losses = agent.train_step(batch)
                        losses["reward_event_sample_fraction"] = float(
                            batch["reward_event_sample_fraction"].item()
                        )

                        for k, v in losses.items():
                            writer.add_scalar(f"train/{k}", v, agent.global_step)
                        if wandb_run is not None:
                            payload = {f"train/{k}": float(v) for k, v in losses.items()}
                            if agent.global_step % args.log_interval == 0:
                                import wandb
                                from learning_machines.distributional import logits_to_value
                                with torch.no_grad():
                                    diagnostic = agent.world_model.observe_sequence(
                                        batch["obs"],
                                        batch["action"],
                                        ir_seq=batch.get("ir"),
                                    )
                                if cfg.use_images or cfg.use_multimodal:
                                    original = batch["obs"][0, 1].detach().cpu()
                                    reconstruction = diagnostic["obs_pred"][0, 0].detach().cpu()
                                    payload["diagnostics/original_frame"] = wandb.Image(
                                        original.permute(1, 2, 0).numpy()
                                    )
                                    payload["diagnostics/reconstructed_frame"] = wandb.Image(
                                        reconstruction.permute(1, 2, 0).numpy()
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
                            payload["global_step"] = agent.global_step
                            wandb_run.log(payload)
                        logger.record_train(losses)

            # === Periodic log ===
            logger.maybe_log(agent.global_step)

            # === Save checkpoint ===
            if agent.global_step % args.checkpoint_every == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                agent.save(model_path)

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
            algorithm_config={
                "world_lr": cfg.world_lr,
                "actor_lr": cfg.actor_lr,
                "critic_lr": cfg.critic_lr,
                "use_multimodal": cfg.use_multimodal,
                "ir_dim": cfg.ir_dim,
                "domain_randomization": True,
                "hardware_calibration": args.hardware_calibration,
                "sequence_length": cfg.sequence_length,
                "imagination_horizon": cfg.imagination_horizon,
                "gamma": cfg.gamma,
                "reward_event_fraction": cfg.reward_event_fraction,
                "reward_event_threshold": cfg.reward_event_threshold,
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
