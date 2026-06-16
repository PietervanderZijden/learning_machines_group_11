"""DreamerV4 training on recorded episodes.

Trains a transformer world model on (obs, action, reward) sequences,
then uses imagination to train an actor-critic policy.

Usage:
    python train_dreamerv4.py --record-dir recorded_episodes
    python train_dreamerv4.py --record-dir recorded_episodes --resume
    python train_dreamerv4.py --use-synthetic --num-synthetic-episodes 100
"""
from __future__ import annotations
import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tqdm


def main():
    parser = argparse.ArgumentParser(description="DreamerV4 training on recorded data")
    parser.add_argument("--record-dir", type=str, default="recorded_episodes",
                        help="Directory with recorded episodes")
    parser.add_argument("--checkpoint-dir", type=str, default="dreamerv4_models")
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--update-every", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--use-synthetic", action="store_true",
                        help="Generate synthetic data for testing")
    parser.add_argument("--num-synthetic-episodes", type=int, default=100)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "learning_machines" / "src"))

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.wandb_run_name or f"dreamerv4-{datetime.datetime.now().strftime('%m%d-%H%M%S')}"

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project="learning-machines",
            entity="Learningmachine",
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
        )

    from learning_machines.dreamerv4.config import DreamerV4Config
    from learning_machines.dreamerv4.dreamerv4 import DreamerV4Agent
    from learning_machines.dreamerv4.replay_buffer import SequenceReplayBuffer

    cfg = DreamerV4Config(
        obs_dim=12,
        act_dim=2,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        context_length=args.context_length,
        imagination_horizon=args.horizon,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = DreamerV4Agent(cfg).to(device)

    buffer = SequenceReplayBuffer(
        obs_dim=cfg.obs_dim,
        act_dim=cfg.act_dim,
        context_length=cfg.context_length,
    )

    if args.use_synthetic:
        print(f"Generating {args.num_synthetic_episodes} synthetic episodes...")
        from generate_synthetic_data import generate_synthetic_episode
        for i in range(args.num_synthetic_episodes):
            ep = generate_synthetic_episode(np.random.randint(20, 100))
            buffer.add_episode(
                observations=ep["observations"],
                actions=ep["actions"],
                rewards=ep["rewards"],
                dones=ep["dones"],
            )
        print(f"Loaded {buffer.size} synthetic episodes")
    else:
        loaded = buffer.load_from_directory(args.record_dir)
        if loaded == 0:
            print(f"No episodes found in {args.record_dir}")
            print("Use --use-synthetic to generate test data")
            return

    model_path = checkpoint_dir / "dreamerv4_latest.pt"

    if args.resume and model_path.exists():
        print(f"Resuming from {model_path}")
        agent = DreamerV4Agent.load(str(model_path), device)

    agent.warmup_stats(buffer)

    print(f"Training DreamerV4 for {args.total_steps:,} steps")
    print(f"  Buffer: {buffer.size} episodes")
    print(f"  Context length: {cfg.context_length}")
    print(f"  Dream horizon: {cfg.imagination_horizon}")
    print(f"  Model: d={cfg.d_model}, heads={cfg.n_heads}, layers={cfg.n_layers}")

    step = 0
    episode_count = 0
    start_time = time.time()
    recent_metrics: dict[str, list[float]] = {}

    pbar = tqdm.tqdm(range(args.total_steps), desc="DreamerV4")

    for step in pbar:
        actual_batch = min(cfg.batch_size, buffer.size)
        if actual_batch < 4:
            continue

        batch = buffer.sample_batch(actual_batch)
        batch = {k: torch.from_numpy(v).to(device) for k, v in batch.items()}

        wm_metrics = agent.update_world_model(batch)

        if step % args.update_every == 0:
            ac_metrics = agent.update_actor_critic(batch)
            wm_metrics.update(ac_metrics)

        for k, v in wm_metrics.items():
            recent_metrics.setdefault(k, []).append(v)

        if step % args.log_interval == 0 and recent_metrics:
            avg = {k: np.mean(v[-args.log_interval:]) for k, v in recent_metrics.items() if v}
            elapsed = time.time() - start_time
            fps = step / max(elapsed, 1e-6)

            desc_parts = [f"step={step:,}", f"fps={fps:.0f}"]
            for k, v in avg.items():
                desc_parts.append(f"{k}={v:.3f}")
            pbar.set_description(" | ".join(desc_parts[:5]))

            if wandb_run is not None:
                wandb_run.log({"step": step, **{f"train/{k}": v for k, v in avg.items()}})

        if step > 0 and step % args.save_freq == 0:
            agent.save(str(model_path))
            print(f"\nSaved checkpoint at step {step:,}")

    agent.save(str(model_path))
    elapsed = time.time() - start_time
    print(f"\nTraining complete: {step:,} steps in {elapsed:.1f}s ({step / max(elapsed, 1e-6):.1f} fps)")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
