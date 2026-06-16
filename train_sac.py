"""
SAC + HER training for Robobo food collection.

Uses stable-baselines3 SAC with HerReplayBuffer.
Observation: Dict with "observation" (12-dim), "achieved_goal" (1-dim), "desired_goal" (1-dim).
Reward: +1 per food collected (sparse via HER relabeling).

Usage:
    python train_sac.py
    python train_sac.py --total-timesteps 500000
    python train_sac.py --resume
    python train_sac.py --no-wandb
"""
from __future__ import annotations

import argparse
import os
import sys
import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class RoboboGoalEnv(gym.Env):
    """Wraps RoboboCompactEnv for HER.

    Observation space is Dict with:
      - "observation": [blob_x, blob_y, blob_area, blob_found, ir0..ir7] (12)
      - "achieved_goal": [food_collected] (1)
      - "desired_goal": [target_food_count] (1)
    """

    metadata = {"render_modes": []}

    def __init__(self, rob=None, max_episode_steps=300, randomize_food_positions=True):
        super().__init__()
        from learning_machines.rl_robobo_compact_env import (
            RoboboCompactEnv,
            RoboboCompactEnvConfig,
        )

        self._config = RoboboCompactEnvConfig(
            max_episode_steps=max_episode_steps,
            randomize_food_positions=randomize_food_positions,
        )
        self._inner = RoboboCompactEnv(rob=rob, config=self._config)
        self._num_food = 7

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32,
            ),
            "achieved_goal": spaces.Box(
                low=0.0, high=float(self._num_food), shape=(1,), dtype=np.float32,
            ),
            "desired_goal": spaces.Box(
                low=0.0, high=float(self._num_food), shape=(1,), dtype=np.float32,
            ),
        })
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
        food = self._get_food_count(info)
        obs = {
            "observation": self._flatten_obs(obs_dict),
            "achieved_goal": np.array([food], dtype=np.float32),
            "desired_goal": np.array([food], dtype=np.float32),
        }
        return obs, info

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self._inner.step(action)
        food = self._get_food_count(info)

        obs = {
            "observation": self._flatten_obs(obs_dict),
            "achieved_goal": np.array([food], dtype=np.float32),
            "desired_goal": np.array([food], dtype=np.float32),
        }

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """HER reward: +1.0 if achieved >= desired, else 0.0."""
        ag = np.asarray(achieved_goal, dtype=np.float32)
        dg = np.asarray(desired_goal, dtype=np.float32)
        if ag.ndim == 1:
            ag = ag.reshape(-1, 1)
        if dg.ndim == 1:
            dg = dg.reshape(-1, 1)
        return (ag >= dg).astype(np.float32).squeeze(-1)

    def close(self):
        self._inner.close()


def main():
    parser = argparse.ArgumentParser(description="SAC + HER for Robobo food collection")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--checkpoint-dir", type=str, default="sac_her_models")
    parser.add_argument("--learning-starts", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--her-n-samples", type=int, default=16)
    parser.add_argument("--her-goal-selection", type=str, default="future")
    parser.add_argument("--no-wandb", action="store_true")
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

    run_name = args.wandb_run_name or f"sac-her-{datetime.datetime.now().strftime('%m%d-%H%M%S')}"

    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project="learning-machines",
            entity="Learningmachine",
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
            resume="allow",
        )

    from stable_baselines3 import SAC
    from stable_baselines3.her import HerReplayBuffer
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    model_path = checkpoint_dir / "sac_her_latest"
    buffer_path = checkpoint_dir / "replay_buffer.pkl"

    env = Monitor(
        RoboboGoalEnv(
            max_episode_steps=args.max_episode_steps,
            randomize_food_positions=True,
        ),
        filename=str(log_dir / "monitor.csv"),
    )

    if args.resume and model_path.exists():
        print(f"Resuming from {model_path}")
        model = SAC.load(str(model_path), env=env, device="auto")
        if buffer_path.exists():
            print(f"Loading replay buffer from {buffer_path}")
            model.load_replay_buffer(str(buffer_path))
            print(f"Buffer size: {model.replay_buffer.size():,}")
    else:
        print("Starting fresh training run")
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=args.her_n_samples,
                goal_selection_strategy=args.her_goal_selection,
            ),
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
        name_prefix="sac_her",
        save_replay_buffer=True,
    )

    try:
        model.learn(
            total_timesteps=remaining,
            callback=[checkpoint_cb],
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving...")
    finally:
        model.save(str(model_path))
        model.save_replay_buffer(str(buffer_path))
        print(f"Saved model to {model_path}")
        print(f"Saved buffer to {buffer_path}")
        if wandb_run is not None:
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    main()
