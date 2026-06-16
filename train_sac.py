"""
SAC + HER training for Robobo food collection.

Uses stable-baselines3 SAC with HerReplayBuffer.
Observation: Dict with "observation" (12-dim), "achieved_goal" (1-dim), "desired_goal" (1-dim).
Reward: +1 per food collected (sparse via HER relabeling).

Records episodes to disk for DreamerV4 offline training.

Usage:
    python train_sac.py
    python train_sac.py --total-timesteps 500000
    python train_sac.py --resume
    python train_sac.py --no-wandb
    python train_sac.py --no-record
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeRecorder(BaseCallback):
    """Records episodes to disk as NPZ files for DreamerV4 offline training.

    Saves each episode as a single NPZ with keys:
      observations: (T+1, 12)  — includes final observation
      actions:      (T, 2)
      rewards:      (T,)
      dones:        (T,)

    Also writes episodes.jsonl manifest with per-episode metadata.
    """

    def __init__(self, record_dir: str, save_freq: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.record_dir = Path(record_dir)
        self.save_freq = save_freq
        self._episodes: list[dict] = []
        self._episode_count = 0
        self._current_obs: list[np.ndarray] = []
        self._current_actions: list[np.ndarray] = []
        self._current_rewards: list[float] = []
        self._current_dones: list[bool] = []

    def _on_training_start(self):
        self.record_dir.mkdir(parents=True, exist_ok=True)
        (self.record_dir / "episodes").mkdir(parents=True, exist_ok=True)
        print(f"[EpisodeRecorder] Saving episodes to {self.record_dir}")

    def _on_step(self) -> bool:
        obs_dict = self.locals.get("obs")
        action = self.locals.get("actions")
        reward = self.locals.get("rewards")
        done = self.locals.get("dones")
        info = self.locals.get("infos", [{}])

        if obs_dict is None or action is None:
            return True

        obs = obs_dict["observation"] if isinstance(obs_dict, dict) else obs_dict
        if isinstance(obs, dict):
            obs = obs.get("observation", np.zeros(12, dtype=np.float32))

        self._current_obs.append(obs.copy())
        self._current_actions.append(action.copy() if hasattr(action, 'copy') else np.array(action))
        self._current_rewards.append(float(reward))
        self._current_dones.append(bool(done))

        if done:
            self._save_episode(info)
            self._current_obs.clear()
            self._current_actions.clear()
            self._current_rewards.clear()
            self._current_dones.clear()

        return True

    def _save_episode(self, info):
        if len(self._current_obs) < 2:
            return

        obs_arr = np.array(self._current_obs, dtype=np.float32)
        act_arr = np.array(self._current_actions, dtype=np.float32)
        rew_arr = np.array(self._current_rewards, dtype=np.float32)
        done_arr = np.array(self._current_dones, dtype=bool)

        ep_data = {
            "observations": obs_arr,
            "actions": act_arr,
            "rewards": rew_arr,
            "dones": done_arr,
        }

        ep_path = self.record_dir / "episodes" / f"ep_{self._episode_count:06d}.npz"
        np.savez_compressed(ep_path, **ep_data)

        ep_info = info[0] if isinstance(info, list) else info
        self._episodes.append({
            "id": self._episode_count,
            "file": str(ep_path.name),
            "length": len(self._current_rewards),
            "total_reward": float(rew_arr.sum()),
            "food_collected": ep_info.get("food_collected", 0),
        })

        self._episode_count += 1

        if self._episode_count % self.save_freq == 0:
            self._save_manifest()

        if self.verbose >= 1 and self._episode_count % 10 == 0:
            print(f"[EpisodeRecorder] Recorded {self._episode_count} episodes "
                  f"(last: {self._episodes[-1]['length']} steps, "
                  f"reward={self._episodes[-1]['total_reward']:.1f})")

    def _save_manifest(self):
        manifest_path = self.record_dir / "episodes.jsonl"
        with open(manifest_path, "w") as f:
            for ep in self._episodes:
                f.write(json.dumps(ep) + "\n")

    def _on_training_end(self):
        self._save_manifest()
        print(f"[EpisodeRecorder] Saved {self._episode_count} episodes total to {self.record_dir}")


class RoboboGoalEnv(gym.Env):
    """Wraps RoboboCompactEnv for HER.

    Observation space is Dict with:
      - "observation": [blob_x, blob_y, blob_area, blob_found, ir0..ir7] (12)
      - "achieved_goal": [food_collected] (1)
      - "desired_goal": [target_food_count] (1)

    Optional domain randomization adds noise to observations and actions.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rob=None,
        max_episode_steps=300,
        randomize_food_positions=True,
        domain_randomization=False,
        ir_noise_std=0.02,
        ir_noise_prob=0.5,
        pose_jitter=0.02,
        wheel_noise_std=0.03,
        wheel_noise_prob=0.3,
    ):
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

        self._dr_enabled = domain_randomization
        self._ir_noise_std = ir_noise_std
        self._ir_noise_prob = ir_noise_prob
        self._pose_jitter = pose_jitter
        self._wheel_noise_std = wheel_noise_std
        self._wheel_noise_prob = wheel_noise_prob

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
        action = np.asarray(action, dtype=np.float32).copy()

        if self._dr_enabled:
            wheel_mask = np.random.random(2) < self._wheel_noise_prob
            wheel_noise = np.random.normal(0, self._wheel_noise_std, 2)
            action = np.clip(action + wheel_noise * wheel_mask, -1.0, 1.0)

        obs_dict, reward, terminated, truncated, info = self._inner.step(action)
        food = self._get_food_count(info)

        blob = obs_dict["blob"].copy()
        ir = obs_dict["ir"].copy()

        if self._dr_enabled:
            ir_mask = np.random.random(len(ir)) < self._ir_noise_prob
            ir_noise = np.random.normal(0, self._ir_noise_std, len(ir))
            ir = np.clip(ir + ir_noise * ir_mask, 0.0, 1.0)

            if blob[3] > 0.5:
                blob[0] += np.random.normal(0, self._pose_jitter)
                blob[1] += np.random.normal(0, self._pose_jitter)
                blob[2] *= (1.0 + np.random.normal(0, self._pose_jitter * 0.5))
                blob[2] = max(0.0, blob[2])

        obs = {
            "observation": np.concatenate([blob, ir]).astype(np.float32),
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
    parser.add_argument("--no-record", action="store_true",
                        help="Disable episode recording to disk")
    parser.add_argument("--record-dir", type=str, default="recorded_episodes",
                        help="Directory for recorded episodes")
    parser.add_argument("--record-freq", type=int, default=50,
                        help="Save manifest every N episodes")
    parser.add_argument("--domain-randomization", action="store_true",
                        help="Enable IR noise, pose jitter, wheel noise")
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
            domain_randomization=args.domain_randomization,
        ),
        filename=str(log_dir / "monitor.csv"),
    )

    callbacks = []

    if not args.no_record:
        recorder = EpisodeRecorder(
            record_dir=args.record_dir,
            save_freq=args.record_freq,
            verbose=1,
        )
        callbacks.append(recorder)

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
        model.save(str(model_path))
        model.save_replay_buffer(str(buffer_path))
        print(f"Saved model to {model_path}")
        print(f"Saved buffer to {buffer_path}")
        if wandb_run is not None:
            wandb_run.finish()
        env.close()


if __name__ == "__main__":
    main()
