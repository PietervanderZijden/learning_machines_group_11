"""Sequence replay buffer for DreamerV4.

Stores complete episodes and samples random subsequences for transformer training.
"""
from __future__ import annotations
import random
from pathlib import Path
from typing import Iterator

import numpy as np


class SequenceReplayBuffer:
    """Stores episodes, samples random subsequences for transformer training."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        context_length: int = 64,
        max_episodes: int = 100_000,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_length = context_length
        self.max_episodes = max_episodes

        self._episodes: list[dict[str, np.ndarray]] = []
        self._episode_rewards: list[float] = []

    @property
    def size(self) -> int:
        return len(self._episodes)

    def add_episode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray | None = None,
    ):
        """Add a complete episode to the buffer.

        Args:
            observations: (T+1, obs_dim) — includes final observation
            actions: (T, act_dim)
            rewards: (T,)
            dones: (T,) — optional, defaults to all False
        """
        if len(self._episodes) >= self.max_episodes:
            idx = random.randint(0, len(self._episodes) - 1)
            self._episodes.pop(idx)
            self._episode_rewards.pop(idx)

        if dones is None:
            dones = np.zeros(len(rewards), dtype=bool)

        self._episodes.append({
            "observations": observations.astype(np.float32),
            "actions": actions.astype(np.float32),
            "rewards": rewards.astype(np.float32),
            "dones": dones.astype(bool),
        })
        self._episode_rewards.append(float(rewards.sum()))

    def load_from_directory(self, record_dir: str) -> int:
        """Load episodes from recorded NPZ files + manifest."""
        record_path = Path(record_dir)
        manifest_path = record_path / "episodes.jsonl"

        if not manifest_path.exists():
            print(f"[SequenceReplayBuffer] No manifest at {manifest_path}")
            return 0

        count = 0
        with open(manifest_path) as f:
            for line in f:
                ep_info = __import__("json").loads(line.strip())
                ep_file = record_path / "episodes" / ep_info["file"]
                if not ep_file.exists():
                    continue

                data = np.load(ep_file)
                self.add_episode(
                    observations=data["observations"],
                    actions=data["actions"],
                    rewards=data["rewards"],
                    dones=data["dones"],
                )
                count += 1

        print(f"[SequenceReplayBuffer] Loaded {count} episodes from {record_dir}")
        return count

    def sample_batch(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch of random subsequences.

        Returns:
            dict with keys: observations, actions, rewards, dones, masks
            shapes: (batch_size, context_length, ...)
            masks: (batch_size, context_length) — 1 for valid timesteps
        """
        if len(self._episodes) == 0:
            raise ValueError("Buffer is empty")

        batch_obs = []
        batch_act = []
        batch_rew = []
        batch_done = []
        batch_mask = []

        for _ in range(batch_size):
            ep = random.choice(self._episodes)
            T = len(ep["rewards"])

            if T <= self.context_length:
                start = 0
                length = T
            else:
                start = random.randint(0, T - self.context_length)
                length = self.context_length

            obs_slice = ep["observations"][start : start + length + 1]
            act_slice = ep["actions"][start : start + length]
            rew_slice = ep["rewards"][start : start + length]
            done_slice = ep["dones"][start : start + length]

            mask = np.ones(length, dtype=np.float32)

            if T < self.context_length:
                pad_len = self.context_length - length
                obs_slice = np.concatenate([
                    obs_slice,
                    np.zeros((pad_len, self.obs_dim), dtype=np.float32),
                ])
                act_slice = np.concatenate([
                    act_slice,
                    np.zeros((pad_len, self.act_dim), dtype=np.float32),
                ])
                rew_slice = np.concatenate([
                    rew_slice,
                    np.zeros(pad_len, dtype=np.float32),
                ])
                done_slice = np.concatenate([
                    done_slice,
                    np.ones(pad_len, dtype=bool),
                ])
                mask = np.concatenate([
                    mask,
                    np.zeros(pad_len, dtype=np.float32),
                ])

            batch_obs.append(obs_slice)
            batch_act.append(act_slice)
            batch_rew.append(rew_slice)
            batch_done.append(done_slice)
            batch_mask.append(mask)

        return {
            "observations": np.array(batch_obs, dtype=np.float32),
            "actions": np.array(batch_act, dtype=np.float32),
            "rewards": np.array(batch_rew, dtype=np.float32),
            "dones": np.array(batch_done, dtype=bool),
            "masks": np.array(batch_mask, dtype=np.float32),
        }

    def get_all_rewards(self) -> list[float]:
        return list(self._episode_rewards)

    def clear(self):
        self._episodes.clear()
        self._episode_rewards.clear()
