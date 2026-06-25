"""Replay buffer with episode-based storage for DreamerV3.

Stores complete episodes and samples random subsequences from them.
This avoids circular boundary issues and ensures clean episode boundaries.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ReplayBuffer:
    """Episode-based replay buffer that stores complete episodes.

    Samples random subsequences of length `sequence_length` from stored episodes.
    This avoids circular boundary crossing issues present in flat circular buffers.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        sequence_length: int = 50,
        num_chunks: int = 4,
        obs_shape: tuple | None = None,
        ir_dim: int = 0,
        reward_event_fraction: float = 0.25,
        reward_event_threshold: float = 1.0,
        online_queue_capacity: int = 4096,
        store_latent_states: bool = True,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.num_chunks = num_chunks
        self.obs_shape = obs_shape if obs_shape is not None else (obs_dim,)
        self.ir_dim = ir_dim
        self.reward_event_fraction = float(reward_event_fraction)
        self.reward_event_threshold = float(reward_event_threshold)
        self.online_queue_capacity = int(online_queue_capacity)
        self.store_latent_states = bool(store_latent_states)
        if not 0.0 <= self.reward_event_fraction <= 1.0:
            raise ValueError("reward_event_fraction must be in [0, 1]")

        # Store complete episodes
        self._episodes: list[dict[str, np.ndarray]] = []
        self._online_queue: list[dict[str, np.ndarray]] = []
        self._online_transitions = 0
        self._total_transitions = 0
        self._current_episode: dict[str, list] = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "ir": [] if ir_dim > 0 else None,
            "latent_state": [] if store_latent_states else None,
        }

    def restore_recorded_episodes(self, record_dir: str | Path) -> int:
        """Restore the newest aligned image episodes up to replay capacity."""
        episode_dir = Path(record_dir) / "episodes"
        if not episode_dir.exists():
            return 0
        selected: list[Path] = []
        transitions = 0
        for path in reversed(sorted(episode_dir.glob("ep_*.npz"))):
            try:
                with np.load(path, allow_pickle=False) as data:
                    length = len(data["actions"])
                    observation_contract = data.get(
                        "observation_contract", np.array("")
                    ).item()
                    reward_contract = data.get(
                        "reward_contract", np.array("")
                    ).item()
                    if (
                        observation_contract == "robobo-push-obs-v1"
                        and reward_contract
                        and reward_contract != "robobo-push-phased-dense-v1"
                    ):
                        raise ValueError(
                            f"incompatible recorded push reward contract "
                            f"{reward_contract}; expected "
                            "robobo-push-phased-dense-v1"
                        )
                    valid = (
                        "images" in data
                        and "rewards" in data
                        and "dones" in data
                        and "terminals" in data
                        and data.get("observation_contract", np.array("")).item()
                        == "robobo-push-obs-v1"
                        and data.get("reward_contract", np.array("")).item()
                        == "robobo-push-phased-dense-v1"
                        and np.isclose(
                            data.get(
                                "control_interval_seconds", np.array(-1.0)
                            ).item(),
                            0.4,
                        )
                        and len(data["images"]) == length + 1
                        and len(data["rewards"]) == length
                    )
                    if self.ir_dim > 0:
                        valid = valid and "irs" in data and len(data["irs"]) == length + 1
                if not valid:
                    continue
            except (OSError, KeyError):
                continue
            selected.append(path)
            transitions += length
            if transitions >= self.capacity:
                break

        restored = 0
        for path in reversed(selected):
            with np.load(path, allow_pickle=False) as data:
                images = data["images"].astype(np.float32) / 255.0
                ir = data["irs"].astype(np.float32) if self.ir_dim > 0 else None
                self.add_episode(
                    images,
                    data["actions"].astype(np.float32),
                    data["rewards"].astype(np.float32),
                    data["terminals"].astype(np.float32),
                    ir=ir,
                )
                restored += len(data["actions"])
        return self.size

    @property
    def size(self) -> int:
        return self._total_transitions

    def state_dict(self) -> dict:
        """Return complete replay state for checkpoint/resume."""
        return {
            "episodes": self._episodes,
            "online_queue": self._online_queue,
            "online_transitions": self._online_transitions,
            "total_transitions": self._total_transitions,
            "current_episode": self._current_episode,
        }

    def load_state_dict(self, state: dict) -> None:
        self._episodes = list(state.get("episodes", ()))
        self._online_queue = list(state.get("online_queue", ()))
        self._online_transitions = int(state.get("online_transitions", 0))
        self._total_transitions = int(state.get("total_transitions", 0))
        self._current_episode = state.get("current_episode", self._current_episode)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        terminal: bool | None = None,
        ir: np.ndarray | None = None,
        next_obs: np.ndarray | None = None,
        next_ir: np.ndarray | None = None,
        latent_state: np.ndarray | None = None,
        next_latent_state: np.ndarray | None = None,
    ):
        """Add a single transition to the current episode.

        Args:
            obs: observation (obs_dim,) or (C, H, W) for images
            action: action (action_dim,)
            reward: scalar reward
            done: episode-boundary flag (success or time limit)
            terminal: true task terminal; false for time-limit truncation
            ir: optional IR data (ir_dim,) for multi-modal
        """
        self._current_episode["obs"].append(obs)
        self._current_episode["action"].append(action)
        self._current_episode["reward"].append(reward)
        self._current_episode["done"].append(
            float(done if terminal is None else terminal)
        )

        if self.ir_dim > 0 and ir is not None:
            self._current_episode["ir"].append(ir)
        if self.store_latent_states and latent_state is not None:
            self._current_episode["latent_state"].append(latent_state)

        if done:
            if next_obs is None:
                raise ValueError("terminal transitions require next_obs for T+1 alignment")
            self._current_episode["obs"].append(next_obs)
            if self.ir_dim > 0:
                if next_ir is None:
                    raise ValueError("terminal multimodal transitions require next_ir")
                self._current_episode["ir"].append(next_ir)
            if self.store_latent_states and self._current_episode["latent_state"]:
                if next_latent_state is None:
                    raise ValueError("terminal transitions with latent replay require next_latent_state")
                self._current_episode["latent_state"].append(next_latent_state)
            self._finalize_episode()

    def _finalize_episode(self):
        """Convert current episode lists to arrays and store."""
        episode = {
            "obs": np.array(self._current_episode["obs"], dtype=np.float32),
            "action": np.array(self._current_episode["action"], dtype=np.float32),
            "reward": np.array(self._current_episode["reward"], dtype=np.float32),
            "done": np.array(self._current_episode["done"], dtype=np.float32),
        }
        if len(episode["obs"]) != len(episode["action"]) + 1:
            raise ValueError("Dreamer episodes require T+1 observations for T actions")

        if self.ir_dim > 0 and len(self._current_episode["ir"]) > 0:
            episode["ir"] = np.array(self._current_episode["ir"], dtype=np.float32)
        if (
            self.store_latent_states
            and self._current_episode["latent_state"]
        ):
            episode["latent_state"] = np.array(
                self._current_episode["latent_state"], dtype=np.float32
            )

        ep_len = len(episode["reward"])
        self._total_transitions += ep_len

        # Remove oldest episodes if over capacity
        while self._total_transitions > self.capacity and len(self._episodes) > 0:
            removed = self._episodes.pop(0)
            self._total_transitions -= len(removed["reward"])

        self._episodes.append(episode)
        self._append_online_episode(episode)

        # Reset current episode
        self._current_episode = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "ir": [] if self.ir_dim > 0 else None,
            "latent_state": [] if self.store_latent_states else None,
        }

    def add_episode(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        ir: np.ndarray | None = None,
        latent_state: np.ndarray | None = None,
    ):
        """Add a full episode directly.

        Args:
            obs: (T+1, obs_dim) or (T+1, C, H, W)
            action: (T, action_dim)
            reward: (T,)
            done: (T,) true terminal flags; time limits remain false
            ir: (T+1, ir_dim) optional
        """
        if len(obs) != len(action) + 1:
            raise ValueError("Dreamer episodes require T+1 observations for T actions")
        if len(reward) != len(action) or len(done) != len(action):
            raise ValueError("actions, rewards, and dones must all have length T")
        if ir is not None and len(ir) != len(obs):
            raise ValueError("multimodal IR must have the same T+1 length as observations")
        episode = {
            "obs": np.array(obs, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "reward": np.array(reward, dtype=np.float32),
            "done": np.array(done, dtype=np.float32),
        }

        if ir is not None:
            episode["ir"] = np.array(ir, dtype=np.float32)
        if latent_state is not None:
            if len(latent_state) != len(obs):
                raise ValueError("latent_state replay must have the same T+1 length as observations")
            episode["latent_state"] = np.array(latent_state, dtype=np.float32)

        ep_len = len(reward)
        self._total_transitions += ep_len

        # Remove oldest episodes if over capacity
        while self._total_transitions > self.capacity and len(self._episodes) > 0:
            removed = self._episodes.pop(0)
            self._total_transitions -= len(removed["reward"])

        self._episodes.append(episode)
        self._append_online_episode(episode)

    def _append_online_episode(self, episode: dict[str, np.ndarray]) -> None:
        if self.online_queue_capacity <= 0:
            return
        self._online_queue.append(episode)
        self._online_transitions += len(episode["reward"])
        while (
            self._online_transitions > self.online_queue_capacity
            and self._online_queue
        ):
            removed = self._online_queue.pop(0)
            self._online_transitions -= len(removed["reward"])

    def refresh_latent_states(self, episode_index: int, latent_state: np.ndarray) -> None:
        episode = self._episodes[episode_index]
        if len(latent_state) != len(episode["obs"]):
            raise ValueError("latent_state replay must have the same T+1 length as observations")
        episode["latent_state"] = np.array(latent_state, dtype=np.float32)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Sample random subsequences of length `sequence_length` from episodes.

        Returns dict with:
          obs: (batch, seq_len + 1, obs_dim)   [for next_obs in last step]
          action: (batch, seq_len, action_dim)
          reward: (batch, seq_len)
          done: (batch, seq_len)
          discount: (batch, seq_len)
          index: (batch,) starting indices (episode_idx * 1000 + offset)
        """
        episode_pool = self._episodes + self._online_queue
        if len(episode_pool) == 0:
            raise ValueError("No episodes in buffer")

        batch_obs = []
        batch_action = []
        batch_reward = []
        batch_done = []
        batch_discount = []
        batch_ir = []
        batch_latent = []
        batch_indices = []
        event_samples = 0

        attempts = 0
        max_attempts = batch_size * 100
        event_episode_indices = [
            index
            for index, episode in enumerate(episode_pool)
            if np.any(
                episode["reward"] >= self.reward_event_threshold
            )
        ]

        while len(batch_indices) < batch_size and attempts < max_attempts:
            attempts += 1

            request_event = (
                bool(event_episode_indices)
                and np.random.random() < self.reward_event_fraction
            )
            ep_idx = (
                int(np.random.choice(event_episode_indices))
                if request_event
                else np.random.randint(0, len(episode_pool))
            )
            episode = episode_pool[ep_idx]
            ep_len = len(episode["reward"])

            # For short episodes, use the full episode length as the sequence length
            seq_len = min(self.sequence_length, ep_len)

            # Oversample windows containing sparse reward events.
            max_start = ep_len - seq_len
            event_indices = np.flatnonzero(
                episode["reward"] >= self.reward_event_threshold
            )
            if request_event and event_indices.size > 0:
                event = int(np.random.choice(event_indices))
                lower = max(0, event - seq_len + 1)
                upper = min(event, max_start)
                start = np.random.randint(lower, upper + 1)
            else:
                start = np.random.randint(0, max_start + 1)
            end = start + seq_len

            # Check that no done=True in the middle (only at the end is ok)
            mid_dones = episode["done"][start:end - 1]
            if np.any(mid_dones > 0.5):
                continue

            # Extract subsequence
            batch_obs.append(episode["obs"][start:end + 1])  # seq_len + 1
            batch_action.append(episode["action"][start:end])
            batch_reward.append(episode["reward"][start:end])
            batch_done.append(episode["done"][start:end])
            batch_discount.append(1.0 - episode["done"][start:end])  # discount=0 at done

            if "ir" in episode and episode["ir"] is not None:
                batch_ir.append(episode["ir"][start:end + 1])
            if "latent_state" in episode and episode["latent_state"] is not None:
                batch_latent.append(episode["latent_state"][start:end + 1])

            # Index encodes episode and position for debugging
            batch_indices.append(ep_idx * 10000 + start)
            event_samples += int(
                np.any(
                    episode["reward"][start:end]
                    >= self.reward_event_threshold
                )
            )

        if len(batch_indices) < batch_size:
            # Fallback: sample without strict episode boundary checking
            while len(batch_indices) < batch_size:
                ep_idx = np.random.randint(0, len(episode_pool))
                episode = episode_pool[ep_idx]
                ep_len = len(episode["reward"])

                # Use full episode length if shorter than sequence_length
                seq_len = min(self.sequence_length, ep_len)
                start = np.random.randint(0, ep_len - seq_len + 1)
                end = start + seq_len

                batch_obs.append(episode["obs"][start:end + 1])
                batch_action.append(episode["action"][start:end])
                batch_reward.append(episode["reward"][start:end])
                batch_done.append(episode["done"][start:end])
                batch_discount.append(1.0 - episode["done"][start:end])

                if "ir" in episode and episode["ir"] is not None:
                    batch_ir.append(episode["ir"][start:end + 1])
                if "latent_state" in episode and episode["latent_state"] is not None:
                    batch_latent.append(episode["latent_state"][start:end + 1])

                batch_indices.append(ep_idx * 10000 + start)
                event_samples += int(
                    np.any(
                        episode["reward"][start:end]
                        >= self.reward_event_threshold
                    )
                )

        # Truncate all samples to the minimum length in the batch
        # to avoid padding issues (padded positions would contribute to loss)
        min_obs_len = min(arr.shape[0] for arr in batch_obs)
        min_action_len = min(arr.shape[0] for arr in batch_action)
        # Use the min across obs (seq_len+1) and action (seq_len)
        # obs has one more step than action
        effective_seq_len = min(min_action_len, min_obs_len - 1)

        result = {
            "obs": torch.tensor(
                np.array([arr[:effective_seq_len + 1] for arr in batch_obs]),
                device=device,
            ),
            "action": torch.tensor(
                np.array([arr[:effective_seq_len] for arr in batch_action]),
                device=device,
            ),
            "reward": torch.tensor(
                np.array([arr[:effective_seq_len] for arr in batch_reward]),
                device=device,
            ),
            "done": torch.tensor(
                np.array([arr[:effective_seq_len] for arr in batch_done]),
                device=device,
            ),
            "discount": torch.tensor(
                np.array([arr[:effective_seq_len] for arr in batch_discount]),
                device=device,
            ),
            "index": torch.tensor(np.array(batch_indices), device=device),
            "reward_event_sample_fraction": torch.tensor(
                event_samples / max(1, batch_size),
                dtype=torch.float32,
                device=device,
            ),
        }

        # Include IR data if available
        if self.ir_dim > 0 and len(batch_ir) > 0:
            result["ir"] = torch.tensor(
                np.array([arr[:effective_seq_len + 1] for arr in batch_ir]),
                device=device,
            )
        if len(batch_latent) == len(batch_indices):
            result["latent_state"] = torch.tensor(
                np.array([arr[:effective_seq_len + 1] for arr in batch_latent]),
                device=device,
            )

        return result
