'Replay buffer with episode-based storage for DreamerV3.'
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ReplayBuffer:
    'Episode-based replay buffer that stores complete episodes.'

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
        if not 0.0 <= self.reward_event_fraction <= 1.0:
            raise ValueError("reward_event_fraction must be in [0, 1]")


        self._episodes: list[dict[str, np.ndarray]] = []
        self._total_transitions = 0
        self._current_episode: dict[str, list] = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "ir": [] if ir_dim > 0 else None,
        }

    def restore_recorded_episodes(self, record_dir: str | Path) -> int:
        'Restore the newest aligned image episodes up to replay capacity.'
        episode_dir = Path(record_dir) / "episodes"
        if not episode_dir.exists():
            return 0
        selected: list[Path] = []
        transitions = 0
        for path in reversed(sorted(episode_dir.glob("ep_*.npz"))):
            try:
                with np.load(path, allow_pickle=False) as data:
                    length = len(data["actions"])
                    valid = (
                        "images" in data
                        and "rewards" in data
                        and "dones" in data
                        and data.get("observation_contract", np.array("")).item()
                        == "robobo-obs-v2"
                        and data.get("reward_contract", np.array("")).item()
                        == "robobo-reward-v4"
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
            except (OSError, KeyError, ValueError):
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
                    data["dones"].astype(np.float32),
                    ir=ir,
                )
                restored += len(data["actions"])
        return self.size

    @property
    def size(self) -> int:
        return self._total_transitions

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        ir: np.ndarray | None = None,
        next_obs: np.ndarray | None = None,
        next_ir: np.ndarray | None = None,
    ):
        'Add a single transition to the current episode.'
        self._current_episode["obs"].append(obs)
        self._current_episode["action"].append(action)
        self._current_episode["reward"].append(reward)
        self._current_episode["done"].append(float(done))

        if self.ir_dim > 0 and ir is not None:
            self._current_episode["ir"].append(ir)

        if done:
            if next_obs is None:
                raise ValueError("terminal transitions require next_obs for T+1 alignment")
            self._current_episode["obs"].append(next_obs)
            if self.ir_dim > 0:
                if next_ir is None:
                    raise ValueError("terminal multimodal transitions require next_ir")
                self._current_episode["ir"].append(next_ir)
            self._finalize_episode()

    def _finalize_episode(self):
        'Convert current episode lists to arrays and store.'
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

        ep_len = len(episode["reward"])
        self._total_transitions += ep_len


        while self._total_transitions > self.capacity and len(self._episodes) > 0:
            removed = self._episodes.pop(0)
            self._total_transitions -= len(removed["reward"])

        self._episodes.append(episode)


        self._current_episode = {
            "obs": [],
            "action": [],
            "reward": [],
            "done": [],
            "ir": [] if self.ir_dim > 0 else None,
        }

    def add_episode(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        ir: np.ndarray | None = None,
    ):
        'Add a full episode directly.'
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

        ep_len = len(reward)
        self._total_transitions += ep_len


        while self._total_transitions > self.capacity and len(self._episodes) > 0:
            removed = self._episodes.pop(0)
            self._total_transitions -= len(removed["reward"])

        self._episodes.append(episode)

    def sample(
        self, batch_size: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        'Sample random subsequences of length `sequence_length` from episodes.'
        if len(self._episodes) == 0:
            raise ValueError("No episodes in buffer")

        batch_obs = []
        batch_action = []
        batch_reward = []
        batch_done = []
        batch_discount = []
        batch_ir = []
        batch_indices = []
        event_samples = 0

        attempts = 0
        max_attempts = batch_size * 100
        event_episode_indices = [
            index
            for index, episode in enumerate(self._episodes)
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
                else np.random.randint(0, len(self._episodes))
            )
            episode = self._episodes[ep_idx]
            ep_len = len(episode["reward"])


            seq_len = min(self.sequence_length, ep_len)


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


            mid_dones = episode["done"][start:end - 1]
            if np.any(mid_dones > 0.5):
                continue


            batch_obs.append(episode["obs"][start:end + 1])
            batch_action.append(episode["action"][start:end])
            batch_reward.append(episode["reward"][start:end])
            batch_done.append(episode["done"][start:end])
            batch_discount.append(1.0 - episode["done"][start:end])

            if "ir" in episode and episode["ir"] is not None:
                batch_ir.append(episode["ir"][start:end + 1])


            batch_indices.append(ep_idx * 10000 + start)
            event_samples += int(
                np.any(
                    episode["reward"][start:end]
                    >= self.reward_event_threshold
                )
            )

        if len(batch_indices) < batch_size:

            while len(batch_indices) < batch_size:
                ep_idx = np.random.randint(0, len(self._episodes))
                episode = self._episodes[ep_idx]
                ep_len = len(episode["reward"])


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

                batch_indices.append(ep_idx * 10000 + start)
                event_samples += int(
                    np.any(
                        episode["reward"][start:end]
                        >= self.reward_event_threshold
                    )
                )



        min_obs_len = min(arr.shape[0] for arr in batch_obs)
        min_action_len = min(arr.shape[0] for arr in batch_action)


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


        if self.ir_dim > 0 and len(batch_ir) > 0:
            result["ir"] = torch.tensor(
                np.array([arr[:effective_seq_len + 1] for arr in batch_ir]),
                device=device,
            )

        return result
