"""Tests for food dataset recording."""

from __future__ import annotations

import numpy as np

from record_food import new_episode, reactive_action, save_episode
from train_dreamerv4_image import scan_recorded_episodes


def test_reactive_action_turns_toward_blob() -> None:
    """Steer toward a visible target."""
    left = reactive_action(
        {"blob": np.array([0.2, 0.5, 0.1, 1.0], dtype=np.float32)}
    )
    right = reactive_action(
        {"blob": np.array([0.8, 0.5, 0.1, 1.0], dtype=np.float32)}
    )
    assert left[0] < left[1]
    assert right[0] > right[1]


def test_saved_episode_matches_offline_scanner(tmp_path) -> None:
    """Write a dataset accepted by DreamerV4-full scanning."""
    observation = {
        "image": np.zeros((3, 16, 16), dtype=np.uint8),
        "ir": np.zeros(8, dtype=np.float32),
    }
    episode = new_episode(observation)
    episode["actions"].append(np.zeros(2, dtype=np.float32))
    episode["rewards"].append(0.0)
    episode["dones"].append(False)
    episode["images"].append(observation["image"])
    episode["irs"].append(observation["ir"])
    save_episode(tmp_path / "episodes/ep_000000.npz", episode, "simulation")
    scan = scan_recorded_episodes(tmp_path, "simulation")
    assert len(scan.episodes) == 1
    assert scan.transitions == 1
