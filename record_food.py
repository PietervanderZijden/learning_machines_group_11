#!/usr/bin/env python3
"""Record food-collection episodes for offline world-model training."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "catkin_ws/src/learning_machines/src"))
sys.path.insert(0, str(ROOT / "catkin_ws/src/robobo_interface/src"))

from learning_machines.coppelia_startup import check_coppelia_service  # noqa: E402
from learning_machines.rl_robobo_compact_env import (  # noqa: E402
    RoboboCompactEnv,
    RoboboCompactEnvConfig,
)
from learning_machines.transfer import (  # noqa: E402
    CalibrationProfile,
    REWARD_CONTRACT_VERSION,
)


def reactive_action(observation: dict[str, np.ndarray]) -> np.ndarray:
    """Choose a blob-seeking action from the compact observation."""
    blob = observation["blob"]
    if blob[3] <= 0.5:
        return np.array([-0.25, 0.25], dtype=np.float32)
    steering = float(np.clip((blob[0] - 0.5) * 1.8, -0.65, 0.65))
    speed = float(np.clip(0.65 - blob[2], 0.2, 0.65))
    return np.array([speed + steering, speed - steering], dtype=np.float32)


def save_episode(
    path: Path,
    episode: dict[str, list],
    calibration: str,
) -> None:
    """Write one episode with the transfer-contract metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        images=np.asarray(episode["images"], dtype=np.uint8),
        irs=np.asarray(episode["irs"], dtype=np.float32),
        actions=np.asarray(episode["actions"], dtype=np.float32),
        rewards=np.asarray(episode["rewards"], dtype=np.float32),
        dones=np.asarray(episode["dones"], dtype=bool),
        observation_contract=np.array("robobo-obs-v2"),
        reward_contract=np.array(REWARD_CONTRACT_VERSION),
        control_interval_seconds=np.array(0.4),
        calibration_profile=np.array(calibration),
        phone_tilt=np.array(100),
    )
    temporary.replace(path)


def new_episode(observation: dict[str, np.ndarray]) -> dict[str, list]:
    """Initialize episode arrays with the first observation."""
    return {
        "images": [observation["image"].copy()],
        "irs": [observation["ir"].copy()],
        "actions": [],
        "rewards": [],
        "dones": [],
    }


def main() -> int:
    """Collect and persist a bounded simulator dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--output", default="recorded_episodes")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument(
        "--calibration", default="config/calibration/simulation.json"
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    check_coppelia_service(args.host, args.port)
    os.environ["COPPELIA_SIM_IP"] = args.host
    os.environ["COPPELIA_SIM_PORT"] = str(args.port)
    calibration = CalibrationProfile.load(args.calibration)
    episode_dir = Path(args.output) / "episodes"
    existing = sorted(episode_dir.glob("ep_*.npz"))
    episode_index = (
        int(existing[-1].stem.split("_")[-1]) + 1 if existing else 0
    )
    env = RoboboCompactEnv(
        config=RoboboCompactEnvConfig(
            return_image=True,
            image_obs_size=(args.image_size, args.image_size),
            calibration_profile=calibration,
        )
    )
    observation, _ = env.reset(seed=args.seed)
    episode = new_episode(observation)
    try:
        for _ in range(args.steps):
            action = reactive_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode["actions"].append(
                np.asarray(info["executed_action"], dtype=np.float32)
            )
            episode["rewards"].append(float(reward))
            episode["dones"].append(done)
            episode["images"].append(observation["image"].copy())
            episode["irs"].append(observation["ir"].copy())
            if done:
                save_episode(
                    episode_dir / f"ep_{episode_index:06d}.npz",
                    episode,
                    calibration.name,
                )
                episode_index += 1
                observation, _ = env.reset()
                episode = new_episode(observation)
        if episode["actions"]:
            save_episode(
                episode_dir / f"ep_{episode_index:06d}.npz",
                episode,
                calibration.name,
            )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
