"""Generate synthetic episodes for testing DreamerV4.

Creates fake episode data in the same format as recorded_episodes/ so we can
test the DreamerV4 pipeline without running CoppeliaSim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def generate_synthetic_episode(
    length: int,
    obs_dim: int = 12,
    act_dim: int = 2,
    num_food: int = 7,
) -> dict[str, np.ndarray]:
    """Generate a synthetic episode with realistic-ish dynamics.

    Simulates a robot wandering in an arena with IR sensors responding
    to nearby walls and food blobs. Food is collected randomly.
    """
    observations = np.zeros((length + 1, obs_dim), dtype=np.float32)
    actions = np.zeros((length, act_dim), dtype=np.float32)
    rewards = np.zeros(length, dtype=np.float32)
    dones = np.zeros(length, dtype=bool)

    x, y = np.random.uniform(-3.5, -2.5), np.random.uniform(0.3, 1.3)
    heading = np.random.uniform(-np.pi, np.pi)
    food_collected = 0

    for t in range(length):
        steer = np.random.uniform(-1.0, 1.0)
        speed = np.random.uniform(0.0, 1.0)
        actions[t] = [steer, speed]

        x += speed * 0.05 * np.cos(heading + steer * 0.5)
        y += speed * 0.05 * np.sin(heading + steer * 0.5)
        heading += steer * 0.1

        blob_x = np.random.uniform(-1.0, 1.0) if food_collected < num_food else 0.0
        blob_y = np.random.uniform(-1.0, 1.0) if food_collected < num_food else 0.0
        blob_area = np.random.uniform(500, 3000) if food_collected < num_food else 0.0
        blob_found = 1.0 if food_collected < num_food and np.random.random() > 0.3 else 0.0

        ir = np.random.uniform(0.1, 0.9, size=8).astype(np.float32)
        ir[4] = max(0, 1.0 - abs(x - (-3.125)) * 0.5)

        observations[t] = np.concatenate([
            [blob_x, blob_y, blob_area / 3000.0, blob_found],
            ir,
        ])

        if food_collected < num_food and np.random.random() < 0.05:
            food_collected += 1
            rewards[t] = 100.0
        else:
            rewards[t] = 0.0

        if t == length - 1:
            dones[t] = True

    observations[length] = observations[length - 1]

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic episodes")
    parser.add_argument("--output-dir", type=str, default="recorded_episodes")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--min-length", type=int, default=20)
    parser.add_argument("--max-length", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for i in range(args.num_episodes):
        length = np.random.randint(args.min_length, args.max_length + 1)
        ep = generate_synthetic_episode(length)

        ep_path = episodes_dir / f"ep_{i:06d}.npz"
        np.savez_compressed(ep_path, **ep)

        manifest.append({
            "id": i,
            "file": f"ep_{i:06d}.npz",
            "length": length,
            "total_reward": float(ep["rewards"].sum()),
            "food_collected": int(ep["rewards"].sum() // 100),
        })

    manifest_path = output_dir / "episodes.jsonl"
    with open(manifest_path, "w") as f:
        for ep in manifest:
            f.write(json.dumps(ep) + "\n")

    total_reward = sum(ep["total_reward"] for ep in manifest)
    print(f"Generated {args.num_episodes} synthetic episodes to {output_dir}")
    print(f"Total reward: {total_reward:.0f}, avg: {total_reward / args.num_episodes:.1f}")


if __name__ == "__main__":
    main()
