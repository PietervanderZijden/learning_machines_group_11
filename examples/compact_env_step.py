#!/usr/bin/env python3
"""Construct and step the compact Robobo environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "catkin_ws/src/learning_machines/src"))
sys.path.insert(0, str(ROOT / "catkin_ws/src/robobo_interface/src"))

from learning_machines.rl_robobo_compact_env import (  # noqa: E402
    RoboboCompactEnv,
    RoboboCompactEnvConfig,
)
from robobo_interface import SimulationRobobo  # noqa: E402


def main() -> int:
    """Run random actions for a bounded number of simulator steps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    rob = SimulationRobobo(ip_adress=args.host, api_port=args.port)
    env = RoboboCompactEnv(
        rob=rob,
        config=RoboboCompactEnvConfig(max_episode_steps=args.steps),
    )
    try:
        env.reset()
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
