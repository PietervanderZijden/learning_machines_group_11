"""Benchmark RoboboCompactEnv simulation throughput with random actions."""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RoboboCompactEnv step/reset speed")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--max-episode-steps", type=int, default=150)
    parser.add_argument("--return-image", action="store_true")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--no-blob-camera", action="store_true")
    parser.add_argument("--include-position-info", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "learning_machines" / "src"))
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))
    os.environ["COPPELIA_SIM_PORT"] = str(args.port)

    from learning_machines.rl_robobo_compact_env import (  # noqa: PLC0415
        RoboboCompactEnv,
        RoboboCompactEnvConfig,
    )

    config = RoboboCompactEnvConfig(
        max_episode_steps=args.max_episode_steps,
        randomize_food_positions=not args.no_randomize,
        return_image=args.return_image,
        image_obs_size=(args.image_size, args.image_size),
        detect_blob_from_camera=not args.no_blob_camera,
        include_position_info=args.include_position_info,
    )
    env = RoboboCompactEnv(config=config)

    reset_times: list[float] = []
    step_times: list[float] = []
    phase_times: dict[str, list[float]] = defaultdict(list)

    try:
        start = time.perf_counter()
        reset_start = time.perf_counter()
        env.reset()
        reset_times.append(time.perf_counter() - reset_start)

        completed = 0
        while completed < args.steps:
            action = env.action_space.sample()
            step_start = time.perf_counter()
            _obs, _reward, terminated, truncated, _info = env.step(action)
            step_times.append(time.perf_counter() - step_start)
            for key, value in env._last_step_timing.items():
                phase_times[key].append(value)

            completed += 1
            if terminated or truncated:
                reset_start = time.perf_counter()
                env.reset()
                reset_times.append(time.perf_counter() - reset_start)

        elapsed = time.perf_counter() - start
    finally:
        env.close()

    print(f"steps: {len(step_times)}")
    print(f"elapsed_sec: {elapsed:.3f}")
    print(f"steps_per_sec: {len(step_times) / elapsed:.3f}")
    print(f"avg_reset_sec: {_mean(reset_times):.4f}")
    print(f"avg_step_sec: {_mean(step_times):.4f}")
    for key in sorted(phase_times):
        print(f"avg_{key}_sec: {_mean(phase_times[key]):.4f}")


if __name__ == "__main__":
    main()
