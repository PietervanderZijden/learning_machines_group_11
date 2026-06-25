#!/usr/bin/env python3
"""Evaluation-only NM512 DreamerV3 rollout in CoppeliaSim."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from deploy_dreamerv3_reference_hardware import (
    REFERENCE_IMAGE_SIZE,
    ReferenceDreamerV3Policy,
    validate_contract_file,
)


def episode_summary(
    episode: int,
    reward: float,
    steps: int,
    success: bool,
    info: dict,
) -> dict:
    return {
        "episode": int(episode),
        "success": bool(success),
        "reward": float(reward),
        "steps": int(steps),
        "simulated_seconds": float(steps * 0.4),
        "block_goal_distance": float(
            info.get("block_goal_distance", float("nan"))
        ),
        "collisions": int(info.get("collisions", 0)),
        "red_block_visible": bool(info.get("red_block_visible", False)),
        "green_goal_visible": bool(info.get("green_goal_visible", False)),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Showcase an NM512 DreamerV3 checkpoint in the fixed Robobo "
            "push scene without training or domain randomization."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--contract",
        default=None,
        help="Optional reward_contract.json; defaults beside the checkpoint.",
    )
    parser.add_argument("--reference-dir", default="dreamerv3_reference")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--max-wheel-speed",
        type=int,
        default=100,
        help="Use 100 to match reference training.",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pace each 400 ms policy transition in wall-clock time.",
    )
    parser.add_argument(
        "--simulation-manifest",
        default="config/calibration/simulation.json",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 1 <= args.max_wheel_speed <= 100:
        parser.error("--max-wheel-speed must be between 1 and 100")

    checkpoint_path = Path(args.checkpoint).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    calibration_path = Path(args.simulation_manifest).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "catkin_ws/src/learning_machines/src"))
    sys.path.insert(0, str(root / "catkin_ws/src/robobo_interface/src"))

    from learning_machines.coppelia_startup import check_coppelia_service
    from learning_machines.rl_robobo_compact_env import (
        RoboboCompactEnv,
        RoboboCompactEnvConfig,
    )
    from learning_machines.transfer import (
        CONTROL_INTERVAL_SECONDS,
        CalibrationProfile,
    )
    from robobo_interface import SimulationRobobo

    check_coppelia_service(args.host, args.port)
    policy = ReferenceDreamerV3Policy(checkpoint_path, reference_dir)
    contract_path = (
        Path(args.contract).resolve()
        if args.contract
        else checkpoint_path.parent / "reward_contract.json"
    )
    validate_contract_file(policy.contract, contract_path)
    calibration = CalibrationProfile.load(calibration_path)

    rob = SimulationRobobo(api_port=args.port, ip_adress=args.host)
    env = RoboboCompactEnv(
        rob=rob,
        config=RoboboCompactEnvConfig(
            task="push",
            return_image=True,
            image_obs_size=REFERENCE_IMAGE_SIZE,
            phone_tilt=100,
            max_episode_steps=int(policy.contract["max_episode_steps"]),
            max_wheel_speed=args.max_wheel_speed,
            calibration_profile=calibration,
            randomize_push_layout=False,
            push_curriculum_stage=0,
            action_smoothing=False,
            pre_action_safety=False,
            max_action_delta=2.0,
            push_discount=float(policy.contract["discount"]),
            push_block_goal_weight=float(
                policy.contract["push_block_goal_weight"]
            ),
            push_robot_pose_weight=float(
                policy.contract["push_robot_pose_weight"]
            ),
            push_standoff_distance=0.22,
            push_time_penalty_per_second=2.5,
            push_action_change_penalty=0.0,
        ),
    )

    summaries = []
    print(
        "DreamerV3 simulation showcase: fixed layout, deterministic policy, "
        "96x96 RGB + IR, no training, no domain randomization.",
        flush=True,
    )
    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            policy.reset()
            total_reward = 0.0
            steps = 0
            success = False

            while True:
                cycle_start = time.monotonic()
                action = policy.act(obs)
                if not np.isfinite(action).all():
                    raise RuntimeError("policy produced a non-finite action")
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                success = bool(terminated or info.get("push_success", False))

                if args.realtime:
                    remaining = (
                        CONTROL_INTERVAL_SECONDS
                        - (time.monotonic() - cycle_start)
                    )
                    if remaining > 0:
                        time.sleep(remaining)
                if terminated or truncated:
                    break

            summary = episode_summary(
                episode, total_reward, steps, success, info
            )
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    except KeyboardInterrupt:
        print("Showcase interrupted.", flush=True)
    finally:
        env.close()

    successes = sum(item["success"] for item in summaries)
    aggregate = {
        "checkpoint": str(checkpoint_path),
        "scene": "scenes/arena_push_easy.ttt",
        "episodes_completed": len(summaries),
        "successes": successes,
        "success_rate": successes / len(summaries) if summaries else 0.0,
        "episodes": summaries,
    }
    print(json.dumps(aggregate, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(aggregate, indent=2) + "\n")


if __name__ == "__main__":
    main()
