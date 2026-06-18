#!/usr/bin/env python3
"""Evaluate SAC, DreamerV3, or DreamerV4 across transfer domains."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=["sac", "dreamerv3", "dreamerv4"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--domain",
        choices=["fixed", "training", "heldout", "calibration"],
        default="fixed",
    )
    parser.add_argument("--calibration", default="config/calibration/simulation.json")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--output", default="evaluation/results.csv")
    parser.add_argument("--wandb-project", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "catkin_ws/src/learning_machines/src"))
    sys.path.insert(0, str(root / "catkin_ws/src/robobo_interface/src"))
    os.environ["COPPELIA_SIM_PORT"] = str(args.port)

    import torch
    from deploy_hardware import DreamerV3Policy, DreamerV4Policy, SACPolicy
    from learning_machines.domain_randomization import (
        DomainRandomizationWrapper,
        RandomizationRanges,
    )
    from learning_machines.rl_robobo_compact_env import (
        RoboboCompactEnv,
        RoboboCompactEnvConfig,
    )
    from learning_machines.transfer import CalibrationProfile, CheckpointManifest

    profile = CalibrationProfile.load(args.calibration)
    manifest_path = Path(args.manifest) if args.manifest else Path(args.checkpoint).parent / "manifest.json"
    manifest = CheckpointManifest.load(manifest_path)
    manifest.validate(
        args.algorithm,
        manifest.calibration_profile if args.domain == "calibration" else profile.name,
        args.image_size,
        manifest.phone_tilt,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.algorithm == "sac":
        policy = SACPolicy(args.checkpoint)
    elif args.algorithm == "dreamerv3":
        policy = DreamerV3Policy(args.checkpoint)
    else:
        policy = DreamerV4Policy(args.checkpoint, device)

    env = RoboboCompactEnv(config=RoboboCompactEnvConfig(
        return_image=args.algorithm != "sac",
        image_obs_size=(args.image_size, args.image_size),
        calibration_profile=profile,
        randomize_food_positions=args.domain != "fixed",
        phone_tilt=manifest.phone_tilt,
        max_episode_seconds=args.max_seconds,
    ))
    if args.domain in {"training", "heldout"}:
        ranges = RandomizationRanges()
        if args.domain == "heldout":
            ranges = RandomizationRanges(
                ir_gain=(0.75, 1.25),
                ir_bias=(-0.1, 0.1),
                motor_gain=(0.75, 1.25),
                motor_bias=(-0.1, 0.1),
                motor_deadband=(0.0, 0.14),
                motor_latency_steps=(0, 2),
                camera_exposure=(-0.2, 0.2),
                camera_contrast=(0.7, 1.3),
                camera_tilt_offset=(-7, 7),
                camera_shift_pixels=(-6.0, 6.0),
                ir_noise_std=0.03,
                image_noise_std=0.02,
                action_jitter_std=0.03,
            )
        env = DomainRandomizationWrapper(env, ranges=ranges)

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f"eval-{args.algorithm}-{args.domain}",
            config=vars(args),
        )

    rows = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=10_000 + episode)
            policy.reset()
            total_reward = 0.0
            final_info = {}
            while True:
                requested = policy.act(obs)
                obs, reward, terminated, truncated, info = env.step(requested)
                policy.observe_executed(np.asarray(info["executed_action"], dtype=np.float32))
                total_reward += float(reward)
                final_info = info
                if terminated or truncated:
                    break
            row = {
                "algorithm": args.algorithm,
                "domain": args.domain,
                "episode": episode,
                "return": total_reward,
                "completion": int(final_info.get("completion_time") is not None),
                "completion_seconds": final_info.get("completion_time") or np.nan,
                "elapsed_seconds": final_info.get("elapsed_seconds", 0.0),
                "food_collected": final_info.get("food_collected", 0),
                "food_per_minute": final_info.get("food_per_minute", 0.0),
                "collisions": final_info.get("collisions", 0),
                "safety_overrides": final_info.get("safety_overrides", 0),
                "mean_action_change": final_info.get("mean_action_change", 0.0),
                "action_saturation_rate": final_info.get("action_saturation_rate", 0.0),
            }
            rows.append(row)
            if wandb_run is not None:
                wandb_run.log({f"episode/{key}": value for key, value in row.items()
                               if isinstance(value, (int, float, np.number))}, step=episode)
    finally:
        env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    summary = {
        "algorithm": args.algorithm,
        "domain": args.domain,
        "episodes": len(rows),
        "completion_rate": float(np.mean([row["completion"] for row in rows])),
        "median_completion_seconds": float(np.nanmedian([
            row["completion_seconds"] for row in rows
        ])) if any(row["completion"] for row in rows) else None,
        "mean_food_per_minute": float(np.mean([row["food_per_minute"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_safety_overrides": float(np.mean([row["safety_overrides"] for row in rows])),
        "mean_action_change": float(np.mean([row["mean_action_change"] for row in rows])),
        "mean_action_saturation": float(np.mean([
            row["action_saturation_rate"] for row in rows
        ])),
    }
    print(json.dumps(summary, indent=2))
    if wandb_run is not None:
        wandb_run.log({f"summary/{key}": value for key, value in summary.items()
                       if isinstance(value, (int, float))})
        wandb_run.finish()


if __name__ == "__main__":
    main()
