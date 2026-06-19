#!/usr/bin/env python3
"""Evaluate SAC, DreamerV3, or DreamerV4 across transfer domains."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
from pathlib import Path

import numpy as np


def _load_repository_env(path: Path) -> list[str]:
    """Load simple .env values without overriding the caller's environment."""
    loaded = []
    if not path.exists():
        return loaded
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not name or name in os.environ:
            continue
        try:
            values = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError(f"invalid .env value for {name}") from exc
        value = values[0] if values else ""
        os.environ[name] = value
        loaded.append(name)
    return loaded


def _dreamerv3_prior_prediction(policy, executed_action: np.ndarray) -> dict:
    """Predict x[t+1] from the filtered state at t without observing x[t+1]."""
    import torch
    import torch.nn.functional as F
    from learning_machines.distributional import logits_to_value

    if policy.state is None:
        raise RuntimeError("DreamerV3 policy state is unavailable")
    agent = policy.agent
    world_model = agent.world_model
    h, z = policy.state
    action = torch.as_tensor(
        executed_action, dtype=torch.float32, device=agent.device
    ).reshape(1, agent.cfg.action_dim)
    with torch.no_grad():
        next_h, _sampled_z, prior_logits = world_model.rssm.imagine(
            action, h, z
        )
        next_z = world_model.rssm.get_stochastic(prior_logits)
        next_state = torch.cat([next_h, next_z], dim=-1)
        image = world_model.obs_decoder(next_state)
        reward_features = F.silu(
            world_model.reward_norm(world_model.reward_hidden(next_state))
        )
        reward_logits = world_model.reward_head(reward_features)
        result = {
            "image": image[0].cpu().numpy(),
            "reward": float(logits_to_value(reward_logits)[0].cpu()),
            "continue_probability": float(
                torch.sigmoid(world_model.continue_head(reward_features))[0, 0].cpu()
            ),
            "value": float(agent.critic.value(next_state)[0].cpu()),
        }
        if world_model.use_multimodal:
            result["ir"] = world_model.ir_decoder(next_state)[0].cpu().numpy()
        return result


def _prediction_metrics(
    prediction: dict,
    next_obs: dict,
    reward: float,
    done: bool,
) -> dict[str, float]:
    actual_image = next_obs["image"].astype(np.float32) / 255.0
    predicted_image = np.clip(
        np.asarray(prediction["image"], dtype=np.float32), 0.0, 1.0
    )
    error = predicted_image - actual_image
    mse = float(np.mean(np.square(error)))
    actual_green = np.maximum(
        0.0,
        actual_image[1] - np.maximum(actual_image[0], actual_image[2]),
    )
    predicted_green = np.maximum(
        0.0,
        predicted_image[1] - np.maximum(predicted_image[0], predicted_image[2]),
    )
    metrics = {
        "image_mse": mse,
        "image_mae": float(np.mean(np.abs(error))),
        "image_psnr_db": (
            float(-10.0 * math.log10(max(mse, 1e-12)))
        ),
        "green_saliency_mse": float(
            np.mean(np.square(predicted_green - actual_green))
        ),
        "predicted_reward": float(prediction["reward"]),
        "reward_error": float(prediction["reward"] - reward),
        "predicted_continue_probability": float(
            prediction["continue_probability"]
        ),
        "continue_target": float(not done),
        "predicted_value": float(prediction["value"]),
    }
    if "ir" in prediction:
        actual_ir = np.asarray(next_obs["ir"], dtype=np.float32)
        predicted_ir = np.asarray(prediction["ir"], dtype=np.float32)
        metrics["ir_mse"] = float(np.mean(np.square(predicted_ir - actual_ir)))
        metrics["ir_mae"] = float(np.mean(np.abs(predicted_ir - actual_ir)))
    return metrics


def _save_prediction_comparison(
    path: Path,
    prediction: np.ndarray,
    actual: np.ndarray,
) -> None:
    import cv2

    predicted_rgb = np.transpose(
        np.clip(prediction, 0.0, 1.0), (1, 2, 0)
    )
    actual_rgb = np.transpose(actual.astype(np.float32) / 255.0, (1, 2, 0))
    error_rgb = np.abs(predicted_rgb - actual_rgb)
    panels = [
        ("Predicted next", predicted_rgb),
        ("Actual next", actual_rgb),
        ("Absolute error", error_rgb),
    ]
    rendered = []
    for label, panel in panels:
        image = np.clip(panel * 255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.copyMakeBorder(
            image, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(
            image, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (0, 0, 0), 1, cv2.LINE_AA,
        )
        rendered.append(image)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(rendered, axis=1))


def _save_blob_overlay(path: Path, image: np.ndarray, blob: np.ndarray) -> None:
    """Save the camera frame with SAC's detected green-blob output."""
    import cv2

    rgb = np.transpose(image, (1, 2, 0))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    x, y, area, visible = [float(value) for value in blob]
    if visible > 0.5:
        center = (
            int(np.clip(round(x * width), 0, width - 1)),
            int(np.clip(round(y * height), 0, height - 1)),
        )
        radius = max(3, int(round(math.sqrt(max(area, 0.0) * width * height / math.pi))))
        cv2.circle(bgr, center, radius, (0, 0, 255), 2)
        cv2.drawMarker(bgr, center, (255, 0, 0), cv2.MARKER_CROSS, 8, 1)
    label = f"visible={int(visible > 0.5)} x={x:.3f} y={y:.3f} area={area:.4f}"
    bgr = cv2.copyMakeBorder(
        bgr, 22, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    cv2.putText(
        bgr, label, (3, 15), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 0), 1, cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


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
    parser.add_argument(
        "--diagnostics-dir",
        default=None,
        help="Per-step metrics and diagnostic images directory.",
    )
    parser.add_argument(
        "--diagnostic-image-every",
        type=int,
        default=25,
        help="Save one predicted/actual comparison every N transitions.",
    )
    parser.add_argument(
        "--max-diagnostic-images",
        type=int,
        default=40,
        help="Maximum report images saved across the evaluation.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.diagnostic_image_every <= 0:
        parser.error("--diagnostic-image-every must be positive")
    if args.max_diagnostic_images < 0:
        parser.error("--max-diagnostic-images cannot be negative")

    root = Path(__file__).resolve().parent
    loaded_env = _load_repository_env(root / ".env")
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
        return_image=True,
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
        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            parser.error(
                "W&B logging requires WANDB_API_KEY in the shell or repository .env"
            )
        try:
            wandb.login(key=api_key, verify=True)
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=os.environ.get("WANDB_ENTITY"),
                name=f"eval-{args.algorithm}-{args.domain}",
                config={
                    **vars(args),
                    "env_values_loaded": [
                        name for name in loaded_env if name != "WANDB_API_KEY"
                    ],
                },
            )
        except Exception as exc:
            raise RuntimeError(
                "W&B authentication failed. Verify WANDB_API_KEY and, if the "
                "project belongs to a team, set WANDB_ENTITY to that team name."
            ) from exc
        wandb.define_metric("evaluation/transition")
        wandb.define_metric("diagnostics/*", step_metric="evaluation/transition")
        wandb.define_metric("evaluation/episode")
        wandb.define_metric("episode/*", step_metric="evaluation/episode")

    output = Path(args.output)
    diagnostics_dir = (
        Path(args.diagnostics_dir)
        if args.diagnostics_dir
        else output.with_suffix("").parent / f"{output.stem}_diagnostics"
    )
    diagnostic_rows = []
    saved_diagnostic_images = 0
    global_transition = 0
    rows = []
    try:
        for episode in range(args.episodes):
            obs, _ = env.reset(seed=10_000 + episode)
            policy.reset()
            total_reward = 0.0
            final_info = {}
            episode_step = 0
            while True:
                requested = policy.act(obs)
                next_obs, reward, terminated, truncated, info = env.step(requested)
                executed = np.asarray(info["executed_action"], dtype=np.float32)
                done = bool(terminated or truncated)
                diagnostic_row = None
                image_path = None
                if args.algorithm == "dreamerv3":
                    prediction = _dreamerv3_prior_prediction(policy, executed)
                    metrics = _prediction_metrics(
                        prediction, next_obs, float(reward), done
                    )
                    diagnostic_row = {
                        "episode": episode,
                        "step": episode_step,
                        "actual_reward": float(reward),
                        "terminated": int(terminated),
                        "truncated": int(truncated),
                        "requested_left": float(requested[0]),
                        "requested_right": float(requested[1]),
                        "executed_left": float(executed[0]),
                        "executed_right": float(executed[1]),
                        **metrics,
                    }
                    diagnostic_rows.append(diagnostic_row)
                    if (
                        saved_diagnostic_images < args.max_diagnostic_images
                        and episode_step % args.diagnostic_image_every == 0
                    ):
                        image_path = diagnostics_dir / "images" / (
                            f"episode_{episode:03d}_step_{episode_step:04d}.png"
                        )
                        _save_prediction_comparison(
                            image_path, prediction["image"], next_obs["image"]
                        )
                        saved_diagnostic_images += 1
                elif args.algorithm == "sac":
                    blob = np.asarray(next_obs["blob"], dtype=np.float32)
                    raw_ir = np.asarray(info.get("raw_ir", [np.nan] * 8), dtype=np.float32)
                    normalized_ir = np.asarray(next_obs["ir"], dtype=np.float32)
                    diagnostic_row = {
                        "episode": episode,
                        "step": episode_step,
                        "reward": float(reward),
                        "terminated": int(terminated),
                        "truncated": int(truncated),
                        "blob_x": float(blob[0]),
                        "blob_y": float(blob[1]),
                        "blob_area": float(blob[2]),
                        "blob_visible": int(blob[3] > 0.5),
                        "requested_left": float(requested[0]),
                        "requested_right": float(requested[1]),
                        "executed_left": float(executed[0]),
                        "executed_right": float(executed[1]),
                        "collision": int(bool(info.get("collision", False))),
                        "safety_override": info.get("safety_override") or "",
                        "food_collected": int(info.get("food_collected", 0)),
                        **{
                            f"ir_{index}": float(value)
                            for index, value in enumerate(normalized_ir)
                        },
                        **{
                            f"raw_ir_{index}": float(value)
                            for index, value in enumerate(raw_ir)
                        },
                    }
                    diagnostic_rows.append(diagnostic_row)
                    if (
                        saved_diagnostic_images < args.max_diagnostic_images
                        and episode_step % args.diagnostic_image_every == 0
                    ):
                        image_path = diagnostics_dir / "images" / (
                            f"episode_{episode:03d}_step_{episode_step:04d}_blob.png"
                        )
                        _save_blob_overlay(
                            image_path, next_obs["image"], blob
                        )
                        saved_diagnostic_images += 1
                if wandb_run is not None and diagnostic_row is not None:
                    import wandb

                    payload = {
                        "evaluation/transition": global_transition,
                        **{
                            f"diagnostics/{key}": value
                            for key, value in diagnostic_row.items()
                            if isinstance(value, (int, float, np.number))
                        },
                    }
                    if image_path is not None:
                        image_key = (
                            "diagnostics/world_model_prediction"
                            if args.algorithm == "dreamerv3"
                            else "diagnostics/blob_detection"
                        )
                        payload[image_key] = wandb.Image(
                            str(image_path),
                            caption=(
                                f"{args.algorithm} episode={episode} "
                                f"step={episode_step}"
                            ),
                        )
                    wandb_run.log(payload)
                policy.observe_executed(executed)
                total_reward += float(reward)
                final_info = info
                obs = next_obs
                episode_step += 1
                global_transition += 1
                if done:
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
                wandb_run.log({
                    "evaluation/episode": episode,
                    **{
                        f"episode/{key}": value
                        for key, value in row.items()
                        if isinstance(value, (int, float, np.number))
                    },
                })
    finally:
        env.close()

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
    if diagnostic_rows:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        step_metrics_path = diagnostics_dir / "step_metrics.csv"
        with step_metrics_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(diagnostic_rows[0])
            )
            writer.writeheader()
            writer.writerows(diagnostic_rows)
        for key in (
            "image_mse",
            "image_mae",
            "image_psnr_db",
            "green_saliency_mse",
            "ir_mse",
            "ir_mae",
            "reward_error",
            "predicted_continue_probability",
            "predicted_value",
        ):
            values = [
                float(row[key]) for row in diagnostic_rows if key in row
            ]
            if values:
                summary[f"world_model_mean_{key}"] = float(np.mean(values))
        summary["diagnostic_steps"] = len(diagnostic_rows)
        summary["diagnostic_images"] = saved_diagnostic_images
        summary["step_metrics"] = str(step_metrics_path)
        summary["diagnostic_images_dir"] = str(diagnostics_dir / "images")
        if args.algorithm == "dreamerv3":
            summary["world_model_prediction_steps"] = len(diagnostic_rows)
            summary["world_model_comparison_images"] = saved_diagnostic_images
        elif args.algorithm == "sac":
            visible_rows = [
                row for row in diagnostic_rows if row["blob_visible"]
            ]
            summary["blob_visible_fraction"] = float(np.mean([
                row["blob_visible"] for row in diagnostic_rows
            ]))
            summary["blob_visible_steps"] = len(visible_rows)
            summary["mean_visible_blob_area"] = (
                float(np.mean([row["blob_area"] for row in visible_rows]))
                if visible_rows else 0.0
            )
            summary["mean_visible_blob_x"] = (
                float(np.mean([row["blob_x"] for row in visible_rows]))
                if visible_rows else None
            )
            summary["mean_visible_blob_y"] = (
                float(np.mean([row["blob_y"] for row in visible_rows]))
                if visible_rows else None
            )
        (diagnostics_dir / "report.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
    print(json.dumps(summary, indent=2))
    if wandb_run is not None:
        import wandb

        wandb_run.log({
            f"summary/{key}": value
            for key, value in summary.items()
            if isinstance(value, (int, float))
        })
        artifact = wandb.Artifact(
            name=f"evaluation-{args.algorithm}-{args.domain}-{wandb_run.id}",
            type="evaluation",
            metadata={
                "algorithm": args.algorithm,
                "domain": args.domain,
                "checkpoint": args.checkpoint,
                "episodes": args.episodes,
            },
        )
        artifact.add_file(str(output), name=output.name)
        if diagnostics_dir.exists():
            artifact.add_dir(
                str(diagnostics_dir), name=diagnostics_dir.name
            )
        wandb_run.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()
