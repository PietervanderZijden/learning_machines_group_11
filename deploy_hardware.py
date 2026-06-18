#!/usr/bin/env python3
"""Unified deterministic hardware deployment for SAC, DreamerV3, and DreamerV4."""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


class OperatorControls:
    """Non-blocking stdin controls: f=food event, e/q=emergency stop."""

    def __init__(self):
        self.events: queue.SimpleQueue[str] = queue.SimpleQueue()
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self):
        self.thread.start()

    def _read(self):
        while True:
            try:
                command = input().strip().lower()
            except EOFError:
                return
            if command in {"f", "food"}:
                self.events.put("food")
            elif command in {"e", "estop", "q", "quit"}:
                self.events.put("estop")
                return

    def drain(self) -> list[str]:
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                return events


class Policy:
    def reset(self):
        pass

    def act(self, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def observe_executed(self, action: np.ndarray):
        pass


class SACPolicy(Policy):
    def __init__(self, checkpoint: str):
        from stable_baselines3 import SAC
        self.model = SAC.load(checkpoint)

    def act(self, obs: dict) -> np.ndarray:
        vector = np.concatenate([obs["blob"], obs["ir"]]).astype(np.float32)
        action, _ = self.model.predict(vector, deterministic=True)
        return np.asarray(action, dtype=np.float32)


class DreamerV3Policy(Policy):
    def __init__(self, checkpoint: str):
        import torch
        from learning_machines.dreamerv3 import DreamerV3
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.agent = DreamerV3(saved["cfg"], device="auto")
        self.agent.load(checkpoint)
        self.state = None

    def reset(self):
        self.state = None
        self.agent.reset_policy_state()

    def act(self, obs: dict) -> np.ndarray:
        image = obs["image"].astype(np.float32) / 255.0
        action, self.state = self.agent.select_action(
            image, state=self.state, ir=obs["ir"], deterministic=True
        )
        return action

    def observe_executed(self, action: np.ndarray):
        self.agent.set_executed_action(action)


class DreamerV4Policy(Policy):
    def __init__(self, checkpoint: str, device):
        from learning_machines.dreamerv4.dreamerv4_image import ImageDreamerV4Agent
        self.agent = ImageDreamerV4Agent.load(checkpoint, device)
        self.device = device

    def act(self, obs: dict) -> np.ndarray:
        import torch
        image = torch.from_numpy(obs["image"]).float().unsqueeze(0).to(self.device) / 255.0
        ir = torch.from_numpy(obs["ir"]).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.agent.tokenizer.encode(image, ir)
            action = self.agent.act(latent, deterministic=True)
        return action.squeeze(0).cpu().numpy()


def stop_robot(rob) -> None:
    try:
        rob.set_wheel_speeds(0, 0, duration_s=0.4)
    except Exception:
        try:
            rob.move(0, 0, 200)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=["sac", "dreamerv3", "dreamerv4"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--phone-tilt", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--stale-timeout", type=float, default=1.0)
    parser.add_argument("--log-dir", default="hardware_logs")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "catkin_ws/src/learning_machines/src"))
    sys.path.insert(0, str(root / "catkin_ws/src/robobo_interface/src"))

    import rospy
    import torch
    from robobo_interface import HardwareRobobo
    from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
    from learning_machines.transfer import (
        CONTROL_INTERVAL_SECONDS,
        CalibrationProfile,
        CheckpointManifest,
    )

    calibration = CalibrationProfile.load(args.calibration)
    manifest_path = Path(args.manifest) if args.manifest else Path(args.checkpoint).parent / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("checkpoint manifest is required; legacy checkpoints are rejected")
    manifest = CheckpointManifest.load(manifest_path)
    manifest.validate(
        args.algorithm,
        calibration.name,
        args.image_size,
        args.phone_tilt,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.algorithm == "sac":
        policy = SACPolicy(args.checkpoint)
    elif args.algorithm == "dreamerv3":
        policy = DreamerV3Policy(args.checkpoint)
    else:
        policy = DreamerV4Policy(args.checkpoint, device)

    rob = HardwareRobobo(camera=True)
    env = RoboboCompactEnv(rob=rob, config=RoboboCompactEnvConfig(
        return_image=True,
        image_obs_size=(args.image_size, args.image_size),
        phone_tilt=args.phone_tilt,
        calibration_profile=calibration,
        max_episode_seconds=args.max_seconds,
        randomize_food_positions=False,
    ))
    controls = OperatorControls()
    controls.start()
    print("Controls: f + Enter = food event; e/q + Enter = emergency stop")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    step_log_path = log_dir / f"{episode_id}.jsonl"
    episode_path = log_dir / f"{episode_id}.npz"

    observations = []
    raw_irs = []
    images = []
    requested_actions = []
    executed_actions = []
    rewards = []
    dones = []
    inference_latencies = []
    overruns = []
    safety_events = []
    food_events = []
    watchdog_failure = None
    food_count = 0
    policy.reset()

    try:
        obs, info = env.reset()
        observations.append(np.concatenate([obs["blob"], obs["ir"]]).astype(np.float32))
        raw_irs.append(np.asarray(info["raw_ir"], dtype=np.float32))
        images.append(obs["image"].copy())
        started = time.monotonic()
        with step_log_path.open("w") as step_log:
            while time.monotonic() - started < args.max_seconds:
                if rospy.is_shutdown():
                    watchdog_failure = "lost_ros_communication"
                    break
                events = controls.drain()
                if "estop" in events:
                    watchdog_failure = "operator_emergency_stop"
                    break
                annotated_food = events.count("food")
                food_count += annotated_food

                cycle_start = time.monotonic()
                inference_start = time.perf_counter()
                try:
                    requested = np.asarray(policy.act(obs), dtype=np.float32)
                except Exception as exc:
                    watchdog_failure = f"inference_failure:{type(exc).__name__}"
                    break
                inference_latency = time.perf_counter() - inference_start
                if not np.isfinite(requested).all():
                    watchdog_failure = "non_finite_action"
                    break

                try:
                    next_obs, reward, terminated, truncated, info = env.step(requested)
                except Exception as exc:
                    watchdog_failure = f"sensor_or_command_failure:{type(exc).__name__}"
                    break
                observation_age = time.monotonic() - cycle_start
                overrun = max(0.0, observation_age - CONTROL_INTERVAL_SECONDS)
                if observation_age > args.stale_timeout:
                    watchdog_failure = "stale_observation"
                    break

                reward += 100.0 * annotated_food
                done = bool(terminated or truncated)
                executed = np.asarray(info["executed_action"], dtype=np.float32)
                policy.observe_executed(executed)
                requested_actions.append(requested.copy())
                executed_actions.append(executed.copy())
                rewards.append(float(reward))
                dones.append(done)
                inference_latencies.append(inference_latency)
                overruns.append(overrun)
                safety_events.append(info.get("safety_override"))
                food_events.append(annotated_food)
                observations.append(np.concatenate([next_obs["blob"], next_obs["ir"]]).astype(np.float32))
                raw_irs.append(np.asarray(info["raw_ir"], dtype=np.float32))
                images.append(next_obs["image"].copy())

                row = {
                    "step": len(rewards) - 1,
                    "elapsed_seconds": info["elapsed_seconds"],
                    "raw_ir": raw_irs[-1].tolist(),
                    "normalized_ir": next_obs["ir"].tolist(),
                    "requested_action": requested.tolist(),
                    "executed_action": executed.tolist(),
                    "inference_latency": inference_latency,
                    "timing_overrun": overrun,
                    "safety_event": info.get("safety_override"),
                    "collision": bool(info["collision"]),
                    "food_events": annotated_food,
                    "food_count": food_count,
                }
                step_log.write(json.dumps(row) + "\n")
                step_log.flush()
                obs = next_obs
                if done:
                    break
    finally:
        stop_robot(rob)
        env.close()
        np.savez_compressed(
            episode_path,
            observations=np.asarray(observations, dtype=np.float32),
            raw_irs=np.asarray(raw_irs, dtype=np.float32),
            images=np.asarray(images, dtype=np.uint8),
            requested_actions=np.asarray(requested_actions, dtype=np.float32),
            actions=np.asarray(executed_actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=bool),
            inference_latencies=np.asarray(inference_latencies, dtype=np.float32),
            timing_overruns=np.asarray(overruns, dtype=np.float32),
            safety_events=np.asarray(safety_events, dtype="U32"),
            food_events=np.asarray(food_events, dtype=np.int16),
            manifest=json.dumps(asdict(manifest)),
            watchdog_failure=watchdog_failure or "",
        )

    if watchdog_failure:
        raise RuntimeError(f"watchdog stopped deployment: {watchdog_failure}")
    print(f"Saved aligned hardware episode to {episode_path}")


if __name__ == "__main__":
    main()
