#!/usr/bin/env python3
"""Run the NM512 DreamerV3 push policy on a physical Robobo."""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np


REFERENCE_SOURCE_COMMIT = "6ef8646d807cd10ce0c88e10a7e943211e7fc44c"
REFERENCE_CHECKPOINT_VERSION = 1
REFERENCE_REWARD_CONTRACT = "robobo-push-sparse-v1"
REFERENCE_IMAGE_SIZE = (96, 96)
MAX_DEPLOY_WHEEL_SPEED = 70
RAISED_WHEEL_CONFIRMATION = "WHEELS RAISED"


class WheelCommandCancelled(RuntimeError):
    """Raised when a retrying wheel command is interrupted safely."""


class NullLogger:
    """Minimal logger required to construct the reference Dreamer agent."""

    def __init__(self, step: int = 0):
        self.step = int(step)

    def scalar(self, *_args, **_kwargs):
        pass

    def image(self, *_args, **_kwargs):
        pass

    def video(self, *_args, **_kwargs):
        pass

    def write(self, *_args, **_kwargs):
        pass


class OperatorControls:
    """Read emergency-stop commands without blocking the rollout."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.reason = ""
        self.thread = threading.Thread(target=self._read, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _read(self) -> None:
        while not self.stop_event.is_set():
            try:
                command = input().strip().lower()
            except EOFError:
                return
            if command in {"e", "estop", "q", "quit"}:
                self.reason = "operator_emergency_stop"
                self.stop_event.set()
                return


class ReliableWheelCommander:
    """Retry one exact wheel command until its service and unlock replies arrive."""

    def __init__(
        self,
        rob,
        *,
        reply_timeout: float = 5.0,
        cancel_check: Callable[[], bool] | None = None,
        logger: Callable[[str], None] = print,
        poll_interval: float = 0.02,
    ):
        if reply_timeout <= 0:
            raise ValueError("reply_timeout must be positive")
        self.rob = rob
        self.reply_timeout = float(reply_timeout)
        self.cancel_check = cancel_check or (lambda: False)
        self.logger = logger
        self.poll_interval = max(0.001, float(poll_interval))
        self.last_retry_count = 0
        self.last_retry_reasons: list[str] = []
        self.total_retries = 0
        self._blockid_cursor = 1

    def __call__(self, left: int, right: int, millis: int) -> None:
        command = (int(left), int(right), int(millis))
        self.last_retry_count = 0
        self.last_retry_reasons = []

        while True:
            self._raise_if_cancelled()
            blockid = self._next_blockid()
            deadline = time.monotonic() + self.reply_timeout
            response_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

            def send() -> None:
                try:
                    returned = self.rob.move(*command, blockid=blockid)
                    response_queue.put_nowait(("ok", returned))
                except BaseException as exc:
                    try:
                        response_queue.put_nowait(("error", exc))
                    except queue.Full:
                        pass

            threading.Thread(target=send, daemon=True).start()
            response = self._wait_for_service_response(response_queue, deadline)
            if response is not None and response[0] == "ok":
                if self._wait_for_unlock(blockid, deadline):
                    return
                reason = "unlock_timeout"
            elif response is not None:
                reason = f"service_error:{type(response[1]).__name__}"
            else:
                reason = "service_timeout"

            self._discard_blockid(blockid)
            if hasattr(self.rob, "refresh_move_service"):
                try:
                    self.rob.refresh_move_service()
                except Exception:
                    pass
            self._record_retry(command, reason)
            self._wait_until(deadline)

    def _wait_for_service_response(
        self,
        response_queue: queue.Queue[tuple[str, Any]],
        deadline: float,
    ) -> tuple[str, Any] | None:
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                return response_queue.get(
                    timeout=min(self.poll_interval, remaining)
                )
            except queue.Empty:
                pass

    def _wait_for_unlock(self, blockid: int, deadline: float) -> bool:
        while self.rob.is_blocked(blockid):
            self._raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(self.poll_interval, remaining))
        return True

    def _wait_until(self, deadline: float) -> None:
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(self.poll_interval, remaining))

    def _record_retry(self, command: tuple[int, int, int], reason: str) -> None:
        self.last_retry_count += 1
        self.total_retries += 1
        self.last_retry_reasons.append(reason)
        self.logger(
            "Wheel command reply failed "
            f"({reason}); resending {command} after "
            f"{self.reply_timeout:.1f}s attempt timeout."
        )

    def _discard_blockid(self, blockid: int) -> None:
        try:
            self.rob._used_pids.discard(blockid)
        except Exception:
            pass

    def _next_blockid(self) -> int:
        used = getattr(self.rob, "_used_pids", ())
        for _ in range(767):
            blockid = self._blockid_cursor
            self._blockid_cursor = blockid % 767 + 1
            if blockid not in used:
                return blockid
        raise RuntimeError("no Robobo block IDs available for wheel command")

    def _raise_if_cancelled(self) -> None:
        if self.cancel_check():
            raise WheelCommandCancelled("wheel command retry cancelled")


def normalize_compiled_state_dict(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Convert torch.compile state keys to the eager module layout."""
    normalized = {}
    for key, value in state_dict.items():
        eager_key = key.replace("._orig_mod.", ".")
        if eager_key.startswith("_orig_mod."):
            eager_key = eager_key[len("_orig_mod.") :]
        if eager_key in normalized:
            raise ValueError(f"duplicate eager checkpoint key: {eager_key}")
        normalized[eager_key] = value
    return normalized


def load_reference_contract(checkpoint: dict[str, Any]) -> dict[str, Any]:
    required = {
        "checkpoint_version",
        "agent_state_dict",
        "training_step",
        "reward_contract",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            "reference checkpoint is missing: " + ", ".join(sorted(missing))
        )
    if checkpoint["checkpoint_version"] != REFERENCE_CHECKPOINT_VERSION:
        raise ValueError(
            "unsupported reference checkpoint version: "
            f"{checkpoint['checkpoint_version']}"
        )
    contract = dict(checkpoint["reward_contract"])
    expected = {
        "reward_contract": REFERENCE_REWARD_CONTRACT,
        "image_size": list(REFERENCE_IMAGE_SIZE),
        "action_smoothing": False,
        "pre_action_safety": False,
        "max_action_delta": 2.0,
        "dyn_hidden": 512,
        "dyn_deter": 1024,
        "units": 512,
        "cnn_depth": 32,
        "cnn_minres": 3,
    }
    mismatches = {
        key: (expected_value, contract.get(key))
        for key, expected_value in expected.items()
        if contract.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"incompatible NM512 checkpoint contract: {mismatches}"
        )
    return contract


def validate_contract_file(
    contract: dict[str, Any],
    contract_path: Path | None,
) -> None:
    if contract_path is None or not contract_path.exists():
        return
    file_contract = json.loads(contract_path.read_text())
    if file_contract != contract:
        raise ValueError(
            f"{contract_path} does not match the checkpoint reward contract"
        )


def validate_reference_source(reference_dir: Path) -> None:
    required = ("dreamer.py", "models.py", "networks.py", "tools.py", "configs.yaml")
    missing = [name for name in required if not (reference_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"missing NM512 reference files in {reference_dir}: {missing}. "
            "Clone https://github.com/NM512/dreamerv3-torch there."
        )
    if (reference_dir / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(reference_dir), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        if result.returncode == 0 and commit != REFERENCE_SOURCE_COMMIT:
            raise ValueError(
                "NM512 reference source commit mismatch: expected "
                f"{REFERENCE_SOURCE_COMMIT}, got {commit}"
            )


def _native(value):
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native(item) for item in value]
    try:
        if not isinstance(value, (int, float, str, bool, type(None))):
            return value.__class__.__bases__[0](value)
    except (AttributeError, TypeError, ValueError):
        pass
    return value


def build_reference_config(reference_dir: Path, contract: dict[str, Any]):
    from ruamel.yaml import YAML

    yaml = YAML(typ="safe")
    all_configs = yaml.load((reference_dir / "configs.yaml").read_text())
    cfg = _native(all_configs["defaults"])
    cfg.update(_native(all_configs.get("robobo", {})))
    cfg.update(
        {
            "task": "robobo_push",
            "size": list(contract["image_size"]),
            "envs": 1,
            "action_repeat": 1,
            "time_limit": int(contract["max_episode_steps"]),
            "grayscale": False,
            "prefill": 0,
            "dyn_hidden": int(contract["dyn_hidden"]),
            "dyn_deter": int(contract["dyn_deter"]),
            "units": int(contract["units"]),
            "model_lr": 4e-5,
            "grad_clip": 100.0,
            "batch_size": int(contract["batch_size"]),
            "batch_length": int(contract["batch_length"]),
            "train_ratio": int(contract["train_ratio"]),
            "dataset_size": 1,
            "device": "cpu",
            "compile": False,
            "video_pred_log": False,
            "num_actions": 2,
        }
    )
    cfg["actor"] = {
        "layers": 3,
        "dist": "normal",
        "entropy": 3e-4,
        "unimix_ratio": 0.01,
        "std": "learned",
        "min_std": 0.1,
        "max_std": 1.0,
        "temp": 0.1,
        "lr": 4e-5,
        "eps": 1e-5,
        "grad_clip": 100.0,
        "outscale": 1.0,
    }
    cfg["critic"] = {
        "layers": 3,
        "dist": "symlog_disc",
        "slow_target": True,
        "slow_target_update": 1,
        "slow_target_fraction": 0.02,
        "lr": 4e-5,
        "eps": 1e-5,
        "grad_clip": 100.0,
        "outscale": 0.0,
    }
    cfg["encoder"] = {
        "mlp_keys": "ir",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": int(contract["cnn_depth"]),
        "kernel_size": 4,
        "minres": int(contract["cnn_minres"]),
        "mlp_layers": 5,
        "mlp_units": 1024,
        "symlog_inputs": True,
    }
    cfg["decoder"] = {
        "mlp_keys": "ir",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": int(contract["cnn_depth"]),
        "kernel_size": 4,
        "minres": int(contract["cnn_minres"]),
        "mlp_layers": 5,
        "mlp_units": 1024,
        "cnn_sigmoid": False,
        "image_dist": "mse",
        "vector_dist": "symlog_mse",
        "outscale": 1.0,
    }
    return SimpleNamespace(**cfg)


def force_reference_cpu_defaults() -> None:
    """Override NM512's hard-coded CUDA default for encoder/decoder MLPs."""
    import networks

    defaults = list(networks.MLP.__init__.__defaults__ or ())
    if len(defaults) < 2:
        raise RuntimeError("unexpected NM512 MLP constructor signature")
    # `device` is the penultimate optional argument, immediately before name.
    defaults[-2] = "cpu"
    networks.MLP.__init__.__defaults__ = tuple(defaults)


class ReferenceDreamerV3Policy:
    def __init__(self, checkpoint_path: Path, reference_dir: Path):
        import gymnasium as gym
        import torch

        validate_reference_source(reference_dir)
        sys.path.insert(0, str(reference_dir))
        import dreamer

        force_reference_cpu_defaults()
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.contract = load_reference_contract(checkpoint)
        config = build_reference_config(reference_dir, self.contract)
        image_height, image_width = self.contract["image_size"]
        obs_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    0,
                    255,
                    shape=(image_height, image_width, 3),
                    dtype=np.uint8,
                ),
                "ir": gym.spaces.Box(
                    0.0, 1.0, shape=(8,), dtype=np.float32
                ),
            }
        )
        action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(2,), dtype=np.float32
        )
        self.agent = dreamer.Dreamer(
            obs_space,
            action_space,
            config,
            NullLogger(checkpoint["training_step"]),
            iter(()),
        ).to("cpu")
        eager_state = normalize_compiled_state_dict(
            checkpoint["agent_state_dict"]
        )
        self.agent.load_state_dict(eager_state, strict=True)
        self.agent.requires_grad_(False)
        self.agent.eval()
        self.state = None

    def reset(self) -> None:
        self.state = None

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        import torch

        image = np.asarray(obs["image"], dtype=np.uint8)
        if image.shape == (3, *REFERENCE_IMAGE_SIZE):
            image = np.transpose(image, (1, 2, 0))
        if image.shape != (*REFERENCE_IMAGE_SIZE, 3):
            raise ValueError(f"expected 96x96 RGB observation, got {image.shape}")
        policy_obs = {
            "image": image[None],
            "ir": np.asarray(obs["ir"], dtype=np.float32)[None],
            "is_first": np.asarray([self.state is None], dtype=bool),
            "is_terminal": np.asarray([False], dtype=bool),
        }
        with torch.no_grad():
            output, self.state = self.agent(
                policy_obs,
                np.asarray([self.state is None], dtype=bool),
                self.state,
                training=False,
            )
        action = output["action"][0].detach().cpu().numpy()
        return np.asarray(action, dtype=np.float32)


def stop_robot(rob, timeout: float = 1.0) -> None:
    """Send a stop command without allowing a broken ROS call to hang exit."""
    completed = threading.Event()

    def send_stop() -> None:
        try:
            rob.set_wheel_speeds(0, 0, duration_s=0.4)
        except Exception:
            try:
                rob.move(0, 0, 200)
            except Exception:
                pass
        finally:
            completed.set()

    threading.Thread(target=send_stop, daemon=True).start()
    completed.wait(max(0.0, timeout))


def main(rob=None, argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Deploy the NM512 DreamerV3 push policy on physical Robobo"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--hardware-manifest",
        "--calibration",
        dest="hardware_manifest",
        default="config/calibration/hardware.json",
        help="Measured hardware IR calibration/observation manifest.",
    )
    parser.add_argument(
        "--contract",
        default=None,
        help="Optional reward_contract.json; defaults beside the checkpoint.",
    )
    parser.add_argument(
        "--reference-dir", default="dreamerv3_reference"
    )
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-wheel-speed", type=int, default=MAX_DEPLOY_WHEEL_SPEED
    )
    parser.add_argument("--reply-timeout", type=float, default=5.0)
    parser.add_argument("--camera-tilt", type=int, default=100)
    parser.add_argument("--leave-camera-tilt", action="store_true")
    parser.add_argument("--raised-wheel-test", action="store_true")
    parser.add_argument("--wheel-confirmation", default=None)
    parser.add_argument("--log-dir", default="hardware_logs/dreamerv3_reference")
    args = parser.parse_args(argv)

    if args.max_seconds <= 0:
        parser.error("--max-seconds must be positive")
    if not 1 <= args.max_wheel_speed <= MAX_DEPLOY_WHEEL_SPEED:
        parser.error("--max-wheel-speed must be between 1 and 70")
    if args.reply_timeout <= 0:
        parser.error("--reply-timeout must be positive")
    if not 5 <= args.camera_tilt <= 110:
        parser.error("--camera-tilt must be between 5 and 110")

    checkpoint_path = Path(args.checkpoint).resolve()
    manifest_path = Path(args.hardware_manifest).resolve()
    reference_dir = Path(args.reference_dir).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "catkin_ws/src/learning_machines/src"))
    sys.path.insert(0, str(root / "catkin_ws/src/robobo_interface/src"))

    import rospy
    from learning_machines.rl_robobo_compact_env import (
        RoboboCompactEnv,
        RoboboCompactEnvConfig,
    )
    from learning_machines.transfer import CalibrationProfile

    calibration = CalibrationProfile.load(manifest_path)
    if calibration.name != "hardware":
        raise ValueError(
            f"hardware manifest must be named 'hardware', got {calibration.name!r}"
        )

    policy = ReferenceDreamerV3Policy(checkpoint_path, reference_dir)
    contract_path = (
        Path(args.contract).resolve()
        if args.contract
        else checkpoint_path.parent / "reward_contract.json"
    )
    validate_contract_file(policy.contract, contract_path)

    run_seconds = min(args.max_seconds, 10.0) if args.raised_wheel_test else args.max_seconds
    if args.raised_wheel_test:
        confirmation = args.wheel_confirmation
        if confirmation is None:
            confirmation = input(
                "Raise the robot so every wheel is clear. "
                f"Type {RAISED_WHEEL_CONFIRMATION}: "
            )
        if confirmation != RAISED_WHEEL_CONFIRMATION:
            raise RuntimeError("raised-wheel confirmation was not accepted")

    if rob is None:
        from robobo_interface import HardwareRobobo

        rob = HardwareRobobo(camera=True)

    controls = OperatorControls()
    controls.start()
    deadline = float("inf")

    def retry_cancelled() -> bool:
        return (
            controls.stop_event.is_set()
            or rospy.is_shutdown()
            or time.monotonic() >= deadline
        )

    commander = ReliableWheelCommander(
        rob,
        reply_timeout=args.reply_timeout,
        cancel_check=retry_cancelled,
        logger=lambda message: print(message, flush=True),
    )
    env = RoboboCompactEnv(
        rob=rob,
        config=RoboboCompactEnvConfig(
            task="push",
            return_image=True,
            image_obs_size=REFERENCE_IMAGE_SIZE,
            phone_tilt=args.camera_tilt,
            initialize_phone_tilt=not args.leave_camera_tilt,
            calibration_profile=calibration,
            max_episode_seconds=run_seconds,
            max_wheel_speed=args.max_wheel_speed,
            randomize_push_layout=False,
            action_smoothing=False,
            pre_action_safety=False,
            max_action_delta=2.0,
            hardware_inference_only=True,
            hardware_wheel_command=commander,
        ),
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_id = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    jsonl_path = log_dir / f"{episode_id}.jsonl"
    episode_path = log_dir / f"{episode_id}.npz"
    summary_path = log_dir / f"{episode_id}_summary.json"

    images = []
    irs = []
    raw_irs = []
    requested_actions = []
    actions = []
    inference_latencies = []
    transition_seconds = []
    retry_counts = []
    retry_reasons = []
    shutdown_reason = ""
    policy.reset()

    print(
        "Controls: e/q + Enter = emergency stop. "
        f"Using 96x96 RGB and {manifest_path}.",
        flush=True,
    )
    try:
        obs, info = env.reset()
        deadline = time.monotonic() + run_seconds
        images.append(obs["image"].copy())
        irs.append(obs["ir"].copy())
        raw_irs.append(np.asarray(info["raw_ir"], dtype=np.float32))
        with jsonl_path.open("w") as log_file:
            while time.monotonic() < deadline:
                if controls.stop_event.is_set():
                    shutdown_reason = controls.reason
                    break
                if rospy.is_shutdown():
                    shutdown_reason = "ros_shutdown"
                    break

                inference_start = time.perf_counter()
                requested = policy.act(obs)
                inference_latency = time.perf_counter() - inference_start
                if not np.isfinite(requested).all():
                    shutdown_reason = "non_finite_action"
                    break

                transition_start = time.monotonic()
                try:
                    next_obs, _reward, _terminated, truncated, info = env.step(
                        requested
                    )
                except WheelCommandCancelled:
                    if controls.stop_event.is_set():
                        shutdown_reason = controls.reason
                    elif rospy.is_shutdown():
                        shutdown_reason = "ros_shutdown"
                    else:
                        shutdown_reason = "max_seconds"
                    break
                transition_duration = time.monotonic() - transition_start

                executed = np.asarray(
                    info["executed_action"], dtype=np.float32
                )
                images.append(next_obs["image"].copy())
                irs.append(next_obs["ir"].copy())
                raw_irs.append(np.asarray(info["raw_ir"], dtype=np.float32))
                requested_actions.append(requested.copy())
                actions.append(executed.copy())
                inference_latencies.append(inference_latency)
                transition_seconds.append(transition_duration)
                retry_counts.append(commander.last_retry_count)
                retry_reasons.append(",".join(commander.last_retry_reasons))
                row = {
                    "step": len(actions) - 1,
                    "elapsed_seconds": info["elapsed_seconds"],
                    "requested_action": requested.tolist(),
                    "executed_action": executed.tolist(),
                    "raw_ir": raw_irs[-1].tolist(),
                    "normalized_ir": next_obs["ir"].tolist(),
                    "inference_latency": inference_latency,
                    "transition_wall_seconds": transition_duration,
                    "wheel_retry_count": commander.last_retry_count,
                    "wheel_retry_reasons": commander.last_retry_reasons,
                    "collision": bool(info["collision"]),
                    "red_block_visible": bool(info["red_block_visible"]),
                    "green_goal_visible": bool(info["green_goal_visible"]),
                }
                log_file.write(json.dumps(row) + "\n")
                log_file.flush()
                obs = next_obs
                if truncated:
                    shutdown_reason = "max_seconds"
                    break
    except KeyboardInterrupt:
        shutdown_reason = "keyboard_interrupt"
    finally:
        stop_robot(rob)
        env.close()

    np.savez_compressed(
        episode_path,
        images=np.asarray(images, dtype=np.uint8),
        ir=np.asarray(irs, dtype=np.float32),
        raw_ir=np.asarray(raw_irs, dtype=np.float32),
        requested_actions=np.asarray(requested_actions, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        inference_latencies=np.asarray(inference_latencies, dtype=np.float32),
        transition_seconds=np.asarray(transition_seconds, dtype=np.float32),
        retry_counts=np.asarray(retry_counts, dtype=np.int32),
        retry_reasons=np.asarray(retry_reasons, dtype="U128"),
        checkpoint=str(checkpoint_path),
        hardware_manifest=str(manifest_path),
        reward_contract=json.dumps(policy.contract),
        shutdown_reason=shutdown_reason,
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "hardware_manifest": str(manifest_path),
        "image_size": list(REFERENCE_IMAGE_SIZE),
        "steps": len(actions),
        "total_wheel_retries": commander.total_retries,
        "shutdown_reason": shutdown_reason or "completed",
        "episode": str(episode_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
