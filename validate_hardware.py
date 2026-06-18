#!/usr/bin/env python3
"""Safely commission, validate, and calibrate a physical Robobo."""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import time
import xmlrpc.client
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
LEARNING_SRC = ROOT / "catkin_ws/src/learning_machines/src"
ROBOBO_SRC = ROOT / "catkin_ws/src/robobo_interface/src"
for source in (LEARNING_SRC, ROBOBO_SRC):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from learning_machines.calibrate_ir import collect_phase, robust_profile
from learning_machines.transfer import IR_LABELS


REQUIRED_SERVICES = ("robot/moveWheels", "robot/movePanTilt")
CALIBRATION_PHASES = ("open_space", "wall_40cm", "wall_25cm", "wall_15cm", "near_obstacle")
WHEEL_CONFIRMATION = "WHEELS RAISED"


def summarize_ir(samples: np.ndarray) -> dict[str, dict[str, float]]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"expected N x 8 IR samples, got {values.shape}")
    return {
        label: {
            "min": float(np.min(values[:, index])),
            "median": float(np.median(values[:, index])),
            "max": float(np.max(values[:, index])),
            "std": float(np.std(values[:, index])),
        }
        for index, label in enumerate(IR_LABELS)
    }


def calibration_quality(
    samples: dict[str, np.ndarray], minimum_span: float
) -> dict[str, Any]:
    if "open_space" not in samples or "near_obstacle" not in samples:
        return {"passed": False, "errors": ["open_space and near_obstacle are required"]}
    free = np.median(samples["open_space"], axis=0)
    near = np.median(samples["near_obstacle"], axis=0)
    spans = np.abs(near - free)
    weak = [IR_LABELS[i] for i, span in enumerate(spans) if span < minimum_span]
    return {
        "passed": not weak,
        "minimum_required_span": float(minimum_span),
        "spans": {label: float(spans[i]) for i, label in enumerate(IR_LABELS)},
        "weak_sensors": weak,
    }


def stop_wheels(robot: Any) -> None:
    """Send an immediate, short zero-speed command."""
    robot.move(0, 0, 200)


def run_wheel_test(
    robot: Any,
    speed: int,
    duration_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    speed = int(np.clip(speed, 1, 20))
    millis = int(np.clip(round(duration_seconds * 1000), 100, 500))
    commands = [
        ("left_wheel_forward", speed, 0),
        ("right_wheel_forward", 0, speed),
        ("both_forward", speed, speed),
        ("both_reverse", -speed, -speed),
    ]
    events = []
    stop_wheels(robot)
    try:
        for name, left, right in commands:
            stop_wheels(robot)
            robot.move(left, right, millis)
            sleep(millis / 1000.0 + 0.1)
            stop_wheels(robot)
            events.append(
                {"name": name, "left": left, "right": right, "duration_ms": millis}
            )
    finally:
        stop_wheels(robot)
    return events


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _collect_ir(robot: Any, duration: float, hz: float) -> np.ndarray:
    deadline = time.monotonic() + duration
    rows: list[list[float]] = []
    while time.monotonic() < deadline:
        row = np.asarray(robot.read_irs(), dtype=np.float64)
        if row.shape == (8,) and np.isfinite(row).all():
            rows.append(row.tolist())
        time.sleep(1.0 / hz)
    if not rows:
        raise RuntimeError("No valid eight-sensor IR readings were received")
    return np.asarray(rows)


def _write_calibration(
    robot: Any,
    output_dir: Path,
    duration: float,
    hz: float,
    minimum_span: float,
) -> dict[str, Any]:
    samples: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    instructions = {
        "open_space": "Keep all eight sensors clear of nearby objects.",
        "wall_40cm": "Place barriers approximately 40 cm from all sensor directions.",
        "wall_25cm": "Place barriers approximately 25 cm from all sensor directions.",
        "wall_15cm": "Place barriers approximately 15 cm from all sensor directions.",
        "near_obstacle": "Place barriers close to every sensor without touching the robot.",
    }
    for phase in CALIBRATION_PHASES:
        input(f"\n{instructions[phase]}\nPress Enter to collect '{phase}'. ")
        values = collect_phase(robot, phase, duration, hz)
        samples[phase] = values
        for sample_index, value in enumerate(values):
            rows.append(
                {"phase": phase, "sample": sample_index}
                | {label: float(value[i]) for i, label in enumerate(IR_LABELS)}
            )

    profile_path = output_dir / "hardware.json"
    csv_path = output_dir / "ir_calibration_samples.csv"
    profile = robust_profile(samples, "hardware", f"hardware:{csv_path.name}")
    profile.save(profile_path)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["phase", "sample", *IR_LABELS])
        writer.writeheader()
        writer.writerows(rows)

    summary = {phase: summarize_ir(values) for phase, values in samples.items()}
    quality = calibration_quality(samples, minimum_span)
    (output_dir / "ir_calibration_summary.json").write_text(
        json.dumps({"phases": summary, "quality": quality}, indent=2) + "\n"
    )
    return {
        "profile": str(profile_path),
        "samples": str(csv_path),
        "quality": quality,
    }


def _check_ros_environment() -> dict[str, str]:
    master = os.environ.get("ROS_MASTER_URI", "")
    advertised = os.environ.get("ROS_IP") or os.environ.get("ROS_HOSTNAME")
    if not master:
        raise RuntimeError("ROS_MASTER_URI is not set")
    if not advertised:
        raise RuntimeError(
            "Set ROS_IP to this computer's robot-network IP. "
            "The robot must be able to connect back to it."
        )
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(5.0)
    try:
        response = xmlrpc.client.ServerProxy(master).getUri(
            "/robobo_hardware_validation_preflight"
        )
        if not isinstance(response, (list, tuple)) or response[0] != 1:
            raise RuntimeError(f"ROS master returned an invalid response: {response!r}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot contact the ROS master from this process at {master}: {exc}. "
            "On macOS, verify Docker Desktop can access the robot's LAN."
        ) from exc
    finally:
        socket.setdefaulttimeout(previous_timeout)
    return {
        "ROS_MASTER_URI": master,
        "ROS_IP_OR_HOSTNAME": advertised,
        "hostname": socket.gethostname(),
    }


def _wait_for_ros(timeout: float, include_camera: bool) -> dict[str, Any]:
    import rospy
    from robobo_msgs.msg import IRs
    from sensor_msgs.msg import CompressedImage

    result: dict[str, Any] = {"services": {}, "topics": {}}
    for service in REQUIRED_SERVICES:
        started = time.monotonic()
        rospy.wait_for_service(service, timeout=timeout)
        result["services"][service] = {"latency_seconds": time.monotonic() - started}

    started = time.monotonic()
    rospy.wait_for_message("robot/irs", IRs, timeout=timeout)
    result["topics"]["robot/irs"] = {"latency_seconds": time.monotonic() - started}
    if include_camera:
        started = time.monotonic()
        message = rospy.wait_for_message(
            "robot/camera/image/compressed", CompressedImage, timeout=timeout
        )
        result["topics"]["robot/camera/image/compressed"] = {
            "latency_seconds": time.monotonic() - started
        }
        result["camera_message"] = message
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only-by-default physical Robobo commissioning tool"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--sensor-seconds", type=float, default=5.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--test-tilt", action="store_true")
    parser.add_argument("--tilt-position", type=int, default=100)
    parser.add_argument("--calibrate-ir", action="store_true")
    parser.add_argument("--calibration-seconds", type=float, default=8.0)
    parser.add_argument("--minimum-ir-span", type=float, default=5.0)
    parser.add_argument("--test-wheels", action="store_true")
    parser.add_argument("--wheels-raised", action="store_true")
    parser.add_argument("--wheel-speed", type=int, default=10)
    parser.add_argument("--wheel-duration", type=float, default=0.4)
    parser.add_argument(
        "--wheel-confirmation",
        help=f"Must equal {WHEEL_CONFIRMATION!r}; interactive entry is safer.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.test_wheels and not args.wheels_raised:
        raise SystemExit("--test-wheels requires --wheels-raised")
    if not 1 <= args.wheel_speed <= 20:
        raise SystemExit("--wheel-speed must be between 1 and 20")
    if not 0.1 <= args.wheel_duration <= 0.5:
        raise SystemExit("--wheel-duration must be between 0.1 and 0.5 seconds")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_dir or f"hardware_logs/diagnostics/{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "output_dir": str(output_dir),
    }
    robot = None
    try:
        report["ros_environment"] = _check_ros_environment()
        from robobo_interface import HardwareRobobo

        robot = HardwareRobobo(camera=not args.no_camera)
        ros_result = _wait_for_ros(args.timeout, include_camera=not args.no_camera)
        camera_message = ros_result.pop("camera_message", None)
        report["ros"] = ros_result

        ir_samples = _collect_ir(robot, args.sensor_seconds, args.sample_hz)
        report["ir"] = {
            "sample_count": int(len(ir_samples)),
            "sensors": summarize_ir(ir_samples),
        }
        report["battery"] = {
            "robot_percent": float(robot.robot_battery()),
            "phone_percent": float(robot.phone_battery()),
            "note": "Battery topics update infrequently; 100 may be the startup default.",
        }
        report["phone_pose"] = {
            "pan": int(robot.read_phone_pan()),
            "tilt": int(robot.read_phone_tilt()),
        }
        report["acceleration"] = _json_value(robot.read_accel())
        report["orientation"] = _json_value(robot.read_orientation())
        report["wheels"] = _json_value(robot.read_wheels())

        if camera_message is not None:
            import cv2

            encoded = np.frombuffer(camera_message.data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Camera topic arrived but JPEG decoding failed")
            image = cv2.flip(image, 1)
            image_path = output_dir / "camera.jpg"
            cv2.imwrite(str(image_path), image)
            report["camera"] = {
                "path": str(image_path),
                "shape": list(image.shape),
                "mean": float(image.mean()),
                "std": float(image.std()),
            }

        if args.test_tilt:
            target = int(np.clip(args.tilt_position, 26, 109))
            before = int(robot.read_phone_tilt())
            robot.set_phone_tilt_blocking(target, 10)
            report["tilt_test"] = {
                "requested": target,
                "before": before,
                "after": int(robot.read_phone_tilt()),
            }

        if args.calibrate_ir:
            report["calibration"] = _write_calibration(
                robot,
                output_dir,
                args.calibration_seconds,
                args.sample_hz,
                args.minimum_ir_span,
            )

        if args.test_wheels:
            confirmation = args.wheel_confirmation
            if confirmation is None:
                confirmation = input(
                    f"\nRaise the robot so every wheel is clear. Type {WHEEL_CONFIRMATION}: "
                )
            if confirmation != WHEEL_CONFIRMATION:
                raise RuntimeError("Wheel test confirmation was not accepted")
            report["wheel_test"] = run_wheel_test(
                robot, args.wheel_speed, args.wheel_duration
            )

        report["status"] = "passed"
        return 0
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(report["error"], file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            try:
                stop_wheels(robot)
            except Exception as exc:
                report["stop_error"] = f"{type(exc).__name__}: {exc}"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, default=_json_value) + "\n")
        print(f"Hardware validation report: {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
