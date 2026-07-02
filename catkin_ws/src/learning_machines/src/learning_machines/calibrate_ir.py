'Collect robust eight-sensor IR calibration profiles.'
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from learning_machines.transfer import (
    CalibrationProfile,
    IR_LABELS,
    IRSensorCalibration,
)


def robust_profile(samples: dict[str, np.ndarray], name: str, source: str) -> CalibrationProfile:
    if "open_space" not in samples or "near_obstacle" not in samples:
        raise ValueError("samples require open_space and near_obstacle phases")
    free = np.percentile(samples["open_space"], 50, axis=0)
    near_median = np.percentile(samples["near_obstacle"], 50, axis=0)


    near = np.where(
        near_median >= free,
        np.percentile(samples["near_obstacle"], 90, axis=0),
        np.percentile(samples["near_obstacle"], 10, axis=0),
    )
    sensors = []
    for i in range(8):
        polarity = 1 if near[i] >= free[i] else -1
        if abs(near[i] - free[i]) < 1e-6:
            near[i] = free[i] + polarity
        sensors.append(IRSensorCalibration(
            free_space=float(free[i]),
            near_obstacle=float(near[i]),
            polarity=polarity,
            exponent=1.0,
        ))
    return CalibrationProfile(name=name, sensors=tuple(sensors), source=source)


def collect_phase(rob, phase: str, duration: float, hz: float) -> np.ndarray:
    print(f"Collecting {phase!r} for {duration:.1f}s...")
    deadline = time.monotonic() + duration
    rows = []
    while time.monotonic() < deadline:
        raw = rob.read_irs()
        cleaned = [float(v) if v is not None and v is not False else np.nan for v in raw]
        if len(cleaned) == 8 and np.isfinite(cleaned).all():
            rows.append(cleaned)
        time.sleep(1.0 / hz)
    if not rows:
        raise RuntimeError(f"no valid readings collected for {phase}")
    return np.asarray(rows, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Create a versioned Robobo IR calibration")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--sample-hz", type=float, default=20.0)
    parser.add_argument("--output", default="config/calibration/hardware.json")
    parser.add_argument("--raw-output", default="ir_calibration_samples.csv")
    parser.add_argument("--name", default="hardware")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["open_space", "wall_40cm", "wall_25cm", "wall_15cm", "near_obstacle"],
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[5]
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))
    if args.port is not None:
        os.environ["COPPELIA_SIM_PORT"] = str(args.port)
        from robobo_interface import SimulationRobobo
        rob = SimulationRobobo()
        if rob.is_stopped():
            rob.play_simulation()
        mode = "simulation"
    else:
        from robobo_interface import HardwareRobobo
        rob = HardwareRobobo(camera=False)
        time.sleep(1.0)
        mode = "hardware"

    samples = {}
    all_rows = []
    try:
        for phase in args.phases:
            input(f"Place the robot for phase '{phase}', then press Enter. ")
            phase_samples = collect_phase(rob, phase, args.duration, args.sample_hz)
            samples[phase] = phase_samples
            for timestamp_index, row in enumerate(phase_samples):
                all_rows.append({
                    "mode": mode,
                    "phase": phase,
                    "sample": timestamp_index,
                    **{label: float(row[i]) for i, label in enumerate(IR_LABELS)},
                })
    finally:
        try:
            if hasattr(rob, "set_wheel_speeds"):
                rob.set_wheel_speeds(0, 0, duration_s=0.4)
            else:
                rob.move(0, 0, 200)
        except Exception:
            pass

    profile = robust_profile(samples, args.name, f"{mode}:{Path(args.raw_output).name}")
    profile.save(args.output)
    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "phase", "sample", *IR_LABELS])
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {
        phase: {
            label: {
                "p10": float(np.percentile(values[:, i], 10)),
                "median": float(np.percentile(values[:, i], 50)),
                "p90": float(np.percentile(values[:, i], 90)),
            }
            for i, label in enumerate(IR_LABELS)
        }
        for phase, values in samples.items()
    }
    raw_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved calibration profile to {args.output}")


if __name__ == "__main__":
    main()
