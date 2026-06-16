"""IR calibration for real Robobo robot.

Reads IR sensors at various orientations, saves to CSV, compares with sim ranges.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

IR_LABELS = ["BackL", "BackR", "FrontL", "FrontR", "FrontC", "FrontRR", "BackC", "FrontLL"]


def main():
    parser = argparse.ArgumentParser(description="IR calibration for Robobo")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds per orientation")
    parser.add_argument("--output", type=str, default="ir_calibration.csv")
    parser.add_argument("--port", type=int, default=None, help="CoppeliaSim port (for sim mode)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))

    if args.port is not None:
        import os
        os.environ["COPPELIA_SIM_PORT"] = str(args.port)
        from robobo_interface import SimulationRobobo
        rob = SimulationRobobo()
        if rob.is_stopped():
            rob.play_simulation()
        mode = "sim"
    else:
        from robobo_interface import HardwareRobobo
        rob = HardwareRobobo(camera=True)
        time.sleep(1.0)
        mode = "hardware"

    orientations = [
        ("forward", (50, 50)),
        ("backward", (-50, -50)),
        ("left", (-50, 50)),
        ("right", (50, -50)),
        ("stop", (0, 0)),
    ]

    rows = []
    for name, (left, right) in orientations:
        print(f"\n--- Orientation: {name} (L={left}, R={right}) ---")
        if left != 0 or right != 0:
            rob.set_wheel_speeds(left, right, duration_s=0.5)
        time.sleep(args.duration)

        readings = []
        for _ in range(10):
            try:
                raw = rob.read_irs()
                cleaned = [float(v) if v is not None and v is not False else 0.0 for v in raw]
                while len(cleaned) < 8:
                    cleaned.append(0.0)
                readings.append(cleaned[:8])
            except Exception:
                pass
            time.sleep(0.1)

        if readings:
            arr = np.array(readings)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            row = {"orientation": name, "mode": mode}
            for i, label in enumerate(IR_LABELS):
                row[f"{label}_mean"] = float(mean[i])
                row[f"{label}_std"] = float(std[i])
            rows.append(row)
            print(f"  IR means: {', '.join(f'{label}={mean[i]:.1f}' for i, label in enumerate(IR_LABELS))}")

        if left != 0 or right != 0:
            rob.set_wheel_speeds(0, 0, duration_s=0.3)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["orientation", "mode"] + [f"{l}_{s}" for l in IR_LABELS for s in ("mean", "std")])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved calibration to {args.output}")


if __name__ == "__main__":
    main()
