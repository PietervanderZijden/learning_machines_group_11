from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class StepLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"deploy_{ts}.csv"
        self.json_path = self.log_dir / f"deploy_{ts}_summary.json"
        self.rows = []
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "step", "timestamp", "left_action", "right_action",
            "ir_front", "ir_back", "blob_x", "blob_y", "blob_found",
            "food_collected", "collision",
        ])

    def log_step(self, step, left, right, ir, blob, food, collision):
        row = [
            step,
            time.time(),
            left,
            right,
            max(ir[2], ir[3], ir[4], ir[5], ir[7]) if len(ir) >= 8 else 0,
            max(ir[0], ir[1], ir[6]) if len(ir) >= 8 else 0,
            blob[0],
            blob[1],
            blob[3],
            food,
            collision,
        ]
        self.writer.writerow(row)
        self.csv_file.flush()
        self.rows.append(row)

    def finish(self):
        self.csv_file.close()
        summary = {
            "total_steps": len(self.rows),
            "food_collected": self.rows[-1][9] if self.rows else 0,
            "collision": any(r[10] for r in self.rows),
        }
        with open(self.json_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary


def main():
    parser = argparse.ArgumentParser(description="Deploy trained TD-MPC2 policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=str, default="deploy_logs")
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "learning_machines" / "src"))
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))
    tdmpc2_dir = project_root / "tdmpc2" / "tdmpc2"
    sys.path.insert(0, str(tdmpc2_dir))

    import torch
    from tdmpc2 import TDMPC2

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        raise ValueError("Checkpoint missing config. Re-train with updated tdmpc2.py.")

    cfg.device = args.device
    agent = TDMPC2(cfg)
    agent.load(ckpt)
    agent.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    if args.hardware:
        from robobo_interface import HardwareRobobo
        rob = HardwareRobobo(camera=True)
        time.sleep(1.0)
    else:
        import os
        os.environ["COPPELIA_SIM_PORT"] = str(args.port)
        from robobo_interface import SimulationRobobo
        rob = SimulationRobobo()
        if rob.is_stopped():
            rob.play_simulation()

    from rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
    from safety_wrapper import SafetyWrapper

    env_config = RoboboCompactEnvConfig(max_episode_steps=args.max_steps)
    env = RoboboCompactEnv(rob=rob, config=env_config)
    env = SafetyWrapper(
        env,
        max_wheel_speed=env_config.max_wheel_speed,
        front_ir_indices=[2, 3, 4, 5, 7],
        danger_threshold=0.7,
        critical_threshold=0.9,
    )

    logger = StepLogger(args.log_dir)
    obs_dict, info = env.reset()
    obs = np.concatenate([obs_dict["blob"], obs_dict["ir"]]).astype(np.float32)
    obs = torch.from_numpy(obs)

    print(f"Running for up to {args.max_steps} steps...")
    for step in range(args.max_steps):
        action = agent.act(obs, t0=(step == 0), eval_mode=True)
        action_np = action.numpy().astype(np.float32)
        action_np = np.clip(action_np, -1.0, 1.0)

        obs_dict, reward, terminated, truncated, info = env.step(action_np)
        obs = np.concatenate([obs_dict["blob"], obs_dict["ir"]]).astype(np.float32)
        obs = torch.from_numpy(obs)

        logger.log_step(
            step, action_np[0], action_np[1],
            obs_dict["ir"], obs_dict["blob"],
            info.get("food_collected", 0), info.get("collision", False),
        )

        if step % 25 == 0:
            print(f"  step={step:4d} R={reward:.1f} food={info.get('food_collected', 0)}")

        if terminated or truncated:
            break

    summary = logger.finish()
    print(f"\nDone: {summary}")
    env.close()


if __name__ == "__main__":
    main()
