#!/usr/bin/env python3
'Long-running simulator contract and task-event validation.'
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Skip camera checks (camera validation is enabled by default).",
    )
    parser.add_argument(
        "--camera-sample",
        type=Path,
        help="Optionally save the first ground-facing observation as a PNG.",
    )
    parser.add_argument("--task-events", action="store_true")
    parser.add_argument("--calibration", default="config/calibration/simulation.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "catkin_ws/src/learning_machines/src"))
    sys.path.insert(0, str(root / "catkin_ws/src/robobo_interface/src"))
    os.environ["COPPELIA_SIM_PORT"] = str(args.port)

    from learning_machines.domain_randomization import DomainRandomizationWrapper
    from learning_machines.rl_robobo_compact_env import (
        RoboboCompactEnv,
        RoboboCompactEnvConfig,
    )

    base_env = RoboboCompactEnv(config=RoboboCompactEnvConfig(
        return_image=not args.no_camera,
        detect_blob_from_camera=not args.no_camera,
        calibration_path=args.calibration,
        randomize_food_positions=True,
        reset_settle_time=0.05,
    ))
    env = base_env
    resets = 0
    started = time.perf_counter()
    max_memory_kb = 0
    initial_memory_kb = 0
    try:
        obs, reset_info = env.reset()
        if not args.no_camera:
            image = np.asarray(obs["image"])
            assert image.shape == (3, 64, 64)
            assert np.isfinite(image).all() and float(image.std()) > 1.0
            actual_tilt = reset_info.get("phone_tilt")
            assert actual_tilt is not None and abs(actual_tilt - 100) <= 5, (
                f"camera did not reach the ground-facing pose: tilt={actual_tilt}"
            )
            if args.camera_sample:
                output = np.transpose(image, (1, 2, 0))
                if np.issubdtype(output.dtype, np.floating) and output.max() <= 1.0:
                    output = output * 255.0
                output = np.clip(output, 0, 255).astype(np.uint8)
                args.camera_sample.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(args.camera_sample),
                    cv2.cvtColor(output, cv2.COLOR_RGB2BGR),
                )

        timing_start = base_env.rob.get_sim_time()
        obs, _, terminated, truncated, timing_info = env.step(
            np.zeros(2, dtype=np.float32)
        )
        timing_delta = base_env.rob.get_sim_time() - timing_start
        simulation_dt = base_env.rob._sim.getSimulationTimeStep()
        dynamics_dt = base_env.rob._sim.getFloatParam(
            base_env.rob._sim.floatparam_physicstimestep
        )
        assert abs(simulation_dt - 0.4) < 1e-9
        assert abs(dynamics_dt - 0.005) < 1e-9
        assert not timing_info.get("simulation_stopped", False)
        assert abs(timing_delta - 0.4) < 1e-6, (
            f"one policy transition advanced {timing_delta:.6f}s instead of 0.400s"
        )
        if terminated or truncated:
            obs, _ = env.reset()
        if args.task_events:
            base = base_env
            sim = base.rob._sim
            assert reset_info.get("food_collected", 0) == 0


            obs, _ = env.reset()
            assert len(base._food_handles) >= 2
            robot_position = sim.getObjectPosition(
                base.rob._robobo, sim.handle_world
            )
            sim.setObjectPosition(
                base.rob._robobo,
                [base._arena_cx + 5.0, base._arena_cy + 5.0, 1.0],
            )
            candidates = [
                [base._arena_cx + dx, base._arena_cy + dy, 0.04]
                for dx, dy in ((0.8, 0.0), (-0.8, 0.0), (0.0, 0.8), (0.0, -0.8))
            ]
            away = max(
                candidates,
                key=lambda point: (
                    (point[0] - robot_position[0]) ** 2
                    + (point[1] - robot_position[1]) ** 2
                ),
            )
            for offset, handle in enumerate(base._food_handles[2:], start=1):
                sim.setObjectPosition(
                    handle,
                    [away[0], away[1] + 0.03 * offset, 2.0 + 0.1 * offset],
                )
            sim.setObjectPosition(base._food_handles[0], away)
            sim.setObjectPosition(base._food_handles[1], away)
            _, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
            assert info["food_collected"] == 0, (
                "food-food or food-floor contact was incorrectly collected; "
                f"robot={robot_position[:2]}, food={away[:2]}, "
                f"positions={[sim.getObjectPosition(handle, sim.handle_world) for handle in base._food_handles]}"
            )



            for index in range(len(base._food_handles)):
                obs, reset_info = env.reset()
                assert reset_info["food_collected"] == 0
                robot_position = sim.getObjectPosition(
                    base.rob._robobo, sim.handle_world
                )
                sim.setObjectPosition(
                    base._food_handles[index],
                    [robot_position[0], robot_position[1], 0.04],
                )
                _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
                assert info["newly_collected"] == 1
                assert info["food_collected"] == 1
                assert reward > 99.0
                _, _, _, _, repeated = env.step(np.zeros(2, dtype=np.float32))
                assert repeated["food_collected"] == 1
                assert repeated["newly_collected"] == 0

            obs, _ = env.reset()
            sim.setObjectPosition(base.rob._robobo, [-2.18, 0.8, 0.04])
            collision_seen = False
            override_seen = False
            for _ in range(10):
                obs, _, _, _, info = env.step(np.zeros(2, dtype=np.float32))
                collision_seen |= bool(info["collision"])
                override_seen |= info.get("safety_override") is not None
                if collision_seen and override_seen:
                    break
            assert collision_seen and override_seen
            obs, _ = env.reset()
            print(
                "task events verified: reset count, non-robot food contacts, "
                "single collection, wall collision, emergency recovery"
            )
        env = DomainRandomizationWrapper(base_env)
        obs, _ = env.reset()
        for step in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward)
            assert np.isfinite(obs["ir"]).all()
            if info.get("simulation_stopped"):
                raise RuntimeError(
                    f"CoppeliaSim stopped unexpectedly at validation step {step}"
                )
            assert np.isfinite(info["executed_action"]).all()
            if not args.no_camera:
                assert np.isfinite(obs["image"]).all()
            if terminated or truncated:
                obs, _ = env.reset()
                resets += 1
                assert abs(
                    env.unwrapped.rob._sim.getSimulationTimeStep() - 0.4
                ) < 1e-9
                assert abs(
                    env.unwrapped.rob._sim.getFloatParam(
                        env.unwrapped.rob._sim.floatparam_physicstimestep
                    )
                    - 0.005
                ) < 1e-9
            if step % 100 == 0:
                import resource
                current_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                if initial_memory_kb == 0:
                    initial_memory_kb = current_memory_kb
                max_memory_kb = max(max_memory_kb, current_memory_kb)
                print(
                    f"step={step} resets={resets} elapsed={time.perf_counter()-started:.1f}s "
                    f"maxrss_kb={max_memory_kb}"
                )
    finally:
        env.close()
    print(
        f"validation complete: steps={args.steps} resets={resets} "
        f"wall_seconds={time.perf_counter()-started:.2f} maxrss_kb={max_memory_kb}"
    )
    assert max_memory_kb - initial_memory_kb < 200_000, (
        f"memory grew by {max_memory_kb - initial_memory_kb} KB"
    )


if __name__ == "__main__":
    main()
