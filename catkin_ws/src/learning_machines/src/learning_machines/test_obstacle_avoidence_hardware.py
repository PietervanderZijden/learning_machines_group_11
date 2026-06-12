from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from learning_machines.rl_robobo_env import RoboboObstacleEnvConfig
from learning_machines.robobo_sac_policy import RoboboCombinedExtractor
from robobo_interface import HardwareRobobo
from stable_baselines3 import SAC

MODEL_PATH = Path("/root/results/runs/robobo_obstacle_sac/models/robobo_sac_final.zip")


def read_hardware_observation(
    rob: HardwareRobobo,
    config: RoboboObstacleEnvConfig,
) -> dict[str, np.ndarray]:
    """
    Read the hardware Robobo camera and IR sensors and format them exactly like
    the training observation.

    Returns:
        {
            "image": uint8 array with shape (1, H, W),
            "ir": float32 array with shape (8,)
        }
    """

    image_bgr = rob.read_image_front()

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(
        gray,
        config.image_size[::-1],
        interpolation=cv2.INTER_AREA,
    )

    image_obs = gray[None, :, :].astype(np.uint8)

    raw_irs = rob.read_irs()

    cleaned_irs: list[float] = []
    for value in raw_irs:
        if value is None or value is False:
            cleaned_irs.append(0.0)
        else:
            cleaned_irs.append(float(value))

    if len(cleaned_irs) != 8:
        cleaned_irs = (cleaned_irs + [0.0] * 8)[:8]

    ir_obs = np.asarray(cleaned_irs, dtype=np.float32)
    ir_obs = np.nan_to_num(
        ir_obs,
        nan=0.0,
        posinf=config.max_ir_value,
        neginf=0.0,
    )
    ir_obs = np.clip(ir_obs / config.max_ir_value, 0.0, 1.0).astype(np.float32)

    return {
        "image": image_obs,
        "ir": ir_obs,
    }


def run_hardware_demo(
    model_path: str | Path = MODEL_PATH,
    duration_seconds: float = 60.0,
    deterministic: bool = True,
    max_wheel_speed: int = 70,
    step_millis: int = 100,
    image_size: tuple[int, int] = (128, 128),
    max_ir_value: float = 400.0,
    phone_tilt: Optional[int] = 100,
    phone_tilt_speed: int = 100,
) -> None:
    """
    Run the trained obstacle-avoidance policy on the physical Robobo.

    Args:
        model_path:
            Path to the trained SAC model.
        duration_seconds:
            How long to run the policy on the hardware robot.
        deterministic:
            Whether to use deterministic model actions.
        max_wheel_speed:
            Maximum wheel speed used to scale actions from [-1, 1].
        step_millis:
            Duration of each wheel command.
        image_size:
            Image size expected by the trained model.
        max_ir_value:
            IR normalization value used during training.
        phone_tilt:
            Optional phone tilt angle before starting. Use None to skip.
        phone_tilt_speed:
            Speed for phone tilt movement.
    """

    config = RoboboObstacleEnvConfig(
        image_size=image_size,
        max_wheel_speed=max_wheel_speed,
        step_millis=step_millis,
        max_ir_value=max_ir_value,
    )

    print(f"Loading model from: {model_path}")
    model = SAC.load(str(model_path), device="auto")

    print("Connecting to hardware Robobo...")
    rob = HardwareRobobo(camera=True)

    try:
        if phone_tilt is not None:
            print(f"Setting phone tilt to {phone_tilt}...")
            rob.set_phone_tilt_blocking(phone_tilt, phone_tilt_speed)

        print(f"Running hardware demo for {duration_seconds:.1f} seconds.")
        start_time = time.monotonic()

        while time.monotonic() - start_time < duration_seconds:
            obs = read_hardware_observation(rob, config)

            action, _state = model.predict(
                obs,
                deterministic=deterministic,
            )

            action = np.asarray(action, dtype=np.float32)
            action = np.clip(action, -1.0, 1.0)

            left_speed = int(action[0] * config.max_wheel_speed)
            right_speed = int(action[1] * config.max_wheel_speed)

            rob.move_blocking(
                left_speed,
                right_speed,
                config.step_millis,
            )

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        print("Stopping robot.")
        try:
            rob.move_blocking(0, 0, 100)
        except Exception:
            pass

        print("Hardware demo finished.")


if __name__ == "__main__":
    run_hardware_demo()
