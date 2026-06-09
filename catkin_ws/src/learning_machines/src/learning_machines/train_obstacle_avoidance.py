from __future__ import annotations

from pathlib import Path
from typing import Any

import torch as th
import wandb
from learning_machines.rl_robobo_env import (
    RoboboObstacleAvoidanceEnv,
    RoboboObstacleEnvConfig,
)
from learning_machines.robobo_sac_policy import RoboboCombinedExtractor
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

RUN_DIR = Path("results/runs/robobo_obstacle_sac")
MODEL_DIR = RUN_DIR / "models"
LOG_DIR = RUN_DIR / "logs"


class WandbInfoCallback(BaseCallback):
    """
    Logs useful custom environment metrics to Weights & Biases.

    SB3 logs losses, entropy coefficient, episode reward, etc. through its own
    logger. This callback adds Robobo-specific metrics from env.step() info.
    """

    def __init__(self, log_freq: int = 10, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        infos: list[dict[str, Any]] = self.locals.get("infos", [])

        if len(infos) == 0:
            return True

        info = infos[0]

        metrics = {}

        keys_to_log = [
            "reward",
            "distance_from_start",
            "front_obstacle_closeness",
            "back_obstacle_closeness",
            "max_obstacle_closeness",
            "collision",
            "x",
            "y",
            "z",
            "left_speed",
            "right_speed",
            "action_left",
            "action_right",
            "step_count",
        ]

        for key in keys_to_log:
            if key in info:
                metrics[f"env/{key}"] = info[key]

        if len(metrics) > 0:
            wandb.log(metrics, step=self.num_timesteps)

        return True


def make_env(config: RoboboObstacleEnvConfig) -> Monitor:
    env = RoboboObstacleAvoidanceEnv(config=config)
    env = Monitor(env, filename=str(LOG_DIR / "monitor.csv"))

    return env


def main(
    total_timesteps: int = 300_000,
    wandb_project: str = "learning-machines",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    check_environment: bool = True,
) -> SAC:
    """
    Start SAC training.

    Example:
        from package import main

        main()

    Or:
        main(total_timesteps=100_000, wandb_mode="offline")
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config = RoboboObstacleEnvConfig(
        image_size=(84, 84),
        max_wheel_speed=100,
        step_millis=200,
        max_episode_steps=500,
        max_ir_value=100.0,
        obstacle_penalty_threshold=0.35,
        collision_ir_threshold=0.85,
        progress_normalizer_m=0.05,
        progress_reward_scale=1.0,
        distance_bonus_scale=0.02,
        obstacle_penalty_scale=0.4,
        front_obstacle_penalty_scale=0.6,
        action_penalty_scale=0.02,
        turning_penalty_scale=0.02,
        alive_bonus=0.01,
        collision_penalty=5.0,
        reset_settle_seconds=0.25,
    )

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
        config={
            "algorithm": "SAC",
            "policy": "MultiInputPolicy",
            "total_timesteps": total_timesteps,
            "image_size": config.image_size,
            "max_wheel_speed": config.max_wheel_speed,
            "step_millis": config.step_millis,
            "max_episode_steps": config.max_episode_steps,
            "max_ir_value": config.max_ir_value,
            "obstacle_penalty_threshold": config.obstacle_penalty_threshold,
            "collision_ir_threshold": config.collision_ir_threshold,
            "progress_normalizer_m": config.progress_normalizer_m,
            "progress_reward_scale": config.progress_reward_scale,
            "distance_bonus_scale": config.distance_bonus_scale,
            "obstacle_penalty_scale": config.obstacle_penalty_scale,
            "front_obstacle_penalty_scale": config.front_obstacle_penalty_scale,
            "action_penalty_scale": config.action_penalty_scale,
            "turning_penalty_scale": config.turning_penalty_scale,
            "alive_bonus": config.alive_bonus,
            "collision_penalty": config.collision_penalty,
        },
    )

    env = make_env(config)

    if check_environment:
        check_env(env.unwrapped, warn=True, skip_render_check=True)

    policy_kwargs = {
        "features_extractor_class": RoboboCombinedExtractor,
        "features_extractor_kwargs": {
            "cnn_features_dim": 128,
            "ir_features_dim": 32,
            "combined_features_dim": 256,
        },
        "net_arch": {
            "pi": [256, 256],
            "qf": [256, 256],
        },
        "activation_fn": th.nn.ReLU,
        "share_features_extractor": False,
    }

    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=2_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device="auto",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_DIR),
        name_prefix="robobo_sac",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    wandb_callback = WandbInfoCallback(log_freq=10)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, wandb_callback],
            log_interval=10,
            progress_bar=True,
            tb_log_name=run.name,
        )

        final_model_path = MODEL_DIR / "robobo_sac_final"
        model.save(str(final_model_path))

        wandb.save(str(final_model_path) + ".zip")

    finally:
        env.close()
        run.finish()

    return model


if __name__ == "__main__":
    main()
