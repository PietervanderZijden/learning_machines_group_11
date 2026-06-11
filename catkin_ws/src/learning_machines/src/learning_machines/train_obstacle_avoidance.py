from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch as th
import wandb
from learning_machines.multi_robobo_env import (
    DomainRandomizationConfig,
    MultiRoboboObstacleAvoidanceEnv,
    RoboboDomainRandomizationWrapper,
)
from learning_machines.rl_robobo_env import RoboboObstacleEnvConfig
from learning_machines.robobo_sac_policy import RoboboCombinedExtractor
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

RUN_DIR = Path("/root/results/runs/robobo_obstacle_sac")
MODEL_DIR = RUN_DIR / "models"
LOG_DIR = RUN_DIR / "logs"
RUN_ID_FILE = RUN_DIR / "wandb_run_id.txt"


def save_wandb_run_id(run_id: str) -> None:
    """Persist the W&B run ID to disk so it survives crashes."""
    RUN_ID_FILE.write_text(run_id)


def load_wandb_run_id() -> str | None:
    """
    Return the previously saved W&B run ID, or None if this is a fresh start.
    An empty file is treated the same as no file.
    """
    if RUN_ID_FILE.exists():
        return RUN_ID_FILE.read_text().strip() or None
    return None


def _make_fresh_model(env: Monitor, policy_kwargs: dict) -> SAC:
    return SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=2_000,
        batch_size=128,
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


def resume_from_latest_checkpoint(
    model_dir: Path,
    env: Monitor,
    policy_kwargs: dict,
) -> tuple[SAC, int]:
    """
    Scan model_dir for the latest CheckpointCallback snapshot and load it.

    Returns:
        (model, steps_already_done)
        steps_already_done is 0 when no checkpoint exists and a fresh
        model is created instead.
    """
    checkpoint_paths = sorted(
        model_dir.glob("robobo_sac_*_steps.zip"),
        key=lambda p: int(re.search(r"_(\d+)_steps", p.stem).group(1)),
    )

    if not checkpoint_paths:
        print("No checkpoint found — starting fresh.")
        return _make_fresh_model(env, policy_kwargs), 0

    latest = checkpoint_paths[-1]
    steps_done = int(re.search(r"_(\d+)_steps", latest.stem).group(1))
    print(f"Resuming from checkpoint: {latest.name}  ({steps_done:,} steps done)")

    model = SAC.load(str(latest), env=env, device="auto")

    buffer_path = model_dir / (
        latest.stem.replace("_steps", "_steps_replay_buffer") + ".pkl"
    )
    if buffer_path.exists():
        print(f"Loading replay buffer: {buffer_path.name}")
        model.load_replay_buffer(str(buffer_path))
    else:
        print(
            "Warning: no replay buffer found — SAC will explore from scratch "
            "until learning_starts is reached."
        )

    return model, steps_done


class WandbInfoCallback(BaseCallback):
    """
    Logs Robobo-specific env metrics to W&B.
    """

    def __init__(self, log_freq: int = 10, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq != 0:
            return True

        infos: list[dict[str, Any]] = self.locals.get("infos", [])
        if not infos:
            return True

        info = infos[0]
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
            "active_robot_identifier",
            "robot_switch_due",
            "steps_since_robot_switch",
            "domain_randomization_enabled",
            "executed_action_left",
            "executed_action_right",
        ]

        metrics = {f"env/{key}": info[key] for key in keys_to_log if key in info}

        if metrics:
            wandb.log(metrics, step=self.num_timesteps)

        return True


def make_env(
    config: RoboboObstacleEnvConfig,
    identifiers: tuple[int, ...] = (0, 1, 2),
    switch_every_steps: int = 500,
    use_domain_randomization: bool = True,
) -> Monitor:
    env = MultiRoboboObstacleAvoidanceEnv(
        identifiers=identifiers,
        config=config,
        switch_every_steps=switch_every_steps,
        avoid_immediate_repeat=True,
    )

    if use_domain_randomization:
        domain_randomization_config = DomainRandomizationConfig(
            enabled=True,
            ir_scale_range=(0.6, 1.4),
            ir_bias_range=(-0.08, 0.08),
            ir_noise_std=0.03,
            ir_dropout_prob=0.02,
            image_contrast_range=(0.75, 1.25),
            image_brightness_range=(-25.0, 25.0),
            image_noise_std=6.0,
            image_blur_prob=0.10,
            action_scale_range=(0.85, 1.15),
            action_bias_range=(-0.04, 0.04),
            action_noise_std=0.025,
            action_latency_prob=0.05,
        )

        env = RoboboDomainRandomizationWrapper(
            env,
            config=domain_randomization_config,
        )

    return Monitor(env, filename=str(LOG_DIR / "monitor.csv"))


def main(
    total_timesteps: int = 300_000,
    wandb_project: str = "learning-machines",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    check_environment: bool = True,
    force_new_wandb_run: bool = False,
) -> SAC:
    """
    Start (or resume) SAC training.

    Args:
        total_timesteps:
            Total environment steps for the *full* training run.
            Steps already completed in previous runs are subtracted
            automatically so the overall budget stays correct.
        force_new_wandb_run:
            When True, ignore any saved run ID and start a brand-new
            W&B run. Useful when you intentionally want a clean slate
            after a completed or abandoned run. The old run-ID file is
            deleted so subsequent resumes start fresh too.

    Example:
        from package import main

        main()                                    # start or resume
        main(force_new_wandb_run=True)            # always start fresh
        main(total_timesteps=100_000, wandb_mode="offline")
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config = RoboboObstacleEnvConfig(
        image_size=(100, 100),
        max_wheel_speed=100,
        step_millis=200,
        max_episode_steps=500,
        max_ir_value=400.0,
        obstacle_penalty_threshold=0.15,
        collision_ir_threshold=0.85,
        progress_normalizer_m=0.05,
        progress_reward_scale=1.0,
        distance_bonus_scale=0.02,
        obstacle_penalty_scale=0.15,
        front_obstacle_penalty_scale=0.25,
        action_penalty_scale=0.02,
        turning_penalty_scale=0.02,
        alive_bonus=0.01,
        collision_penalty=5.0,
        reset_settle_seconds=0.1,
    )

    if force_new_wandb_run and RUN_ID_FILE.exists():
        RUN_ID_FILE.unlink()

    existing_run_id = load_wandb_run_id()

    if existing_run_id:
        print(f"Resuming W&B run: {existing_run_id}")
    else:
        print("Starting a new W&B run.")
    robobo_identifiers = (0, 1, 2)
    switch_every_steps = 500
    use_domain_randomization = True

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        id=existing_run_id,
        resume="allow",
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
            "robobo_identifiers": robobo_identifiers,
            "switch_every_steps": switch_every_steps,
            "use_domain_randomization": use_domain_randomization,
        },
    )

    save_wandb_run_id(run.id)

    env = make_env(
        config=config,
        identifiers=robobo_identifiers,
        switch_every_steps=switch_every_steps,
        use_domain_randomization=use_domain_randomization,
    )

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

    model, steps_done = resume_from_latest_checkpoint(MODEL_DIR, env, policy_kwargs)
    remaining_steps = total_timesteps - steps_done

    if remaining_steps <= 0:
        print(
            f"Training already complete ({steps_done:,} / {total_timesteps:,} steps). "
            "Nothing to do. Pass force_new_wandb_run=True to start over."
        )
        env.close()
        run.finish()
        return model

    print(f"Training for {remaining_steps:,} more steps ({steps_done:,} already done).")

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(MODEL_DIR),
        name_prefix="robobo_sac",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    wandb_callback = WandbInfoCallback(log_freq=10)

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=[checkpoint_callback, wandb_callback],
            log_interval=10,
            progress_bar=True,
            tb_log_name=run.name,
            reset_num_timesteps=False,
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
