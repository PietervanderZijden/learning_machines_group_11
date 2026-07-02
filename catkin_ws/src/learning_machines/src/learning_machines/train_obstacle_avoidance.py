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
    'Persist the W&B run ID to disk so it survives crashes.'
    RUN_ID_FILE.write_text(run_id)


def load_wandb_run_id() -> str | None:
    'Return the previously saved W&B run ID, or None if this is a fresh start.'
    if RUN_ID_FILE.exists():
        return RUN_ID_FILE.read_text().strip() or None
    return None


def _make_fresh_model(env: Monitor, policy_kwargs: dict) -> SAC:
    return SAC(
        policy="MultiInputPolicy",
        env=env,
        #policy_kwargs=policy_kwargs,
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


def _checkpoint_step(path: Path) -> int | None:
    match = re.search(r"_(\d+)_steps", path.stem)
    if match is None:
        return None

    return int(match.group(1))


def _replay_buffer_path_for_checkpoint(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}_replay_buffer.pkl",
    )


def resume_from_latest_checkpoint(
    model_dir: Path,
    env: Monitor,
    policy_kwargs: dict,
    warmup_steps_if_no_buffer: int = 2_000,
) -> tuple[SAC, int]:
    'Load the latest SAC checkpoint and matching replay buffer.'

    checkpoint_paths = []

    for path in model_dir.glob("robobo_sac_*_steps.zip"):
        step = _checkpoint_step(path)
        if step is not None:
            checkpoint_paths.append((step, path))

    checkpoint_paths = sorted(checkpoint_paths, key=lambda item: item[0])

    if not checkpoint_paths:
        print("No checkpoint found — starting fresh.")
        return _make_fresh_model(env, policy_kwargs), 0

    checkpoint_step, latest = checkpoint_paths[-1]

    print(
        f"Resuming from checkpoint: {latest.name} "
        f"({checkpoint_step:,} steps from filename)"
    )

    model = SAC.load(str(latest), env=env, device="auto")

    if model.num_timesteps > 0:
        steps_done = int(model.num_timesteps)
    else:
        steps_done = checkpoint_step

    buffer_path = _replay_buffer_path_for_checkpoint(latest)

    if buffer_path.exists():
        print(f"Loading replay buffer: {buffer_path.name}")
        model.load_replay_buffer(str(buffer_path))

        if model.replay_buffer is not None:
            buffer_size = model.replay_buffer.size()
            print(f"Replay buffer size after loading: {buffer_size:,}")

            if buffer_size < model.batch_size:
                print(
                    "Warning: replay buffer is smaller than batch_size. "
                    "Delaying learning for additional warmup."
                )
                model.learning_starts = model.num_timesteps + warmup_steps_if_no_buffer
    else:
        print(
            "Warning: no matching replay buffer found. "
            "The policy and critics were loaded, but SAC will not have the "
            "old off-policy data. Delaying learning to collect fresh data."
        )

        model.learning_starts = model.num_timesteps + warmup_steps_if_no_buffer

    return model, steps_done


class WandbInfoCallback(BaseCallback):
    'Logs Robobo-specific env metrics to W&B.'

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
            "step_displacement",
            "wheel_difference",
            "step_displacement",
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
            ir_bias_range=(-0.04, 0.04),
            ir_noise_std=0.03,
            ir_dropout_prob=0.01,
            image_contrast_range=(0.8, 1.20),
            image_brightness_range=(-25.0, 25.0),
            image_noise_std=4.0,
            image_blur_prob=0.05,
            action_scale_range=(0.9, 1.10),
            action_bias_range=(-0.03, 0.03),
            action_noise_std=0.020,
            action_latency_prob=0.00,
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
    check_environment: bool = False,
    force_new_wandb_run: bool = False,
) -> SAC:
    'Start (or resume) SAC training.'

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    config = RoboboObstacleEnvConfig(
        image_size=(128, 128),
        max_wheel_speed=70,
        step_millis=100,
        max_episode_steps=300,
        max_ir_value=400.0,
        obstacle_penalty_threshold=0.15,
        collision_ir_threshold=0.85,
        progress_normalizer_m=0.05,
        progress_reward_scale=1.5,
        distance_bonus_scale=0,
        obstacle_penalty_scale=0.15,
        front_obstacle_penalty_scale=0.25,
        action_penalty_scale=0.01,
        turning_penalty_scale=0.02,
        spin_penalty_scale=0.08,
        alive_bonus=0.0,
        collision_penalty=5.0,
        idle_penalty_scale=0.10,
        idle_speed_threshold=0.15,
        movement_bonus_scale=0.04,
        low_displacement_penalty_scale=0.15,
        low_displacement_threshold_m=0.02,
        near_start_penalty_scale=0.03,
        near_start_distance_threshold_m=0.10,
        near_start_grace_steps=20,
        reset_settle_seconds=0.1,
    )

    if force_new_wandb_run and RUN_ID_FILE.exists():
        RUN_ID_FILE.unlink()

    existing_run_id = load_wandb_run_id()

    if existing_run_id:
        print(f"Resuming W&B run: {existing_run_id}")
    else:
        print("Starting a new W&B run.")
    robobo_identifiers = (0, 1)
    switch_every_steps = 1000
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
        "activation_fn": th.nn.SELU,
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
        model.save_replay_buffer(str(MODEL_DIR / "robobo_sac_final_replay_buffer.pkl"))

        wandb.save(str(final_model_path) + ".zip")
        wandb.save(str(MODEL_DIR / "robobo_sac_final_replay_buffer.pkl"))

    finally:
        env.close()
        run.finish()
    return model


if __name__ == "__main__":
    main()
