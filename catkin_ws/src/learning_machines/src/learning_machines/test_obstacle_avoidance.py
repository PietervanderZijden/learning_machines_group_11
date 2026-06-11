from __future__ import annotations

from pathlib import Path

import numpy as np
import wandb
from learning_machines.multi_robobo_env import MultiRoboboObstacleAvoidanceEnv
from learning_machines.rl_robobo_env import RoboboObstacleEnvConfig
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

MODEL_PATH = Path("/root/results/runs/robobo_obstacle_sac/models/robobo_sac_final.zip")


def test(
    model_path: str | Path = MODEL_PATH,
    episodes_per_environment: int = 5,
    identifiers: tuple[int, ...] = (0, 1, 2),
    wandb_project: str = "learning-machines",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
) -> None:
    """
    Evaluate a trained SAC model on all Robobo environments.

    Example:
        test()

        test(
            model_path="runs/robobo_obstacle_sac/models/robobo_sac_final.zip",
            episodes_per_environment=10,
        )
    """

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        name="robobo_sac_evaluation_all_envs",
        config={
            "model_path": str(model_path),
            "episodes_per_environment": episodes_per_environment,
            "identifiers": identifiers,
        },
    )

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
        turning_penalty_scale=0.0,
        alive_bonus=0.01,
        collision_penalty=5.0,
        reset_settle_seconds=0.1,
    )

    env = Monitor(
        MultiRoboboObstacleAvoidanceEnv(
            identifiers=identifiers,
            config=config,
            switch_every_steps=config.max_episode_steps,
            avoid_immediate_repeat=False,
        )
    )

    model = SAC.load(str(model_path), env=env, device="auto")

    all_episode_rewards: list[float] = []
    all_episode_lengths: list[int] = []
    all_episode_distances: list[float] = []
    all_episode_collisions: list[bool] = []

    try:
        for identifier in identifiers:
            env_rewards: list[float] = []
            env_lengths: list[int] = []
            env_distances: list[float] = []
            env_collisions: list[bool] = []

            for episode in range(episodes_per_environment):
                obs, info = env.reset(options={"identifier": identifier})

                terminated = False
                truncated = False
                total_reward = 0.0
                step = 0

                while not terminated and not truncated:
                    action, _state = model.predict(obs, deterministic=True)

                    obs, reward, terminated, truncated, info = env.step(action)

                    total_reward += float(reward)
                    step += 1

                    wandb.log(
                        {
                            "eval/identifier": identifier,
                            "eval/episode": episode,
                            "eval/step": step,
                            "eval/reward": reward,
                            "eval/total_reward": total_reward,
                            "eval/distance_from_start": info["distance_from_start"],
                            "eval/front_obstacle_closeness": info[
                                "front_obstacle_closeness"
                            ],
                            "eval/back_obstacle_closeness": info[
                                "back_obstacle_closeness"
                            ],
                            "eval/max_obstacle_closeness": info[
                                "max_obstacle_closeness"
                            ],
                            "eval/collision": info["collision"],
                            "eval/x": info["x"],
                            "eval/y": info["y"],
                            "eval/z": info["z"],
                        }
                    )

                final_distance = float(info["distance_from_start"])
                final_collision = bool(info["collision"])

                env_rewards.append(total_reward)
                env_lengths.append(step)
                env_distances.append(final_distance)
                env_collisions.append(final_collision)

                all_episode_rewards.append(total_reward)
                all_episode_lengths.append(step)
                all_episode_distances.append(final_distance)
                all_episode_collisions.append(final_collision)

                wandb.log(
                    {
                        "eval_episode/identifier": identifier,
                        "eval_episode/episode": episode,
                        "eval_episode/total_reward": total_reward,
                        "eval_episode/distance_from_start": final_distance,
                        "eval_episode/collision": final_collision,
                        "eval_episode/steps": step,
                    }
                )

                print(
                    f"Identifier {identifier} | "
                    f"Episode {episode} | "
                    f"Reward {total_reward:.3f} | "
                    f"Steps {step} | "
                    f"Distance {final_distance:.3f} | "
                    f"Collision {final_collision}"
                )

            env_summary = {
                f"eval_summary/env_{identifier}/mean_reward": float(
                    np.mean(env_rewards)
                ),
                f"eval_summary/env_{identifier}/std_reward": float(np.std(env_rewards)),
                f"eval_summary/env_{identifier}/mean_length": float(
                    np.mean(env_lengths)
                ),
                f"eval_summary/env_{identifier}/mean_distance": float(
                    np.mean(env_distances)
                ),
                f"eval_summary/env_{identifier}/collision_rate": float(
                    np.mean(env_collisions)
                ),
            }

            wandb.log(env_summary)

            print(
                f"\nEnvironment {identifier} summary:\n"
                f"  Mean reward: {env_summary[f'eval_summary/env_{identifier}/mean_reward']:.3f}\n"
                f"  Mean length: {env_summary[f'eval_summary/env_{identifier}/mean_length']:.1f}\n"
                f"  Mean distance: {env_summary[f'eval_summary/env_{identifier}/mean_distance']:.3f}\n"
                f"  Collision rate: {env_summary[f'eval_summary/env_{identifier}/collision_rate']:.3f}\n"
            )

        wandb.log(
            {
                "eval_summary/all/mean_reward": float(np.mean(all_episode_rewards)),
                "eval_summary/all/std_reward": float(np.std(all_episode_rewards)),
                "eval_summary/all/mean_length": float(np.mean(all_episode_lengths)),
                "eval_summary/all/mean_distance": float(np.mean(all_episode_distances)),
                "eval_summary/all/collision_rate": float(
                    np.mean(all_episode_collisions)
                ),
            }
        )

    finally:
        env.close()
        run.finish()


if __name__ == "__main__":
    test()
