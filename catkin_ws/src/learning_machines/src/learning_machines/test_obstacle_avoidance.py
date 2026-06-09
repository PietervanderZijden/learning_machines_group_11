from __future__ import annotations

from pathlib import Path

import wandb
from rl_robobo_env import RoboboObstacleAvoidanceEnv, RoboboObstacleEnvConfig
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

MODEL_PATH = Path("results/runs/robobo_obstacle_sac/models/robobo_sac_final.zip")


def test(
    model_path: str | Path = MODEL_PATH,
    episodes: int = 10,
    wandb_project: str = "learning-machines",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
) -> None:
    """
    Test/evaluate a trained SAC model.

    Example:
        from package import test

        test()

    Or:
        test("runs/robobo_obstacle_sac/models/robobo_sac_final.zip")
    """

    run = wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        name="robobo_sac_evaluation",
        config={
            "model_path": str(model_path),
            "episodes": episodes,
        },
    )

    config = RoboboObstacleEnvConfig(
        image_size=(84, 84),
        max_wheel_speed=100,
        step_millis=200,
        max_episode_steps=500,
        max_ir_value=100.0,
        obstacle_penalty_threshold=0.35,
        collision_ir_threshold=0.85,
    )

    env = Monitor(RoboboObstacleAvoidanceEnv(config=config))

    model = SAC.load(str(model_path), env=env, device="auto")

    try:
        for episode in range(episodes):
            obs, info = env.reset()

            terminated = False
            truncated = False
            total_reward = 0.0
            step = 0

            while not terminated and not truncated:
                action, _state = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                step += 1

                wandb.log(
                    {
                        "eval/episode": episode,
                        "eval/step": step,
                        "eval/reward": reward,
                        "eval/total_reward": total_reward,
                        "eval/distance_from_start": info["distance_from_start"],
                        "eval/front_obstacle_closeness": info[
                            "front_obstacle_closeness"
                        ],
                        "eval/back_obstacle_closeness": info["back_obstacle_closeness"],
                        "eval/max_obstacle_closeness": info["max_obstacle_closeness"],
                        "eval/collision": info["collision"],
                        "eval/x": info["x"],
                        "eval/y": info["y"],
                        "eval/z": info["z"],
                    }
                )

                print(
                    f"Episode {episode} | "
                    f"Step {step} | "
                    f"Reward {reward:.3f} | "
                    f"Total {total_reward:.3f} | "
                    f"Distance {info['distance_from_start']:.3f} | "
                    f"Front IR {info['front_obstacle_closeness']:.3f} | "
                    f"Collision {info['collision']}"
                )

            wandb.log(
                {
                    "eval_episode/final_episode": episode,
                    "eval_episode/final_total_reward": total_reward,
                    "eval_episode/final_distance_from_start": info[
                        "distance_from_start"
                    ],
                    "eval_episode/final_collision": info["collision"],
                    "eval_episode/final_steps": step,
                }
            )

            print(
                f"Finished episode {episode}. "
                f"Total reward: {total_reward:.3f}. "
                f"Distance: {info['distance_from_start']:.3f}. "
                f"Collision: {info['collision']}."
            )

    finally:
        env.close()
        run.finish()


if __name__ == "__main__":
    test()
