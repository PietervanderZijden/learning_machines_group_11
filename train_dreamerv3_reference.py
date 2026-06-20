#!/usr/bin/env python3
"""Train NM512 DreamerV3 on Robobo food collection using our env."""
from __future__ import annotations

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
NM512_DIR = PROJECT_ROOT / "dreamerv3_reference"

sys.path.insert(0, str(NM512_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "catkin_ws" / "src" / "learning_machines" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "catkin_ws" / "src" / "robobo_interface" / "src"))

from rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
from robobo_env_wrapper import RoboboNM512Wrapper
from transfer import default_calibration_profile

import dreamer
from envs import wrappers
from tools import Logger


def make_robobo_env(config, mode, robobo_id=0):
    env_config = RoboboCompactEnvConfig(
        initialize_phone_tilt=False,
        max_episode_steps=200,
        step_millis=400,
    )
    from robobo_interface import SimulationRobobo
    rob = SimulationRobobo()
    base_env = RoboboCompactEnv(rob=rob, config=env_config)
    env = RoboboNM512Wrapper(base_env, include_ir=True)
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train NM512 DreamerV3 on Robobo")
    parser.add_argument("--logdir", type=str, default="results/nm512_robobo",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Total training steps")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-length", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()

    import yaml
    config_path = NM512_DIR / "configs.yaml"
    with open(config_path) as f:
        all_configs = yaml.safe_load(f)

    cfg = all_configs["defaults"]
    cfg.update(all_configs.get("robobo", {}))

    cfg["task"] = "robobo_food_collection"
    cfg["steps"] = args.steps
    cfg["device"] = args.device
    cfg["seed"] = args.seed
    cfg["batch_size"] = args.batch_size
    cfg["batch_length"] = args.batch_length
    cfg["eval_every"] = args.eval_every
    cfg["log_every"] = args.log_every
    cfg["logdir"] = args.logdir
    cfg["traindir"] = os.path.join(args.logdir, "train")
    cfg["evaldir"] = os.path.join(args.logdir, "eval")
    cfg["size"] = [64, 64]
    cfg["envs"] = 1
    cfg["action_repeat"] = 1
    cfg["time_limit"] = 200
    cfg["grayscale"] = False
    cfg["prefill"] = 500
    cfg["encoder"] = {
        "mlp_keys": "ir",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 32,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 5,
        "mlp_units": 1024,
        "symlog_inputs": True,
    }
    cfg["decoder"] = {
        "mlp_keys": "ir",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": 32,
        "kernel_size": 4,
        "minres": 4,
        "mlp_layers": 5,
        "mlp_units": 1024,
        "cnn_sigmoid": False,
        "image_dist": "mse",
        "vector_dist": "symlog_mse",
        "outscale": 1.0,
    }

    from types import SimpleNamespace
    config = SimpleNamespace(**cfg)

    config.num_actions = 2

    os.makedirs(config.logdir, exist_ok=True)
    os.makedirs(config.traindir, exist_ok=True)
    os.makedirs(config.evaldir, exist_ok=True)

    print(f"Starting NM512 DreamerV3 training on Robobo")
    print(f"  Log dir: {config.logdir}")
    print(f"  Steps: {config.steps}")
    print(f"  Device: {config.device}")
    print(f"  Batch: {config.batch_size} x {config.batch_length}")

    train_env = make_robobo_env(config, "train")
    eval_env = make_robobo_env(config, "eval")

    agent = dreamer.Dreamer(config, train_env.observation_space, train_env.action_space)
    logger = Logger(config.logdir, 0)

    step = 0
    episode = 0
    obs = train_env.reset()
    agent._should_train = True

    print("Prefilling replay buffer...")
    while step < config.prefill:
        action = train_env.action_space.sample()
        obs, reward, done, info = train_env.step(action)
        agent._dataset.add({"obs": obs, "action": action, "reward": reward, "done": done})
        step += 1
        if done:
            obs = train_env.reset()
            episode += 1
    print(f"Prefill complete. {step} steps, {episode} episodes")

    print("Starting training...")
    episode_reward = 0.0
    episode_count = 0
    while step < config.steps:
        policy_output = agent.policy(obs)
        action = policy_output["action"]
        next_obs, reward, done, info = train_env.step(action)
        agent._dataset.add({
            "obs": obs,
            "action": action,
            "reward": reward,
            "done": done,
        })
        obs = next_obs
        episode_reward += reward
        step += 1

        if done:
            print(f"Step {step:>8d} | Episode {episode_count} | Return {episode_reward:.1f}")
            logger.scalar("episode_return", episode_reward, step)
            logger.scalar("episode_count", episode_count, step)
            logger.dump()
            episode_reward = 0.0
            episode_count += 1
            obs = train_env.reset()

        if step % config.log_every == 0 and step > config.prefill:
            metrics = agent.train(step)
            for key, val in metrics.items():
                logger.scalar(f"train/{key}", float(val), step)
            logger.dump()

        if step % config.eval_every == 0 and step > config.prefill:
            eval_reward = 0.0
            eval_obs = eval_env.reset()
            eval_done = False
            eval_steps = 0
            while not eval_done:
                eval_action = agent.policy(eval_obs)["action"]
                eval_obs, eval_r, eval_done, _ = eval_env.step(eval_action)
                eval_reward += eval_r
                eval_steps += 1
            logger.scalar("eval_return", eval_reward, step)
            logger.scalar("eval_steps", eval_steps, step)
            logger.dump()
            print(f"  Eval @ step {step}: return={eval_reward:.1f}, steps={eval_steps}")

    train_env.close()
    eval_env.close()
    print(f"Training complete. {step} steps, {episode_count} episodes")


if __name__ == "__main__":
    main()
