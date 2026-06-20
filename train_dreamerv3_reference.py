#!/usr/bin/env python3
"""Train NM512 DreamerV3 on Robobo food collection using our env."""
from __future__ import annotations

import functools
import os
import sys
import pathlib
import argparse
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import torch
from ruamel.yaml import YAML

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
NM512_DIR = PROJECT_ROOT / "dreamerv3_reference"

sys.path.insert(0, str(NM512_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "catkin_ws" / "src" / "learning_machines" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "catkin_ws" / "src" / "learning_machines" / "src" / "learning_machines"))
sys.path.insert(0, str(PROJECT_ROOT / "catkin_ws" / "src" / "robobo_interface" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "catkin_ws" / "src" / "robobo_interface" / "src" / "robobo_interface"))

from rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
from robobo_env_wrapper import RoboboNM512Wrapper

import dreamer
from envs import wrappers
import tools
from parallel import Damy


def make_robobo_env(config, mode, robobo_id=0):
    env_config = RoboboCompactEnvConfig(
        initialize_phone_tilt=False,
        max_episode_steps=200,
        step_millis=400,
        collect_reward=1.0,
        time_penalty_per_second=0.0,
        action_change_penalty=0.0,
        return_image=True,
        image_obs_size=(64, 64),
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
    yaml = YAML()
    config_path = NM512_DIR / "configs.yaml"
    with open(config_path) as f:
        all_configs = yaml.load(f)

    def to_native(v):
        if isinstance(v, dict):
            return {k: to_native(val) for k, val in v.items()}
        if isinstance(v, list):
            return [to_native(x) for x in v]
        try:
            return v.__class__.__bases__[0](v) if not isinstance(v, (int, float, str, bool, type(None))) else v
        except (TypeError, ValueError):
            return v

    cfg = to_native(all_configs["defaults"])
    cfg.update(to_native(all_configs.get("robobo", {})))

    cfg["task"] = "robobo_food_collection"
    cfg["size"] = [64, 64]
    cfg["envs"] = 1
    cfg["action_repeat"] = 1
    cfg["time_limit"] = 200
    cfg["grayscale"] = False
    cfg["prefill"] = 5000
    cfg["model_lr"] = 4e-5
    cfg["actor"] = {
        "layers": 3,
        "dist": "normal",
        "entropy": 3e-4,
        "unimix_ratio": 0.01,
        "std": "learned",
        "min_std": 0.1,
        "max_std": 1.0,
        "temp": 0.1,
        "lr": 4e-5,
        "eps": 1e-5,
        "grad_clip": 100.0,
        "outscale": 1.0,
    }
    cfg["critic"] = {
        "layers": 3,
        "dist": "symlog_disc",
        "slow_target": True,
        "slow_target_update": 1,
        "slow_target_fraction": 0.02,
        "lr": 4e-5,
        "eps": 1e-5,
        "grad_clip": 100.0,
        "outscale": 0.0,
    }
    cfg["grad_clip"] = 100.0
    cfg["batch_size"] = 32
    cfg["batch_length"] = 50
    cfg["dataset_size"] = 100_000
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="results/nm512_robobo")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-length", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=2)
    cli = parser.parse_args()

    cfg["steps"] = cli.steps
    cfg["device"] = cli.device
    cfg["seed"] = cli.seed
    cfg["batch_size"] = cli.batch_size
    cfg["batch_length"] = cli.batch_length
    cfg["eval_every"] = cli.eval_every
    cfg["log_every"] = cli.log_every
    cfg["eval_episode_num"] = cli.eval_episodes
    cfg["logdir"] = cli.logdir

    from types import SimpleNamespace
    config = SimpleNamespace(**cfg)
    config.num_actions = 2

    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = str(logdir / "train_eps")
    config.evaldir = str(logdir / "eval_eps")
    os.makedirs(config.traindir, exist_ok=True)
    os.makedirs(config.evaldir, exist_ok=True)

    tools.set_seed_everywhere(config.seed)

    step = dreamer.count_steps(pathlib.Path(config.traindir))
    logger = tools.Logger(logdir, config.action_repeat * step)

    print(f"Starting NM512 DreamerV3 training on Robobo")
    print(f"  Log dir: {logdir}")
    print(f"  Steps: {config.steps}")
    print(f"  Device: {config.device}")
    print(f"  Batch: {config.batch_size} x {config.batch_length}")
    print(f"  Eval episodes: {config.eval_episode_num}")

    train_eps = tools.load_episodes(pathlib.Path(config.traindir), limit=config.dataset_size)
    eval_eps = tools.load_episodes(pathlib.Path(config.evaldir), limit=1)

    train_env = Damy(make_robobo_env(config, "train"))
    eval_env = Damy(make_robobo_env(config, "eval"))

    acts = train_env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    prefill = max(0, config.prefill - dreamer.count_steps(pathlib.Path(config.traindir)))
    print(f"Prefill dataset ({prefill} steps).")
    random_actor = torch.distributions.independent.Independent(
        torch.distributions.uniform.Uniform(
            torch.tensor(acts.low).repeat(config.envs, 1),
            torch.tensor(acts.high).repeat(config.envs, 1),
        ),
        1,
    )

    def random_agent(o, d, s):
        action = random_actor.sample()
        return {"action": action}, None

    state = tools.simulate(
        random_agent,
        [train_env],
        train_eps,
        config.traindir,
        logger,
        limit=config.dataset_size,
        steps=prefill,
    )
    logger.step += prefill * config.action_repeat
    print(f"Prefill done. Logger step: {logger.step}")

    train_dataset = dreamer.make_dataset(train_eps, config)
    eval_dataset = dreamer.make_dataset(eval_eps, config)

    agent = dreamer.Dreamer(
        train_env.observation_space,
        train_env.action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    print("Start training.")
    while agent._step < config.steps + config.eval_every:
        logger.write()
        print(f"Step {agent._step}/{config.steps} | Evaluating ({config.eval_episode_num} episodes)...")
        eval_policy = functools.partial(agent, training=False)
        tools.simulate(
            eval_policy,
            [eval_env],
            eval_eps,
            config.evaldir,
            logger,
            is_eval=True,
            episodes=config.eval_episode_num,
        )
        if config.video_pred_log:
            from dreamer import to_np
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_openl", to_np(video_pred))
        print(f"Step {agent._step}/{config.steps} | Training {config.eval_every} steps...")
        state = tools.simulate(
            agent,
            [train_env],
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        items_to_save = {
            "agent_state_dict": agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        }
        torch.save(items_to_save, logdir / "latest.pt")
        print(f"Step {agent._step}/{config.steps} | Saved checkpoint.")

    for env in [train_env, eval_env]:
        try:
            env.close()
        except Exception:
            pass
    print(f"Training complete.")


if __name__ == "__main__":
    main()
