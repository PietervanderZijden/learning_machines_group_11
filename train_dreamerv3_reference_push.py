#!/usr/bin/env python3
"""Train NM512 DreamerV3 on Robobo push task using domain randomization."""

from __future__ import annotations

import argparse
import atexit
import collections
import contextlib
import functools
import json
import os
import pathlib
import sys

import numpy as np
import torch
from ruamel.yaml import YAML
from tqdm import tqdm

import wandb

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
NM512_DIR = PROJECT_ROOT / "dreamerv3_reference"

sys.path.insert(0, str(NM512_DIR))
sys.path.insert(
    0,
    str(PROJECT_ROOT / "catkin_ws/src/learning_machines/src"),
)
sys.path.insert(
    0,
    str(PROJECT_ROOT / "catkin_ws/src/learning_machines/src/learning_machines"),
)
sys.path.insert(
    0,
    str(PROJECT_ROOT / "catkin_ws/src/robobo_interface/src"),
)
sys.path.insert(
    0,
    str(PROJECT_ROOT / "catkin_ws/src/robobo_interface/src/robobo_interface"),
)

import dreamer
import tools
from domain_randomization import DomainRandomizationWrapper, RandomizationRanges
from envs import wrappers
from learning_machines.push_curriculum import (
    PushCurriculumConfig,
    PushCurriculumController,
)
from learning_machines.reference_checkpoint import (
    REFERENCE_CHECKPOINT_VERSION,
    collect_optimizer_state_dicts,
    load_optimizer_state_dicts,
    save_checkpoint_atomic,
    validate_checkpoint,
)
from learning_machines.reference_wandb_media import WandbLogger
from parallel import Damy
from rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
from robobo_env_wrapper import RoboboNM512Wrapper

PUSH_REWARD_CONTRACT = "robobo-push-phased-dense-v1"
PUSH_TIME_PENALTY_PER_SECOND = 0.05
PUSH_APPROACH_POTENTIAL_SCALE = 2.0
PUSH_GOAL_POTENTIAL_OFFSET = 2.0
PUSH_GOAL_POTENTIAL_SCALE = 4.0
PUSH_CONTACT_BONUS = 1.0
PUSH_APPROACH_COMPLETION_BONUS = 5.0
PUSH_GOAL_COMPLETION_BONUS = 15.0
REFERENCE_IMAGE_SIZE = (96, 96)
REFERENCE_DYN_HIDDEN = 512
REFERENCE_DYN_DETER = 1024
REFERENCE_MLP_UNITS = 512
REFERENCE_CNN_DEPTH = 32
REFERENCE_CNN_MINRES = 3
REFERENCE_BATCH_SIZE = 4
REFERENCE_BATCH_LENGTH = 48
REFERENCE_TRAIN_RATIO = 128


class EpisodeStats:
    """Tracks rolling averages of push success over recent episodes."""

    def __init__(self, window=100):
        self._rewards = collections.deque(maxlen=window)
        self._successes = collections.deque(maxlen=window)

    def add(self, reward, success):
        self._rewards.append(reward)
        self._successes.append(int(success))

    @property
    def avg_reward(self):
        return sum(self._rewards) / len(self._rewards) if self._rewards else 0.0

    @property
    def success_rate(self):
        return sum(self._successes) / len(self._successes) if self._successes else 0.0

    @property
    def count(self):
        return len(self._rewards)


def make_robobo_env(config, curriculum, curriculum_state_path):
    curriculum_stage = curriculum.stage
    env_config = RoboboCompactEnvConfig(
        task="push",
        initialize_phone_tilt=True,
        max_episode_steps=config.time_limit,
        step_millis=400,
        return_image=True,
        image_obs_size=tuple(config.size),
        randomize_push_layout=True,
        push_curriculum_stage=curriculum_stage,
        push_goal_jitter_radius=config.curriculum_goal_jitter_radius,
        action_smoothing=False,
        pre_action_safety=False,
        max_action_delta=2.0,
        push_discount=config.discount,
        push_approach_potential_scale=PUSH_APPROACH_POTENTIAL_SCALE,
        push_goal_potential_offset=PUSH_GOAL_POTENTIAL_OFFSET,
        push_goal_potential_scale=PUSH_GOAL_POTENTIAL_SCALE,
        push_contact_bonus=PUSH_CONTACT_BONUS,
        push_approach_completion_bonus=PUSH_APPROACH_COMPLETION_BONUS,
        push_goal_completion_bonus=PUSH_GOAL_COMPLETION_BONUS,
        push_standoff_distance=0.22,
        push_time_penalty_per_second=PUSH_TIME_PENALTY_PER_SECOND,
        push_action_change_penalty=0.0,
    )
    from robobo_interface import SimulationRobobo

    rob = SimulationRobobo()
    base_env = RoboboCompactEnv(rob=rob, config=env_config)
    env = DomainRandomizationWrapper(
        base_env,
        enabled=(config.domain_randomization and curriculum_stage == 2),
        ranges=RandomizationRanges(camera_color_balance_enabled=False),
    )
    env = RoboboNM512Wrapper(
        env,
        include_ir=True,
        curriculum_controller=curriculum,
        curriculum_state_path=curriculum_state_path,
        domain_randomization=config.domain_randomization,
    )
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
            return (
                v.__class__.__bases__[0](v)
                if not isinstance(v, (int, float, str, bool, type(None)))
                else v
            )
        except (TypeError, ValueError):
            return v

    cfg = to_native(all_configs["defaults"])
    cfg.update(to_native(all_configs.get("robobo", {})))

    cfg["task"] = "robobo_push"
    cfg["size"] = list(REFERENCE_IMAGE_SIZE)
    cfg["envs"] = 1
    cfg["action_repeat"] = 1
    cfg["time_limit"] = 200
    cfg["grayscale"] = False
    cfg["prefill"] = 5000
    cfg["dyn_hidden"] = REFERENCE_DYN_HIDDEN
    cfg["dyn_deter"] = REFERENCE_DYN_DETER
    cfg["units"] = REFERENCE_MLP_UNITS
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
    cfg["batch_size"] = REFERENCE_BATCH_SIZE
    cfg["batch_length"] = REFERENCE_BATCH_LENGTH
    cfg["train_ratio"] = REFERENCE_TRAIN_RATIO
    cfg["dataset_size"] = 100_000
    cfg["encoder"] = {
        "mlp_keys": "ir",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": REFERENCE_CNN_DEPTH,
        "kernel_size": 4,
        "minres": REFERENCE_CNN_MINRES,
        "mlp_layers": 5,
        "mlp_units": 1024,
        "symlog_inputs": True,
    }
    cfg["decoder"] = {
        "mlp_keys": "ir",
        "cnn_keys": "image",
        "act": "SiLU",
        "norm": True,
        "cnn_depth": REFERENCE_CNN_DEPTH,
        "kernel_size": 4,
        "minres": REFERENCE_CNN_MINRES,
        "mlp_layers": 5,
        "mlp_units": 1024,
        "cnn_sigmoid": False,
        "image_dist": "mse",
        "vector_dist": "symlog_mse",
        "outscale": 1.0,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir", type=str, default="results/nm512d1024_push_96_light"
    )
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=REFERENCE_BATCH_SIZE)
    parser.add_argument("--batch-length", type=int, default=REFERENCE_BATCH_LENGTH)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--domain-randomization",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("COPPELIA_SIM_IP", "127.0.0.1"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("COPPELIA_SIM_PORT", "23000")),
    )
    parser.add_argument("--time-limit", type=int, default=200)
    parser.add_argument(
        "--curriculum",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--curriculum-start-stage", type=int, choices=(0, 1, 2), default=0
    )
    parser.add_argument("--curriculum-success-threshold", type=float, default=0.80)
    parser.add_argument("--curriculum-window", type=int, default=100)
    parser.add_argument("--curriculum-min-stage-steps", type=int, default=20_000)
    parser.add_argument("--curriculum-goal-jitter-radius", type=float, default=0.20)
    cli = parser.parse_args()

    os.environ["COPPELIA_SIM_IP"] = cli.host
    os.environ["COPPELIA_SIM_PORT"] = str(cli.port)

    cfg["steps"] = cli.steps
    cfg["device"] = cli.device
    cfg["seed"] = cli.seed
    cfg["batch_size"] = cli.batch_size
    cfg["batch_length"] = cli.batch_length
    cfg["eval_every"] = cli.eval_every
    cfg["log_every"] = cli.log_every
    cfg["eval_episode_num"] = cli.eval_episodes
    cfg["logdir"] = cli.logdir
    cfg["time_limit"] = cli.time_limit
    cfg["domain_randomization"] = cli.domain_randomization
    cfg["curriculum"] = cli.curriculum
    cfg["curriculum_start_stage"] = cli.curriculum_start_stage
    cfg["curriculum_success_threshold"] = cli.curriculum_success_threshold
    cfg["curriculum_window"] = cli.curriculum_window
    cfg["curriculum_min_stage_steps"] = cli.curriculum_min_stage_steps
    cfg["curriculum_goal_jitter_radius"] = cli.curriculum_goal_jitter_radius

    from types import SimpleNamespace

    config = SimpleNamespace(**cfg)
    config.num_actions = 2
    config.domain_randomization = cli.domain_randomization
    config.curriculum = cli.curriculum
    config.curriculum_goal_jitter_radius = cli.curriculum_goal_jitter_radius
    config.video_pred_log = True

    logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = str(logdir / "train_eps")
    config.evaldir = str(logdir / "eval_eps")
    contract_path = logdir / "reward_contract.json"
    checkpoint_path = logdir / "latest.pt"
    curriculum_state_path = logdir / "curriculum_state.json"
    curriculum_config = PushCurriculumConfig(
        enabled=cli.curriculum,
        start_stage=cli.curriculum_start_stage,
        success_threshold=cli.curriculum_success_threshold,
        window=cli.curriculum_window,
        min_stage_steps=cli.curriculum_min_stage_steps,
        goal_jitter_radius=cli.curriculum_goal_jitter_radius,
    )
    if cli.resume:
        if not curriculum_state_path.exists():
            raise ValueError(
                f"--resume requires curriculum state {curriculum_state_path}"
            )
        curriculum = PushCurriculumController.load(
            curriculum_state_path, curriculum_config
        )
    else:
        curriculum = PushCurriculumController(curriculum_config)
    expected_contract = {
        "reward_contract": PUSH_REWARD_CONTRACT,
        "max_episode_steps": config.time_limit,
        "discount": config.discount,
        "push_time_penalty_per_second": PUSH_TIME_PENALTY_PER_SECOND,
        "push_approach_potential_scale": PUSH_APPROACH_POTENTIAL_SCALE,
        "push_goal_potential_offset": PUSH_GOAL_POTENTIAL_OFFSET,
        "push_goal_potential_scale": PUSH_GOAL_POTENTIAL_SCALE,
        "push_contact_bonus": PUSH_CONTACT_BONUS,
        "push_approach_completion_bonus": PUSH_APPROACH_COMPLETION_BONUS,
        "push_goal_completion_bonus": PUSH_GOAL_COMPLETION_BONUS,
        "action_smoothing": False,
        "pre_action_safety": False,
        "max_action_delta": 2.0,
        "image_size": list(REFERENCE_IMAGE_SIZE),
        "dyn_hidden": REFERENCE_DYN_HIDDEN,
        "dyn_deter": REFERENCE_DYN_DETER,
        "units": REFERENCE_MLP_UNITS,
        "cnn_depth": REFERENCE_CNN_DEPTH,
        "cnn_minres": REFERENCE_CNN_MINRES,
        "batch_size": config.batch_size,
        "batch_length": config.batch_length,
        "train_ratio": config.train_ratio,
        "curriculum": cli.curriculum,
        "curriculum_success_threshold": cli.curriculum_success_threshold,
        "curriculum_window": cli.curriculum_window,
        "curriculum_min_stage_steps": cli.curriculum_min_stage_steps,
        "curriculum_goal_jitter_radius": cli.curriculum_goal_jitter_radius,
    }
    checkpoint = None
    if cli.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"--resume requires checkpoint {checkpoint_path}")
        if not contract_path.exists():
            raise ValueError(f"--resume requires reward contract {contract_path}")
        contract = json.loads(contract_path.read_text())
        if contract != expected_contract:
            raise ValueError(
                f"incompatible reference replay reward contract: {contract}; "
                "use a fresh --logdir"
            )
    elif logdir.exists() and any(logdir.iterdir()):
        raise ValueError(
            f"refusing to start a fresh run in non-empty logdir {logdir}; "
            "use a new --logdir or pass --resume"
        )

    os.makedirs(config.traindir, exist_ok=True)
    os.makedirs(config.evaldir, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(json.dumps(expected_contract, indent=2) + "\n")
    curriculum.save(curriculum_state_path)

    tools.set_seed_everywhere(config.seed)

    step = dreamer.count_steps(pathlib.Path(config.traindir))
    if cli.resume:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        validate_checkpoint(
            checkpoint,
            reward_contract=expected_contract,
            replay_step=step,
            max_replay_lag=config.time_limit - 1,
            max_replay_lead=config.eval_every,
        )
        replay_lead = step - int(checkpoint["training_step"])
        if replay_lead > 0:
            print(
                f"Replay is {replay_lead:,} transitions ahead of the model "
                "checkpoint due to a mid-block shutdown. Keeping those "
                "off-policy transitions and resuming model state from "
                f"step {checkpoint['training_step']:,}."
            )
        step = int(checkpoint["training_step"])

    if not cli.no_wandb:
        wandb_init_kwargs = dict(
            project="learning-machines",
            config={
                k: v
                for k, v in cfg.items()
                if isinstance(v, (int, float, str, bool, list, dict))
            },
            name=logdir.name,
        )
        if cli.resume and checkpoint.get("wandb_run_id"):
            wandb_init_kwargs["id"] = checkpoint["wandb_run_id"]
            wandb_init_kwargs["resume"] = "must"
        else:
            wandb_init_kwargs["resume"] = "never"
        wandb.init(**wandb_init_kwargs)
    logger = WandbLogger(
        config.action_repeat * step,
        enabled=not cli.no_wandb,
    )
    atexit.register(logger.close)

    print(f"Starting NM512 DreamerV3 push training on Robobo")
    print(f"  Log dir: {logdir}")
    print(f"  Steps: {config.steps}")
    print(f"  Device: {config.device}")
    print(f"  Batch: {config.batch_size} x {config.batch_length}")
    print(f"  Train ratio: {config.train_ratio}")
    print(f"  Image size: {config.size[0]}x{config.size[1]}")
    print(
        "  Network: "
        f"RSSM hidden={config.dyn_hidden}, "
        f"deter={config.dyn_deter}, "
        f"MLP units={config.units}, "
        f"CNN depth={config.encoder['cnn_depth']}"
    )
    print(f"  Eval episodes: {config.eval_episode_num}")
    print(f"  Domain randomization: {config.domain_randomization}")
    print(
        f"  Curriculum: stage={curriculum.stage} "
        f"({curriculum.stage_name}), enabled={curriculum.config.enabled}"
    )
    print(f"  Resume: {cli.resume}")
    print("  Actions: direct (smoothing and pre-action safety disabled)")
    print("  Reward: phased dense approach/contact/push shaping")

    train_eps = tools.load_episodes(
        pathlib.Path(config.traindir), limit=config.dataset_size
    )
    eval_eps = tools.load_episodes(pathlib.Path(config.evaldir), limit=1)

    # Training and evaluation run sequentially, so they must share one
    # simulator connection. Constructing separate environments here connects
    # two SimulationRobobo clients to the same CoppeliaSim instance and lets
    # either client reset or stop the other's simulation.
    env = Damy(make_robobo_env(config, curriculum, curriculum_state_path))
    env.set_push_curriculum_stage(curriculum.stage, config.domain_randomization)

    acts = env.action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    prefill = max(
        0, config.prefill - dreamer.count_steps(pathlib.Path(config.traindir))
    )
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
        logprob = random_actor.log_prob(action)
        return {"action": action, "logprob": logprob}, None

    pbar = tqdm(
        total=config.steps,
        initial=step,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
    )
    ep_stats = EpisodeStats(window=100)

    def checkpoint_payload(current_agent):
        return {
            "checkpoint_version": REFERENCE_CHECKPOINT_VERSION,
            "agent_state_dict": current_agent.state_dict(),
            "optims_state_dict": collect_optimizer_state_dicts(current_agent),
            "training_step": current_agent._step,
            "reward_contract": expected_contract,
            "wandb_run_id": wandb.run.id if not cli.no_wandb else None,
        }

    if prefill > 0:
        pbar.set_description("Prefilling buffer")
        state = tools.simulate(
            random_agent,
            [env],
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        pbar.update(prefill)
        tqdm.write(f"Prefill done. Logger step: {logger.step}")
    else:
        state = None

    pbar.set_description("Training")

    train_dataset = dreamer.make_dataset(train_eps, config)
    eval_dataset = dreamer.make_dataset(eval_eps, config)

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            agent = dreamer.Dreamer(
                env.observation_space,
                env.action_space,
                config,
                logger,
                train_dataset,
            ).to(config.device)
    agent.requires_grad_(requires_grad=False)

    def on_curriculum_promotion(promotion):
        tqdm.write(
            f"[Curriculum] Promoted to stage {curriculum.stage} "
            f"({curriculum.stage_name}) at transition "
            f"{curriculum.transition_count:,}"
        )
        payload = checkpoint_payload(agent)
        save_checkpoint_atomic(payload, checkpoint_path)
        save_checkpoint_atomic(
            payload,
            logdir / f"promotion_stage_{curriculum.stage}.pt",
        )

    env.set_curriculum_promotion_callback(on_curriculum_promotion)
    if checkpoint is not None:
        agent.load_state_dict(checkpoint["agent_state_dict"])
        load_optimizer_state_dicts(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False
        tqdm.write(f"Resumed checkpoint at step {checkpoint['training_step']}.")
        checkpoint = None

    tqdm.write("Start training.")

    class StepTracker:
        def __init__(self, agent, pbar, obs_space, act_space):
            self._agent = agent
            self._pbar = pbar
            self._last_step = agent._step
            self.observation_space = obs_space
            self.action_space = act_space

        def __call__(self, obs, reset, state=None, training=True):
            result = self._agent(obs, reset, state, training)
            new_steps = self._agent._step - self._last_step
            if new_steps > 0:
                self._pbar.update(new_steps)
                self._last_step = self._agent._step
            return result

        def __getattr__(self, name):
            return getattr(self._agent, name)

    tracked_agent = StepTracker(agent, pbar, env.observation_space, env.action_space)

    while agent._step < config.steps + config.eval_every:
        logger.write()
        tqdm.write(
            f"Step {agent._step}/{config.steps} | "
            f"Evaluating ({config.eval_episode_num} episodes)..."
        )
        eval_policy = functools.partial(tracked_agent, training=False)
        env.set_curriculum_tracking_enabled(False)
        try:
            tools.simulate(
                eval_policy,
                [env],
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
        finally:
            env.set_curriculum_tracking_enabled(True)
        # Evaluation has reset and advanced the shared physical environment.
        # Do not reuse the previous training observation or latent policy state.
        state = None

        if config.video_pred_log and eval_dataset is not None:
            from dreamer import to_np

            try:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
                logger.write(step=logger.step)
            except StopIteration:
                pass

        prev_step = agent._step
        tqdm.write(
            f"Step {agent._step}/{config.steps} | "
            f"Training {config.eval_every} steps..."
        )
        state = tools.simulate(
            tracked_agent,
            [env],
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )

        new_steps = agent._step - prev_step

        train_eps_this_block = dict(list(train_eps.items())[-50:])
        block_rewards = []
        block_successes = []
        for ep_data in train_eps_this_block.values():
            ep_len = len(ep_data["reward"]) - 1
            if ep_len < 1:
                continue
            ep_reward = float(np.array(ep_data["reward"]).sum())
            block_rewards.append(ep_reward)
            block_successes.append(
                float(np.asarray(ep_data["is_terminal"], dtype=bool).any())
            )
        if block_rewards:
            recent_reward = block_rewards[-1]
            recent_success = block_successes[-1]
            ep_stats.add(recent_reward, recent_success)

        pbar.set_postfix(
            {
                "reward": f"{ep_stats.avg_reward:.0f}",
                "success": f"{ep_stats.success_rate:.2f}",
                "eps": ep_stats.count,
            }
        )

        items_to_save = checkpoint_payload(agent)
        save_checkpoint_atomic(items_to_save, checkpoint_path)
        curriculum.save(curriculum_state_path)
        tqdm.write(f"Step {agent._step}/{config.steps} | Saved checkpoint.")

    pbar.close()

    try:
        env.close()
    except Exception:
        pass

    logger.close()
    atexit.unregister(logger.close)
    tqdm.write("Training complete.")


if __name__ == "__main__":
    main()
