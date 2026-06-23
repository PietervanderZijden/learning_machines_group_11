import os
import sys
import argparse
import datetime
import colorama
import numpy as np
import warnings
from collections import deque
from tqdm import tqdm
from typing import Union
from pathlib import Path

import torch
from einops import rearrange

# Voeg je catkin_ws toe aan het pad (pas dit aan als je mappenstructuur anders is)
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "learning_machines" / "src"))
sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))

from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.world_models import WorldModel, GITWorldModel

# --- Jouw specifieke Robobo imports ---
from learning_machines.coppelia_startup import check_coppelia_service
from learning_machines.rl_robobo_compact_env import RoboboCompactEnvConfig
from learning_machines.domain_randomization import RandomizationRanges
from learning_machines.transfer import CalibrationProfile


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: Union[WorldModel, GITWorldModel], batch_size, demonstration_batch_size, batch_length, logger, wandb_run=None, step=0):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    world_model.update(obs, action, reward, termination, logger=logger)
    
    # Als GIT-STORM interne logging gebruikt, kunnen we specifieke losses eventueel later uithalen.
    # Voor nu stuurt hun model het al naar TensorBoard (logger).

@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: Union[WorldModel, GITWorldModel], agent: agents.ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger):
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat


def joint_train_world_model_agent(env_config, ranges, args, conf,
                                  replay_buffer: ReplayBuffer,
                                  world_model: Union[WorldModel, GITWorldModel], agent: agents.ActorCriticAgent,
                                  logger, wandb_run=None):
    os.makedirs(f"ckpt/{conf.BasicSettings.n}", exist_ok=True)

    # 1. Start de Robobo Vector Omgeving!
    vec_env = env_wrapper.build_robobo_vec_env(env_config, num_envs=conf.JointTrainAgent.NumEnvs, ranges=ranges)
    print("Current env: " + colorama.Fore.YELLOW + "Robobo CoppeliaSim" + colorama.Style.RESET_ALL)

    sum_reward = np.zeros(conf.JointTrainAgent.NumEnvs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # Curriculum Helper
    def apply_curriculum(step: int) -> tuple[int, bool]:
        if not args.curriculum:
            return 7, True
        if step < args.curriculum_one_food_steps:
            active_food = 1
        elif step < args.curriculum_three_food_steps:
            active_food = 3
        else:
            active_food = 7
        env_config.active_food_count = active_food
        randomization_enabled = step >= args.randomization_start_steps
        # Pas de interne wrapper aan (DomainRandomization zit als base in onze wrapper)
        vec_env.envs[0].env.enabled = randomization_enabled
        return active_food, randomization_enabled

    max_steps = conf.JointTrainAgent.SampleMaxSteps
    num_envs = conf.JointTrainAgent.NumEnvs

    # Training Loop
    for total_steps in tqdm(range(max_steps // num_envs), disable=conf.BasicSettings.silent):
        
        # --- Curriculum check ---
        active_food, is_randomized = apply_curriculum(total_steps)

        # --- Sample Part ---
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )

            context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W") / 255)
            context_action.append(action)
        else:
            action = vec_env.action_space.sample()

        obs, reward, done, truncated, info = vec_env.step(action)
        
        # Sla op in replay buffer
        # Let op: de GIT-STORM buffer verwacht termination als boolean of float.
        is_terminal = np.logical_or(done, truncated)
        replay_buffer.append(current_obs, action, reward, is_terminal)

        if is_terminal.any():
            for i in range(num_envs):
                if is_terminal[i]:
                    logger.log(f"episode/reward", sum_reward[i])
                    logger.log("replay_buffer/length", len(replay_buffer))
                    
                    if wandb_run is not None:
                        wandb_run.log({
                            "episode/return": float(sum_reward[i]),
                            "curriculum/active_food_count": active_food,
                            "curriculum/randomization_enabled": float(is_randomized),
                            "global_step": total_steps
                        })
                    sum_reward[i] = 0

        sum_reward += reward
        current_obs = obs
        current_info = info

        # --- Train World Model ---
        if replay_buffer.ready() and total_steps % (conf.JointTrainAgent.TrainDynamicsEverySteps // num_envs) == 0:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=conf.JointTrainAgent.BatchSize * num_envs,
                demonstration_batch_size=0,
                batch_length=conf.JointTrainAgent.BatchLength,
                logger=logger,
                wandb_run=wandb_run,
                step=total_steps
            )

        # --- Train Agent ---
        if replay_buffer.ready() and total_steps % (conf.JointTrainAgent.TrainAgentEverySteps // num_envs) == 0 and total_steps * num_envs >= 0:
            log_video = (total_steps % (conf.JointTrainAgent.SaveEverySteps // num_envs) == 0)

            imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
                imagine_demonstration_batch_size=0,
                imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
                imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
                log_video=log_video,
                logger=logger
            )

            # De update functie geeft een zooi aan metrics terug in de GIT-STORM code.
            # We vangen ze op (aangepast naar hun return statement) om naar wandb te sturen.
            metrics = agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger
            )
            
            # W&B Logging voor de agent losses (als de update functie dit teruggeeft)
            if wandb_run is not None and isinstance(metrics, tuple) and len(metrics) >= 4:
                wandb_run.log({
                    "train/actor_loss": float(metrics[1]),
                    "train/critic_loss": float(metrics[2]),
                    "train/entropy": float(metrics[3]),
                    "global_step": total_steps
                })

        # --- Save Checkpoints ---
        if total_steps % (conf.JointTrainAgent.SaveEverySteps // num_envs) == 0 and total_steps > 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), f"ckpt/{conf.BasicSettings.n}/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), f"ckpt/{conf.BasicSettings.n}/agent_{total_steps}.pth")


def build_world_model(conf, action_dim, device: torch.device):
    if conf.JointTrainAgent.ModelType == "STORM":
        return WorldModel(
            in_channels=conf.Models.WorldModel.InChannels,
            action_dim=action_dim,
            transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
            transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
            transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
            transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
        ).cuda()
    elif conf.JointTrainAgent.ModelType == "GITSTORM":
        return GITWorldModel(
            in_channels=conf.Models.WorldModel.InChannels,
            action_dim=action_dim,
            transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
            transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
            transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
            transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
            device=device,
            conf=conf
        ).cuda()


def build_agent(conf, action_dim):
    return agents.ActorCriticAgent(
        feat_dim=32*32+conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
    ).cuda()


def main():
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser(description="GIT-STORM for Robobo")
    parser.add_argument("--config-path", type=str, default="config_files/robobo_gitstorm.yaml")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--host", default=os.environ.get("COPPELIA_SIM_IP", "127.0.0.1"))
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--calibration", default="config/calibration/simulation.json")
    parser.add_argument("--hardware-calibration", default=None)
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--curriculum-one-food-steps", type=int, default=100_000)
    parser.add_argument("--curriculum-three-food-steps", type=int, default=250_000)
    parser.add_argument("--randomization-start-steps", type=int, default=300_000)
    args = parser.parse_args()

    # Pre-flight check voor CoppeliaSim
    os.environ["COPPELIA_SIM_PORT"] = str(args.port)
    os.environ["COPPELIA_SIM_IP"] = args.host
    print(f"Connecting to simulator at {args.host}:{args.port}...")
    check_coppelia_service(args.host, args.port)

    # Laad configuratie met de YACS parser van GIT-STORM
    conf = load_config(args.config_path)
    seed_np_torch(seed=conf.BasicSettings.Seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = Logger(path=f"runs/{conf.BasicSettings.n}")

    # Initialiseer W&B
    wandb_run = None
    if not args.no_wandb:
        import wandb
        run_name = args.wandb_run_name or f"gitstorm-{datetime.datetime.now().strftime('%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project="learning-machines",
            entity="Learningmachine",
            name=run_name,
            config=vars(args),
            sync_tensorboard=True
        )

    # Configuratie voor Robobo
    env_config = RoboboCompactEnvConfig(
        max_episode_steps=150,
        return_image=True, # We pakken de image
        image_obs_size=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize),
        calibration_path=args.calibration,
        active_food_count=1 if args.curriculum else None,
    )
    
    ranges = None
    if args.hardware_calibration:
        ranges = RandomizationRanges.from_calibration_profiles(
            CalibrationProfile.load(args.calibration),
            CalibrationProfile.load(args.hardware_calibration),
        )

    action_dim = 25  # Omdat we 25 discrete knoppen hebben gemaakt!

    world_model = build_world_model(conf, action_dim, device)
    world_model = torch.compile(world_model, mode="max-autotune")
    agent = build_agent(conf, action_dim)
    agent = torch.compile(agent, mode="max-autotune")

    # Buffer (Verwacht (H, W, C) = (64, 64, 3))
    replay_buffer = ReplayBuffer(
        obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
        num_envs=conf.JointTrainAgent.NumEnvs,
        max_length=conf.JointTrainAgent.BufferMaxLength,
        warmup_length=conf.JointTrainAgent.BufferWarmUp,
        store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
        device=device
    )

    joint_train_world_model_agent(
        env_config=env_config,
        ranges=ranges,
        args=args,
        conf=conf,
        replay_buffer=replay_buffer,
        world_model=world_model,
        agent=agent,
        logger=logger,
        wandb_run=wandb_run
    )

if __name__ == "__main__":
    main()