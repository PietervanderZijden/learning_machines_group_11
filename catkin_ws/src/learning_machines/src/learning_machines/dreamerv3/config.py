from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DreamerV3Config:
    obs_dim: int = 12
    action_dim: int = 2
    action_type: str = "continuous"


    deterministic_size: int = 512
    stochastic_classes: int = 32
    stochastic_bins: int = 32
    hidden_size: int = 512


    embed_size: int = 512
    mlp_hidden: int = 512
    mlp_layers: int = 3


    actor_hidden: int = 512
    actor_layers: int = 3
    critic_hidden: int = 512
    critic_layers: int = 3
    horizon: int = 15
    imagination_horizon: int = 15


    world_lr: float = 4e-5
    actor_lr: float = 4e-5
    critic_lr: float = 4e-5
    optimizer: str = "laprop"
    optimizer_eps: float = 1e-20
    optimizer_warmup: int = 1000
    agc: float = 0.3
    free_nats: float = 1.0
    gamma: float = 0.997
    lam: float = 0.95
    entropy_weight: float = 3e-4
    actor_mean_limit: float = 2.5
    actor_std_min: float = 0.1
    actor_std_max: float = 1.0
    reward_weight: float = 1.0
    recon_weight: float = 1.0
    food_recon_weight: float = 2.0
    kl_rep_weight: float = 0.1
    continue_weight: float = 1.0
    replay_value_weight: float = 0.3
    slow_value_regularization: float = 1.0
    target_tau: float = 0.02
    advantage_clip: float = 0.0


    imagination_starts: int = 8


    buffer_capacity: int = 100_000
    sequence_length: int = 50
    batch_size: int = 32
    reward_event_fraction: float = 0.25
    reward_event_threshold: float = 1.0


    prefill_steps: int = 5000
    train_ratio: float = 512.0
    update_every: int = 1
    grad_clip: float = 100.0


    max_episode_steps: int = 150
    num_envs: int = 1
    use_images: bool = False
    use_multimodal: bool = True
    image_size: int = 64
    ir_dim: int = 8


    log_every: int = 100
    checkpoint_every: int = 10000
