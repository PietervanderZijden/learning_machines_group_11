from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DreamerV3Config:
    obs_dim: int = 12
    action_dim: int = 2
    action_type: str = "continuous"

    # RSSM
    deterministic_size: int = 512
    stochastic_classes: int = 32
    stochastic_bins: int = 32
    hidden_size: int = 512

    # Networks
    embed_size: int = 512
    mlp_hidden: int = 512
    mlp_layers: int = 3

    # Actor-Critic
    actor_hidden: int = 512
    actor_layers: int = 3
    critic_hidden: int = 512
    critic_layers: int = 3
    horizon: int = 16
    imagination_horizon: int = 16

    # Training
    world_lr: float = 3e-4
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    free_nats: float = 1.0
    kl_balance: float = 0.8
    gamma: float = 0.994009
    lam: float = 0.95
    entropy_weight: float = 3e-4  # Paper value: η = 3×10⁻⁴
    reward_weight: float = 1.0
    recon_weight: float = 1.0
    kl_rep_weight: float = 0.1
    continue_weight: float = 1.0
    target_tau: float = 0.01

    # Buffer
    buffer_capacity: int = 1_000_000
    sequence_length: int = 50
    batch_size: int = 32
    reward_event_fraction: float = 0.25
    reward_event_threshold: float = 1.0

    # Training loop
    prefill_steps: int = 5000
    train_ratio: int = 512
    update_every: int = 1
    grad_clip: float = 100.0  # Tightened from 1000.0 for stability

    # Environment
    max_episode_steps: int = 150
    num_envs: int = 1
    use_images: bool = False
    use_multimodal: bool = True  # Use images + IR data
    image_size: int = 64
    ir_dim: int = 8  # Number of IR sensors

    # Logging
    log_every: int = 100
    checkpoint_every: int = 10000
