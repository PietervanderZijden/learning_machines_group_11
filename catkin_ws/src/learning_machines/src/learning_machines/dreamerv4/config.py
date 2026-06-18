"""DreamerV4 configuration.

Based on: "Training Agents Inside of Scalable World Models" (Hafner et al., 2025)
arXiv:2509.24527v1
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DreamerV4Config:
    obs_dim: int = 12
    act_dim: int = 2
    reward_dim: int = 1

    # Transformer architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1

    # Sequence settings
    context_length: int = 64
    imagination_horizon: int = 16

    # Training
    lr: float = 1e-4
    weight_decay: float = 0.01  # Paper uses 0.01
    grad_clip: float = 1.0

    batch_size: int = 32

    # RL settings
    gamma: float = 0.994009
    lam: float = 0.95
    entropy_coef: float = 3e-4  # Paper uses 3e-4

    # Replay buffer
    buffer_size: int = 100_000
    update_every: int = 10

    # Reward processing
    clip_reward: bool = True
    reward_scale: float = 10.0

    # Target network
    target_update_rate: float = 0.01

    # Shortcut forcing settings (DreamerV4 paper)
    k_inference: int = 4  # Number of sampling steps at inference (K=4)
    k_max: int = 4  # Maximum number of sampling steps for training

    # PMPO settings
    pmpo_alpha: float = 0.5  # Balance between positive/negative advantages
    pmpo_prior_beta: float = 0.3  # Behavioral prior weight

    # Multi-token prediction (MTP)
    mtp_length: int = 8  # Number of future steps to predict (L=8 in paper)

    # Image settings
    image_size: int = 64
    latent_dim: int = 256
    ir_dim: int = 0  # IR sensor dimension (0 for no IR)
