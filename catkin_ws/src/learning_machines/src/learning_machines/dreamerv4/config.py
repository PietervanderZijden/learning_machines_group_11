'DreamerV4 configuration.'
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DreamerV4Config:
    obs_dim: int = 12
    act_dim: int = 2
    reward_dim: int = 1


    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    ff_dim: int = 1024
    dropout: float = 0.1


    context_length: int = 64
    imagination_horizon: int = 16


    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    batch_size: int = 32


    gamma: float = 0.994009
    lam: float = 0.95
    entropy_coef: float = 3e-4


    buffer_size: int = 100_000
    update_every: int = 10


    clip_reward: bool = True
    reward_scale: float = 10.0


    target_update_rate: float = 0.01


    k_inference: int = 4
    k_max: int = 4


    pmpo_alpha: float = 0.5
    pmpo_prior_beta: float = 0.3


    mtp_length: int = 8


    image_size: int = 64
    latent_dim: int = 256
    ir_dim: int = 0
