"""DreamerV4 configuration."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class DreamerV4Config:
    obs_dim: int = 12
    act_dim: int = 2
    reward_dim: int = 1

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.0

    context_length: int = 64
    dream_horizon: int = 15
    imagination_horizon: int = 15

    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 0.5

    horizon: int = 64
    batch_size: int = 32

    gamma: float = 0.997
    lam: float = 0.95
    entropy_coef: float = 1e-3
    free_nats: float = 1.0

    buffer_size: int = 100_000
    min_episode_length: int = 10
    update_every: int = 100
    prefill_episodes: int = 20

    clip_reward: bool = True
    reward_scale: float = 10.0
    target_update_rate: float = 0.01
