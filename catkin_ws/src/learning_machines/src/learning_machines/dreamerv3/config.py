from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DreamerV3Config:
    model_size: str = "12m"
    obs_dim: int = 12
    action_dim: int = 2
    action_type: str = "continuous"

    # RSSM
    deterministic_size: int = 1024
    stochastic_classes: int = 16
    stochastic_bins: int = 16
    hidden_size: int = 256
    block_gru_blocks: int = 8

    # Networks
    embed_size: int = 256
    mlp_hidden: int = 256
    mlp_layers: int = 3
    cnn_base_channels: int = 16

    # Actor-Critic
    actor_hidden: int = 256
    actor_layers: int = 3
    critic_hidden: int = 256
    critic_layers: int = 3
    horizon: int = 15
    imagination_horizon: int = 15

    # Training
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
    entropy_weight: float = 3e-4  # Paper value: η = 3×10⁻⁴
    actor_mean_limit: float = 2.5
    actor_std_min: float = 0.1
    actor_std_max: float = 1.0
    reward_weight: float = 1.0
    recon_weight: float = 1.0
    food_recon_weight: float = 0.0
    kl_rep_weight: float = 0.1
    continue_weight: float = 1.0
    replay_value_weight: float = 0.3
    slow_value_regularization: float = 1.0
    target_tau: float = 0.02
    advantage_clip: float = 0.0
    # 0 uses every replay state. Eight is a memory-conscious Robobo default,
    # with food-reward states explicitly included by the start selector.
    imagination_starts: int = 0

    # Buffer
    buffer_capacity: int = 100_000
    sequence_length: int = 64
    batch_size: int = 16
    reward_event_fraction: float = 0.0
    reward_event_threshold: float = 1.0
    online_queue_capacity: int = 4096
    store_latent_states: bool = True

    # Training loop
    prefill_steps: int = 5000
    train_ratio: float = 512.0
    update_every: int = 1
    grad_clip: float = 100.0

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


MODEL_SIZE_PRESETS: dict[str, dict[str, int | str]] = {
    "12m": {
        "model_size": "12m",
        "hidden_size": 256,
        "deterministic_size": 1024,
        "embed_size": 256,
        "mlp_hidden": 256,
        "actor_hidden": 256,
        "critic_hidden": 256,
        "cnn_base_channels": 16,
        "stochastic_classes": 16,
        "stochastic_bins": 16,
    },
    "25m": {
        "model_size": "25m",
        "hidden_size": 384,
        "deterministic_size": 1536,
        "embed_size": 384,
        "mlp_hidden": 384,
        "actor_hidden": 384,
        "critic_hidden": 384,
        "cnn_base_channels": 24,
        "stochastic_classes": 24,
        "stochastic_bins": 16,
    },
    "50m": {
        "model_size": "50m",
        "hidden_size": 512,
        "deterministic_size": 2048,
        "embed_size": 512,
        "mlp_hidden": 512,
        "actor_hidden": 512,
        "critic_hidden": 512,
        "cnn_base_channels": 32,
        "stochastic_classes": 32,
        "stochastic_bins": 16,
    },
}


def apply_model_size_preset(cfg: DreamerV3Config, size: str) -> DreamerV3Config:
    if size not in MODEL_SIZE_PRESETS:
        raise ValueError(f"unknown DreamerV3 model size preset: {size}")
    for key, value in MODEL_SIZE_PRESETS[size].items():
        setattr(cfg, key, value)
    return cfg
