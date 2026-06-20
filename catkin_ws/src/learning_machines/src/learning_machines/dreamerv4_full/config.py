"""Configuration for the paper-aligned DreamerV4 implementation."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DreamerV4FullConfig:
    image_size: int = 64
    patch_size: int = 8
    image_channels: int = 3
    ir_dim: int = 8
    action_dim: int = 2
    num_tasks: int = 1

    model_dim: int = 128
    latent_tokens: int = 16
    latent_channels: int = 16
    register_tokens: int = 4
    tokenizer_layers: int = 4
    dynamics_layers: int = 8
    heads: int = 8
    kv_heads: int = 2
    ff_multiplier: float = 4.0
    temporal_every: int = 4
    attention_softcap: float = 50.0
    dropout: float = 0.0
    context_length: int = 32

    shortcut_steps: int = 4
    context_signal: float = 0.1
    mtp_length: int = 8
    imagination_horizon: int = 16

    gamma: float = 0.997
    lambda_: float = 0.95
    pmpo_alpha: float = 0.5
    prior_kl_weight: float = 0.3
    entropy_weight: float = 1e-4
    normalize_advantages: bool = True

    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    agent_grad_clip: float | None = None
    warmup_fraction: float = 0.05
    minimum_lr_ratio: float = 0.1
    mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"
    rms_decay: float = 0.99
    rms_floor_ratio: float = 0.1
    lpips_weight: float = 0.2
    mask_ratio: float = 0.75

    def __post_init__(self):
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        if self.heads % self.kv_heads:
            raise ValueError("heads must be divisible by kv_heads")
        if self.shortcut_steps <= 0 or (
            self.shortcut_steps & (self.shortcut_steps - 1)
        ):
            raise ValueError("shortcut_steps must be a positive power of two")
        if self.temporal_every <= 0:
            raise ValueError("temporal_every must be positive")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not 0 <= self.minimum_lr_ratio <= 1:
            raise ValueError("minimum_lr_ratio must be in [0, 1]")
        if self.mixed_precision_dtype not in {"bfloat16", "float16"}:
            raise ValueError(
                "mixed_precision_dtype must be 'bfloat16' or 'float16'"
            )
        if not 0.0 < self.mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in (0, 1)")

    @property
    def patches_per_frame(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

    @property
    def latent_dim(self) -> int:
        return self.latent_tokens * self.latent_channels

    def to_dict(self) -> dict:
        return asdict(self)
