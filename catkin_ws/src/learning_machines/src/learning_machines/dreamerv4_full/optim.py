'Initialization and optimization utilities for DreamerV4.'
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .transformer import GroupedQueryAttention, RMSNorm, SwiGLU


def initialize_dreamerv4(module: nn.Module, transformer_layers: int) -> None:
    'Initialize transformer paths conservatively for stable deep training.'
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.trunc_normal_(child.weight, std=0.02)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            nn.init.trunc_normal_(child.weight, std=0.02)
        elif isinstance(child, RMSNorm):
            nn.init.ones_(child.weight)

    residual_scale = 1.0 / math.sqrt(max(1, 2 * transformer_layers))
    for child in module.modules():
        if isinstance(child, GroupedQueryAttention):
            child.out_proj.weight.data.mul_(residual_scale)
        elif isinstance(child, SwiGLU):
            child.out.weight.data.mul_(residual_scale)


def adamw_parameter_groups(
    named_parameters,
    weight_decay: float,
) -> list[dict]:
    'Decay matrix weights only; exclude norms, biases, embeddings and tokens.'
    decay, no_decay = [], []
    seen: set[int] = set()
    for name, parameter in named_parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if parameter.ndim >= 2 and "embed" not in name and "token" not in name:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def warmup_cosine_lambda(
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
):
    total_steps = max(1, int(total_steps))
    warmup_steps = min(
        total_steps - 1,
        max(1, round(total_steps * warmup_fraction)),
    )

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(
            1, total_steps - warmup_steps - 1
        )
        cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        return minimum_ratio + (1 - minimum_ratio) * cosine

    return schedule
