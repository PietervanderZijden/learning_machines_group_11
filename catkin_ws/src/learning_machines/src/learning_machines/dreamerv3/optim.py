"""DreamerV3 optimization utilities.

The reference implementation applies adaptive gradient clipping (AGC),
RMS-normalized gradients, and momentum. The latter two operations form the
LaProp optimizer update.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim import Optimizer


def _unitwise_norm(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim <= 1:
        return torch.linalg.vector_norm(tensor)
    dims = tuple(range(1, tensor.ndim))
    return torch.linalg.vector_norm(tensor, dim=dims, keepdim=True)


@torch.no_grad()
def adaptive_clip_grad_(
    parameters: Iterable[torch.nn.Parameter],
    clipping: float = 0.3,
    eps: float = 1e-3,
) -> float:
    """Apply unit-wise adaptive gradient clipping and return the raw norm."""
    parameters = [
        parameter for parameter in parameters if parameter.grad is not None
    ]
    if not parameters:
        return 0.0
    finite = torch.stack([
        torch.isfinite(parameter.grad).all() for parameter in parameters
    ]).all()
    if not bool(finite):
        raise FloatingPointError("non-finite DreamerV3 gradient")

    squared_norms = []
    for parameter in parameters:
        gradient = parameter.grad
        squared_norms.append(gradient.detach().double().square().sum())

        parameter_norm = _unitwise_norm(parameter.detach()).clamp_min(eps)
        gradient_norm = _unitwise_norm(gradient.detach()).clamp_min(1e-6)
        maximum_norm = parameter_norm * clipping
        scale = torch.clamp(maximum_norm / gradient_norm, max=1.0)
        gradient.mul_(scale)
    return math.sqrt(float(torch.stack(squared_norms).sum().cpu()))


class LaProp(Optimizer):
    """PyTorch implementation of LaProp with optional linear warmup."""

    def __init__(
        self,
        params,
        lr: float = 4e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-20,
        weight_decay: float = 0.0,
        warmup_steps: int = 1000,
    ):
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if eps < 0:
            raise ValueError("epsilon must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("optimizer betas must be in [0, 1)")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=int(warmup_steps),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("LaProp does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if group["weight_decay"]:
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])

                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2
                )
                rms = (
                    exp_avg_sq / (1.0 - beta2**step)
                ).sqrt().add_(group["eps"])
                normalized_gradient = gradient / rms
                exp_avg.mul_(beta1).add_(
                    normalized_gradient, alpha=1.0 - beta1
                )
                update = exp_avg / (1.0 - beta1**step)

                warmup_steps = group["warmup_steps"]
                warmup = (
                    min(1.0, step / warmup_steps)
                    if warmup_steps > 0
                    else 1.0
                )
                parameter.add_(update, alpha=-group["lr"] * warmup)
        return loss
