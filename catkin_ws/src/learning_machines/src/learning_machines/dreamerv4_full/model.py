'Interactive dynamics and agent heads for the full DreamerV4 implementation.'
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from learning_machines.distributional import _NUM_BINS

from .config import DreamerV4FullConfig
from .transformer import RMSNorm, SpaceTimeTransformer, SwiGLU


def sample_shortcut_schedule(
    shape: tuple[int, ...],
    device: torch.device,
    k_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    levels = int(math.log2(k_max)) + 1
    indices = torch.randint(levels, shape, device=device)
    step = torch.pow(2.0, -indices.float())
    grid_size = step.reciprocal().long()
    signal = (
        torch.rand(shape, device=device) * grid_size.float()
    ).floor() * step
    return signal, step


class MLPHead(nn.Module):
    def __init__(self, dim: int, output_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.hidden = SwiGLU(dim, dim * 2)
        self.output = nn.Linear(dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x + self.hidden(self.norm(x)))


class TanhGaussianMTPPolicy(nn.Module):
    "Continuous-action adaptation of the paper's MTP policy head."

    def __init__(self, cfg: DreamerV4FullConfig):
        super().__init__()
        self.action_dim = cfg.action_dim
        self.mtp_length = cfg.mtp_length
        self.heads = nn.ModuleList([
            MLPHead(cfg.model_dim, 2 * cfg.action_dim)
            for _ in range(cfg.mtp_length)
        ])

    def parameters_for(
        self, hidden: torch.Tensor, distance: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.heads[min(distance, self.mtp_length - 1)](hidden)
        mean, log_std = output.chunk(2, -1)
        return mean, log_std.clamp(-5.0, 2.0)

    def sample(
        self, hidden: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.parameters_for(hidden)
        distribution = Normal(mean, log_std.exp())
        raw = mean if deterministic else distribution.sample()
        action = raw.tanh()
        log_prob = distribution.log_prob(raw).sum(-1)
        log_prob -= torch.log1p(-action.square() + 1e-6).sum(-1)
        return action, log_prob

    def log_prob(
        self,
        hidden: torch.Tensor,
        action: torch.Tensor,
        distance: int = 0,
    ) -> torch.Tensor:
        mean, log_std = self.parameters_for(hidden, distance)
        action = action.clamp(-0.999, 0.999)
        raw = torch.atanh(action)
        distribution = Normal(mean, log_std.exp())
        result = distribution.log_prob(raw).sum(-1)
        return result - torch.log1p(-action.square() + 1e-6).sum(-1)

    def reverse_kl(
        self, hidden: torch.Tensor, prior: "TanhGaussianMTPPolicy"
    ) -> torch.Tensor:
        mean, log_std = self.parameters_for(hidden)
        with torch.no_grad():
            prior_mean, prior_log_std = prior.parameters_for(hidden)
        variance = (2 * log_std).exp()
        prior_variance = (2 * prior_log_std).exp()
        return (
            prior_log_std
            - log_std
            + (variance + (mean - prior_mean).square()) / (2 * prior_variance)
            - 0.5
        ).sum(-1)


class MTPDistributionalHead(nn.Module):
    def __init__(self, cfg: DreamerV4FullConfig):
        super().__init__()
        self.heads = nn.ModuleList([
            MLPHead(cfg.model_dim, _NUM_BINS)
            for _ in range(cfg.mtp_length)
        ])

    def forward(self, hidden: torch.Tensor, distance: int = 0) -> torch.Tensor:
        return self.heads[min(distance, len(self.heads) - 1)](hidden)


class InteractiveDynamics(nn.Module):
    'Space-time shortcut model with an isolated task/agent token.'

    def __init__(self, cfg: DreamerV4FullConfig):
        super().__init__()
        self.cfg = cfg
        self.latent_embed = nn.Linear(cfg.latent_channels, cfg.model_dim)
        self.latent_output = nn.Linear(cfg.model_dim, cfg.latent_channels)
        self.action_embed = nn.Linear(cfg.action_dim, cfg.model_dim)


        self.unknown_action = nn.Parameter(torch.randn(cfg.model_dim) * 0.02)
        half = cfg.model_dim // 2
        self.signal_embed = nn.Embedding(cfg.shortcut_steps + 1, half)
        self.step_embed = nn.Embedding(
            int(math.log2(cfg.shortcut_steps)) + 1,
            cfg.model_dim - half,
        )
        self.registers = nn.Parameter(
            torch.randn(cfg.register_tokens, cfg.model_dim) * 0.02
        )


        self.agent_token = nn.Parameter(torch.randn(cfg.model_dim) * 0.02)
        self.task_embed = nn.Embedding(cfg.num_tasks, cfg.model_dim)
        self.transformer = SpaceTimeTransformer(
            cfg.model_dim,
            cfg.dynamics_layers,
            cfg.heads,
            cfg.kv_heads,
            cfg.ff_multiplier,
            cfg.temporal_every,
            cfg.attention_softcap,
            cfg.dropout,
        )

    @property
    def action_index(self) -> int:
        return self.cfg.latent_tokens

    @property
    def condition_index(self) -> int:
        return self.action_index + 1

    @property
    def agent_index(self) -> int:
        return (
            self.cfg.latent_tokens + 2 + self.cfg.register_tokens
        )

    def _signal_indices(self, signal: torch.Tensor) -> torch.Tensor:
        return (signal * self.cfg.shortcut_steps).round().long().clamp(
            0, self.cfg.shortcut_steps
        )

    def _step_indices(self, step: torch.Tensor) -> torch.Tensor:
        return torch.log2(step.reciprocal().clamp_min(1)).round().long().clamp(
            0, self.step_embed.num_embeddings - 1
        )

    def _spatial_mask(self, device: torch.device) -> torch.Tensor:
        count = self.agent_index + 1
        mask = torch.ones(count, count, dtype=torch.bool, device=device)


        mask[: self.agent_index, self.agent_index] = False
        return mask

    def forward(
        self,
        corrupted_latents: torch.Tensor,
        previous_actions: torch.Tensor,
        signal: torch.Tensor,
        step: torch.Tensor,
        task_ids: torch.Tensor | None = None,
        action_known: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, time, latent_tokens, _ = corrupted_latents.shape
        if latent_tokens != self.cfg.latent_tokens:
            raise ValueError("unexpected number of tokenizer latent tokens")
        if task_ids is None:
            task_ids = torch.zeros(batch, dtype=torch.long, device=signal.device)
        latent = self.latent_embed(corrupted_latents)
        action = self.action_embed(previous_actions)
        if action_known is not None:
            action = torch.where(
                action_known.bool().unsqueeze(-1),
                action,
                self.unknown_action.view(1, 1, -1),
            )
        action = action.unsqueeze(2)
        condition = torch.cat([
            self.signal_embed(self._signal_indices(signal)),
            self.step_embed(self._step_indices(step)),
        ], -1).unsqueeze(2)
        registers = self.registers.view(
            1, 1, self.cfg.register_tokens, -1
        ).expand(batch, time, -1, -1)
        agent = (
            self.agent_token.view(1, 1, 1, -1)
            + self.task_embed(task_ids).view(batch, 1, 1, -1)
        ).expand(batch, time, -1, -1)
        hidden = self.transformer(
            torch.cat([latent, action, condition, registers, agent], 2),
            spatial_mask=self._spatial_mask(corrupted_latents.device),
            temporal_window=self.cfg.context_length,
        )
        return {
            "latent": self.latent_output(
                hidden[:, :, : self.cfg.latent_tokens]
            ),
            "agent": hidden[:, :, self.agent_index],
        }
