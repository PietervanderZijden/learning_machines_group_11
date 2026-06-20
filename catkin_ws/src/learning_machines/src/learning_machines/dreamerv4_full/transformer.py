"""DreamerV4 transformer primitives.

The paper starts from pre-norm RMSNorm, RoPE, and SwiGLU, then factorizes video
attention into space-only layers and sparse time-only layers. Dynamics attention
uses grouped-query attention, QK normalization, and logit soft-capping.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.float().square().mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * scale.to(x.dtype)) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.value = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.dropout(F.silu(self.gate(x)) * self.value(x)))


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to (B, H, N, D) tensors."""
    dim = x.shape[-1]
    rotary_dim = dim - dim % 2
    if rotary_dim == 0:
        return x
    half = rotary_dim // 2
    inv_freq = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=x.device, dtype=torch.float32)
        / max(1, half)
    )
    angles = positions.float().unsqueeze(-1) * inv_freq
    cos = angles.cos().to(x.dtype).view(1, 1, positions.numel(), half)
    sin = angles.sin().to(x.dtype).view(1, 1, positions.numel(), half)
    left, right = x[..., :half], x[..., half:rotary_dim]
    rotated = torch.cat([left * cos - right * sin, right * cos + left * sin], -1)
    return torch.cat([rotated, x[..., rotary_dim:]], -1)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        softcap: float = 50.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.group_size = heads // kv_heads
        self.softcap = softcap
        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q = _rope(q, positions)
        k = _rope(k, positions)
        q = F.normalize(q.float(), dim=-1).to(x.dtype)
        k = F.normalize(k.float(), dim=-1).to(x.dtype)
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        # QK normalization already bounds dot products to [-1, 1], so the
        # conventional 1/sqrt(head_dim) scaling is unnecessary and would
        # over-amplify the logits. DreamerV4 uses QK norm + logit soft-capping
        # without this extra scaling.
        logits = torch.matmul(q, k.transpose(-2, -1))
        if self.softcap > 0:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        if mask is not None:
            logits = logits.masked_fill(~mask.view(1, 1, length, length), -1e9)
        attention = F.softmax(logits.float(), dim=-1).to(x.dtype)
        attention = F.dropout(attention, self.dropout, self.training)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.out_proj(output)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        kv_heads: int,
        ff_multiplier: float,
        softcap: float,
        dropout: float,
    ):
        super().__init__()
        hidden = int(dim * ff_multiplier)
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(
            dim, heads, kv_heads, softcap=softcap, dropout=dropout
        )
        self.ff_norm = RMSNorm(dim)
        self.ff = SwiGLU(dim, hidden, dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        positions: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.residual_dropout(
            self.attn(self.attn_norm(x), positions=positions, mask=mask)
        )
        return x + self.residual_dropout(self.ff(self.ff_norm(x)))


class SpaceTimeTransformer(nn.Module):
    """Factorized space-time transformer with sparse temporal layers."""

    def __init__(
        self,
        dim: int,
        layers: int,
        heads: int,
        kv_heads: int,
        ff_multiplier: float,
        temporal_every: int,
        softcap: float,
        dropout: float,
    ):
        super().__init__()
        self.temporal_every = temporal_every
        self.spatial = nn.ModuleList([
            TransformerLayer(
                dim, heads, kv_heads, ff_multiplier, softcap, dropout
            )
            for _ in range(layers)
        ])
        self.temporal = nn.ModuleDict({
            str(index): TransformerLayer(
                dim, heads, kv_heads, ff_multiplier, softcap, dropout
            )
            for index in range(layers)
            if (index + 1) % temporal_every == 0
        })
        self.output_norm = RMSNorm(dim)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.ones(length, length, dtype=torch.bool, device=device).tril()

    def forward(
        self,
        x: torch.Tensor,
        *,
        spatial_mask: torch.Tensor | None = None,
        temporal_causal: bool = True,
        temporal_window: int | None = None,
    ) -> torch.Tensor:
        """Transform (B, T, S, D) tokens."""
        batch, time, space, dim = x.shape
        spatial_positions = torch.arange(space, device=x.device)
        temporal_positions = torch.arange(time, device=x.device)
        for index, spatial_layer in enumerate(self.spatial):
            spatial_x = x.reshape(batch * time, space, dim)
            spatial_x = spatial_layer(
                spatial_x, positions=spatial_positions, mask=spatial_mask
            )
            x = spatial_x.reshape(batch, time, space, dim)
            temporal_layer = (
                self.temporal[str(index)] if str(index) in self.temporal else None
            )
            if temporal_layer is not None and time > 1:
                temporal_x = x.transpose(1, 2).reshape(batch * space, time, dim)
                temporal_mask = (
                    self._causal_mask(time, x.device) if temporal_causal else None
                )
                if temporal_mask is not None and temporal_window is not None:
                    positions = torch.arange(time, device=x.device)
                    temporal_mask &= (
                        positions[:, None] - positions[None, :]
                    ) < temporal_window
                temporal_x = temporal_layer(
                    temporal_x,
                    positions=temporal_positions,
                    mask=temporal_mask,
                )
                x = temporal_x.reshape(batch, space, time, dim).transpose(1, 2)
        return self.output_norm(x)
