"""Causal transformer dynamics model for DreamerV4.

Processes (observation, action, reward) sequences with causal masking.
Learns to predict next observation, reward, and terminal from past context.
"""
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformer(nn.Module):
    """Causal transformer for sequence modeling of (obs, action, reward) tokens.

    Uses causal self-attention mask so each position can only attend to
    past and current positions (no future information leakage).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        context_length: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length

        self.obs_embed = nn.Linear(12, d_model)
        self.act_embed = nn.Linear(2, d_model)
        self.rew_embed = nn.Linear(1, d_model)

        self.pos_embed = nn.Embedding(context_length + 1, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.obs_head = nn.Linear(d_model, 12)
        self.rew_head = nn.Linear(d_model, 1)
        self.done_head = nn.Linear(d_model, 1)

        self._context_length = context_length
        self._causal_mask = self._build_causal_mask(context_length + 1)

    def _build_causal_mask(self, size: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.bool()
        return mask

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for sequence modeling.

        Args:
            observations: (B, T+1, obs_dim)
            actions: (B, T, act_dim)
            rewards: (B, T, 1)
            mask: (B, T) — 1 for valid, 0 for padded

        Returns:
            dict with predictions for positions 1..T
            obs_pred: (B, T, obs_dim) — next observation prediction
            rew_pred: (B, T, 1) — reward prediction
            done_pred: (B, T, 1) — terminal prediction
        """
        B, Tp1, _ = observations.shape
        T = Tp1 - 1

        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)

        obs_tok = self.obs_embed(observations)
        act_tok = self.act_embed(actions)
        rew_tok = self.rew_embed(rewards)

        tokens = []
        for t in range(T):
            tokens.append(obs_tok[:, t] + act_tok[:, t])
        tokens.append(obs_tok[:, T])
        tokens = torch.stack(tokens, dim=1)

        positions = torch.arange(T + 1, device=tokens.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embed(positions)
        tokens = self.embed_dropout(tokens)

        causal_mask = self._causal_mask[:T + 1, :T + 1].to(tokens.device)

        if mask is not None:
            key_padding_mask = ~mask.bool()
            key_padding_mask = F.pad(key_padding_mask, (0, 1), value=True)
        else:
            key_padding_mask = None

        hidden = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )

        pred_obs = self.obs_head(hidden[:, :T])
        pred_rew = self.rew_head(hidden[:, :T])
        pred_done = self.done_head(hidden[:, :T])

        return {
            "obs_pred": pred_obs,
            "rew_pred": pred_rew,
            "done_pred": pred_done,
        }

    def imagine(
        self,
        initial_obs: torch.Tensor,
        actor: nn.Module,
        horizon: int,
        clip_reward: bool = True,
        reward_scale: float = 10.0,
    ) -> dict[str, torch.Tensor]:
        """Imagine future trajectories using the world model + actor.

        Args:
            initial_obs: (B, obs_dim) — starting observation
            actor: callable that takes (obs, deterministic) → (action, log_prob)
            horizon: number of steps to imagine
            clip_reward: whether to clip rewards
            reward_scale: reward scaling factor

        Returns:
            dict with imagined trajectory tensors
        """
        B = initial_obs.shape[0]
        device = initial_obs.device

        obs_list = [initial_obs]
        act_list = []
        rew_list = []
        done_list = []
        log_prob_list = []

        h_obs = initial_obs

        for t in range(horizon):
            act, log_prob = actor(h_obs, deterministic=False)

            if clip_reward:
                act = torch.clamp(act, -1.0, 1.0)

            act_list.append(act)
            log_prob_list.append(log_prob)

            obs_tok = self.obs_embed(h_obs)
            act_tok = self.act_embed(act)

            token = obs_tok + act_tok

            pos = torch.arange(t, t + 1, device=device).unsqueeze(0).expand(B, -1)
            token = token.unsqueeze(1) + self.pos_embed(pos)
            token = self.embed_dropout(token)

            hidden = token.squeeze(1)

            h_obs = self.obs_head(hidden)
            rew = self.rew_head(hidden)
            done = self.done_head(hidden)

            if clip_reward:
                rew = torch.clamp(rew, -reward_scale, reward_scale)

            obs_list.append(h_obs)
            rew_list.append(rew)
            done_list.append(done)

        obs_arr = torch.stack(obs_list, dim=1)
        act_arr = torch.stack(act_list, dim=1)
        rew_arr = torch.cat(rew_list, dim=1)
        done_arr = torch.cat(done_list, dim=1)
        log_prob_arr = torch.stack(log_prob_list, dim=1)

        return {
            "observations": obs_arr,
            "actions": act_arr,
            "rewards": rew_arr,
            "dones": done_arr,
            "log_probs": log_prob_arr,
        }
