'Causal transformer dynamics model for DreamerV4.'
from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class CausalTransformer(nn.Module):
    'Causal transformer for sequence modeling of (obs, action, reward) tokens.'

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        context_length: int = 64,
        obs_dim: int = 12,
        act_dim: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.act_embed = nn.Linear(act_dim, d_model)
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

        self.obs_head = nn.Linear(d_model, obs_dim)
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
        'Forward pass for sequence modeling.'
        B, Tp1, _ = observations.shape
        T = Tp1 - 1

        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)


        obs_tok = self.obs_embed(observations)
        act_tok = self.act_embed(actions)
        rew_tok = self.rew_embed(rewards)




        tokens = []
        for t in range(T):
            tokens.append(obs_tok[:, t] + act_tok[:, t] + rew_tok[:, t])
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
        'Imagine future trajectories using autoregressive causal transformer.'
        B = initial_obs.shape[0]
        device = initial_obs.device

        obs_list = [initial_obs]
        act_list = []
        rew_list = []
        done_list = []
        log_prob_list = []

        token_embeddings: list[torch.Tensor] = []

        for t in range(horizon):
            h_obs = obs_list[-1].detach()

            act, log_prob = actor(h_obs, deterministic=False)

            if clip_reward:
                act = torch.clamp(act, -1.0, 1.0)

            act_list.append(act)
            log_prob_list.append(log_prob)


            obs_symlog = symlog(h_obs)
            obs_tok = self.obs_embed(obs_symlog)
            act_tok = self.act_embed(act)


            if t > 0:
                prev_rew = rew_list[-1].detach()
                prev_rew_symlog = symlog(prev_rew)
                rew_tok = self.rew_embed(prev_rew_symlog)
            else:
                rew_tok = torch.zeros(B, self.d_model, device=device)

            token_embeddings.append(obs_tok + act_tok + rew_tok)

            tokens = torch.stack(token_embeddings, dim=1)
            positions = torch.arange(t + 1, device=device).unsqueeze(0).expand(B, -1)
            tokens = tokens + self.pos_embed(positions)
            tokens = self.embed_dropout(tokens)

            causal_mask = self._causal_mask[:t + 1, :t + 1].to(device)
            hidden = self.transformer(tokens, mask=causal_mask)

            last_hidden = hidden[:, -1]
            next_obs = self.obs_head(last_hidden)
            rew = self.rew_head(last_hidden)
            done = self.done_head(last_hidden)


            next_obs_raw = symexp(next_obs)
            rew_raw = symexp(rew)

            if clip_reward:
                rew_raw = torch.clamp(rew_raw, -reward_scale, reward_scale)

            obs_list.append(next_obs_raw)
            rew_list.append(rew_raw)
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
