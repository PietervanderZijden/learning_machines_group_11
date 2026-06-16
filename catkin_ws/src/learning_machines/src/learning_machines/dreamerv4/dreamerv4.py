"""DreamerV4 agent: transformer world model + imagination actor-critic.

Training:
  1. World model: predict next obs, reward, done from (obs, action, reward) sequences
  2. Actor-critic: imagine future rollouts with world model, optimize policy via TD-lambda

Reference: Hafner et al., "DreamerV4: Scalable World Models Without World Models" (2025)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning_machines.dreamerv4.config import DreamerV4Config
from learning_machines.dreamerv4.transformer import CausalTransformer
from learning_machines.dreamerv4.actor_critic import (
    SquashedGaussianActor,
    TwinCritic,
    symlog,
    symexp,
    compute_td_lambda_returns,
)


class RunningStats:
    """Fixed-scale normalization from pre-computed stats."""

    def __init__(self, shape: tuple):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)

    def warmup(self, data: np.ndarray):
        self.mean = data.mean(axis=0).astype(np.float32)
        self.var = data.var(axis=0).astype(np.float32)

    def normalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.from_numpy(self.mean).to(x.device).float()
        std = torch.sqrt(torch.from_numpy(self.var).to(x.device).float() + 1e-8)
        return (x - mean) / std


class DreamerV4Agent(nn.Module):
    """DreamerV4 agent with transformer dynamics model."""

    def __init__(self, cfg: DreamerV4Config):
        super().__init__()
        self.cfg = cfg

        self.obs_stats = RunningStats((cfg.obs_dim,))
        self.rew_stats = RunningStats((cfg.reward_dim,))

        self.world_model = CausalTransformer(
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
            context_length=cfg.context_length,
        )

        self.actor = SquashedGaussianActor(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dim=cfg.d_model,
        )

        self.critic = TwinCritic(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dim=cfg.d_model,
        )

        self.target_critic = TwinCritic(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dim=cfg.d_model,
        )
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.wm_optimizer = torch.optim.AdamW(
            self.world_model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.ac_optimizer = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self._update_count = 0

    def warmup_stats(self, buffer):
        """Pre-compute normalization stats from the replay buffer."""
        all_obs = []
        all_rew = []
        for ep in buffer._episodes:
            all_obs.append(ep["observations"])
            all_rew.append(ep["rewards"].reshape(-1, 1))
        self.obs_stats.warmup(np.concatenate(all_obs, axis=0))
        self.rew_stats.warmup(np.concatenate(all_rew, axis=0))

    def _has_nan(self, *tensors) -> bool:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                if torch.isnan(t).any() or torch.isinf(t).any():
                    return True
        return False

    def update_world_model(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Update world model on a batch of subsequences."""
        self.wm_optimizer.zero_grad()

        obs = batch["observations"].float()
        act = batch["actions"].float()
        rew = batch["rewards"].float()
        mask = batch["masks"].float()

        obs_norm = self.obs_stats.normalize_tensor(obs)
        rew_norm = self.rew_stats.normalize_tensor(rew.unsqueeze(-1)).squeeze(-1)

        preds = self.world_model(obs_norm, act, rew_norm, mask)

        target_obs = obs_norm[:, 1:]
        target_rew = rew_norm

        obs_loss = F.mse_loss(preds["obs_pred"], target_obs, reduction="none")
        obs_loss = (obs_loss.mean(-1) * mask).sum() / mask.sum().clamp(min=1)

        rew_loss = F.mse_loss(preds["rew_pred"].squeeze(-1), target_rew, reduction="none")
        rew_loss = (rew_loss * mask).sum() / mask.sum().clamp(min=1)

        target_done = batch["dones"].float()
        pred_done = preds["done_pred"].squeeze(-1)
        target_done = target_done[..., :pred_done.shape[-1]]
        done_loss = F.binary_cross_entropy_with_logits(pred_done, target_done, reduction="none")
        done_loss = (done_loss * mask).sum() / mask.sum().clamp(min=1)

        total_loss = obs_loss + rew_loss + done_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.grad_clip)

        if self._has_nan(total_loss):
            self.wm_optimizer.zero_grad()
            return {"wm/obs_loss": 0.0, "wm/rew_loss": 0.0, "wm/done_loss": 0.0, "wm/total_loss": 0.0}

        self.wm_optimizer.step()

        return {
            "wm/obs_loss": obs_loss.item(),
            "wm/rew_loss": rew_loss.item(),
            "wm/done_loss": done_loss.item(),
            "wm/total_loss": total_loss.item(),
        }

    def update_actor_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Update actor and critic via imagination."""
        self.ac_optimizer.zero_grad()

        initial_obs = batch["observations"][:, 0]
        initial_obs_norm = self.obs_stats.normalize_tensor(initial_obs)

        imagined = self.world_model.imagine(
            initial_obs_norm,
            self.actor,
            horizon=self.cfg.imagination_horizon,
            clip_reward=self.cfg.clip_reward,
            reward_scale=self.cfg.reward_scale,
        )

        imag_obs = imagined["observations"]
        imag_act = imagined["actions"]
        imag_rew = imagined["rewards"]
        imag_done = imagined["dones"]
        imag_log_prob = imagined["log_probs"]

        B, H = imag_rew.shape

        imag_obs_flat = imag_obs[:, :H].reshape(-1, self.cfg.obs_dim)
        imag_act_flat = imag_act.reshape(-1, self.cfg.act_dim)

        q1, q2 = self.critic(imag_obs_flat, imag_act_flat)
        q1 = symexp(q1).reshape(B, H)
        q2 = symexp(q2).reshape(B, H)
        q_min = torch.min(q1, q2)

        with torch.no_grad():
            tq1, tq2 = self.target_critic(imag_obs_flat, imag_act_flat)
            tq1 = symexp(tq1).reshape(B, H)
            tq2 = symexp(tq2).reshape(B, H)
            tq_min = torch.min(tq1, tq2)

        returns = compute_td_lambda_returns(
            imag_rew, tq_min, torch.zeros(B, 1, device=tq_min.device),
            imag_done, self.cfg.gamma, self.cfg.lam,
        )

        critic_loss = F.mse_loss(q1, returns.detach()) + F.mse_loss(q2, returns.detach())

        actor_loss = -(q_min.mean() + self.cfg.entropy_coef * imag_log_prob.mean())

        total_loss = critic_loss + actor_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.cfg.grad_clip,
        )

        if self._has_nan(total_loss):
            self.ac_optimizer.zero_grad()
            return {"ac/critic_loss": 0.0, "ac/actor_loss": 0.0, "ac/q_mean": 0.0, "ac/return_mean": 0.0}

        self.ac_optimizer.step()

        self._soft_update_target()

        return {
            "ac/critic_loss": critic_loss.item(),
            "ac/actor_loss": actor_loss.item(),
            "ac/q_mean": q_min.mean().item(),
            "ac/return_mean": returns.mean().item(),
        }

    def _soft_update_target(self):
        self._update_count += 1
        if self._update_count % self.cfg.target_update_rate == 0:
            for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
                tp.data.copy_(self.cfg.target_update_rate * p.data + (1 - self.cfg.target_update_rate) * tp.data)

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Select action for deployment."""
        with torch.no_grad():
            action, _ = self.actor(obs, deterministic=deterministic)
        return action

    def save(self, path: str):
        torch.save({
            "model": self.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "ac_optimizer": self.ac_optimizer.state_dict(),
            "obs_stats_mean": self.obs_stats.mean,
            "obs_stats_var": self.obs_stats.var,
            "rew_stats_mean": self.rew_stats.mean,
            "rew_stats_var": self.rew_stats.var,
            "cfg": self.cfg,
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")) -> "DreamerV4Agent":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint["cfg"]
        agent = cls(cfg).to(device)
        agent.load_state_dict(checkpoint["model"])
        agent.wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])
        agent.ac_optimizer.load_state_dict(checkpoint["ac_optimizer"])
        agent.obs_stats.mean = checkpoint["obs_stats_mean"]
        agent.obs_stats.var = checkpoint["obs_stats_var"]
        agent.rew_stats.mean = checkpoint["rew_stats_mean"]
        agent.rew_stats.var = checkpoint["rew_stats_var"]
        return agent
