'TransformerDreamer agent: transformer world model + imagination actor-critic.'
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning_machines.dreamerv4.config import DreamerV4Config
from learning_machines.dreamerv4.transformer import CausalTransformer, symlog, symexp
from learning_machines.dreamerv4.actor_critic import (
    SquashedGaussianActor,
    DistributionalValueCritic,
    two_hot_loss,
    logits_to_value,
    ReturnEMA,
    compute_td_lambda_returns,
)
from learning_machines.dreamerv4.dreamerv4_image import ImageDreamerV4Agent


class RunningStats:
    'Fixed-scale normalization from pre-computed stats.'

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


class TransformerDreamerAgent(nn.Module):
    'Transformer-based world model agent with Dreamer-style actor-critic.'

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
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
        )

        self.actor = SquashedGaussianActor(
            obs_dim=cfg.obs_dim,
            act_dim=cfg.act_dim,
            hidden_dim=cfg.d_model,
        )


        self.critic = DistributionalValueCritic(
            state_dim=cfg.obs_dim,
            hidden_dim=cfg.d_model,
        )

        self.target_critic = DistributionalValueCritic(
            state_dim=cfg.obs_dim,
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

        self.return_ema = ReturnEMA(decay=0.99)

    def warmup_stats(self, buffer):
        'Pre-compute normalization stats from the replay buffer.'
        all_obs = []
        all_rew = []
        for ep in buffer._episodes:
            all_obs.append(ep["observations"])
            all_rew.append(ep["rewards"].reshape(-1, 1))
        self.obs_stats.warmup(np.concatenate(all_obs, axis=0))

        all_rew_arr = np.concatenate(all_rew, axis=0)
        rew_symlog = np.sign(all_rew_arr) * np.log1p(np.abs(all_rew_arr))
        self.rew_stats.warmup(rew_symlog)

    def _has_nan(self, *tensors) -> bool:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                if torch.isnan(t).any() or torch.isinf(t).any():
                    return True
        return False

    def update_world_model(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        'Update world model on a batch of subsequences.'
        self.wm_optimizer.zero_grad()

        self._wm_state_before = {k: v.clone() for k, v in self.world_model.state_dict().items()}

        obs = batch["observations"].float()
        act = batch["actions"].float()
        rew = batch["rewards"].float()
        mask = batch["masks"].float()

        obs_norm = self.obs_stats.normalize_tensor(obs)

        rew_symlog = symlog(rew.unsqueeze(-1)).squeeze(-1)
        rew_norm = self.rew_stats.normalize_tensor(rew_symlog.unsqueeze(-1)).squeeze(-1)

        preds = self.world_model(obs_norm, act, rew_norm, mask)


        target_obs = obs_norm[:, 1:]
        obs_loss = F.mse_loss(preds["obs_pred"], target_obs, reduction="none")
        obs_loss = (obs_loss.mean(-1) * mask).sum() / mask.sum().clamp(min=1)


        rew_loss = F.mse_loss(preds["rew_pred"].squeeze(-1), rew_norm, reduction="none")
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

        has_nan_grad = any(
            p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in self.world_model.parameters()
        )
        if has_nan_grad:
            self.wm_optimizer.zero_grad()
            return {"wm/obs_loss": 0.0, "wm/rew_loss": 0.0, "wm/done_loss": 0.0, "wm/total_loss": 0.0}

        self.wm_optimizer.step()

        has_nan_weight = any(
            torch.isnan(p).any() or torch.isinf(p).any()
            for p in self.world_model.parameters()
        )
        if has_nan_weight:
            self.wm_optimizer.zero_grad()
            self.world_model.load_state_dict(self._wm_state_before)
            return {"wm/obs_loss": 0.0, "wm/rew_loss": 0.0, "wm/done_loss": 0.0, "wm/total_loss": 0.0}

        return {
            "wm/obs_loss": obs_loss.item(),
            "wm/rew_loss": rew_loss.item(),
            "wm/done_loss": done_loss.item(),
            "wm/total_loss": total_loss.item(),
        }

    def update_actor_critic(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        'Update actor and critic via imagination using REINFORCE.'
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
        imag_rew = imagined["rewards"]
        imag_done = imagined["dones"]
        imag_log_prob = imagined["log_probs"]

        B, H = imag_rew.shape


        imag_obs_after = imag_obs[:, 1:H + 1]
        imag_obs_norm = self.obs_stats.normalize_tensor(imag_obs_after)

        imag_obs_norm = torch.clamp(imag_obs_norm, -5.0, 5.0)


        imag_logits = self.critic(imag_obs_norm.reshape(-1, self.cfg.obs_dim)).reshape(B, H, -1)
        imag_values = logits_to_value(imag_logits)

        with torch.no_grad():

            last_obs_norm = self.obs_stats.normalize_tensor(imag_obs[:, -1])
            last_obs_norm = torch.clamp(last_obs_norm, -5.0, 5.0)
            bootstrap_logits = self.target_critic(last_obs_norm)
            bootstrap = logits_to_value(bootstrap_logits).unsqueeze(1)


        returns = compute_td_lambda_returns(
            imag_rew,
            torch.cat([imag_values, bootstrap], dim=1)[:, :H + 1],
            bootstrap,
            imag_done,
            self.cfg.gamma,
            self.cfg.lam,
        )



        returns_flat = returns.reshape(-1)
        logits_flat = imag_logits.reshape(-1, imag_logits.shape[-1])
        critic_loss = two_hot_loss(logits_flat, returns_flat)



        with torch.no_grad():
            return_range = self.return_ema.update(returns)
            S = max(1.0, return_range)
            advantages = (returns - imag_values.detach()) / S
            advantages = torch.clamp(advantages, -5.0, 5.0)


        actor_loss = -(advantages.detach() * imag_log_prob.squeeze(-1)).mean()

        actor_loss += self.cfg.entropy_coef * imag_log_prob.mean()

        total_loss = critic_loss + actor_loss
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.cfg.grad_clip,
        )

        if self._has_nan(total_loss):
            self.ac_optimizer.zero_grad()
            return {"ac/critic_loss": 0.0, "ac/actor_loss": 0.0, "ac/return_mean": 0.0}

        self.ac_optimizer.step()
        self._soft_update_target()

        return {
            "ac/critic_loss": critic_loss.item(),
            "ac/actor_loss": actor_loss.item(),
            "ac/return_mean": returns.mean().item(),
            "ac/return_range": self.return_ema.range,
            "ac/actor_std": self.actor.std(imag_obs_norm.reshape(-1, self.cfg.obs_dim)).mean().item(),
            "ac/action_saturation": (imagined["actions"].abs() >= 0.99).float().mean().item(),
            "ac/advantage_p05": torch.quantile(advantages, 0.05).item(),
            "ac/advantage_p50": torch.quantile(advantages, 0.50).item(),
            "ac/advantage_p95": torch.quantile(advantages, 0.95).item(),
            "ac/grad_norm": float(grad_norm),
            "ac/actor_entropy": float(-imag_log_prob.mean().item()),
        }

    def _soft_update_target(self):
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, self.cfg.target_update_rate)

    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        'Select action for deployment.'
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
            "return_ema_range": self.return_ema.range,
            "cfg": self.cfg,
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")) -> "TransformerDreamerAgent":
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
        agent.return_ema.range = checkpoint.get("return_ema_range", 1.0)
        return agent





DreamerV4Agent = ImageDreamerV4Agent
