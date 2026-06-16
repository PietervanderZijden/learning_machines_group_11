"""Actor-critic for DreamerV4 imagination training.

Squashed Gaussian actor + symlog critic, trained on imagined trajectories.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class SquashedGaussianActor(nn.Module):
    """Squashed Gaussian policy."""

    def __init__(self, obs_dim: int = 12, act_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

        if deterministic:
            action = torch.tanh(mu)
            log_prob = torch.zeros(obs.shape[0], 1, device=obs.device)
            return action, log_prob

        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        action_tanh = torch.tanh(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1, keepdim=True)

        return action_tanh, log_prob

    def evaluate(self, obs: torch.Tensor):
        return self.forward(obs, deterministic=False)


class SymlogCritic(nn.Module):
    """Critic network with symlog prediction."""

    def __init__(self, obs_dim: int = 12, act_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return symlog(self.net(x))


class TwinCritic(nn.Module):
    """Twin Q-networks for clipped double-Q learning."""

    def __init__(self, obs_dim: int = 12, act_dim: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.q1 = SymlogCritic(obs_dim, act_dim, hidden_dim)
        self.q2 = SymlogCritic(obs_dim, act_dim, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        return self.q1(obs, action), self.q2(obs, action)


def compute_td_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.997,
    lam: float = 0.95,
) -> torch.Tensor:
    """Compute TD-lambda returns for imagination training.

    Args:
        rewards: (B, T) — imagined rewards
        values: (B, T) — critic values
        bootstrap: (B, 1) — bootstrap value
        dones: (B, T) — done flags
        gamma: discount factor
        lam: GAE lambda

    Returns:
        returns: (B, T) — TD-lambda returns
    """
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)

    last_gae = bootstrap
    for t in reversed(range(T)):
        next_non_terminal = (1.0 - dones[:, t : t + 1]).float()
        next_value = values[:, t + 1 : t + 2] if t < T - 1 else bootstrap
        delta = rewards[:, t : t + 1] + gamma * next_value * next_non_terminal - values[:, t : t + 1]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        returns[:, t : t + 1] = last_gae + values[:, t : t + 1]

    return returns
