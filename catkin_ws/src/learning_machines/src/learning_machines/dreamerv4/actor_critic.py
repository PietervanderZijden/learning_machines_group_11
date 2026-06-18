"""Actor-critic for DreamerV4 imagination training.

Dreamer-style: actor picks actions, critic evaluates states (not state-action pairs).
Critic uses two-hot distributional loss for robust value prediction.

Uses shared distributional utilities from distributional.py:
- Bins in symlog space (linear linspace -20 to 20)
- two_hot_loss applies symlog to targets automatically
- logits_to_value applies symexp to output automatically
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from learning_machines.distributional import (
    symlog, symexp, two_hot_loss, logits_to_value, _NUM_BINS,
)

# For positive entropy: σ ≥ 0.25 → log(σ) ≥ -1.39
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class SquashedGaussianActor(nn.Module):
    """Squashed Gaussian policy for continuous actions.

    Supports multi-token prediction (MTP) for paper-accurate DreamerV4.
    With mtp_length > 1, the actor predicts actions for multiple future steps
    from the same hidden state.
    """

    def __init__(self, obs_dim: int = 12, act_dim: int = 2, hidden_dim: int = 256,
                 mtp_length: int = 1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.mtp_length = mtp_length

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # MTP: separate output heads for each future step
        # Paper uses L=8 for multi-token prediction
        self.mu_heads = nn.ModuleList([nn.Linear(hidden_dim, act_dim) for _ in range(mtp_length)])
        self.log_std_heads = nn.ModuleList([nn.Linear(hidden_dim, act_dim) for _ in range(mtp_length)])

    def forward(self, obs: torch.Tensor, deterministic: bool = False,
                mtp_step: int = 0):
        """Forward pass with optional MTP.

        Args:
            obs: (B, obs_dim) observation
            deterministic: if True, return deterministic action
            mtp_step: which MTP head to use (0 = default, 1..L-1 = future steps)

        Returns:
            action: (B, act_dim)
            log_prob: (B, 1)
        """
        h = self.net(obs)

        # Use the specified MTP head
        step = min(mtp_step, self.mtp_length - 1)
        mu = self.mu_heads[step](h)
        log_std = self.log_std_heads[step](h).clamp(LOG_STD_MIN, LOG_STD_MAX)

        if deterministic:
            action = torch.tanh(mu)
            log_prob = torch.zeros(obs.shape[0], 1, device=obs.device)
            return action, log_prob

        std = log_std.exp()
        dist = Normal(mu, std)
        action = dist.sample().detach()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        action_tanh = torch.tanh(action)
        log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1, keepdim=True)

        return action_tanh, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor,
                 mtp_step: int = 0) -> torch.Tensor:
        """Compute log probability of given actions under this policy.

        Args:
            obs: (B, obs_dim)
            action: (B, act_dim) — actions in [-1, 1] (tanh-squashed)
            mtp_step: which MTP head to use

        Returns:
            log_prob: (B, 1)
        """
        h = self.net(obs)
        step = min(mtp_step, self.mtp_length - 1)
        mu = self.mu_heads[step](h)
        log_std = self.log_std_heads[step](h).clamp(LOG_STD_MIN, LOG_STD_MAX)

        # Inverse tanh to get raw action
        action_clamped = action.clamp(-0.999, 0.999)
        raw = torch.atanh(action_clamped)

        std = log_std.exp()
        dist = Normal(mu, std)
        log_prob = dist.log_prob(raw).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)

        return log_prob

    def std(self, obs: torch.Tensor, mtp_step: int = 0) -> torch.Tensor:
        h = self.net(obs)
        step = min(mtp_step, self.mtp_length - 1)
        return self.log_std_heads[step](h).clamp(LOG_STD_MIN, LOG_STD_MAX).exp()

    def forward_mtp(self, obs: torch.Tensor) -> tuple:
        """Forward pass returning all MTP predictions.

        Used during behavior cloning phase to predict L future actions.

        Args:
            obs: (B, obs_dim) observation

        Returns:
            actions: (B, L, act_dim) predicted actions for each future step
            log_probs: (B, L, 1) log probabilities for each future step
        """
        h = self.net(obs)

        actions = []
        log_probs = []

        for step in range(self.mtp_length):
            mu = self.mu_heads[step](h)
            log_std = self.log_std_heads[step](h).clamp(LOG_STD_MIN, LOG_STD_MAX)

            std = log_std.exp()
            dist = Normal(mu, std)
            action = dist.sample().detach()
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            action_tanh = torch.tanh(action)
            log_prob -= torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1, keepdim=True)

            actions.append(action_tanh)
            log_probs.append(log_prob)

        return torch.stack(actions, dim=1), torch.stack(log_probs, dim=1)


class MTPRewardHead(nn.Module):
    """Multi-token prediction reward head.

    Predicts rewards for L future steps from the same hidden state.
    Uses symlog two-hot distributional output.
    """

    def __init__(self, d_model: int = 256, mtp_length: int = 8):
        super().__init__()
        self.mtp_length = mtp_length

        # Separate heads for each future step
        self.heads = nn.ModuleList([
            nn.Linear(d_model, _NUM_BINS) for _ in range(mtp_length)
        ])

        # Initialize to zero for stable early training
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hidden: torch.Tensor, mtp_step: int = 0) -> torch.Tensor:
        """Predict reward distribution for a specific MTP step.

        Args:
            hidden: (B, d_model) hidden state
            mtp_step: which MTP head to use

        Returns:
            logits: (B, NUM_BINS) reward distribution logits
        """
        step = min(mtp_step, self.mtp_length - 1)
        return self.heads[step](hidden)

    def forward_all(self, hidden: torch.Tensor) -> torch.Tensor:
        """Predict reward distributions for all MTP steps.

        Args:
            hidden: (B, d_model) hidden state

        Returns:
            logits: (B, L, NUM_BINS) reward distribution logits for each step
        """
        return torch.stack([head(hidden) for head in self.heads], dim=1)


class DistributionalValueCritic(nn.Module):
    """Dreamer-style state-value critic with two-hot distributional output.

    Predicts V(s_t) as a distribution over symlog-spaced bins.
    logits_to_value() converts to original space via symexp.
    """

    def __init__(self, state_dim: int = 12, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, _NUM_BINS),
        )
        # Initialize output weights to zero for stable early training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return logits over bins: (..., NUM_BINS)"""
        return self.net(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        """Return expected scalar value in original space."""
        return logits_to_value(self.forward(state))


class ReturnEMA:
    """Exponential moving average of return percentiles for normalization.

    Paper: S = EMA(Per(R, 95) - Per(R, 5), 0.99)
    """

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.range = 1.0

    def update(self, returns: torch.Tensor) -> float:
        with torch.no_grad():
            flat = returns.flatten()
            p95 = torch.quantile(flat, 0.95)
            p05 = torch.quantile(flat, 0.05)
            batch_range = (p95 - p05).item()
            self.range = self.decay * self.range + (1 - self.decay) * batch_range
        return self.range


def compute_td_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.994009,
    lam: float = 0.95,
    continuations: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute TD-lambda returns.

    Args:
        rewards: (B, T) — imagined rewards in original space
        values: (B, T+1) — critic values in original space (last is bootstrap)
        bootstrap: (B, 1) — bootstrap value in original space
        dones: (B, T) — done flags or probabilities, ignored when continuations is provided
        continuations: optional (B, T) continuation probabilities
        gamma: discount factor
        lam: GAE lambda

    Returns:
        returns: (B, T) — TD-lambda returns in original space
    """
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)

    last_gae = torch.zeros_like(bootstrap)
    for t in reversed(range(T)):
        if continuations is None:
            next_non_terminal = (1.0 - dones[:, t : t + 1]).float()
        else:
            next_non_terminal = continuations[:, t : t + 1].float()
        next_value = values[:, t + 1 : t + 2] if t < T - 1 else bootstrap
        delta = rewards[:, t : t + 1] + gamma * next_value * next_non_terminal - values[:, t : t + 1]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        returns[:, t : t + 1] = last_gae + values[:, t : t + 1]

    return returns
