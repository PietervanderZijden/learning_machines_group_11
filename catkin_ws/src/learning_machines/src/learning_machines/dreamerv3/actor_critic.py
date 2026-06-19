"""DreamerV3 actor-critic: policy and value function trained in imagination.

Implements:
- Squashed Gaussian actor with REINFORCE + return normalization
- Distributional critic with symlog two-hot output
- Lambda returns computation

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


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 3) -> nn.Sequential:
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    net = []
    for i in range(len(dims) - 1):
        net.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            net.append(nn.SiLU())
    return nn.Sequential(*net)


class Actor(nn.Module):
    """Squashed Gaussian actor for continuous actions.

    Uses a bounded log standard deviation for stable continuous control.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 512,
        layers: int = 3,
        std_min: float = 0.1,
        std_max: float = 1.0,
        mean_limit: float = 2.5,
        action_limit: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.std_min = float(std_min)
        self.std_max = float(std_max)
        if not 0 < self.std_min <= self.std_max:
            raise ValueError("actor standard-deviation bounds are invalid")
        self.mean_limit = float(mean_limit)
        self.action_limit = action_limit
        self.net = mlp(state_dim, hidden, 2 * action_dim, layers)
        nn.init.trunc_normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def _distribution_parameters(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std_param = self.net(state).chunk(2, dim=-1)
        mean = mean.clamp(-self.mean_limit, self.mean_limit)
        std = self.std_min + (
            self.std_max - self.std_min
        ) * torch.sigmoid(std_param)
        return mean, std

    def forward(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample action from policy.

        state: (batch, state_dim)
        Returns: (batch, action_dim)
        """
        mean, std = self._distribution_parameters(state)

        if deterministic:
            action = torch.tanh(mean) * self.action_limit
            return action

        dist = Normal(mean, std)
        sample = dist.rsample()
        action = torch.tanh(sample) * self.action_limit

        return action

    def get_action_and_log_prob(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return log probability.

        Note: Do NOT clamp raw before evaluating log_prob — that would make
        the reported probability inconsistent with the actual sampling distribution.
        Instead, use a stable tanh-normal implementation.
        """
        mean, std = self._distribution_parameters(state)

        dist = Normal(mean, std)
        # Score-function objective: transitions use detached sampled actions,
        # while log_prob retains gradients only through distribution parameters.
        raw = dist.sample().detach()
        action = torch.tanh(raw) * self.action_limit

        # Log probability with tanh squashing correction
        # Formula: log π(a|s) = log N(raw; μ, σ) - Σ log(1 - tanh²(raw))
        log_prob = dist.log_prob(raw).sum(-1)
        # Stable tanh correction: log(1 - tanh²(x)) = 2*(log(2) - x - softplus(-2x))
        log_prob -= (2 * (torch.log(torch.tensor(2.0, device=raw.device)) - raw - F.softplus(-2 * raw))).sum(-1)

        return action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of given actions under this policy.

        Args:
            state: (batch, state_dim)
            action: (batch, action_dim) — actions in [-1, 1] (tanh-squashed)

        Returns:
            log_prob: (batch,)
        """
        mean, std = self._distribution_parameters(state)

        # Inverse tanh to get raw action
        action_clamped = action.clamp(-0.999, 0.999)  # avoid ±1
        raw = torch.atanh(action_clamped)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(raw).sum(-1)
        log_prob -= (2 * (torch.log(torch.tensor(2.0, device=raw.device)) - raw - F.softplus(-2 * raw))).sum(-1)

        return log_prob

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Monte Carlo entropy estimate for the squashed policy."""
        _, log_prob = self.get_action_and_log_prob(state)
        return -log_prob

    def std(self, state: torch.Tensor) -> torch.Tensor:
        return self._distribution_parameters(state)[1]

    def mean(self, state: torch.Tensor) -> torch.Tensor:
        mean, _ = self.net(state).chunk(2, dim=-1)
        return mean.clamp(-self.mean_limit, self.mean_limit)


class Critic(nn.Module):
    """Distributional value function with symlog two-hot output.

    Outputs logits over 255 symlog-space bins.
    logits_to_value() converts to original space via symexp.
    """

    def __init__(
        self,
        state_dim: int,
        hidden: int = 512,
        layers: int = 3,
    ):
        super().__init__()
        self.net = mlp(state_dim, hidden, _NUM_BINS, layers)
        # Zero-initialize output layer for stable early training (paper: Sec 3.2)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return logits over bins: (..., NUM_BINS)"""
        return self.net(state)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        """Return expected scalar value in original space."""
        return logits_to_value(self.forward(state))


class ReturnNormalizer:
    """Exponential moving average return range for normalization.

    Paper: S = EMA(Per(R_t^λ, 95) - Per(R_t^λ, 5), 0.99)
    Used to normalize returns to approximately [0, 1] range.
    """

    def __init__(self, decay: float = 0.99, low_percentile: float = 5.0, high_percentile: float = 95.0):
        self.decay = decay
        self.low_pct = low_percentile
        self.high_pct = high_percentile
        self.range = 1.0

    def update(self, returns: torch.Tensor) -> float:
        with torch.no_grad():
            flat = returns.flatten()
            p_high = torch.quantile(flat, self.high_pct / 100.0)
            p_low = torch.quantile(flat, self.low_pct / 100.0)
            batch_range = (p_high - p_low).item()
            self.range = self.decay * self.range + (1 - self.decay) * batch_range
        return max(1.0, self.range)


def lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continue_logit: torch.Tensor | None,
    gamma: float = 0.997,
    lam: float = 0.95,
    continuation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute lambda Returns (TD-lambda) for imagined trajectories.

    Paper: R_t^λ = r_t + γ c_t ((1-λ) v_t + λ R_{t+1}^λ), R_T^λ = v_T

    rewards: (batch, horizon) — in original space
    values: (batch, horizon + 1) — in original space, last is bootstrap
    continue_logit: (batch, horizon)

    Returns: (batch, horizon) lambda returns in original space
    """
    if continuation is None:
        if continue_logit is None:
            raise ValueError("continue_logit or continuation must be provided")
        continue_prob = torch.sigmoid(continue_logit)
    else:
        continue_prob = continuation
    returns = torch.zeros_like(rewards)
    last = values[:, -1]

    for t in reversed(range(rewards.shape[1])):
        last = rewards[:, t] + gamma * continue_prob[:, t] * (
            (1.0 - lam) * values[:, t + 1] + lam * last
        )
        returns[:, t] = last

    return returns


def soft_cross_entropy(
    logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy to a detached categorical target distribution."""
    target = F.softmax(target_logits.detach(), dim=-1)
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1)
