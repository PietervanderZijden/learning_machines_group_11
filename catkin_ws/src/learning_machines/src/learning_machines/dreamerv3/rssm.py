'Recurrent State-Space Model (RSSM) for DreamerV3.'
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, OneHotCategorical


class RSSM(nn.Module):
    'RSSM with categorical stochastic state.'

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        deterministic_size: int = 512,
        stochastic_classes: int = 32,
        stochastic_bins: int = 32,
        hidden_size: int = 512,
        unimix: float = 0.01,
        obs_is_embedding: bool = False,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.deterministic_size = deterministic_size
        self.stochastic_classes = stochastic_classes
        self.stochastic_bins = stochastic_bins
        self.stochastic_size = stochastic_classes * stochastic_bins
        self.unimix = unimix
        self.obs_is_embedding = obs_is_embedding


        self.obs_embed = nn.Linear(obs_dim, hidden_size)
        self.act_embed = nn.Linear(action_dim, hidden_size)

        self.z_embed = nn.Linear(self.stochastic_size, hidden_size)


        self.gru = nn.GRUCell(hidden_size, deterministic_size)


        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, stochastic_classes * stochastic_bins),
        )


        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_size + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, stochastic_classes * stochastic_bins),
        )


        if not obs_is_embedding:
            self.obs_norm = nn.LayerNorm(obs_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.deterministic_size, device=device)
        z = torch.zeros(batch_size, self.stochastic_size, device=device)
        return h, z

    def observe(
        self, obs: torch.Tensor, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        'Forward pass for observed data.'
        if self.obs_is_embedding:

            obs_embed = F.silu(self.obs_embed(obs))
        else:

            obs_symlog = torch.sign(obs) * torch.log1p(torch.abs(obs))
            obs_normed = self.obs_norm(obs_symlog)
            obs_embed = F.silu(self.obs_embed(obs_normed))
        act_embed = F.silu(self.act_embed(action))
        z_embed = F.silu(self.z_embed(z))


        gru_input = z_embed + act_embed
        h_new = self.gru(gru_input, h)


        prior_logits = self.prior_net(h_new)
        posterior_logits = self.posterior_net(torch.cat([h_new, obs_embed], dim=-1))


        z_new = self._sample_categorical(posterior_logits)

        return h_new, z_new, prior_logits, posterior_logits

    def imagine(
        self, action: torch.Tensor, h: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        'Forward pass for imagined (dreamed) data.'
        act_embed = F.silu(self.act_embed(action))
        z_embed = F.silu(self.z_embed(z))


        gru_input = z_embed + act_embed
        h_new = self.gru(gru_input, h)

        prior_logits = self.prior_net(h_new)
        z_new = self._sample_categorical(prior_logits)

        return h_new, z_new, prior_logits

    def _apply_unimix(self, logits: torch.Tensor) -> torch.Tensor:
        'Apply uniform mixture to logits to prevent overconfident predictions.'
        if self.unimix <= 0:
            return logits

        uniform = torch.ones_like(logits) / self.stochastic_bins
        mixed_probs = (1 - self.unimix) * logits.softmax(-1) + self.unimix * uniform
        return torch.log(mixed_probs + 1e-8)

    def _sample_categorical(self, logits: torch.Tensor) -> torch.Tensor:
        'Sample from Gumbel-Softmax categorical for straight-through gradient.'
        shape = (-1, self.stochastic_classes, self.stochastic_bins)
        logits_3d = logits.view(shape)

        logits_mixed = self._apply_unimix(logits_3d)
        dist = OneHotCategorical(logits=logits_mixed)
        sample = dist.sample()

        sample = sample + logits_mixed.softmax(-1) - logits_mixed.softmax(-1).detach()
        return sample.view(-1, self.stochastic_size)

    def get_stochastic(self, logits: torch.Tensor) -> torch.Tensor:
        'Get mode (argmax) of categorical for evaluation.'
        shape = (-1, self.stochastic_classes, self.stochastic_bins)
        logits_3d = logits.view(shape)
        mode = torch.zeros_like(logits_3d)
        idx = logits_3d.argmax(dim=-1, keepdim=True)
        mode.scatter_(-1, idx, 1.0)
        return mode.view(-1, self.stochastic_size)

    def kl_loss(
        self, prior_logits: torch.Tensor, posterior_logits: torch.Tensor,
        free_nats: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        'KL divergence losses with free nats and KL balance.'
        shape = (-1, self.stochastic_classes, self.stochastic_bins)


        prior_logits_3d = prior_logits.view(shape)
        posterior_logits_3d = posterior_logits.view(shape)
        prior_logits_mixed = self._apply_unimix(prior_logits_3d)
        posterior_logits_mixed = self._apply_unimix(posterior_logits_3d)

        prior_3d = prior_logits_mixed.softmax(-1)
        posterior_3d = posterior_logits_mixed.softmax(-1)

        log_prior = torch.log(prior_3d + 1e-8)
        log_posterior = torch.log(posterior_3d + 1e-8)



        kl_dyn_per_class = (posterior_3d.detach() * (log_posterior.detach() - log_prior)).sum(-1)
        kl_dyn = kl_dyn_per_class.sum(-1)
        kl_dyn = torch.clamp(kl_dyn, min=free_nats).mean()


        kl_rep_per_class = (posterior_3d * (log_posterior - log_prior.detach())).sum(-1)
        kl_rep = kl_rep_per_class.sum(-1)
        kl_rep = torch.clamp(kl_rep, min=free_nats).mean()

        return kl_dyn, kl_rep

    def raw_kl(self, prior_logits: torch.Tensor, posterior_logits: torch.Tensor) -> torch.Tensor:
        'Unclamped posterior-to-prior KL per sample for diagnostics.'
        shape = (-1, self.stochastic_classes, self.stochastic_bins)
        prior = self._apply_unimix(prior_logits.view(shape)).softmax(-1)
        posterior = self._apply_unimix(posterior_logits.view(shape)).softmax(-1)
        return (
            posterior * (
                torch.log(posterior + 1e-8) - torch.log(prior + 1e-8)
            )
        ).sum((-1, -2))
