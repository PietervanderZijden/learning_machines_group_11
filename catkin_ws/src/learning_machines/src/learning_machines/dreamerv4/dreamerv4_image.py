"""Image-based DreamerV4 agent with shortcut forcing objective.

Implements the full DreamerV4 paper: "Training Agents Inside of Scalable World Models"
(Hafner et al., 2025) arXiv:2509.24527v1

Key features:
- Shortcut forcing objective (flow matching with self-consistency)
- X-prediction (predict clean latent z₁)
- Ramp loss weight w(τ) = 0.9τ + 0.1
- K=4 sampling steps at inference
- PMPO policy optimization
- Distributional value critic with two-hot loss
"""
from __future__ import annotations
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning_machines.dreamerv4.tokenizer import ImageTokenizer
from learning_machines.dreamerv4.actor_critic import (
    SquashedGaussianActor,
    DistributionalValueCritic,
    MTPRewardHead,
    two_hot_loss,
    logits_to_value,
    ReturnEMA,
    compute_td_lambda_returns,
    symlog,
    symexp,
)
from learning_machines.distributional import _make_bin_centers

# Number of sampling steps at inference (paper uses K=4)
_K_INFERENCE = 4


def finite_clip_grad_norm_(
    parameters, max_norm: float, value_clip: float = 100.0
) -> float:
    """Clip finite gradients without float32 norm-overflow poisoning weights."""
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    for parameter in parameters:
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("non-finite gradient before optimizer step")
    nn.utils.clip_grad_value_(parameters, value_clip)
    norms = [
        torch.linalg.vector_norm(parameter.grad.detach().double())
        for parameter in parameters
    ]
    total_norm = (
        torch.linalg.vector_norm(torch.stack(norms)).item() if norms else 0.0
    )
    if not np.isfinite(total_norm):
        raise FloatingPointError("non-finite gradient norm before optimizer step")
    scale = min(1.0, float(max_norm) / (total_norm + 1e-12))
    if scale < 1.0:
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return float(total_norm)


def _ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """Ramp loss weight: w(τ) = 0.9τ + 0.1.

    Low signal levels contain less learning signal (predicting dataset mean).
    This weight focuses capacity on higher signal levels.
    """
    return 0.9 * tau + 0.1


def _sample_tau_and_d(
    batch_size: int,
    device: torch.device,
    k_max: int = _K_INFERENCE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample signal levels τ and step sizes d for training.

    From the paper:
    - d is sampled uniformly from powers of 2: {1/1, 1/2, 1/4, ..., 1/K_max}
    - τ is sampled uniformly on the grid reached by step size d

    Args:
        batch_size: number of samples
        device: torch device
        k_max: maximum number of sampling steps (defines d_min = 1/k_max)

    Returns:
        tau: (B,) signal levels in [0, 1)
        d: (B,) step sizes in (0, 1]
    """
    # Sample d from powers of 2
    # d ∈ {1/1, 1/2, 1/4, ..., 1/K_max}
    num_levels = int(np.log2(k_max)) + 1  # e.g., for k_max=4: levels 0,1,2 -> d=1, 0.5, 0.25
    d_indices = torch.randint(0, num_levels, (batch_size,), device=device)
    d = 1.0 / (2.0 ** d_indices.float())  # (B,)

    # Sample τ uniformly on grid {0, d, 2d, ..., 1-d}
    # For each sample, τ is a random multiple of d
    num_steps = (1.0 / d).long()  # (B,) number of steps for each d
    tau_indices = torch.zeros(batch_size, device=device)
    for i in range(batch_size):
        n = num_steps[i].item()
        if n > 0:
            tau_indices[i] = torch.randint(0, n, (1,), device=device).float()
    tau = tau_indices * d  # (B,)

    return tau, d


class ImageDynamicsModel(nn.Module):
    """Transformer-based dynamics model with shortcut forcing.

    Predicts clean latents z₁ from corrupted latents z̃_τ.
    Conditioned on signal level τ and step size d for flow matching.

    Architecture:
    - Latent tokens + action tokens + signal level token
    - Causal transformer
    - X-prediction head (predicts clean latent z₁)
    - Reward and done heads
    """

    def __init__(
        self,
        latent_dim: int = 256,
        act_dim: int = 2,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        context_length: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.context_length = context_length
        self.tokens_per_step = 5
        self.register_index = 4

        # Embed latent and action into d_model
        self.latent_embed = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.act_embed = nn.Linear(act_dim, d_model)

        # Signal level τ and step size d embeddings (discrete embedding lookup)
        # τ is discretized into bins, d is discretized into powers of 2
        self.tau_embed = nn.Embedding(8, d_model)  # 8 bins for τ
        self.d_embed = nn.Embedding(8, d_model)  # 8 bins for d (powers of 2)

        max_tokens = (context_length + 1) * self.tokens_per_step
        self.pos_embed = nn.Embedding(max_tokens, d_model)
        self.token_type_embed = nn.Embedding(self.tokens_per_step, d_model)
        self.register_token = nn.Parameter(torch.zeros(d_model))
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer encoder
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

        # X-prediction head: predicts clean latent z₁
        self.latent_head = nn.Linear(d_model, latent_dim)
        # Reward prediction head (two-hot bins)
        self.reward_head = nn.Linear(d_model, 255)
        # Done prediction head
        self.done_head = nn.Linear(d_model, 1)

        # Initialize output heads to zero for stable early training
        nn.init.zeros_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

        self._causal_mask = self._build_block_causal_mask(context_length + 1)

    def _build_block_causal_mask(self, num_steps: int) -> torch.Tensor:
        size = num_steps * self.tokens_per_step
        token_ids = torch.arange(size)
        blocks = token_ids // self.tokens_per_step
        mask = blocks.unsqueeze(1) < blocks.unsqueeze(0)
        return mask.bool()

    def _ensure_mask_size(self, size: int, device: torch.device) -> torch.Tensor:
        if size <= self._causal_mask.shape[0]:
            return self._causal_mask[:size, :size].to(device)
        steps = int(np.ceil(size / self.tokens_per_step))
        return self._build_block_causal_mask(steps)[:size, :size].to(device)

    def _discretize_tau(self, tau: torch.Tensor) -> torch.Tensor:
        """Discretize τ into 8 bins for embedding lookup."""
        # τ ∈ [0, 1] -> bins 0-7
        bins = (tau * 7).long().clamp(0, 7)
        return bins

    def _discretize_d(self, d: torch.Tensor) -> torch.Tensor:
        """Discretize d into bins for embedding lookup."""
        # d ∈ {1, 0.5, 0.25, 0.125, ...} -> bins 0-7
        # Use log2(1/d) to get the index
        indices = torch.log2(1.0 / d.clamp(min=1e-6)).long().clamp(0, 7)
        return indices

    def _make_tokens(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        tau: torch.Tensor,
        d: torch.Tensor,
        start_step: torch.Tensor | int = 0,
    ) -> torch.Tensor:
        """Create [latent, action, signal-level, step-size, register] blocks."""
        B, T, _ = actions.shape
        type_ids = torch.arange(self.tokens_per_step, device=latents.device)

        lat_tok = self.latent_embed(latents[:, :T])
        act_tok = self.act_embed(actions)
        tau_tok = self.tau_embed(self._discretize_tau(tau))
        d_tok = self.d_embed(self._discretize_d(d))
        reg_tok = self.register_token.view(1, 1, -1).expand(B, T, -1)
        tokens = torch.stack([lat_tok, act_tok, tau_tok, d_tok, reg_tok], dim=2)
        tokens = tokens + self.token_type_embed(type_ids).view(1, 1, self.tokens_per_step, -1)

        if isinstance(start_step, int):
            step_ids = torch.arange(T, device=latents.device).view(1, T) + start_step
        else:
            step_ids = start_step.view(B, 1) + torch.arange(T, device=latents.device).view(1, T)
        pos_ids = step_ids.unsqueeze(-1) * self.tokens_per_step + type_ids.view(1, 1, -1)
        pos_ids = pos_ids.clamp(max=self.pos_embed.num_embeddings - 1)
        tokens = tokens + self.pos_embed(pos_ids)
        return tokens.reshape(B, T * self.tokens_per_step, self.d_model)

    def predict_single_step(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        tau: torch.Tensor,
        d: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
        context_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict clean latent z₁ for a single time step.

        Used for bootstrap loss computation where we need to run the model
        at different (τ, d) values for the same latent.

        Args:
            latent: (N, latent_dim) — corrupted latent z̃_τ
            action: (N, act_dim) — action
            tau: (N,) — signal level
            d: (N,) — step size
            context_tokens: (N, C, d_model) — optional context from previous steps

        Returns:
            z1_pred: (N, latent_dim) — predicted clean latent
        """
        N = latent.shape[0]
        device = latent.device

        if context_tokens is not None and context_lengths is None:
            context_lengths = torch.full(
                (N,), context_tokens.shape[1], device=device, dtype=torch.long
            )
        start_step = (
            context_lengths // self.tokens_per_step
            if context_lengths is not None
            else torch.zeros(N, device=device, dtype=torch.long)
        )
        step_tokens = self._make_tokens(
            latent.unsqueeze(1),
            action.unsqueeze(1),
            tau.unsqueeze(1),
            d.unsqueeze(1),
            start_step=start_step,
        )

        if context_tokens is not None:
            tokens = torch.cat([context_tokens, step_tokens], dim=1)
        else:
            tokens = step_tokens

        # Run transformer
        seq_len = tokens.shape[1]
        causal_mask = self._ensure_mask_size(seq_len, device)
        if context_tokens is not None and context_lengths is not None:
            prefix_len = context_tokens.shape[1]
            positions = torch.arange(seq_len, device=device).view(1, -1)
            valid_context = positions[:, :prefix_len] < context_lengths.view(-1, 1)
            valid_current = torch.ones(N, self.tokens_per_step, device=device, dtype=torch.bool)
            key_padding_mask = ~torch.cat([valid_context, valid_current], dim=1)
        else:
            key_padding_mask = None
        hidden = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=key_padding_mask)

        # Get prediction from last token
        last_hidden = hidden[:, -1]
        z1_pred = self.latent_head(last_hidden)

        return z1_pred

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        tau: torch.Tensor | None = None,
        d: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for sequence modeling with shortcut forcing.

        Args:
            latents: (B, T+1, latent_dim) — corrupted latents z̃_τ
            actions: (B, T, act_dim)
            tau: (B, T) or (B,) — signal levels
            d: (B, T) or (B,) — step sizes
            mask: (B, T) — 1 for valid, 0 for padded

        Returns:
            dict with latent_pred (clean z₁), reward_logits, done_pred
        """
        B, Tp1, _ = latents.shape
        T = Tp1 - 1

        # Create signal level and step size tokens
        if tau is not None and d is not None:
            # Expand tau and d to match sequence length
            if tau.dim() == 1:
                tau = tau.unsqueeze(1).expand(-1, T)
            if d.dim() == 1:
                d = d.unsqueeze(1).expand(-1, T)
        else:
            tau = torch.ones(B, T, device=latents.device)
            d = torch.full((B, T), 1.0 / _K_INFERENCE, device=latents.device)

        tokens = self._make_tokens(latents[:, :T], actions, tau, d)
        tokens = self.embed_dropout(tokens)

        causal_mask = self._ensure_mask_size(tokens.shape[1], tokens.device)

        if mask is not None:
            key_padding_mask = ~mask.bool().repeat_interleave(self.tokens_per_step, dim=1)
        else:
            key_padding_mask = None

        hidden = self.transformer(tokens, mask=causal_mask, src_key_padding_mask=key_padding_mask)

        # X-prediction: predict clean latent z₁ from positions 0..T-1
        register_hidden = hidden[:, self.register_index::self.tokens_per_step][:, :T]
        latent_pred = self.latent_head(register_hidden)
        reward_logits = self.reward_head(register_hidden)
        done_pred = self.done_head(register_hidden)

        return {
            "latent_pred": latent_pred,
            "reward_logits": reward_logits,
            "done_pred": done_pred,
            "context_tokens": tokens.detach(),
        }

    def imagine(
        self,
        initial_latent: torch.Tensor,
        actor: nn.Module,
        horizon: int,
    ) -> dict[str, torch.Tensor]:
        """Imagine future trajectories with shortcut forcing inference.

        Uses K=4 sampling steps with step size d=1/4 to generate each frame.
        Past inputs are corrupted to signal level τ_ctx=0.1 for robustness.

        Args:
            initial_latent: (B, latent_dim) — clean initial latent
            actor: callable (latent) -> (action, log_prob)
            horizon: number of steps

        Returns:
            dict with latents, actions, rewards, dones, log_probs
        """
        B = initial_latent.shape[0]
        device = initial_latent.device
        K = _K_INFERENCE  # 4 sampling steps
        d_step = 1.0 / K  # 0.25

        latent_list = [initial_latent]
        act_list = []
        rew_logits_list = []
        done_list = []
        log_prob_list = []

        token_blocks: list[torch.Tensor] = []

        for t in range(horizon):
            h_lat = latent_list[-1].detach()

            act, log_prob = actor(h_lat, deterministic=False)
            act = torch.clamp(act, -1.0, 1.0)

            act_list.append(act)
            log_prob_list.append(log_prob)

            # Create token with context corruption (τ_ctx = 0.1)
            # This makes the model robust to small imperfections
            tau_ctx = 0.1
            noise = torch.randn_like(h_lat)
            h_lat_corrupted = (1 - tau_ctx) * noise + tau_ctx * h_lat

            tau_ctx_tensor = torch.full((B,), tau_ctx, device=device)
            d_ctx_tensor = torch.full((B,), d_step, device=device)
            block = self._make_tokens(
                h_lat_corrupted.unsqueeze(1),
                act.unsqueeze(1),
                tau_ctx_tensor.unsqueeze(1),
                d_ctx_tensor.unsqueeze(1),
                start_step=t,
            )
            token_blocks.append(block)

            # Stack all tokens
            tokens = torch.cat(token_blocks, dim=1)
            tokens = self.embed_dropout(tokens)

            causal_mask = self._ensure_mask_size(tokens.shape[1], device)
            hidden = self.transformer(tokens, mask=causal_mask)

            last_hidden = hidden[:, -1]

            # Shortcut forcing inference: K=4 sampling steps
            # Start from noise, iteratively denoise to get clean latent
            z_noise = torch.randn(B, self.latent_dim, device=device)
            z_current = z_noise

            for step in range(K):
                tau_step = step * d_step
                tau_tensor = torch.full((B,), tau_step, device=device)

                # Corrupt current estimate
                z_corrupted = (1 - tau_step) * z_noise + tau_step * z_current

                context_lengths = torch.full(
                    (B,), tokens.shape[1], device=device, dtype=torch.long
                )
                z_pred = self.predict_single_step(
                    z_corrupted,
                    act,
                    tau_tensor,
                    d_ctx_tensor,
                    context_tokens=tokens,
                    context_lengths=context_lengths,
                )

                # X-prediction to velocity update: x_{tau+d} = x_tau + v d.
                velocity = (z_pred - z_corrupted) / (1.0 - tau_step + 1e-6)
                z_current = z_corrupted + velocity * d_step

            # Clamp latent to prevent runaway
            next_latent = torch.clamp(z_current, -5.0, 5.0)

            # Get reward and done from final hidden state
            rew_logits = self.reward_head(last_hidden)
            done = self.done_head(last_hidden)

            latent_list.append(next_latent)
            rew_logits_list.append(rew_logits)
            done_list.append(done)

        return {
            "latents": torch.stack(latent_list, dim=1),  # (B, H+1, latent_dim)
            "actions": torch.stack(act_list, dim=1),  # (B, H, act_dim)
            "reward_logits": torch.stack(rew_logits_list, dim=1),  # (B, H, 255)
            "dones": torch.cat(done_list, dim=1),  # (B, H)
            "log_probs": torch.stack(log_prob_list, dim=1),  # (B, H, 1)
        }


class ImageDreamerV4Agent(nn.Module):
    """DreamerV4 agent with shortcut forcing objective.

    Implements the full DreamerV4 architecture:
    - CNN tokenizer with masked autoencoding
    - Transformer dynamics with shortcut forcing
    - PMPO policy optimization
    - Distributional value critic

    Args:
        obs_dim: observation dimension (for compatibility)
        act_dim: action dimension
        latent_dim: latent space dimension
        d_model: transformer hidden dimension
        n_heads: number of attention heads
        n_layers: number of transformer layers
        ff_dim: feedforward dimension
        dropout: dropout rate
        context_length: maximum sequence length
        imagination_horizon: steps to imagine during training
        lr: learning rate
        grad_clip: gradient clipping
        gamma: discount factor
        lam: TD-lambda
        entropy_coef: entropy regularization coefficient
        pmpo_alpha: PMPO balance between positive/negative advantages
        pmpo_prior_beta: PMPO behavioral prior weight
        target_update_rate: soft target update rate
        image_size: input image size
        ir_dim: IR sensor dimension (0 for no IR)
    """

    def __init__(
        self,
        obs_dim: int = 12,
        act_dim: int = 2,
        latent_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        context_length: int = 64,
        imagination_horizon: int = 16,
        lr: float = 1e-4,
        grad_clip: float = 1.0,
        gamma: float = 0.994009,
        lam: float = 0.95,
        entropy_coef: float = 0.001,
        pmpo_alpha: float = 0.5,
        pmpo_prior_beta: float = 0.3,
        target_update_rate: float = 0.01,
        image_size: int = 64,
        ir_dim: int = 0,
        mtp_length: int = 8,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.latent_dim = latent_dim
        self.ir_dim = ir_dim
        self.imagination_horizon = imagination_horizon
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.pmpo_alpha = pmpo_alpha
        self.pmpo_prior_beta = pmpo_prior_beta
        self.target_update_rate = target_update_rate
        self.mtp_length = mtp_length

        # Tokenizer (with optional IR support)
        self.tokenizer = ImageTokenizer(latent_dim=latent_dim, image_size=image_size, ir_dim=ir_dim)
        combined_latent_dim = self.tokenizer.combined_dim
        self.state_dim = combined_latent_dim

        # Dynamics model with shortcut forcing
        self.dynamics = ImageDynamicsModel(
            latent_dim=combined_latent_dim,
            act_dim=act_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            context_length=context_length,
        )

        # Actor (policy head)
        self.actor = SquashedGaussianActor(
            obs_dim=combined_latent_dim,
            act_dim=act_dim,
            hidden_dim=d_model,
            mtp_length=mtp_length,
        )

        # Behavioral prior (frozen copy of actor for PMPO)
        self.prior_actor = SquashedGaussianActor(
            obs_dim=combined_latent_dim,
            act_dim=act_dim,
            hidden_dim=d_model,
            mtp_length=mtp_length,
        )
        self.prior_actor.load_state_dict(self.actor.state_dict())
        self.mtp_feature = nn.Sequential(
            nn.Linear(combined_latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        self.mtp_reward_head = MTPRewardHead(d_model=d_model, mtp_length=mtp_length)

        # Value critic
        self.critic = DistributionalValueCritic(
            state_dim=combined_latent_dim,
            hidden_dim=d_model,
        )
        self.target_critic = DistributionalValueCritic(
            state_dim=combined_latent_dim,
            hidden_dim=d_model,
        )
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.tokenizer_optimizer = torch.optim.AdamW(
            self.tokenizer.parameters(), lr=lr, weight_decay=0.01
        )
        self.dynamics_optimizer = torch.optim.AdamW(
            self.dynamics.parameters(), lr=lr, weight_decay=0.01
        )
        self.ac_optimizer = torch.optim.AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr, weight_decay=0.01,
        )
        self.mtp_optimizer = torch.optim.AdamW(
            list(self.prior_actor.parameters())
            + list(self.mtp_feature.parameters())
            + list(self.mtp_reward_head.parameters()),
            lr=lr, weight_decay=0.01,
        )

        self.return_ema = ReturnEMA(decay=0.99)
        self.register_buffer("dyn_latent_loss_rms", torch.tensor(1.0))
        self.register_buffer("dyn_reward_loss_rms", torch.tensor(1.0))
        self.register_buffer("dyn_done_loss_rms", torch.tensor(1.0))

    def _normalize_loss(self, loss: torch.Tensor, rms_name: str) -> torch.Tensor:
        rms = getattr(self, rms_name)
        with torch.no_grad():
            rms.mul_(0.99).add_(0.01 * loss.detach().square().clamp_min(1e-12).sqrt())
        return loss / rms.detach().clamp_min(1e-6)

    def _has_nan(self, *tensors) -> bool:
        for t in tensors:
            if isinstance(t, torch.Tensor):
                if torch.isnan(t).any() or torch.isinf(t).any():
                    return True
        return False

    def update_tokenizer(self, images: torch.Tensor, ir: torch.Tensor | None = None) -> dict[str, float]:
        """Train tokenizer with masked autoencoding.

        Args:
            images: (B, 3, H, W) float in [0, 1]
            ir: (B, ir_dim) optional IR data for multi-modal
        """
        self.tokenizer_optimizer.zero_grad()

        result = self.tokenizer(images, ir)
        loss = result["loss"]

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite tokenizer loss")
        loss.backward()
        grad_norm = finite_clip_grad_norm_(
            self.tokenizer.parameters(), self.grad_clip
        )
        self.tokenizer_optimizer.step()

        return {
            "tok/loss": loss.item(),
            "tok/mse_loss": result["mse_loss"].item(),
            "tok/perceptual_loss": result["perceptual_loss"].item(),
            "tok/food_saliency_loss": result["food_saliency_loss"].item(),
            "tok/ir_loss": result["ir_loss"].item(),
            "tok/perceptual_metric": result["perceptual_metric"],
            "tok/grad_norm": float(grad_norm),
        }

    def update_dynamics(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        """Train dynamics model with shortcut forcing objective.

        Implements the shortcut forcing loss from the paper:
        1. Sample τ and d
        2. Corrupt latents: z̃ = (1-τ)z₀ + τz₁
        3. Predict clean z₁ (x-prediction)
        4. Flow matching loss at d = d_min
        5. Bootstrap loss with self-consistency for d > d_min
        6. Ramp loss weight w(τ) = 0.9τ + 0.1

        Args:
            latents: (B, T+1, latent_dim) — clean latents from tokenizer
            actions: (B, T, act_dim)
            rewards: (B, T) — scalar rewards
            dones: (B, T) — done flags
            mask: (B, T) — 1 for valid, 0 for padded

        Returns:
            dict with loss metrics
        """
        self.dynamics_optimizer.zero_grad()

        B, Tp1, latent_dim = latents.shape
        T = Tp1 - 1
        device = latents.device

        # Sample τ and d for each sequence element
        tau, d = _sample_tau_and_d(B * T, device)
        tau = tau.reshape(B, T)
        d = d.reshape(B, T)

        # Corrupt latents: z̃_τ = (1-τ)z₀ + τz₁
        # z₀ is noise, z₁ is clean latent
        z0 = torch.randn_like(latents[:, 1:])  # noise
        z1 = latents[:, 1:]  # clean target

        # Expand tau to match latent dimensions
        tau_expanded = tau.unsqueeze(-1)  # (B, T, 1)
        z_tilde = (1 - tau_expanded) * z0 + tau_expanded * z1  # (B, T, latent_dim)

        # Create corrupted sequence: [z̃_0, z̃_1, ..., z̃_{T-1}, z_T]
        # The last latent is clean (used as context)
        corrupted_latents = torch.cat([z_tilde, latents[:, T:T+1]], dim=1)

        # Forward pass: predict clean z₁
        preds = self.dynamics(corrupted_latents, actions, tau, d, mask)

        # X-prediction: model predicts clean z₁
        z1_pred = preds["latent_pred"]  # (B, T, latent_dim)

        # Determine which samples use flow matching vs bootstrap
        d_min = 1.0 / _K_INFERENCE  # 0.25
        is_flow = (d <= d_min + 1e-6)  # flow matching for smallest step size
        is_bootstrap = ~is_flow

        # Flow matching loss: ||ẑ₁ - z₁||²
        flow_loss = F.mse_loss(z1_pred, z1, reduction="none")  # (B, T, latent_dim)
        flow_loss = flow_loss.mean(-1)  # (B, T)

        # Bootstrap loss with self-consistency
        # For d > d_min: compute two half-steps and use their average as target
        # This requires separate forward passes for the two half-steps
        bootstrap_loss = torch.zeros_like(flow_loss)

        if is_bootstrap.any():
            # Get indices of bootstrap samples
            bootstrap_idx = is_bootstrap.nonzero(as_tuple=True)
            if len(bootstrap_idx[0]) > 0:
                # Get tau and d for bootstrap samples
                tau_b = tau[bootstrap_idx]  # (N,)
                d_b = d[bootstrap_idx]  # (N,)
                z_tilde_b = z_tilde[bootstrap_idx]  # (N, latent_dim)
                z1_b = z1[bootstrap_idx]  # (N, latent_dim)
                actions_b = actions[bootstrap_idx]  # (N, act_dim) — actions for these samples
                batch_idx, time_idx = bootstrap_idx
                context_lengths = (time_idx * self.dynamics.tokens_per_step).long()
                max_context_len = int(context_lengths.max().item()) if context_lengths.numel() else 0
                all_context = preds["context_tokens"][batch_idx]
                context_tokens = all_context[:, :max_context_len] if max_context_len > 0 else all_context[:, :0]

                # First half-step: predict clean from z̃ at (τ, d/2)
                d_half = d_b / 2
                tau_mid = tau_b + d_half

                # Predict clean latent at first half-step
                # f(z̃, τ, d/2) predicts clean z₁ given corrupted z̃ at signal level τ with step d/2
                z1_pred_first = self.dynamics.predict_single_step(
                    z_tilde_b,
                    actions_b,
                    tau_b,
                    d_half,
                    context_tokens=context_tokens,
                    context_lengths=context_lengths,
                )  # (N, latent_dim)

                # Compute velocity: b' = (ẑ₁ - z̃) / (1-τ)
                b_prime = (z1_pred_first - z_tilde_b) / (1 - tau_b + 1e-6).unsqueeze(-1)

                # Intermediate point: z' = z̃ + b' * d/2
                z_prime = z_tilde_b + b_prime * d_half.unsqueeze(-1)

                # Second half-step: predict clean from z' at (τ+d/2, d/2)
                # f(z', τ+d/2, d/2) predicts clean z₁ given z' at signal level τ+d/2 with step d/2
                z1_pred_second = self.dynamics.predict_single_step(
                    z_prime,
                    actions_b,
                    tau_mid,
                    d_half,
                    context_tokens=context_tokens,
                    context_lengths=context_lengths,
                )  # (N, latent_dim)

                # Compute velocity: b'' = (ẑ₁ - z') / (1-(τ+d/2))
                b_double_prime = (z1_pred_second - z_prime) / (1 - tau_mid + 1e-6).unsqueeze(-1)

                # Bootstrap target: sg((b' + b'') / 2)
                # This is the self-consistency target: one big step = average of two small steps
                v_target = (b_prime + b_double_prime) / 2

                # Convert to x-space: target = z̃ + v_target * (1-τ)
                z1_bootstrap_target = z_tilde_b + v_target * (1 - tau_b).unsqueeze(-1)

                # Bootstrap loss: ||ẑ₁ - target||²
                bootstrap_loss_sample = F.mse_loss(z1_pred[bootstrap_idx], z1_bootstrap_target.detach(), reduction="none")
                bootstrap_loss_sample = bootstrap_loss_sample.mean(-1)  # (N,)

                bootstrap_loss[bootstrap_idx] = bootstrap_loss_sample

        # Apply ramp loss weight: w(τ) = 0.9τ + 0.1
        weights = _ramp_weight(tau)  # (B, T)

        # Combine losses
        flow_loss_weighted = (flow_loss * weights * is_flow.float() * mask).sum() / (is_flow.float() * mask).sum().clamp(min=1)
        bootstrap_loss_weighted = (bootstrap_loss * weights * is_bootstrap.float() * mask).sum() / (is_bootstrap.float() * mask).sum().clamp(min=1)

        latent_loss = flow_loss_weighted + bootstrap_loss_weighted

        # Reward prediction loss (two-hot)
        rew_logits = preds["reward_logits"]  # (B, T, 255)
        rew_flat = rew_logits.reshape(-1, 255)
        rew_target = rewards.reshape(-1)
        rew_loss_per_step = two_hot_loss(rew_flat, rew_target, reduction="none").reshape(B, T)
        rew_loss = (rew_loss_per_step * mask).sum() / mask.sum().clamp(min=1)

        # Done loss
        done_pred = preds["done_pred"].squeeze(-1)
        done_target = dones[..., :done_pred.shape[-1]].float()
        done_loss = F.binary_cross_entropy_with_logits(done_pred, done_target, reduction="none")
        done_loss = (done_loss * mask).sum() / mask.sum().clamp(min=1)

        total_loss = (
            self._normalize_loss(latent_loss, "dyn_latent_loss_rms")
            + self._normalize_loss(rew_loss, "dyn_reward_loss_rms")
            + self._normalize_loss(done_loss, "dyn_done_loss_rms")
        )
        if not all(
            torch.isfinite(value)
            for value in (latent_loss, rew_loss, done_loss, total_loss)
        ):
            raise FloatingPointError(
                "non-finite DreamerV4 dynamics loss before backward"
            )
        total_loss.backward()
        grad_norm = finite_clip_grad_norm_(
            self.dynamics.parameters(), self.grad_clip
        )
        self.dynamics_optimizer.step()

        return {
            "dyn/latent_loss": latent_loss.item(),
            "dyn/flow_loss": flow_loss_weighted.item(),
            "dyn/bootstrap_loss": bootstrap_loss_weighted.item(),
            "dyn/rew_loss": rew_loss.item(),
            "dyn/done_loss": done_loss.item(),
            "dyn/total_loss": total_loss.item(),
            "dyn/grad_norm": float(grad_norm),
            "dyn/latent_input_mean": latents.mean().item(),
            "dyn/latent_input_std": latents.std().item(),
            "dyn/latent_input_abs_max": latents.abs().max().item(),
            "dyn/action_input_abs_max": actions.abs().max().item(),
            "dyn/reward_target_abs_max": rewards.abs().max().item(),
        }

    def update_mtp_behavior(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        """Pretrain the frozen behavior prior with L-step action and reward prediction."""
        self.mtp_optimizer.zero_grad()
        B, T, _ = actions.shape

        action_losses = []
        reward_losses = []
        for k in range(self.mtp_length):
            valid_len = T - k
            if valid_len <= 0:
                break
            valid_mask = mask[:, :valid_len]
            states = latents[:, :valid_len].reshape(-1, self.state_dim)
            target_actions = actions[:, k:k + valid_len].reshape(-1, self.act_dim)
            target_rewards = rewards[:, k:k + valid_len].reshape(-1)
            valid_flat = valid_mask.reshape(-1)

            logp = self.prior_actor.log_prob(states, target_actions, mtp_step=k).squeeze(-1)
            action_loss = -(logp * valid_flat).sum() / valid_flat.sum().clamp(min=1)

            features = self.mtp_feature(states)
            reward_logits = self.mtp_reward_head(features, mtp_step=k)
            reward_loss_flat = two_hot_loss(reward_logits, target_rewards, reduction="none")
            reward_loss = (reward_loss_flat * valid_flat).sum() / valid_flat.sum().clamp(min=1)

            action_losses.append(action_loss)
            reward_losses.append(reward_loss)

        if not action_losses:
            return {"mtp/action_loss": 0.0, "mtp/reward_loss": 0.0, "mtp/total_loss": 0.0}

        action_loss = torch.stack(action_losses).mean()
        reward_loss = torch.stack(reward_losses).mean()
        total_loss = action_loss + reward_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite MTP loss before backward")
        total_loss.backward()
        grad_norm = finite_clip_grad_norm_(
            list(self.prior_actor.parameters())
            + list(self.mtp_feature.parameters())
            + list(self.mtp_reward_head.parameters()),
            self.grad_clip,
        )
        self.mtp_optimizer.step()

        return {
            "mtp/action_loss": action_loss.item(),
            "mtp/reward_loss": reward_loss.item(),
            "mtp/total_loss": total_loss.item(),
            "mtp/grad_norm": float(grad_norm),
        }

    def update_actor_critic(
        self,
        initial_latents: torch.Tensor,
    ) -> dict[str, float]:
        """Train actor and critic via imagination using PMPO.

        PMPO uses sign of advantages (not magnitude) and balances
        positive and negative sets equally.
        """
        self.ac_optimizer.zero_grad()

        frozen_modules = [self.tokenizer, self.dynamics, self.prior_actor]
        prev_requires_grad = {
            module: [p.requires_grad for p in module.parameters()]
            for module in frozen_modules
        }
        for module in frozen_modules:
            for p in module.parameters():
                p.requires_grad_(False)

        # Imagine trajectories
        imagined = self.dynamics.imagine(
            initial_latents.detach(),
            self.actor,
            horizon=self.imagination_horizon,
        )

        imag_latents = imagined["latents"]  # (B, H+1, latent_dim)
        imag_rew_logits = imagined["reward_logits"]  # (B, H, 255)
        imag_done_logits = imagined["dones"]  # (B, H)
        imag_continue = 1.0 - torch.sigmoid(imag_done_logits)
        imag_log_prob = imagined["log_probs"]  # (B, H, 1)

        B, H = imag_done_logits.shape

        # Reward/value: logits_to_value handles symexp internally
        imag_rewards = logits_to_value(imag_rew_logits)  # (B, H)

        # Get critic values for imagined latents (positions 1..H)
        imag_lat_for_critic = imag_latents[:, 1:H + 1].reshape(-1, self.state_dim)
        imag_logits = self.critic(imag_lat_for_critic).reshape(B, H, -1)
        imag_values = logits_to_value(imag_logits)  # (B, H)

        # Bootstrap value
        with torch.no_grad():
            last_lat = imag_latents[:, -1]
            bootstrap_logits = self.target_critic(last_lat)
            bootstrap = logits_to_value(bootstrap_logits).unsqueeze(1)  # (B, 1)

        # Lambda returns
        values_with_bootstrap = torch.cat([imag_values, bootstrap], dim=1)[:, :H + 1]
        returns = compute_td_lambda_returns(
            imag_rewards,
            values_with_bootstrap,
            bootstrap,
            torch.zeros_like(imag_done_logits),
            self.gamma,
            self.lam,
            continuations=imag_continue,
        )

        # Critic loss: two-hot distributional loss
        # two_hot_loss handles symlog internally — pass RAW returns
        returns_flat = returns.reshape(-1)
        logits_flat = imag_logits.reshape(-1, imag_logits.shape[-1])
        critic_loss = two_hot_loss(logits_flat, returns_flat)

        # PMPO policy loss
        # Paper: uses sign of advantages, balances positive/negative sets
        advantages = (returns - imag_values).detach()  # (B, H)
        self.return_ema.update(returns)
        adv_flat = advantages.reshape(-1)
        log_prob_flat = imag_log_prob.squeeze(-1).reshape(-1)

        # Split into positive and negative advantage sets
        pos_mask = adv_flat >= 0
        neg_mask = ~pos_mask

        # Policy loss: increase likelihood of positive-advantage actions,
        # decrease likelihood of negative-advantage actions
        if pos_mask.sum() > 0:
            pos_loss = -log_prob_flat[pos_mask].mean()  # maximize likelihood
        else:
            pos_loss = torch.tensor(0.0, device=adv_flat.device)

        if neg_mask.sum() > 0:
            neg_loss = +log_prob_flat[neg_mask].mean()  # minimize likelihood
        else:
            neg_loss = torch.tensor(0.0, device=adv_flat.device)

        # Prior KL: KL[π_θ(a|s) || π_prior(a|s)]
        # Use current actor's actions, compute log-prob ratio with prior
        with torch.no_grad():
            prior_lat = imag_latents[:, 1:H + 1].detach()
        # Get actions from current actor (already sampled during imagination)
        imag_actions = imagined["actions"]  # (B, H, act_dim)
        imag_actions_flat = imag_actions.reshape(-1, self.act_dim)

        # Compute log-probs under current and prior policy for the SAME actions
        logp_current = self.actor.log_prob(
            prior_lat.reshape(-1, self.state_dim),
            imag_actions_flat,
        ).squeeze(-1)
        with torch.no_grad():
            logp_prior = self.prior_actor.log_prob(
                prior_lat.reshape(-1, self.state_dim),
                imag_actions_flat,
            ).squeeze(-1)
        prior_kl = (logp_current - logp_prior).mean()

        actor_loss = self.pmpo_alpha * pos_loss + (1 - self.pmpo_alpha) * neg_loss
        actor_loss = actor_loss + self.pmpo_prior_beta * prior_kl
        actor_loss = actor_loss + self.entropy_coef * logp_current.mean()

        total_loss = critic_loss + actor_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("non-finite actor-critic loss before backward")
        total_loss.backward()
        grad_norm = finite_clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            self.grad_clip,
        )

        self.ac_optimizer.step()
        self._soft_update_target()
        for module in frozen_modules:
            for p, req in zip(module.parameters(), prev_requires_grad[module]):
                p.requires_grad_(req)

        return {
            "ac/critic_loss": critic_loss.item(),
            "ac/actor_loss": actor_loss.item(),
            "ac/return_mean": returns.mean().item(),
            "ac/return_range": self.return_ema.range,
            "ac/continue_mean": imag_continue.mean().item(),
            "ac/actor_std": self.actor.std(
                prior_lat.reshape(-1, self.state_dim)
            ).mean().item(),
            "ac/action_saturation": (imag_actions.abs() >= 0.99).float().mean().item(),
            "ac/advantage_p05": torch.quantile(advantages, 0.05).item(),
            "ac/advantage_p50": torch.quantile(advantages, 0.50).item(),
            "ac/advantage_p95": torch.quantile(advantages, 0.95).item(),
            "ac/grad_norm": float(grad_norm),
            "ac/actor_entropy": float(-logp_current.mean().item()),
        }

    def _soft_update_target(self):
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.lerp_(p.data, self.target_update_rate)

    def sync_prior(self):
        """Sync behavioral prior with current actor."""
        self.prior_actor.load_state_dict(self.actor.state_dict())

    def freeze_behavior_prior(self, copy_to_actor: bool = True):
        """Freeze the MTP-trained behavior prior before PMPO."""
        if copy_to_actor:
            self.actor.load_state_dict(self.prior_actor.state_dict())
        for p in self.prior_actor.parameters():
            p.requires_grad_(False)

    def act(self, latent: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Select action given latent state."""
        with torch.no_grad():
            action, _ = self.actor(latent, deterministic=deterministic)
        return action

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "config": {
                "act_dim": self.act_dim,
                "latent_dim": self.latent_dim,
                "ir_dim": self.ir_dim,
                "imagination_horizon": self.imagination_horizon,
                "grad_clip": self.grad_clip,
                "gamma": self.gamma,
                "lam": self.lam,
                "entropy_coef": self.entropy_coef,
                "pmpo_alpha": self.pmpo_alpha,
                "pmpo_prior_beta": self.pmpo_prior_beta,
                "target_update_rate": self.target_update_rate,
                "mtp_length": self.mtp_length,
                "d_model": self.dynamics.d_model,
                "n_heads": self.dynamics.transformer.layers[0].self_attn.num_heads,
                "n_layers": len(self.dynamics.transformer.layers),
                "ff_dim": self.dynamics.transformer.layers[0].linear1.out_features,
                "dropout": self.dynamics.embed_dropout.p,
                "context_length": self.dynamics.context_length,
                "lr": self.tokenizer_optimizer.param_groups[0]["lr"],
                "image_size": self.tokenizer.decoder.image_size,
            },
            "tokenizer": self.tokenizer.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "prior_actor": self.prior_actor.state_dict(),
            "mtp_feature": self.mtp_feature.state_dict(),
            "mtp_reward_head": self.mtp_reward_head.state_dict(),
            "tokenizer_optimizer": self.tokenizer_optimizer.state_dict(),
            "dynamics_optimizer": self.dynamics_optimizer.state_dict(),
            "ac_optimizer": self.ac_optimizer.state_dict(),
            "mtp_optimizer": self.mtp_optimizer.state_dict(),
            "return_ema_range": self.return_ema.range,
            "dyn_latent_loss_rms": self.dyn_latent_loss_rms,
            "dyn_reward_loss_rms": self.dyn_reward_loss_rms,
            "dyn_done_loss_rms": self.dyn_done_loss_rms,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path: str, device: torch.device = torch.device("cpu")) -> "ImageDreamerV4Agent":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        cfg = checkpoint.get("config", {})
        agent = cls(
            act_dim=cfg.get("act_dim", 2),
            latent_dim=cfg.get("latent_dim", 256),
            ir_dim=cfg.get("ir_dim", 0),
            d_model=cfg.get("d_model", 256),
            n_heads=cfg.get("n_heads", 8),
            n_layers=cfg.get("n_layers", 4),
            ff_dim=cfg.get("ff_dim", 1024),
            dropout=cfg.get("dropout", 0.1),
            context_length=cfg.get("context_length", 64),
            imagination_horizon=cfg.get("imagination_horizon", 16),
            lr=cfg.get("lr", 1e-4),
            grad_clip=cfg.get("grad_clip", 1.0),
            gamma=cfg.get("gamma", 0.994009),
            lam=cfg.get("lam", 0.95),
            entropy_coef=cfg.get("entropy_coef", 0.001),
            pmpo_alpha=cfg.get("pmpo_alpha", 0.5),
            pmpo_prior_beta=cfg.get("pmpo_prior_beta", 0.3),
            target_update_rate=cfg.get("target_update_rate", 0.01),
            image_size=cfg.get("image_size", 64),
            mtp_length=cfg.get("mtp_length", 8),
        )
        agent.to(device)
        agent.tokenizer.load_state_dict(checkpoint["tokenizer"])
        agent.dynamics.load_state_dict(checkpoint["dynamics"])
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.target_critic.load_state_dict(checkpoint["target_critic"])
        agent.prior_actor.load_state_dict(checkpoint["prior_actor"])
        if "mtp_feature" in checkpoint:
            agent.mtp_feature.load_state_dict(checkpoint["mtp_feature"])
        if "mtp_reward_head" in checkpoint:
            agent.mtp_reward_head.load_state_dict(checkpoint["mtp_reward_head"])
        agent.tokenizer_optimizer.load_state_dict(checkpoint["tokenizer_optimizer"])
        agent.dynamics_optimizer.load_state_dict(checkpoint["dynamics_optimizer"])
        agent.ac_optimizer.load_state_dict(checkpoint["ac_optimizer"])
        if "mtp_optimizer" in checkpoint:
            agent.mtp_optimizer.load_state_dict(checkpoint["mtp_optimizer"])
        agent.return_ema.range = checkpoint.get("return_ema_range", 1.0)
        agent.dyn_latent_loss_rms.copy_(checkpoint.get("dyn_latent_loss_rms", torch.tensor(1.0, device=device)))
        agent.dyn_reward_loss_rms.copy_(checkpoint.get("dyn_reward_loss_rms", torch.tensor(1.0, device=device)))
        agent.dyn_done_loss_rms.copy_(checkpoint.get("dyn_done_loss_rms", torch.tensor(1.0, device=device)))
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        return agent
