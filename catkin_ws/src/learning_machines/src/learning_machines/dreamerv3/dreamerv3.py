"""DreamerV3 agent: ties world model, actor, critic, and training logic together.

Implements the full DreamerV3 training loop per the paper (arXiv:2301.04104):
- World model: RSSM + encoder + decoder + reward + continue predictors
- Actor: REINFORCE with return normalization
- Critic: Distributional (two-hot) with lambda-returns

Key fixes:
- Uses shared distributional utilities (symlog bins, searchsorted two-hot)
- Stores log-probs during imagination (not resampled)
- Freezes world model during actor-critic updates
- Proper select_action with previous action
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning_machines.distributional import (
    symlog, symexp, two_hot_loss, logits_to_value,
)
from .actor_critic import (
    Actor,
    Critic,
    ReturnNormalizer,
    lambda_return,
    soft_cross_entropy,
)
from .config import DreamerV3Config
from .optim import LaProp, adaptive_clip_grad_
from .replay_buffer import ReplayBuffer
from .rssm import RSSM
from .world_model import WorldModel


def _cpu_byte_rng_states(states) -> list[torch.Tensor]:
    """Normalize saved CUDA RNG states for torch.cuda.set_rng_state_all."""
    return [
        state.detach().to(device="cpu", dtype=torch.uint8)
        if isinstance(state, torch.Tensor)
        else torch.as_tensor(state, dtype=torch.uint8, device="cpu")
        for state in states
    ]


def green_saliency_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Preserve small green food regions that global image MSE can ignore."""
    pred_green = torch.relu(
        prediction[..., 1, :, :]
        - torch.maximum(prediction[..., 0, :, :], prediction[..., 2, :, :])
    )
    target_green = torch.relu(
        target[..., 1, :, :]
        - torch.maximum(target[..., 0, :, :], target[..., 2, :, :])
    )
    weights = 1.0 + 20.0 * (target_green > 0.1).float()
    return ((pred_green - target_green).square() * weights).sum() / weights.sum()


def select_imagination_starts(
    state_sequence: torch.Tensor,
    rewards: torch.Tensor,
    count: int,
    reward_threshold: float,
) -> torch.Tensor:
    """Select replay states while ensuring food-event states are represented."""
    batch, transitions_plus_one, state_dim = state_sequence.shape
    transitions = transitions_plus_one - 1
    candidates = state_sequence[:, :transitions]
    if count <= 0 or count >= transitions:
        return candidates.reshape(batch * transitions, state_dim)

    base = torch.linspace(
        0, transitions - 1, count, device=state_sequence.device
    ).round().long()
    indices = base.unsqueeze(0).expand(batch, -1).clone()
    for batch_index in range(batch):
        events = torch.nonzero(
            rewards[batch_index] >= reward_threshold, as_tuple=False
        ).flatten()
        if events.numel():
            retained = events[-count:]
            indices[batch_index, : retained.numel()] = retained
    gather_index = indices.unsqueeze(-1).expand(-1, -1, state_dim)
    return candidates.gather(1, gather_index).reshape(batch * count, state_dim)


class DreamerV3:
    """DreamerV3 agent.

    Training loop:
      1. Collect experience in real environment
      2. Sample sequences from replay buffer
      3. Train world model (encoder + RSSM + decoder + reward + continue)
      4. Imagine trajectories using world model + actor
      5. Train actor via REINFORCE with return normalization
      6. Train critic via two-hot distributional loss on lambda-returns
    """

    def __init__(self, cfg: DreamerV3Config, device: str = "auto"):
        self.cfg = cfg
        self.device = torch.device(device) if device != "auto" else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # World model
        self.world_model = WorldModel(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            deterministic_size=cfg.deterministic_size,
            stochastic_classes=cfg.stochastic_classes,
            stochastic_bins=cfg.stochastic_bins,
            hidden_size=cfg.hidden_size,
            embed_size=cfg.embed_size,
            mlp_hidden=cfg.mlp_hidden,
            mlp_layers=cfg.mlp_layers,
            use_images=cfg.use_images,
            use_multimodal=cfg.use_multimodal,
            image_size=cfg.image_size,
            ir_dim=cfg.ir_dim,
        ).to(self.device)

        rssm_state_size = cfg.deterministic_size + self.world_model.rssm.stochastic_size

        # Actor and Critic
        self.actor = Actor(
            state_dim=rssm_state_size,
            action_dim=cfg.action_dim,
            hidden=cfg.actor_hidden,
            layers=cfg.actor_layers,
            std_min=getattr(
                cfg,
                "actor_std_min",
                float(np.exp(getattr(cfg, "actor_log_std_min", -2.302585))),
            ),
            std_max=getattr(
                cfg,
                "actor_std_max",
                float(np.exp(getattr(cfg, "actor_log_std_max", 0.0))),
            ),
            mean_limit=getattr(cfg, "actor_mean_limit", 100.0),
        ).to(self.device)

        self.critic = Critic(
            state_dim=rssm_state_size,
            hidden=cfg.critic_hidden,
            layers=cfg.critic_layers,
        ).to(self.device)

        self.critic_target = Critic(
            state_dim=rssm_state_size,
            hidden=cfg.critic_hidden,
            layers=cfg.critic_layers,
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Return normalizer for actor loss
        self.return_normalizer = ReturnNormalizer(decay=0.99)

        # Paper-style LaProp is the default. Adam remains available for old
        # experiments and checkpoint migration.
        self.optimizer_name = getattr(cfg, "optimizer", "laprop").lower()
        if self.optimizer_name == "laprop":
            optimizer = LaProp
            optimizer_kwargs = {
                "eps": getattr(cfg, "optimizer_eps", 1e-20),
                "warmup_steps": getattr(cfg, "optimizer_warmup", 1000),
            }
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam
            optimizer_kwargs = {"eps": getattr(cfg, "optimizer_eps", 1e-5)}
        else:
            raise ValueError(
                f"Unsupported DreamerV3 optimizer: {self.optimizer_name}"
            )
        self.world_optim = optimizer(
            self.world_model.parameters(), lr=cfg.world_lr, **optimizer_kwargs
        )
        self.actor_optim = optimizer(
            self.actor.parameters(), lr=cfg.actor_lr, **optimizer_kwargs
        )
        self.critic_optim = optimizer(
            self.critic.parameters(), lr=cfg.critic_lr, **optimizer_kwargs
        )

        # Replay buffer - use image shape if using images
        if cfg.use_multimodal:
            obs_shape = (3, cfg.image_size, cfg.image_size)
        elif cfg.use_images:
            obs_shape = (3, cfg.image_size, cfg.image_size)
        else:
            obs_shape = (cfg.obs_dim,)

        self.buffer = ReplayBuffer(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            capacity=cfg.buffer_capacity,
            sequence_length=cfg.sequence_length,
            obs_shape=obs_shape,
            ir_dim=cfg.ir_dim if cfg.use_multimodal else 0,
            reward_event_fraction=cfg.reward_event_fraction,
            reward_event_threshold=cfg.reward_event_threshold,
        )

        # Training stats
        self._global_step = 0
        self._train_calls = 0
        self._prev_action = None

    def initial_state(self, batch_size: int = 1):
        return self.world_model.initial_state(batch_size, self.device)

    def reset_policy_state(self):
        """Reset action history used by online RSSM filtering."""
        self._prev_action = None

    def set_executed_action(self, action: np.ndarray) -> None:
        """Use the action that actually caused the next observation."""
        self._prev_action = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).reshape(1, self.cfg.action_dim)

    def select_action(
        self, obs: np.ndarray, state: tuple[torch.Tensor, torch.Tensor] | None = None,
        ir: np.ndarray | None = None, deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
        """Select action given observation.

        Properly updates RSSM state using the PREVIOUS action (not zero).

        Args:
            obs: observation (obs_dim,) or (3, H, W) for images
            state: optional RSSM state (h, z)
            ir: optional IR data (ir_dim,) for multi-modal

        Returns:
            action: (action_dim,) numpy array
            state: updated RSSM state (h, z)
        """
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            if state is None:
                h, z = self.initial_state(1)
                prev_action = torch.zeros(1, self.cfg.action_dim, device=self.device)
            else:
                h, z = state
                prev_action = self._prev_action if self._prev_action is not None else \
                    torch.zeros(1, self.cfg.action_dim, device=self.device)

            # Encode observation and update RSSM with previous action
            if self.cfg.use_multimodal and ir is not None:
                ir_t = torch.tensor(ir, dtype=torch.float32, device=self.device).unsqueeze(0)
                obs_embed = self.world_model.encode_multimodal(obs_t, ir_t)
                h, z, _, _ = self.world_model.rssm.observe(obs_embed, prev_action, h, z)
            elif self.cfg.use_images:
                obs_embed = self.world_model.encode_obs(obs_t)
                h, z, _, _ = self.world_model.rssm.observe(obs_embed, prev_action, h, z)
            else:
                obs_embed = self.world_model.encode_obs(obs_t)
                h, z, _, _ = self.world_model.rssm.observe(obs_embed, prev_action, h, z)

            # Get action from actor
            state_vec = torch.cat([h, z], dim=-1)
            action = self.actor(state_vec, deterministic=deterministic)

            return action.squeeze(0).cpu().numpy(), (h, z)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run one training step: world model + actor + critic.

        Batch keys:
          - obs: (batch, seq_len+1, obs_dim) or (batch, seq_len+1, 3, H, W) for images
          - action: (batch, seq_len, action_dim)
          - reward: (batch, seq_len)
          - done: (batch, seq_len)
          - ir: (batch, seq_len+1, ir_dim) optional for multi-modal
        """
        obs = batch["obs"]       # (batch, seq_len + 1, obs_dim) or images
        action = batch["action"]  # (batch, seq_len, action_dim)
        reward = batch["reward"]  # (batch, seq_len)
        done = batch["done"]      # (batch, seq_len)
        ir = batch.get("ir")      # (batch, seq_len + 1, ir_dim) optional

        batch_size, seq_len = action.shape[:2]

        # === World Model Training ===
        wm_out = self.world_model.observe_sequence(
            obs,          # (batch, seq_len + 1, obs_dim/images)
            action,       # (batch, seq_len, action_dim)
            ir_seq=ir if ir is not None else None,
        )

        # Obs reconstruction loss
        # obs_pred reconstructs x_{t+1} from the posterior reached by action_t.
        if self.cfg.use_multimodal:
            obs_target = obs[:, 1: 1 + wm_out["obs_pred"].shape[1]]
            image_recon_loss = F.mse_loss(wm_out["obs_pred"], obs_target)
            food_recon_loss = green_saliency_loss(
                wm_out["obs_pred"], obs_target
            )
            if ir is None or "ir_pred" not in wm_out:
                ir_recon_loss = torch.zeros((), device=obs.device)
            else:
                # Calibrated IR is already bounded in [0, 1]; applying symlog
                # would make deployment preprocessing differ from training.
                ir_target = ir[:, 1: 1 + wm_out["ir_pred"].shape[1]]
                ir_recon_loss = F.mse_loss(wm_out["ir_pred"], ir_target)
            recon_loss = (
                image_recon_loss
                + getattr(self.cfg, "food_recon_weight", 0.0) * food_recon_loss
                + ir_recon_loss
            )
        elif self.cfg.use_images:
            obs_target = obs[:, 1: 1 + wm_out["obs_pred"].shape[1]]
            image_recon_loss = F.mse_loss(wm_out["obs_pred"], obs_target)
            food_recon_loss = green_saliency_loss(
                wm_out["obs_pred"], obs_target
            )
            recon_loss = (
                image_recon_loss
                + getattr(self.cfg, "food_recon_weight", 0.0) * food_recon_loss
            )
            ir_recon_loss = torch.zeros((), device=obs.device)
        else:
            obs_target_symlog = symlog(obs[:, 1: 1 + wm_out["obs_pred"].shape[1]])
            recon_loss = F.mse_loss(wm_out["obs_pred"], obs_target_symlog)
            image_recon_loss = recon_loss
            ir_recon_loss = torch.zeros((), device=obs.device)
            food_recon_loss = torch.zeros((), device=obs.device)

        # Reward prediction loss
        # two_hot_loss handles symlog internally — pass RAW rewards
        # reward_logits has seq_len elements, target is reward[:, :seq_len]
        reward_logits = wm_out["reward_logits"]  # (batch, seq_len, NUM_BINS)
        reward_target = reward[:, :reward_logits.shape[1]]  # (batch, seq_len)
        reward_loss = two_hot_loss(
            reward_logits.reshape(-1, reward_logits.shape[-1]),
            reward_target.reshape(-1),
        )

        # Continue predictor loss (logistic regression)
        continue_logits = wm_out["continue_pred"]  # (batch, seq_len)
        continue_targets = (1.0 - done[:, :continue_logits.shape[1]].float())
        continue_loss = F.binary_cross_entropy_with_logits(
            continue_logits, continue_targets
        )

        # KL divergence losses (dynamics + representation)
        prior_logits = wm_out["prior_logits"].reshape(-1, self.world_model.rssm.stochastic_size)
        posterior_logits = wm_out["posterior_logits"].reshape(-1, self.world_model.rssm.stochastic_size)
        kl_dyn, kl_rep = self.world_model.rssm.kl_loss(
            prior_logits, posterior_logits,
            self.cfg.free_nats,
        )
        with torch.no_grad():
            raw_kl = self.world_model.rssm.raw_kl(prior_logits, posterior_logits)

        # Total world model loss: β_pred * L_pred + L_dyn + β_rep * L_rep
        wm_loss = (
            self.cfg.recon_weight * recon_loss
            + self.cfg.reward_weight * reward_loss
            + self.cfg.continue_weight * continue_loss
            + kl_dyn
            + self.cfg.kl_rep_weight * kl_rep
        )

        self.world_optim.zero_grad()
        wm_loss.backward()
        if getattr(self.cfg, "agc", 0.0) > 0:
            wm_grad_norm = adaptive_clip_grad_(
                self.world_model.parameters(), self.cfg.agc
            )
        else:
            wm_grad_norm = nn.utils.clip_grad_norm_(
                self.world_model.parameters(), self.cfg.grad_clip
            )
        self.world_optim.step()
        self.world_optim.zero_grad(set_to_none=True)

        # === Actor-Critic Training (in imagination) ===
        # Freeze world model — actor/critic should not update world model params
        for p in self.world_model.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            wm_out_detached = self.world_model.observe_sequence(
                obs, action,
                ir_seq=ir if ir is not None else None,
            )
            start_states = select_imagination_starts(
                wm_out_detached["state_all"],
                reward,
                getattr(self.cfg, "imagination_starts", 0),
                self.cfg.reward_event_threshold,
            )
            h_start = start_states[:, : self.cfg.deterministic_size]
            z_start = start_states[:, self.cfg.deterministic_size :]

        # Imagine trajectory — returns log-probs for the ACTUAL imagined actions
        imag_out = self.world_model.imagine_trajectory(
            self.actor, h_start, z_start, horizon=self.cfg.imagination_horizon
        )

        states = imag_out["state"]  # (batch, horizon, state_dim)
        batch_size_imag, horizon, state_dim = states.shape
        states_flat = states.reshape(-1, state_dim)

        # Reward/value logits_to_value handles symexp internally
        # imag_rewards and imag_values are in ORIGINAL space
        imag_rewards = logits_to_value(imag_out["reward_logits"])  # (B, H)

        # Imagined values from distributional critic
        imag_logits = self.critic(states_flat).reshape(batch_size_imag, horizon, -1)
        imag_values = logits_to_value(imag_logits)  # (B, H)

        # Lambda-return bootstraps use the online critic, while the slow critic
        # is reserved for the paper's critic regularization term.
        with torch.no_grad():
            last_state = torch.cat([imag_out["h"][:, -1], imag_out["z"][:, -1]], dim=-1)
            target_state_sequence = torch.cat(
                [states, last_state.unsqueeze(1)], dim=1
            )
            last_value = logits_to_value(
                self.critic(last_state)
            ).unsqueeze(1)
            return_values = torch.cat(
                [imag_values.detach(), last_value], dim=1
            )
            target_imag_logits = self.critic_target(
                target_state_sequence.reshape(-1, state_dim)
            ).reshape(batch_size_imag, horizon + 1, -1)

        returns = lambda_return(
            imag_rewards,
            return_values,
            imag_out["continue_logit"],
            gamma=self.cfg.gamma,
            lam=self.cfg.lam,
        )

        # === Actor Loss: REINFORCE with return normalization ===
        # Paper Eq 6: L(θ) = -Σ sg((R_t^λ - v_t) / max(1, S)) * log π(a_t|s_t) + η H[π]
        # Use log-probs from imagination (same actions that caused transitions)
        log_probs = imag_out["log_probs"]  # (batch, horizon)

        with torch.no_grad():
            S = self.return_normalizer.update(returns)
            advantages = (returns - imag_values.detach()) / S
            advantage_clip = getattr(self.cfg, "advantage_clip", 0.0)
            if advantage_clip > 0:
                advantages = torch.clamp(
                    advantages, -advantage_clip, advantage_clip
                )
            continuation = torch.sigmoid(imag_out["continue_logit"])
            trajectory_weights = torch.ones_like(continuation)
            if horizon > 1:
                trajectory_weights[:, 1:] = torch.cumprod(
                    self.cfg.gamma * continuation[:, :-1], dim=1
                )
            weight_normalizer = trajectory_weights.sum().clamp_min(1.0)

        # REINFORCE loss — use log-probs of the ACTUAL imagined actions
        actor_loss = -(
            trajectory_weights * advantages.detach() * log_probs
        ).sum() / weight_normalizer

        # Entropy bonus
        actor_entropy = -log_probs
        actor_loss -= self.cfg.entropy_weight * (
            trajectory_weights * actor_entropy
        ).sum() / weight_normalizer

        self.actor_optim.zero_grad()
        actor_loss.backward()
        if getattr(self.cfg, "agc", 0.0) > 0:
            actor_grad_norm = adaptive_clip_grad_(
                self.actor.parameters(), self.cfg.agc
            )
        else:
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.grad_clip
            )
        self.actor_optim.step()

        # === Critic Loss: imagination + replay + slow regularization ===
        critic_logits = self.critic(states_flat.detach()).reshape(batch_size_imag, horizon, -1)
        imag_value_loss_per_step = two_hot_loss(
            critic_logits.reshape(-1, critic_logits.shape[-1]),
            returns.detach().reshape(-1),
            reduction="none",
        ).reshape(batch_size_imag, horizon)
        imag_slow_loss = soft_cross_entropy(
            critic_logits, target_imag_logits[:, :-1]
        )
        slow_regularization = getattr(
            self.cfg, "slow_value_regularization", 1.0
        )
        imag_critic_loss = (
            trajectory_weights
            * (imag_value_loss_per_step + slow_regularization * imag_slow_loss)
        ).sum() / weight_normalizer

        # Replay value learning anchors the critic to real rewards. The
        # reward-event replay sampler makes this especially useful for sparse
        # food pickups.
        replay_states = wm_out_detached["state_all"]
        replay_state_dim = replay_states.shape[-1]
        replay_logits = self.critic(
            replay_states[:, :-1].reshape(-1, replay_state_dim).detach()
        ).reshape(batch_size, seq_len, -1)
        with torch.no_grad():
            replay_last_value = logits_to_value(
                self.critic(replay_states[:, -1].detach())
            ).unsqueeze(1)
            replay_return_values = torch.cat(
                [logits_to_value(replay_logits.detach()), replay_last_value],
                dim=1,
            )
            replay_target_logits = self.critic_target(
                replay_states.reshape(-1, replay_state_dim)
            ).reshape(batch_size, seq_len + 1, -1)
            replay_returns = lambda_return(
                reward,
                replay_return_values,
                None,
                gamma=self.cfg.gamma,
                lam=self.cfg.lam,
                continuation=1.0 - done.float(),
            )
        replay_value_loss = two_hot_loss(
            replay_logits.reshape(-1, replay_logits.shape[-1]),
            replay_returns.reshape(-1),
        )
        replay_slow_loss = soft_cross_entropy(
            replay_logits, replay_target_logits[:, :-1]
        ).mean()
        replay_critic_loss = (
            replay_value_loss + slow_regularization * replay_slow_loss
        )
        critic_loss = imag_critic_loss + getattr(
            self.cfg, "replay_value_weight", 0.3
        ) * replay_critic_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        if getattr(self.cfg, "agc", 0.0) > 0:
            critic_grad_norm = adaptive_clip_grad_(
                self.critic.parameters(), self.cfg.agc
            )
        else:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.cfg.grad_clip
            )
        self.critic_optim.step()

        # Update target critic (EMA)
        tau = self.cfg.target_tau
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Unfreeze world model for next training step
        for p in self.world_model.parameters():
            p.requires_grad_(True)

        self._train_calls += 1

        return {
            "wm_loss": wm_loss.item(),
            "recon_loss": recon_loss.item(),
            "image_recon_loss": image_recon_loss.item(),
            "ir_recon_loss": ir_recon_loss.item(),
            "food_recon_loss": food_recon_loss.item(),
            "reward_loss": reward_loss.item(),
            "continue_loss": continue_loss.item(),
            "kl_dyn": kl_dyn.item(),
            "kl_rep": kl_rep.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "critic_imagination_loss": imag_critic_loss.item(),
            "critic_replay_loss": replay_critic_loss.item(),
            "critic_replay_value_loss": replay_value_loss.item(),
            "critic_slow_regularization": (
                imag_slow_loss.mean() + replay_slow_loss
            ).item(),
            "actor_entropy": actor_entropy.mean().item(),
            "imag_returns": returns.mean().item(),
            "return_range": S,
            "imagination_starts": batch_size_imag,
            "actor_std": self.actor.std(states_flat).mean().item(),
            "actor_mean_abs": self.actor.mean(states_flat).abs().mean().item(),
            "action_saturation": (imag_out["action"].abs() >= 0.99).float().mean().item(),
            "advantage_p05": torch.quantile(advantages, 0.05).item(),
            "advantage_p50": torch.quantile(advantages, 0.50).item(),
            "advantage_p95": torch.quantile(advantages, 0.95).item(),
            "world_grad_norm": float(wm_grad_norm),
            "actor_grad_norm": float(actor_grad_norm),
            "critic_grad_norm": float(critic_grad_norm),
            "raw_kl_mean": raw_kl.mean().item(),
            "raw_kl_median": raw_kl.median().item(),
            "raw_kl_p95": torch.quantile(raw_kl, 0.95).item(),
            "raw_kl_fraction_above_one": (raw_kl > 1.0).float().mean().item(),
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "optimizer_type": self.optimizer_name,
            "world_optim": self.world_optim.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "return_normalizer_range": self.return_normalizer.range,
            "cfg": self.cfg,
            "global_step": self._global_step,
            "train_calls": self._train_calls,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }, temporary)
        temporary.replace(path)

    def load(self, path: str | Path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(ckpt["world_model"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        checkpoint_optimizer = ckpt.get("optimizer_type", "adam")
        if checkpoint_optimizer == self.optimizer_name:
            self.world_optim.load_state_dict(ckpt["world_optim"])
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
        else:
            print(
                f"DreamerV3 optimizer changed from {checkpoint_optimizer} to "
                f"{self.optimizer_name}; optimizer moments were reset."
            )
        self.return_normalizer.range = ckpt.get("return_normalizer_range", 1.0)
        self._global_step = ckpt.get("global_step", 0)
        self._train_calls = ckpt.get("train_calls", 0)
        if "python_rng_state" in ckpt:
            random.setstate(ckpt["python_rng_state"])
        if "numpy_rng_state" in ckpt:
            np.random.set_state(ckpt["numpy_rng_state"])
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                _cpu_byte_rng_states(ckpt["cuda_rng_state"])
            )

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value
