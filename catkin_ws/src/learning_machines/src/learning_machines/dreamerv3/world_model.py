"""DreamerV3 world model: encoder, decoder, reward predictor.

Supports both MLP (vector obs) and CNN (image obs) encoder/decoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from learning_machines.distributional import _NUM_BINS
from .rssm import RSSM


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 3) -> nn.Sequential:
    dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
    net = []
    for i in range(len(dims) - 1):
        net.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            net.append(nn.SiLU())
    return nn.Sequential(*net)


class CNNEncoder(nn.Module):
    """CNN encoder for image observations.

    Input: (B, 3, H, W) image
    Output: (B, embed_dim) latent
    """

    def __init__(self, embed_dim: int = 512, image_size: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64->32
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32->16
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16->8
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 8->4
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) float in [0, 1]
        Returns:
            latent: (B, embed_dim)
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return self.conv(image)


class CNNDecoder(nn.Module):
    """CNN decoder for image reconstruction.

    Input: (B, state_dim) latent
    Output: (B, 3, H, W) image in [0, 1]
    """

    def __init__(self, state_dim: int = 512, image_size: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256 * 4 * 4),
            nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4->8
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8->16
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16->32
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),  # 32->64
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent: (B, state_dim)
        Returns:
            image: (B, 3, H, W) float in [0, 1]
        """
        x = self.fc(latent)
        x = x.view(-1, 256, 4, 4)
        return self.deconv(x)


class WorldModel(nn.Module):
    """DreamerV3 world model.

    Components:
      - RSSM for dynamics
      - Observation encoder (MLP, CNN, or multi-modal)
      - Observation decoder (MLP or CNN)
      - Reward predictor (MLP) — outputs logits for two-hot loss
      - Continue predictor (MLP) for episode termination

    Input modes:
      - use_images=False: MLP encoder for vector observations (blob + IR)
      - use_images=True: CNN encoder for images only
      - use_multimodal=True: CNN for images + MLP for IR, concatenated
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        deterministic_size: int = 512,
        stochastic_classes: int = 32,
        stochastic_bins: int = 32,
        hidden_size: int = 512,
        embed_size: int = 512,
        mlp_hidden: int = 512,
        mlp_layers: int = 3,
        use_images: bool = False,
        use_multimodal: bool = False,
        image_size: int = 64,
        ir_dim: int = 8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_images = use_images
        self.use_multimodal = use_multimodal
        self.image_size = image_size
        self.ir_dim = ir_dim

        # Determine RSSM input dimension based on mode
        if use_multimodal:
            # Multi-modal: CNN embed + MLP embed for IR
            rssm_obs_dim = embed_size + embed_size // 2
        elif use_images:
            rssm_obs_dim = embed_size
        else:
            rssm_obs_dim = obs_dim

        self.rssm = RSSM(
            obs_dim=rssm_obs_dim,
            action_dim=action_dim,
            deterministic_size=deterministic_size,
            stochastic_classes=stochastic_classes,
            stochastic_bins=stochastic_bins,
            hidden_size=hidden_size,
            obs_is_embedding=(use_multimodal or use_images),
        )

        rssm_state_size = deterministic_size + self.rssm.stochastic_size

        if use_multimodal:
            # Multi-modal: CNN for images + MLP for IR
            self.image_encoder = CNNEncoder(embed_dim=embed_size, image_size=image_size)
            self.ir_encoder = nn.Sequential(
                nn.Linear(ir_dim, embed_size // 4),
                nn.SiLU(),
                nn.Linear(embed_size // 4, embed_size // 2),
                nn.LayerNorm(embed_size // 2),
            )
            self.obs_decoder = CNNDecoder(state_dim=rssm_state_size, image_size=image_size)
            self.ir_decoder = nn.Sequential(
                mlp(rssm_state_size, mlp_hidden, ir_dim, mlp_layers),
                nn.Sigmoid(),
            )
        elif use_images:
            # CNN encoder: image -> embed
            self.obs_encoder = CNNEncoder(embed_dim=embed_size, image_size=image_size)
            # CNN decoder: state -> image
            self.obs_decoder = CNNDecoder(state_dim=rssm_state_size, image_size=image_size)
        else:
            # MLP encoder: symlog(obs) -> embed (via RSSM obs_embed)
            self.obs_encoder = None
            # MLP decoder: state -> symlog(obs)
            self.obs_decoder = mlp(rssm_state_size, mlp_hidden, obs_dim, mlp_layers)

        # Reward predictor: (h, z) -> logits over two-hot bins
        self.reward_hidden = mlp(rssm_state_size, mlp_hidden, mlp_hidden, 2)
        self.reward_head = nn.Linear(mlp_hidden, _NUM_BINS)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

        # Continue predictor: (h, z) -> P(not done)
        self.continue_head = nn.Sequential(
            nn.Linear(mlp_hidden, 1),
        )

        # Layer norms
        self.reward_norm = nn.LayerNorm(mlp_hidden)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to embedding for RSSM.

        For images: CNN encoder
        For vectors: symlog transform (done in RSSM.observe)
        """
        if self.use_images:
            return self.obs_encoder(obs)
        else:
            # For vector obs, encoding is done inside RSSM.observe
            return obs

    def encode_multimodal(self, image: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        """Encode multi-modal observation (image + IR) to embedding.

        Args:
            image: (B, 3, H, W) float in [0, 1]
            ir: (B, ir_dim) float in [0, 1]

        Returns:
            combined_embed: (B, embed_size + embed_size//2)
        """
        image_embed = self.image_encoder(image)  # (B, embed_size)
        ir_embed = self.ir_encoder(ir)  # (B, embed_size // 2)
        return torch.cat([image_embed, ir_embed], dim=-1)

    def initial_state(self, batch_size: int, device: torch.device):
        return self.rssm.initial_state(batch_size, device)

    def observe_sequence(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        h_init: torch.Tensor | None = None,
        z_init: torch.Tensor | None = None,
        ir_seq: torch.Tensor | None = None,
    ):
        """Process a sequence of observations through the world model.

        Dreamer transition convention:
          1. Infer the initial posterior from obs_0 with a zero previous action.
          2. For each transition t, update the RSSM with action_t and obs_{t+1}.
          3. Predict reward_t, continuation_t, and reconstruct obs_{t+1} from that posterior.

        Args:
            obs_seq: (batch, T+1, obs_dim/images) — observations x_0..x_T
            action_seq: (batch, T, action_dim) — actions a_0..a_{T-1}
            h_init: optional initial hidden state
            z_init: optional initial stochastic state
            ir_seq: (batch, T+1, ir_dim) optional IR data

        Returns dict with all predictions for transitions 0..T-1.
        obs_pred[t] reconstructs x_{t+1}, reward_logits[t] predicts r_t.
        """
        batch, obs_len = obs_seq.shape[:2]
        seq_len = action_seq.shape[1]
        if obs_len != seq_len + 1:
            raise ValueError(
                f"observe_sequence expects T+1 observations for T actions, got "
                f"obs_len={obs_len}, action_len={seq_len}"
            )

        if h_init is None:
            h_init, z_init = self.initial_state(batch, obs_seq.device)

        h_list, z_list = [], []
        state_all = []
        prior_logits_list, posterior_logits_list = [], []
        obs_pred_list = []
        reward_logits_list = []
        continue_list = []
        ir_pred_list = []

        h, z = h_init, z_init

        def observe_one(obs_t, act_t, ir_t=None):
            if self.use_multimodal:
                if ir_t is None:
                    ir_t = torch.zeros(batch, self.ir_dim, device=obs_seq.device)
                obs_embed = self.encode_multimodal(obs_t, ir_t)
                return self.rssm.observe(obs_embed, act_t, h, z)
            if self.use_images:
                obs_embed = self.obs_encoder(obs_t)
                return self.rssm.observe(obs_embed, act_t, h, z)
            return self.rssm.observe(obs_t, act_t, h, z)

        # Initial posterior q(s_0 | obs_0), used only as context for transition a_0.
        zero_action = torch.zeros(batch, self.action_dim, device=obs_seq.device)
        ir0 = ir_seq[:, 0] if ir_seq is not None else None
        h, z, _, _ = observe_one(obs_seq[:, 0], zero_action, ir0)
        state_all.append(torch.cat([h, z], dim=-1))

        for t in range(seq_len):
            obs_next = obs_seq[:, t + 1]
            ir_next = ir_seq[:, t + 1] if ir_seq is not None else None
            h, z, prior_logits, posterior_logits = observe_one(obs_next, action_seq[:, t], ir_next)

            h_list.append(h)
            z_list.append(z)
            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)

            state = torch.cat([h, z], dim=-1)
            state_all.append(state)

            # Decode current observation from posterior state.
            obs_pred = self.obs_decoder(state)
            obs_pred_list.append(obs_pred)
            if self.use_multimodal:
                ir_pred_list.append(self.ir_decoder(state))

            # Reward logits (predicts r_t from state at time t)
            reward_feat = F.silu(self.reward_norm(self.reward_hidden(state)))
            reward_logits = self.reward_head(reward_feat)
            reward_logits_list.append(reward_logits)

            # Continue
            cont_logit = self.continue_head(reward_feat).squeeze(-1)
            continue_list.append(cont_logit)

        result = {
            "h": torch.stack(h_list, dim=1),
            "z": torch.stack(z_list, dim=1),
            "prior_logits": torch.stack(prior_logits_list, dim=1),
            "posterior_logits": torch.stack(posterior_logits_list, dim=1),
            "obs_pred": torch.stack(obs_pred_list, dim=1),  # reconstructs obs[:, 1:]
            "reward_logits": torch.stack(reward_logits_list, dim=1),  # (batch, seq_len, NUM_BINS)
            "continue_pred": torch.stack(continue_list, dim=1),  # (batch, seq_len)
            "state_all": torch.stack(state_all, dim=1),  # posterior states s_0..s_T
        }
        if self.use_multimodal:
            result["ir_pred"] = torch.stack(ir_pred_list, dim=1)
        return result

    def imagine_trajectory(
        self,
        actor: nn.Module,
        h: torch.Tensor,
        z: torch.Tensor,
        horizon: int = 15,
    ):
        """Roll out the world model using the actor's policy.

        h: (batch, deterministic_size)
        z: (batch, stochastic_size)

        Returns dict with imagined trajectory data.
        reward_logits: (batch, horizon, NUM_BINS) — for two-hot loss
        log_probs: (batch, horizon) — log π(a_t|s_t) for the sampled actions
        """
        h_list, z_list = [], []
        action_list = []
        log_prob_list = []
        reward_logits_list = []
        continue_list = []
        state_list = []

        for t in range(horizon):
            state = torch.cat([h, z], dim=-1)
            state_list.append(state)

            # Sample action AND log-prob together (REINFORCE requirement)
            action, log_prob = actor.get_action_and_log_prob(state, deterministic=False)

            h, z, prior_logits = self.rssm.imagine(action, h, z)

            h_list.append(h)
            z_list.append(z)
            action_list.append(action)
            log_prob_list.append(log_prob.reshape(-1))  # preserve batch dim for B=1

            state_new = torch.cat([h, z], dim=-1)
            reward_feat = F.silu(self.reward_norm(self.reward_hidden(state_new)))
            reward_logits = self.reward_head(reward_feat)
            reward_logits_list.append(reward_logits)

            cont_logit = self.continue_head(reward_feat).squeeze(-1)
            continue_list.append(cont_logit)

        return {
            "h": torch.stack(h_list, dim=1),
            "z": torch.stack(z_list, dim=1),
            "state": torch.stack(state_list, dim=1),
            "action": torch.stack(action_list, dim=1),
            "log_probs": torch.stack(log_prob_list, dim=1),  # (batch, horizon)
            "reward_logits": torch.stack(reward_logits_list, dim=1),
            "continue_logit": torch.stack(continue_list, dim=1),
        }
