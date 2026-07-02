'DreamerV3 world model: encoder, decoder, reward predictor.'
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
    'CNN encoder for image observations.'

    def __init__(self, embed_dim: int = 512, image_size: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        'Args:.'
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return self.conv(image)


class CNNDecoder(nn.Module):
    'CNN decoder for image reconstruction.'

    def __init__(self, state_dim: int = 512, image_size: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256 * 4 * 4),
            nn.SiLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        'Args:.'
        x = self.fc(latent)
        x = x.view(-1, 256, 4, 4)
        return self.deconv(x)


class WorldModel(nn.Module):
    'DreamerV3 world model.'

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


        if use_multimodal:

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

            self.obs_encoder = CNNEncoder(embed_dim=embed_size, image_size=image_size)

            self.obs_decoder = CNNDecoder(state_dim=rssm_state_size, image_size=image_size)
        else:

            self.obs_encoder = None

            self.obs_decoder = mlp(rssm_state_size, mlp_hidden, obs_dim, mlp_layers)


        self.reward_hidden = mlp(rssm_state_size, mlp_hidden, mlp_hidden, 2)
        self.reward_head = nn.Linear(mlp_hidden, _NUM_BINS)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)


        self.continue_head = nn.Sequential(
            nn.Linear(mlp_hidden, 1),
        )


        self.reward_norm = nn.LayerNorm(mlp_hidden)

    def encode_obs(self, obs: torch.Tensor) -> torch.Tensor:
        'Encode observation to embedding for RSSM.'
        if self.use_images:
            return self.obs_encoder(obs)
        else:

            return obs

    def encode_multimodal(self, image: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        'Encode multi-modal observation (image + IR) to embedding.'
        image_embed = self.image_encoder(image)
        ir_embed = self.ir_encoder(ir)
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
        'Process a sequence of observations through the world model.'
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


            obs_pred = self.obs_decoder(state)
            obs_pred_list.append(obs_pred)
            if self.use_multimodal:
                ir_pred_list.append(self.ir_decoder(state))


            reward_feat = F.silu(self.reward_norm(self.reward_hidden(state)))
            reward_logits = self.reward_head(reward_feat)
            reward_logits_list.append(reward_logits)


            cont_logit = self.continue_head(reward_feat).squeeze(-1)
            continue_list.append(cont_logit)

        result = {
            "h": torch.stack(h_list, dim=1),
            "z": torch.stack(z_list, dim=1),
            "prior_logits": torch.stack(prior_logits_list, dim=1),
            "posterior_logits": torch.stack(posterior_logits_list, dim=1),
            "obs_pred": torch.stack(obs_pred_list, dim=1),
            "reward_logits": torch.stack(reward_logits_list, dim=1),
            "continue_pred": torch.stack(continue_list, dim=1),
            "state_all": torch.stack(state_all, dim=1),
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
        "Roll out the world model using the actor's policy."
        h_list, z_list = [], []
        action_list = []
        log_prob_list = []
        reward_logits_list = []
        continue_list = []
        state_list = []

        for t in range(horizon):
            state = torch.cat([h, z], dim=-1)
            state_list.append(state)


            action, log_prob = actor.get_action_and_log_prob(state, deterministic=False)

            h, z, prior_logits = self.rssm.imagine(action, h, z)

            h_list.append(h)
            z_list.append(z)
            action_list.append(action)
            log_prob_list.append(log_prob.reshape(-1))

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
            "log_probs": torch.stack(log_prob_list, dim=1),
            "reward_logits": torch.stack(reward_logits_list, dim=1),
            "continue_logit": torch.stack(continue_list, dim=1),
        }
