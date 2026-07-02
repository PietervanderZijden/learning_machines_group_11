'Causal spatial tokenizer from DreamerV4 Section 3.1.'
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DreamerV4FullConfig
from .transformer import SpaceTimeTransformer


class RunningRMS(nn.Module):
    def __init__(self, decay: float, floor_ratio: float = 0.1):
        super().__init__()
        self.decay = decay
        self.floor_ratio = floor_ratio



        self.register_buffer("value", torch.tensor(1.0))
        self.register_buffer("max_value", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    def normalize(self, loss: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            magnitude = loss.detach().square().clamp_min(1e-12).sqrt()
            if not self.initialized:


                self.value.copy_(magnitude.clamp(min=1.0))
                self.max_value.copy_(magnitude)
                self.initialized.fill_(True)
            else:
                self.value.mul_(self.decay).add_((1 - self.decay) * magnitude)
                self.max_value.clamp_(min=magnitude)
        floor = self.max_value * self.floor_ratio
        return loss / self.value.detach().clamp_min(floor)


class LPIPSLoss(nn.Module):
    'Frozen official LPIPS metric, loaded only when the loss is used.'

    def __init__(self):
        super().__init__()



        object.__setattr__(self, "_metric", None)

    def _load(self, device: torch.device):
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError(
                "DreamerV4 tokenizer training requires the `lpips` package. "
                "Install project dependencies with `uv sync`."
            ) from exc
        metric = lpips.LPIPS(net="alex").eval().to(device)
        for parameter in metric.parameters():
            parameter.requires_grad_(False)
        object.__setattr__(self, "_metric", metric)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        metric = self._metric
        if metric is None:
            self._load(prediction.device)
            metric = self._metric
        assert metric is not None
        if next(metric.parameters()).device != prediction.device:
            metric.to(prediction.device)
        with torch.autocast(
            device_type=prediction.device.type,
            enabled=False,
        ):
            return metric(
                prediction.float() * 2 - 1,
                target.float() * 2 - 1,
            ).mean()


class CausalVideoTokenizer(nn.Module):
    'Patch/latent-token video tokenizer with causal temporal compression.'

    def __init__(self, cfg: DreamerV4FullConfig):
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.image_channels * cfg.patch_size * cfg.patch_size
        self.patch_embed = nn.Linear(patch_dim, cfg.model_dim)
        self.patch_decode = nn.Linear(cfg.model_dim, patch_dim)


        self.mask_token = nn.Parameter(torch.randn(cfg.model_dim) * 0.02)
        self.encoder_latents = nn.Parameter(
            torch.randn(cfg.latent_tokens, cfg.model_dim) * 0.02
        )
        self.decoder_queries = nn.Parameter(
            torch.randn(cfg.patches_per_frame, cfg.model_dim) * 0.02
        )
        self.ir_embed = (
            nn.Linear(cfg.ir_dim, cfg.model_dim) if cfg.ir_dim else None
        )
        self.ir_query = (
            nn.Parameter(torch.randn(1, cfg.model_dim) * 0.02)
            if cfg.ir_dim else None
        )
        self.ir_decode = (
            nn.Linear(cfg.model_dim, cfg.ir_dim) if cfg.ir_dim else None
        )
        self.to_bottleneck = nn.Linear(cfg.model_dim, cfg.latent_channels)
        self.from_bottleneck = nn.Linear(cfg.latent_channels, cfg.model_dim)
        self.encoder = self._make_transformer(cfg.tokenizer_layers)
        self.decoder = self._make_transformer(cfg.tokenizer_layers)
        self.lpips = LPIPSLoss()
        self.mse_rms = RunningRMS(cfg.rms_decay, cfg.rms_floor_ratio)
        self.lpips_rms = RunningRMS(cfg.rms_decay, cfg.rms_floor_ratio)
        self.ir_rms = RunningRMS(cfg.rms_decay, cfg.rms_floor_ratio)

    def _make_transformer(self, layers: int) -> SpaceTimeTransformer:
        cfg = self.cfg
        return SpaceTimeTransformer(
            cfg.model_dim,
            layers,
            cfg.heads,
            cfg.heads,
            cfg.ff_multiplier,
            cfg.temporal_every,
            cfg.attention_softcap,
            cfg.dropout,
        )

    def patchify(self, images: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = images.shape
        patch = self.cfg.patch_size
        patches = images.unfold(3, patch, patch).unfold(4, patch, patch)
        patches = patches.permute(0, 1, 3, 4, 2, 5, 6)
        return patches.reshape(batch, time, -1, channels * patch * patch)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch, time, _, patch_dim = patches.shape
        patch = self.cfg.patch_size
        side = self.cfg.image_size // patch
        channels = patch_dim // (patch * patch)
        images = patches.reshape(
            batch, time, side, side, channels, patch, patch
        )
        images = images.permute(0, 1, 4, 2, 5, 3, 6)
        return images.reshape(
            batch, time, channels, self.cfg.image_size, self.cfg.image_size
        )

    def _encoder_mask(self, device: torch.device) -> torch.Tensor:
        patches = self.cfg.patches_per_frame
        ir_tokens = int(self.ir_embed is not None)
        latents = self.cfg.latent_tokens
        total = patches + ir_tokens + latents
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:patches, :patches] = True
        if ir_tokens:
            mask[patches, patches] = True
        mask[patches + ir_tokens :, :] = True
        return mask

    def _decoder_mask(self, device: torch.device) -> torch.Tensor:
        latents = self.cfg.latent_tokens
        patches = self.cfg.patches_per_frame
        ir_tokens = int(self.ir_decode is not None)
        total = latents + patches + ir_tokens
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:latents, :latents] = True
        mask[latents : latents + patches, : latents + patches] = True
        if ir_tokens:
            mask[-1, :latents] = True
            mask[-1, -1] = True
        return mask

    def encode(
        self,
        images: torch.Tensor,
        ir: torch.Tensor | None = None,
        *,
        mask_patches: bool = False,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        if images.dim() == 4:
            images = images.unsqueeze(1)
            if ir is not None and ir.dim() == 2:
                ir = ir.unsqueeze(1)
            squeeze_time = True
        else:
            squeeze_time = False
        patches = self.patch_embed(self.patchify(images))
        patch_mask: torch.Tensor | None = None
        if mask_patches:
            batch, time, count, _ = patches.shape
            patch_mask = torch.rand(
                batch, time, count, 1, device=images.device
            ) < self.cfg.mask_ratio
            patches = torch.where(
                patch_mask, self.mask_token.view(1, 1, 1, -1), patches
            )
        tokens = [patches]
        if self.ir_embed is not None:
            if ir is None:
                raise ValueError("IR input is required by this tokenizer configuration")
            tokens.append(self.ir_embed(ir).unsqueeze(2))
        batch, time = images.shape[:2]
        tokens.append(
            self.encoder_latents.view(1, 1, self.cfg.latent_tokens, -1)
            .expand(batch, time, -1, -1)
        )
        encoded = self.encoder(
            torch.cat(tokens, 2),
            spatial_mask=self._encoder_mask(images.device),
            temporal_window=self.cfg.context_length,
        )
        latent_hidden = encoded[:, :, -self.cfg.latent_tokens :]
        latent = torch.tanh(self.to_bottleneck(latent_hidden))
        if squeeze_time:
            latent = latent[:, 0]
            if patch_mask is not None:
                patch_mask = patch_mask[:, 0]
        if return_mask:
            return latent, patch_mask.squeeze(-1) if patch_mask is not None else None
        return latent

    def decode(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if latent.dim() == 3:
            latent = latent.unsqueeze(1)
            squeeze_time = True
        else:
            squeeze_time = False
        batch, time = latent.shape[:2]
        latent_tokens = self.from_bottleneck(latent)
        patch_queries = self.decoder_queries.view(
            1, 1, self.cfg.patches_per_frame, -1
        ).expand(batch, time, -1, -1)
        tokens = [latent_tokens, patch_queries]
        if self.ir_query is not None:
            tokens.append(self.ir_query.view(1, 1, 1, -1).expand(batch, time, -1, -1))
        decoded = self.decoder(
            torch.cat(tokens, 2),
            spatial_mask=self._decoder_mask(latent.device),
            temporal_window=self.cfg.context_length,
        )
        patch_hidden = decoded[
            :, :, self.cfg.latent_tokens :
            self.cfg.latent_tokens + self.cfg.patches_per_frame
        ]
        images = self.unpatchify(self.patch_decode(patch_hidden)).sigmoid()
        ir_prediction = self.ir_decode(decoded[:, :, -1]) if self.ir_decode else None
        if squeeze_time:
            images = images[:, 0]
            ir_prediction = ir_prediction[:, 0] if ir_prediction is not None else None
        return images, ir_prediction

    def forward(
        self, images: torch.Tensor, ir: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        single_frame = images.dim() == 4
        target_images = images.unsqueeze(1) if single_frame else images
        target_ir = (
            ir.unsqueeze(1) if ir is not None and ir.dim() == 2 else ir
        )
        latent, patch_mask = self.encode(
            target_images, target_ir, mask_patches=True, return_mask=True
        )

        if patch_mask.dim() == 2:
            patch_mask = patch_mask.unsqueeze(1)
        reconstruction, ir_prediction = self.decode(latent)



        B, T, C, H, W = reconstruction.shape
        patch = self.cfg.patch_size
        gh, gw = H // patch, W // patch
        pixel_mask = (
            patch_mask.view(B, T, gh, gw)
            .float()
            .unsqueeze(2)
            .repeat_interleave(patch, dim=-2)
            .repeat_interleave(patch, dim=-1)
        )

        diff_sq = (reconstruction - target_images).pow(2)
        mse = (diff_sq * pixel_mask).sum() / pixel_mask.sum().clamp_min(1.0)

        if self.cfg.lpips_weight:


            hybrid = torch.where(
                pixel_mask.bool().expand_as(reconstruction),
                reconstruction,
                target_images,
            )
            perceptual = self.lpips(hybrid.flatten(0, 1), target_images.flatten(0, 1))
        else:
            perceptual = mse.new_zeros(())

        total = self.mse_rms.normalize(mse)
        if self.cfg.lpips_weight:
            total = total + self.cfg.lpips_weight * self.lpips_rms.normalize(
                perceptual
            )
        if self.ir_decode is not None:
            if target_ir is None or ir_prediction is None:
                raise ValueError("IR input is required by this tokenizer configuration")
            ir_loss = F.mse_loss(ir_prediction, target_ir)
            total = total + self.ir_rms.normalize(ir_loss)
        else:
            ir_loss = mse.new_zeros(())

        return {
            "loss": total,
            "mse_loss": mse,
            "lpips_loss": perceptual,
            "ir_loss": ir_loss,
            "latent": latent[:, 0] if single_frame else latent,
            "reconstruction": reconstruction[:, 0] if single_frame else reconstruction,
        }
