'Image tokenizer for DreamerV4 with LPIPS loss and patch-based MAE.'
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning_machines.distributional import symlog

class ImageEncoder(nn.Module):
    'CNN encoder: image (3, H, W) -> latent (latent_dim,).'

    def __init__(self, latent_dim: int = 256, image_size: int = 64):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.LayerNorm(latent_dim),
        )


        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, return_features: bool = False):
        'Args:.'
        if image.dtype == torch.uint8:
            image = image.float() / 255.0

        features = []
        x = self.conv1(image)
        if return_features:
            features.append(x)
        x = self.conv2(x)
        if return_features:
            features.append(x)
        x = self.conv3(x)
        if return_features:
            features.append(x)
        x = self.conv4(x)
        if return_features:
            features.append(x)

        x = self.flatten(x)
        latent = self.fc(x)
        latent = self.bottleneck(latent)

        if return_features:
            return latent, features
        return latent


class ImageDecoder(nn.Module):
    'CNN decoder: latent (latent_dim,) -> image (3, H, W).'

    def __init__(self, latent_dim: int = 256, image_size: int = 64):
        super().__init__()
        self.image_size = image_size
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
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


class PerceptualLoss(nn.Module):
    'Simplified perceptual loss using encoder features.'

    def __init__(self):
        super().__init__()

    def forward(self, features_recon: list, features_target: list) -> torch.Tensor:
        'Compute perceptual loss between feature lists.'
        loss = 0.0
        for f_recon, f_target in zip(features_recon, features_target):

            f_recon = f_recon / (f_recon.std() + 1e-8)
            f_target = f_target / (f_target.std() + 1e-8)
            loss = loss + F.mse_loss(f_recon, f_target)
        return loss


class ImageTokenizer(nn.Module):
    'Image tokenizer with encoder + decoder, trained with masked autoencoding.'

    def __init__(self, latent_dim: int = 256, image_size: int = 64,
                 ir_dim: int = 0):
        super().__init__()
        self.encoder = ImageEncoder(latent_dim, image_size)
        self.latent_dim = latent_dim
        self.ir_dim = ir_dim


        self.perceptual_loss = PerceptualLoss()


        if ir_dim > 0:
            self.ir_encoder = nn.Sequential(
                nn.Linear(ir_dim, latent_dim // 4),
                nn.SiLU(),
                nn.Linear(latent_dim // 4, latent_dim // 2),
                nn.LayerNorm(latent_dim // 2),
            )

            self.combined_dim = latent_dim + latent_dim // 2
            self.ir_decoder = nn.Sequential(
                nn.Linear(self.combined_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, ir_dim),
            )
        else:
            self.ir_encoder = None
            self.ir_decoder = None
            self.combined_dim = latent_dim


        self.decoder = ImageDecoder(latent_dim, image_size)


        self.mask_token = nn.Parameter(torch.randn(3, 1, 1) * 0.02)
        self.register_buffer("image_loss_rms", torch.tensor(1.0))
        self.register_buffer("ir_loss_rms", torch.tensor(1.0))

    @staticmethod
    def _green_saliency_loss(
        reconstruction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        recon_green = torch.relu(
            reconstruction[:, 1]
            - torch.maximum(reconstruction[:, 0], reconstruction[:, 2])
        )
        target_green = torch.relu(
            target[:, 1] - torch.maximum(target[:, 0], target[:, 2])
        )
        weights = 1.0 + 20.0 * (target_green > 0.1).float()
        return (
            (recon_green - target_green).square() * weights
        ).sum() / weights.sum()

    def _normalize_loss(self, loss: torch.Tensor, rms_name: str) -> torch.Tensor:
        'RMS-normalize objective terms without backpropagating through the scale.'
        rms = getattr(self, rms_name)
        with torch.no_grad():
            rms.mul_(0.99).add_(0.01 * loss.detach().square().clamp_min(1e-12).sqrt())
        return loss / rms.detach().clamp_min(1e-6)

    def _patchify_and_mask(self, image: torch.Tensor, mask_ratio: float) -> tuple:
        'Apply patch-based masking to image.'
        B, C, H, W = image.shape


        patch_size = 4
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size


        mask_patches = torch.rand(B, 1, num_patches_h, num_patches_w, device=image.device) > mask_ratio


        mask = mask_patches.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)


        mask_token_expanded = self.mask_token.expand(B, -1, H, W)
        masked_image = torch.where(mask, image, mask_token_expanded)

        return masked_image, mask

    def encode(self, image: torch.Tensor, ir: torch.Tensor | None = None) -> torch.Tensor:
        'Encode image (and optionally IR) to latent.'
        image_latent = self.encoder(image)

        if self.ir_encoder is not None and ir is not None:
            ir_latent = self.ir_encoder(ir)
            return torch.cat([image_latent, ir_latent], dim=-1)
        else:
            return image_latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        'Decode latent to image (uses only image portion of latent).'
        image_latent = latent[:, :self.latent_dim]
        return self.decoder(image_latent)

    def forward(self, image: torch.Tensor, ir: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        'Forward pass with masked autoencoding for training.'
        B = image.shape[0]
        device = image.device


        mask_ratio = torch.rand(1, device=device).item() * 0.9


        masked_image, mask = self._patchify_and_mask(image, mask_ratio)


        latent = self.encoder(masked_image)


        recon = self.decode(latent)


        mse_loss = F.mse_loss(recon, image)



        _, features_recon = self.encoder(recon, return_features=True)
        with torch.no_grad():
            _, features_original = self.encoder(image, return_features=True)
        perceptual_loss = self.perceptual_loss(features_recon, features_original)
        food_saliency_loss = self._green_saliency_loss(recon, image)

        if self.ir_encoder is not None and ir is not None:
            ir_latent = self.ir_encoder(ir)
            combined_latent = torch.cat([latent, ir_latent], dim=-1)
            ir_pred = self.ir_decoder(combined_latent)

            ir_loss = F.mse_loss(ir_pred, ir)
        else:
            combined_latent = latent
            ir_pred = None
            ir_loss = torch.zeros((), device=image.device)



        image_loss = mse_loss + 0.2 * perceptual_loss + 2.0 * food_saliency_loss
        total_loss = self._normalize_loss(image_loss, "image_loss_rms")
        if self.ir_decoder is not None and ir is not None:
            total_loss = total_loss + self._normalize_loss(ir_loss, "ir_loss_rms")

        return {
            "latent": combined_latent,
            "recon": recon,
            "ir_recon": ir_pred,
            "loss": total_loss,
            "mse_loss": mse_loss,
            "perceptual_loss": perceptual_loss,
            "food_saliency_loss": food_saliency_loss,
            "ir_loss": ir_loss,
            "perceptual_metric": "fallback_encoder_features",
        }
