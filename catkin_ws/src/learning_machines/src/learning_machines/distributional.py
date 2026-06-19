"""Shared distributional utilities for DreamerV3/V4.

Implements symlog-space two-hot bins as described in:
- DreamerV3: "Mastering Diverse Domains through World Models" (Hafner et al.)
- DreamerV4: "Training Agents Inside of Scalable World Models" (Hafner et al.)

Key design:
- Bins are in SYMLOG space (linear linspace from -20 to 20)
- Targets are symlog-transformed before encoding
- Values are symexp-transformed after decoding
- Two-hot encoding uses searchsorted for correct bin ordering
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

_NUM_BINS = 255


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: sign(x) * log(|x| + 1)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)."""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def _make_bin_centers(device: torch.device) -> torch.Tensor:
    """Create bin centers in SYMLOG space.

    Returns:
        (NUM_BINS,) tensor of linearly spaced bins from -20 to 20.
        These are in symlog space. Use symexp() to convert back to original space.
    """
    return torch.linspace(-20.0, 20.0, _NUM_BINS, device=device)


def two_hot_encode(target: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """Encode scalar targets as two-hot vectors using linear interpolation.

    Uses searchsorted for correct bin ordering (not topk which may return
    indices in arbitrary order).

    Args:
        target: (...,) scalar values in SYMLOG space
        bin_centers: (NUM_BINS,) sorted bin centers in SYMLOG space

    Returns:
        soft_target: (..., NUM_BINS) two-hot encoded vectors
    """
    target = target.unsqueeze(-1)  # (..., 1)

    # Use searchsorted to find the right bin index
    # searchsorted returns index where target would be inserted to keep sorted order
    indices = torch.searchsorted(bin_centers, target.squeeze(-1))  # (...,)
    indices = indices.clamp(1, len(bin_centers) - 1)  # ensure we have both neighbors

    idx_low = indices - 1  # (...,)
    idx_high = indices  # (...,)
    val_low = bin_centers[idx_low]  # (...,)
    val_high = bin_centers[idx_high]  # (...,)

    # Linear interpolation weight
    w_high = torch.clamp(
        (target.squeeze(-1) - val_low) / (val_high - val_low + 1e-8), 0.0, 1.0
    )
    w_low = 1.0 - w_high

    # Scatter weights into soft target
    soft_target = torch.zeros(*target.shape[:-1], _NUM_BINS, device=target.device)
    soft_target.scatter_(-1, idx_low.unsqueeze(-1), w_low.unsqueeze(-1))
    soft_target.scatter_add_(-1, idx_high.unsqueeze(-1), w_high.unsqueeze(-1))
    return soft_target


def two_hot_loss(
    logits: torch.Tensor,
    target_raw: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Distributional two-hot loss.

    Args:
        logits: (..., NUM_BINS) predicted logits
        target_raw: (...,) scalar values in ORIGINAL space (will be symlog-transformed)

    Returns:
        loss: scalar for reduction="mean", otherwise per-sample loss
    """
    bins = _make_bin_centers(logits.device)
    # Transform target to symlog space before encoding
    target_symlog = symlog(target_raw).clamp(bins[0], bins[-1])
    soft_target = two_hot_encode(target_symlog, bins)
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_target * log_probs).sum(-1)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction != "mean":
        raise ValueError(f"Unsupported reduction: {reduction}")
    return loss.mean()


def logits_to_value(logits: torch.Tensor) -> torch.Tensor:
    """Convert distributional logits to expected scalar value.

    Args:
        logits: (..., NUM_BINS) predicted logits in symlog space

    Returns:
        value: (...,) expected values in ORIGINAL space (symexp-transformed)
    """
    bins = _make_bin_centers(logits.device)
    probs = F.softmax(logits, dim=-1)
    # The categorical distribution lives on symexp-transformed bin values.
    # Taking symexp after averaging the symlog bins is incorrect because
    # symexp is nonlinear and systematically biases broad distributions.
    return (probs * symexp(bins)).sum(-1)
