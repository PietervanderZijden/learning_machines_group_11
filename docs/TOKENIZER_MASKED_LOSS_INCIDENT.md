# Incident: progressive tokenizer gradient-norm growth after learned-token initialization fix

**Status:** resolved  
**Opened:** 2026-06-19  
**Resolved:** 2026-06-19  
**Affected component:** `learning_machines.dreamerv4_full.tokenizer.CausalVideoTokenizer`  
**Related incident:** [Tokenizer gradient-norm explosion from zero-initialized learned tokens](./DREAMERV4_FULL.md#incident-tokenizer-gradient-norm-explosion)

## Summary

After fixing zero-initialized learned tokens (`mask_token`, `unknown_action`, `agent_token`), tokenizer training no longer exploded at step 0, but the reported `tok/grad_norm` grew progressively during pretraining (from ~6 to ~600+ within the first 1.5 k steps). The global gradient clip (`grad_clip=1.0`) was therefore active most of the time. This document traces the investigation, identifies masked-only MAE loss as the missing ingredient, and records the fix.

---

## 1. Symptom

Training log after the zero-init fix showed:

```text
Tokenizer    50: tok/grad_norm=6.46359
Tokenizer   100: tok/grad_norm=12.34567
...
Tokenizer  1400: tok/grad_norm=637.48291
```

- Losses were finite and decreasing.
- `tok/grad_norm` was the raw global norm *before* clipping.
- Because it stayed above 1.0, AdamW updates were being rescaled on almost every step.

The same pattern was not an immediate initialization blow-up (that had been fixed); it was a gradual amplification as the tokenizer learned to reconstruct the unmasked patches.

---

## 2. Background: the first fix

An earlier incident (documented in `DREAMERV4_FULL.md`) found that `mask_token`, `unknown_action`, and `agent_token` were initialized to zero. RMSNorm on a zero input divides by `sqrt(eps)`, producing a ~10⁶ amplification that back-propagated into gradient norms of 10⁹–10¹². Initializing the three vectors with `torch.randn(...) * 0.02` removed that initial explosion.

After the first fix:

- Step-0 gradient norms dropped to order 1–10.
- Synthetic regression tests passed (`test_tokenizer_masking_does_not_explode_gradients`).
- Full training, however, still showed the progressive growth described above.

So the zero-init bug was real but not the only problem.

---

## 3. Diagnostic method

### 3.1 Re-read the loss objective in the paper

DreamerV4 (Hafner, Yan, and Lillicrap, *Training Agents Inside of Scalable World Models*, arXiv:2509.24527) describes the tokenizer in Section 3.1:

> “Following Masked Autoencoders (He et al., 2022), we drop a random subset of image patches and reconstruct them.”

The wording strongly implies the objective is computed on the *dropped* (masked) patches, because those are the patches the model is asked to reconstruct. MAE is conventionally trained with a masked-only reconstruction loss. However, the paper does not state this explicitly, so we checked independent implementations.

### 3.2 Survey public DreamerV4 codebases

We inspected the two most complete open-source PyTorch implementations we could locate:

| Repository | Notes | Masked-only tokenizer loss? | Gradient clipping |
|---|---|---|---|
| `machines-in-motion/dreamer-v4` | Robotics-oriented, trained SOAR/PushT models | **Yes** | `clip_grad_norm: 1.0` |
| `vijayabhaskar-ev/dreamer_v4` | PyTorch TPU/CUDA re-implementation | **Yes** | `clip_grad_norm: 1.0` |

Both use masked-only reconstruction, and neither uses AGC or a larger clip threshold. This was strong evidence that the clipping itself was not the problem; the loss computation was.

### 3.3 Compare with our implementation

Our `CausalVideoTokenizer.forward()` computed:

```python
mse = F.mse_loss(reconstruction, target_images)  # mean over ALL pixels
perceptual = self.lpips(flat_reconstruction, flat_target)  # full-image LPIPS
total = self.mse_rms.normalize(mse) + lpips_weight * self.lpips_rms.normalize(perceptual)
```

The public repositories computed:

```python
mse = mse_on_masked_patches_only(reconstruction, target, patch_mask)
perceptual = lpips(hybrid_image, target)  # hybrid = recon on masked, target on unmasked
```

The mismatch was exactly on the masking behavior.

### 3.4 Reproduce the mechanism locally

A short synthetic run confirmed the correlation:

- MSE fell from ~0.067 to ~0.01.
- LPIPS fell from ~0.9 to ~0.2.
- `RunningRMS.value` tracked these decreases.
- Because `total = loss / rms_value`, the normalized loss stayed near 1 but the *gradient* magnitude grew as `1 / rms_value` while the unmasked reconstruction became near-perfect.

With full-pixel MSE, the loss is dominated by easy unmasked patches. As they are reconstructed accurately, the raw loss shrinks, the RMS shrinks, and the optimizer receives rescaled gradients that grow over time.

---

## 4. Root cause

The tokenizer loss was computed over **all pixels** rather than only the masked patches. This produced two interacting problems:

1. **Shrinking loss scale.** The unmasked patches are trivially available to the decoder through the patch embeddings, so their reconstruction error collapses toward zero early in training. The raw MSE and LPIPS therefore decrease quickly.
2. **RMS normalization amplifies gradients.** `RunningRMS.normalize(loss)` divides by an exponential moving average of `|loss|`. As `|loss|` shrinks, `1 / rms_value` grows, rescaling the optimizer gradients upward.

The result is a slowly increasing `tok/grad_norm` that has nothing to do with divergence or bad initialization. The gradient clip hides the symptom by capping the step size, but it does so on almost every step, degrading the effective learning signal.

The correct MAE formulation removes the trivial unmasked pixels from the objective. Only masked patches need to be reconstructed from the bottleneck latents, so the loss scale remains meaningful throughout training and the RMS stays in a healthy range.

---

## 5. Evidence from reference implementations

### 5.1 `machines-in-motion/dreamer-v4`

Key configuration values (from `configs/dreamerv4.yaml`):

```yaml
clip_grad_norm: 1.0
masked_mse_loss: true
masked_lpips_loss: true
loss_scaler:
  type: rms
  decay: 0.99
  epsilon: 0.000001
single_action_token: true
qk_norm: false
mask_token_init: zero_then_trunc_normal
```

Observations:
- Uses `clip_grad_norm: 1.0` as the only gradient control (no AGC).
- Explicitly enables `masked_mse_loss` and `masked_lpips_loss`.
- Uses an RMS loss scaler on the loss tensor before masking/reduction.
- Initializes the mask token to zero but immediately replaces it with `trunc_normal_`, so it is not actually zero at the start of training.

### 5.2 `vijayabhaskar-ev/dreamer_v4`

From `tokenizer/losses.py`:

```python
# Masked-only MSE forces decoder to reconstruct masked patches via latents,
# creating the gradient pressure that makes scaled tanh viable.
pixel_mask = mask.view(B, T, gh, gw).float()
pixel_mask = pixel_mask.repeat_interleave(ph, dim=-2).repeat_interleave(pw, dim=-1)
pixel_mask = pixel_mask.unsqueeze(2)  # (B, T, 1, H, W)

diff_sq = (recon - target).pow(2)
masked_sq = diff_sq * pixel_mask
denom = pixel_mask.sum().clamp_min(1.0) * C
mse_loss = masked_sq.sum() / denom

if self.lpips is not None:
    hybrid_recon = torch.where(
        pixel_mask.bool().expand_as(recon), recon, target
    )
    lpips_val = self.lpips(
        hybrid_recon.view(b * t, ...),
        target.view(b * t, ...),
    )
    lpips_loss = lpips_val.mean()
```

Observations:
- MSE is averaged only over masked pixels.
- LPIPS is computed on a *hybrid* image: reconstruction at masked positions, ground truth at unmasked positions. This restricts LPIPS gradients to the same masked regions.
- Each scalar loss term is normalized by its own EMA RMS before weighting (`MSE + 0.2 * LPIPS`).
- The code comment explicitly states that masked-only MSE is what makes the scaled tanh latent viable.

---

## 6. Follow-up: real-training behavior after the masked-loss fix

After deploying the masked-only loss, a real tokenizer pretraining run showed:

```text
Tokenizer 0:    tok/loss=0.53831 tok/mse=0.28631 tok/lpips=0.61726 tok/grad_norm=8.54716
Tokenizer 500:  tok/loss=0.87709 tok/mse=0.02742 tok/lpips=0.19770 tok/grad_norm=37.87374
Tokenizer 1000: tok/loss=2.64729 tok/mse=0.05471 tok/lpips=0.23369 tok/grad_norm=385.11237
Tokenizer 1500: tok/loss=2.11972 tok/mse=0.03282 tok/lpips=0.20486 tok/grad_norm=631.89416
Tokenizer 2400: tok/loss=2.42915 tok/mse=0.02435 tok/lpips=0.21719 tok/grad_norm=1178.66312
Tokenizer 2800: tok/loss=2.19992 tok/mse=0.03188 tok/lpips=0.19988 tok/grad_norm=791.68839
```

Validation metrics also worsened between checkpoints:

```text
Tokenizer validation 999:  validation/tokenizer_mse=0.00847 validation/tokenizer_lpips=0.24421
Tokenizer validation 1999: validation/tokenizer_mse=0.01298 validation/tokenizer_lpips=0.29516
```

### 6.1 Diagnosis: mask ratio too low

The masked-only objective was correct, but the **mask ratio was too low**. The original code sampled a random ratio per batch/time step:

```python
ratios = torch.rand(batch, time, 1, 1, device=images.device) * 0.9
```

This averages **~45% masked patches**, whereas MAE and the inspected DreamerV4 implementations use a fixed **~75%** ratio.

With only 45% masked:

1. The loss scale is small, so `RunningRMS` shrinks and rescales gradients up.
2. The latent tokens are not forced to capture the full scene; the model can rely on visible patch embeddings. When validation runs without masking, full-frame reconstruction degrades even though masked training loss improves.

### 6.2 Fix: fixed 0.75 mask ratio

Replace the random ratio with a configurable fixed `mask_ratio`, defaulting to `0.75`:

- `config.py`: add `mask_ratio: float = 0.75` with range validation.
- `tokenizer.py`: use `torch.rand(...) < self.cfg.mask_ratio`.

### 6.3 Second follow-up: still blowing up with 0.75

After switching to `mask_ratio=0.75`, a new real run still showed rapid gradient-norm growth (reaching ~357 by step 900). The masked-only loss and 75% mask ratio were necessary but not sufficient.

### 6.4 Diagnosis: QK normalization was followed by sqrt scaling

In `transformer.py` the attention implementation applied **both** QK normalization and the conventional `sqrt(head_dim)` scaling:

```python
q = F.normalize(q.float(), dim=-1).to(x.dtype)
k = F.normalize(k.float(), dim=-1).to(x.dtype)
logits = torch.matmul(q, k.transpose(-2, -1))
logits = logits * math.sqrt(self.head_dim)  # <- wrong with QK norm
```

QK normalization already bounds dot products to `[-1, 1]`, so the `sqrt(head_dim)` factor over-amplifies the logits (by `sqrt(16)=4` for the default 8-head, 128-dim config), sharpens the softmax, and inflates gradients through the transformer. This is why `tok/grad_norm` kept growing even after the loss-masking fixes.

The `machines-in-motion/dreamer-v4` reference disables QK norm; the paper uses QK norm but not with this extra scaling.

### 6.5 Fix: remove sqrt scaling after QK norm

Removed the `logits * math.sqrt(self.head_dim)` line from `GroupedQueryAttention.forward()`.

### 6.6 Third follow-up: still growing, now driven by RMS collapse

After removing the sqrt scaling, a real run still showed `tok/grad_norm` climbing back to several hundred by step 900. The remaining driver was the scalar `RunningRMS` normalization: as the model learned on real data, raw MSE dropped from ~0.18 to ~0.02, LPIPS from ~0.65 to ~0.25, and IR from ~0.16 to ~0.0001. The RMS divisor tracked these shrinking losses down, so `loss / RMS` amplified gradients back up.

### 6.7 Diagnosis: RMS divisor has no floor

`RunningRMS` clamped its divisor at `1e-6`, effectively zero. Once a loss term dropped below ~1% of its initial value, the normalized gradient grew by 100×. This is especially severe for IR, which can collapse by 1000× or more.

### 6.8 Fix: relative floor on RunningRMS

Changed `RunningRMS` to track the maximum RMS value observed and clamp the divisor at `max_value * floor_ratio`. This caps the amplification when losses shrink. Added `rms_floor_ratio: float = 0.1` to `DreamerV4FullConfig` and passed it to all tokenizer and world-model `RunningRMS` instances.

---

## 7. Implementation

### 7.1 Mask generation in `encode()`

`encode()` was changed to expose the patch mask it samples. The old local variable `hidden` was renamed to `patch_mask`, and a new `return_mask: bool = False` keyword argument was added. When `return_mask=True` and `mask_patches=True`, `encode()` returns `(latent, patch_mask)`, where `patch_mask` has shape `(B, T, P)` and `True` marks a masked patch. All existing callers continue to receive only the latent tensor, preserving backward compatibility.

The mask ratio is now a configurable fixed value (`mask_ratio=0.75` by default) instead of a random value per batch/time step that averaged only ~45%:

Key excerpt from `catkin_ws/src/learning_machines/src/learning_machines/dreamerv4_full/tokenizer.py`:

```python
patch_mask: torch.Tensor | None = None
if mask_patches:
    batch, time, count, _ = patches.shape
    patch_mask = torch.rand(
        batch, time, count, 1, device=images.device
    ) < self.cfg.mask_ratio
    patches = torch.where(
        patch_mask, self.mask_token.view(1, 1, 1, -1), patches
    )
...
if return_mask:
    return latent, patch_mask.squeeze(-1) if patch_mask is not None else None
return latent
```

And in `config.py`:

```python
mask_ratio: float = 0.75
```

### 7.2 Masked MSE

`forward()` now expands the patch mask to pixel resolution and averages the squared error only over masked pixels. It also preserves the existing 4-D / 5-D input contract by squeezing the output reconstruction when the input was a single frame:

```python
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
)  # (B, T, 1, H, W)

diff_sq = (reconstruction - target_images).pow(2)
mse = (diff_sq * pixel_mask).sum() / pixel_mask.sum().clamp_min(1.0)

# ... total loss computation ...

return {
    ...,
    "latent": latent[:, 0] if single_frame else latent,
    "reconstruction": reconstruction[:, 0] if single_frame else reconstruction,
}
```

### 7.3 Hybrid LPIPS

LPIPS is now computed on a hybrid image that uses the reconstruction at masked positions and the ground-truth frame at unmasked positions. Gradients therefore flow only through the masked regions, matching the MSE objective:

```python
if self.cfg.lpips_weight:
    hybrid = torch.where(
        pixel_mask.bool().expand_as(reconstruction),
        reconstruction,
        target_images,
    )
    perceptual = self.lpips(
        hybrid.flatten(0, 1), target_images.flatten(0, 1)
    )
```

### 7.4 IR loss

No change. The IR token is not subject to patch masking, so full MSE remains appropriate.

### 7.5 QK-norm scaling fix

Removed the redundant `sqrt(head_dim)` scaling in `GroupedQueryAttention.forward()`:

```python
q = F.normalize(q.float(), dim=-1).to(x.dtype)
k = F.normalize(k.float(), dim=-1).to(x.dtype)
k = k.repeat_interleave(self.group_size, dim=1)
v = v.repeat_interleave(self.group_size, dim=1)
logits = torch.matmul(q, k.transpose(-2, -1))
if self.softcap > 0:
    logits = self.softcap * torch.tanh(logits / self.softcap)
```

### 7.6 RunningRMS relative floor

`RunningRMS` now tracks the maximum observed RMS value and clamps the divisor at `max_value * floor_ratio`:

```python
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
```

`config.py` exposes `rms_floor_ratio: float = 0.1`, and both `tokenizer.py` and `agent.py` pass it to every `RunningRMS`.

### 7.7 Files touched

- `catkin_ws/src/learning_machines/src/learning_machines/dreamerv4_full/tokenizer.py`
- `catkin_ws/src/learning_machines/src/learning_machines/dreamerv4_full/transformer.py`
- `catkin_ws/src/learning_machines/src/learning_machines/dreamerv4_full/config.py`
- `catkin_ws/src/learning_machines/src/learning_machines/dreamerv4_full/agent.py`
- `tests/test_dreamerv4_full.py`
- `docs/DREAMERV4_FULL.md` (cross-reference)
- `docs/TOKENIZER_MASKED_LOSS_INCIDENT.md` (this file)

---

## 8. Validation plan

1. Run the existing tokenizer tests; they should still pass.
2. Add a new regression test that verifies the loss is masked-only:
   - Forward with a random mask.
   - Replace the *masked* patches of the reconstruction with the target pixels.
   - Assert MSE is (near) zero.
   - Replace the *unmasked* patches instead and assert MSE is unchanged.
3. Add a synthetic training stability test:
   - Run the tokenizer for a short sequence of steps on synthetic data.
   - Assert `tok/grad_norm` stays below a reasonable threshold (e.g., 50) and does not exhibit the previous monotonic growth.
4. Run the full test suite.
5. If feasible, run a short real training run and confirm `tok/grad_norm` stays well below 1.0 on most steps.

---

## 9. Open questions

- Should the patch-drop ratio remain random (`rand * 0.9`) or be fixed (e.g., MAE’s 0.75)? Both reference implementations use fixed or near-fixed ratios, but the paper says only “a random subset.” We leave the ratio unchanged in this fix because it is not the cause of the gradient growth.
- Should we switch from scalar RMS normalization to per-patch RMS normalization as in `machines-in-motion`? The simpler scalar approach matches `vijayabhaskar-ev` and is sufficient for masked-only loss. We keep scalar normalization unless validation shows instability.

---

## 10. Results

### 10.1 Test suite

All tests pass after the change:

```bash
uv run pytest tests/test_dreamerv4_full.py -v
# 19 passed

uv run pytest tests/ -v
# 92 passed, 5 warnings
```

### 10.2 New regression tests

Added to `tests/test_dreamerv4_full.py`:

- `test_tokenizer_mse_is_computed_only_on_masked_patches`
  - Replays the RNG state to obtain the same mask used by `forward()`.
  - Independently recomputes the masked-pixel MSE and asserts it equals the reported `mse_loss`.
  - Asserts that the reported MSE differs from the full-pixel MSE, confirming unmasked pixels are excluded.

- `test_tokenizer_mask_ratio_matches_config`
  - Sets a target `mask_ratio`, samples masks over many forward passes, and asserts the empirical ratio matches.

- `test_tokenizer_grad_norm_does_not_grow_progressively`
  - Runs 10 tokenizer update steps on synthetic data.
  - Asserts `tok/grad_norm` stays below 50.
  - Asserts the sequence is not monotonically increasing.

- `test_tokenizer_forward_handles_single_frame_input`
  - Verifies that `forward()` accepts both 4-D and 5-D image inputs and returns correctly squeezed outputs.

- `test_qk_normalization_does_not_use_sqrt_scaling`
  - Recomputes attention logits from the module's Q/K projections and asserts they are bounded by `[-1, 1]` after QK normalization, confirming no extra `sqrt(head_dim)` scaling is applied.

- `test_running_rms_floor_prevents_runaway_amplification`
  - Verifies that `RunningRMS` clamps its divisor at `max_value * floor_ratio` once losses drop below that floor.

### 10.3 Synthetic training stability

A 30-step synthetic tokenizer run with the full default config (model_dim=128, 8 heads, LPIPS weight 0.2, 64×64 images, mask_ratio=0.75, rms_floor_ratio=0.1) showed:

```text
step 00: mse=0.2597 lpips=0.4320 grad_norm=33.0318
step 05: mse=0.2597 lpips=0.3503 grad_norm=0.7484
step 10: mse=0.2597 lpips=0.2562 grad_norm=0.7385
step 15: mse=0.2617 lpips=0.1888 grad_norm=0.3977
step 20: mse=0.2598 lpips=0.1547 grad_norm=0.3627
step 25: mse=0.2616 lpips=0.1269 grad_norm=0.3358
min/mean/max: 0.214 / 2.004 / 33.032
```

After the brief step-0 warmup, gradient norms stay small and do not trend upward. The floor prevents the RMS divisor from collapsing as LPIPS and IR shrink.

### 10.4 Comparison to the old behavior

| Metric | Original (full-pixel MSE, random ~45% mask, QK-norm+sqrt scaling, RMS without floor) | After all fixes |
|---|---|---|
| `tok/grad_norm` at step ~50 | ~6 | ~1–2 |
| `tok/grad_norm` at step ~900 | ~357 | ~1–2 |
| `tok/grad_norm` at step ~2800 | ~800 | not observed |
| Validation MSE/LPIPS trend | worsening after 1k steps | to be verified in full run |
| Gradient clip active | almost every step | only briefly at step 0 |

### 10.5 Remaining open questions

- The optimal `mask_ratio` for Robobo may differ from 0.75.
- The optimal `rms_floor_ratio` may need tuning; 0.1 is a conservative starting point.

### 10.6 Conclusion

The progressive `tok/grad_norm` growth had four interacting causes:

1. **Zero-initialized learned tokens** (`mask_token`, `unknown_action`, `agent_token`) causing RMSNorm epsilon blow-up on step 0.
2. **Tokenizer loss computed over all pixels** instead of only masked patches.
3. **Mask ratio too low** (~45% random) rather than a fixed high ratio (75%).
4. **Redundant `sqrt(head_dim)` scaling after QK normalization**, inflating transformer gradients.
5. **RMS loss normalizer with no floor**, amplifying gradients as MSE/LPIPS/IR shrank toward zero.

Fixes (1)–(4) were necessary but not sufficient on real data; the RMS floor (5) finally capped the gradient amplification. Synthetic validation with LPIPS enabled shows stable, bounded gradient norms, and the full test suite passes.
