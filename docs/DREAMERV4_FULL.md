# DreamerV4 full implementation

This repository contains two DreamerV4-related agents:

- `learning_machines.dreamerv4`: the original compact experimental agent.
- `learning_machines.dreamerv4_full`: the paper-aligned implementation.

The full implementation follows *Training Agents Inside of Scalable World
Models* (Hafner, Yan, and Lillicrap, 2025), including:

- causal patch/latent-token video tokenization;
- randomized masked autoencoding with MSE and frozen LPIPS;
- tanh latent bottlenecks;
- a shared factorized space-time transformer architecture;
- pre-layer RMSNorm, RoPE, SwiGLU, QK normalization, attention soft-capping,
  grouped-query attention, and temporal attention every fourth layer;
- spatial latent and register tokens;
- shortcut forcing with x-prediction, power-of-two step sampling,
  stop-gradient two-half-step targets, and ramp weighting;
- four-step interactive generation with corrupted context;
- causally isolated task/agent tokens;
- joint noisy world-model and length-8 multi-token policy/reward finetuning;
- symlog two-hot reward and value heads;
- frozen-world-model imagination training;
- TD-lambda values and sign-based PMPO with reverse behavioral-prior KL.

The paper does not include source code or all architectural and optimizer
hyperparameters. Consequently, this implementation should not be presented as
an exact reproduction of the authors' system. It implements the published
equations and explicit architecture statements, with the adaptations and
remaining uncertainties documented below.

## Intentional Robobo adaptations

The paper trains a 2-billion-parameter Minecraft model with discrete and binary
mouse/keyboard actions. Robobo has two bounded continuous wheel commands, so
the policy is a tanh-Gaussian distribution. Terminal continuation prediction
is retained because Robobo episodes terminate. Model width, token counts, and
batch lengths are configurable to fit a single workstation GPU.

Robobo actions are transition actions. Shortcut world-model training aligns
\(a_t\) with the arrival representation \(z_{t+1}\). Policy states use \(z_t\)
and the preceding action, avoiding action-label leakage into behavior cloning.
Reward and continuation heads follow Equation (9)'s \(r_{t+n}\) indexing;
during imagination each generated transition is annotated after producing its
arrival state.

The default discount is the paper value `0.997`. If discounting should preserve
the same real-time horizon under Robobo's 400 ms control interval, override it
in `DreamerV4FullConfig` and document the conversion.

## Remaining reproduction limits

- The paper's 2B-parameter widths, token counts, dataset scale, and TPU training
  setup are not practical defaults for this project.
- The exact ordering of factorized space/time layers, QK-normalization
  parameterization, initialization, optimizer schedule, and several masking
  details are not fully specified in the paper.
- Robobo uses a tanh-Gaussian continuous policy instead of the paper's
  categorical/vectorized-binary Minecraft policy.
- The repository has only action-labelled Robobo episodes, so the paper's
  large-scale unlabeled-video pretraining and action-alignment experiments are
  not reproduced.
- Agent MTP losses and shortcut prediction share the same transformer and are
  optimized jointly, but use separate noisy forwards to keep transition and
  policy action alignment explicit for the Robobo data contract.
- The implementation is functionally autoregressive but does not implement the
  production KV-cache kernels needed to reproduce the paper's throughput.

## Training

Install the locked dependencies first. The first tokenizer update downloads
the standard frozen AlexNet weights used by LPIPS.

```bash
uv sync

python train_dreamerv4_full.py \
  --record-dir recorded-states/dreamer-v3-states \
  --checkpoint-dir results/dreamer-v4-full-checkpoints
```

Training is separated into the four phases from the paper. Progress and all
optimizer states are checkpointed. Resume with:

```bash
python train_dreamerv4_full.py \
  --record-dir recorded-states/dreamer-v3-states \
  --checkpoint-dir results/dreamer-v4-full-checkpoints \
  --resume
```

Short and long sequence batches are alternated. Both batch lengths must exceed
the configured context length, matching the paper's length-generalization
requirement. World-model pretraining ends with a long-sequence-only finetuning
stage. By default, 30% of its samples are standalone start frames, matching the
paper's start-frame generation setup.

## Training-quality features

The implementation includes the following stability and model-selection
features:

- truncated-normal transformer initialization with depth-scaled residual
  projections;
- zero-initialized dynamics, reward, value, continuation, and policy outputs;
- AdamW decay only on matrix weights, excluding biases, norms, embeddings, and
  learned tokens;
- per-phase linear warmup followed by cosine decay to a configurable minimum
  learning-rate ratio;
- BF16 autocast by default on supported GPUs, with FP16 plus gradient scaling
  available as an alternative;
- global finite-gradient checking and clipping;
- optimizer-moment transfer from world pretraining into joint finetuning and
  from behavior finetuning into PMPO;
- explicit unknown-action embeddings for standalone or unlabeled frames;
- deterministic episode-level held-out validation;
- multiple validation batches per report;
- phase-specific best checkpoints in addition to the latest resumable
  checkpoint;
- cached encoded latents, avoiding a full tokenizer pass on every resume.

The learning-rate schedule and gradient-scaler state are stored in checkpoints,
so resumed training continues at the exact optimization step.

Generated checkpoint files include:

- `dreamerv4_full_latest.pt`: latest complete resumable state;
- `dreamerv4_full_best_tokenizer.pt`;
- `dreamerv4_full_best_world.pt`;
- `dreamerv4_full_best_world_long.pt`;
- `dreamerv4_full_best_finetune.pt`;
- `dreamerv4_full_best_imagination.pt`;
- `best_metrics.json`: validation metrics used for each selection.

Useful controls include:

```text
--warmup-fraction 0.05
--minimum-lr-ratio 0.1
--mixed-precision-dtype bfloat16
--no-mixed-precision
--validation-fraction 0.1
--validation-every 1000
--validation-batches 4
--reencode
```

For stable full runs, keep warmup enabled and do not disable gradient clipping.

## Incident: tokenizer gradient-norm explosion

Date: 2026-06-19.

### Symptom

During tokenizer pretraining the reported `tok/grad_norm` was consistently in
the hundreds to over a thousand, for example:

```text
Tokenizer 2500: tok/grad_norm=1000.17691
Tokenizer 2550: tok/grad_norm=1153.26779
Tokenizer 2600: tok/grad_norm=1031.72395
```

The global gradient clip (`grad_clip=1.0`) was therefore active on almost every
step. The loss was finite and not diverging, but the optimizer was permanently
pinned to the clipping boundary.

### Initial hypothesis

The high norm was first attributed to the LPIPS perceptual loss and the
running-RMS loss normalization: small raw losses (MSE ~0.01, LPIPS ~0.2) are
divided by small RMS values, which rescales gradients up. Under this reading
the clipping was merely suboptimal.

### Diagnostic method

1. Ran the existing test suite. All tests passed, including the finite-gradient
checks, so the gradients were finite but large rather than NaN/Inf.
2. Measured per-parameter gradient norms after a single tokenizer forward/backward
on a small synthetic batch.
3. The largest gradient belonged to `CausalVideoTokenizer.mask_token`, with a
norm of order 10⁹–10¹⁰ on a 64×64-image config.
4. Systematically removed components to isolate the cause:
   - Disabling patch masking made the gradient norm drop to ~10⁻³.
   - Disabling temporal layers did not fix it.
   - Reducing to a single transformer layer did not fix it.
   - Disabling QK normalization (by replacing `F.normalize` with the identity)
     made the gradient norm drop to ~1.
5. The same pattern appeared in `InteractiveDynamics.unknown_action` when all
actions were marked unknown: its gradient norm reached ~10¹¹.

### Root cause

`mask_token`, `unknown_action`, and `agent_token` were initialized to zero.
RMSNorm computes

```
scale = (mean(x²) + eps)^(-1/2)
```

For a zero input this becomes `1/sqrt(eps) ≈ 10⁶`. The forward output is still
zero, but on the backward pass the normalization factor amplifies downstream
gradients by ~10⁶. Through the transformer the effect compounded into
catastrophic gradient norms for the learned embedding parameters.

Because `finite_grad_norm` clips the *global* norm to `grad_clip=1.0`, the
problem was hidden: training looked stable, but the learned mask/action tokens
were receiving only clipped, directionally degraded updates.

### Fix

Initialize the three learned embedding vectors with small random values instead
of zeros:

- `dreamerv4_full/tokenizer.py`: `self.mask_token`
- `dreamerv4_full/model.py`: `self.unknown_action`
- `dreamerv4_full/model.py`: `self.agent_token`

All three now use `torch.randn(model_dim) * 0.02`, matching the scale used for
`encoder_latents`, `decoder_queries`, `ir_query`, and `registers`.

### Why this solves the issue

A non-zero learned token no longer triggers the RMSNorm epsilon path, so the
normalization factor stays near 1 and gradients remain well behaved. After the
change the same synthetic batch produced gradient norms of order 1–10 instead of
10⁹–10¹².

### Regression tests

Added to `tests/test_dreamerv4_full.py`:

- `test_tokenizer_masking_does_not_explode_gradients`
- `test_dynamics_unknown_actions_do_not_explode_gradients`

Both tests run a forward/backward with masking or unknown actions and assert
that every parameter gradient has norm below 10⁴.

### Validation

```text
uv run pytest tests/test_dreamerv4_full.py -v
# 13 passed

uv run pytest tests/ -v
# 86 passed, 5 warnings
```

### Implications for training

With this fix the tokenizer should report `tok/grad_norm` values that are
usually well below `grad_clip` (1.0 by default). Occasional spikes are still
possible from hard batches or LPIPS, but they will be genuine spikes rather than
a systematic initialization blow-up. If you still see frequent clipping, the
remaining cause is likely the LPIPS weight or batch length rather than the
learned-token initialization.

### Follow-up: progressive gradient-norm growth

After the initialization fix, training still showed `tok/grad_norm` growing
progressively (from ~6 to ~600+ over the first 1.5 k steps), keeping the
optimizer pinned to the `grad_clip=1.0` boundary. A long investigation found
four interacting causes:

1. The tokenizer computed MSE and LPIPS over **all** pixels instead of only
   masked patches.
2. The patch mask ratio averaged only **~45%** instead of the MAE-standard 75%.
3. The attention applied **QK normalization followed by `sqrt(head_dim)`
   scaling**, over-amplifying attention logits and inflating transformer
   gradients.
4. The `RunningRMS` loss normalizer had **no floor**, so as MSE/LPIPS/IR
   shrank toward zero the divisor collapsed and amplified gradients back up.

Switching to masked-only MAE loss, fixing the mask ratio to 0.75, removing the
redundant sqrt scaling, and adding a relative floor to `RunningRMS` brought
gradient norms under control. See
[`docs/TOKENIZER_MASKED_LOSS_INCIDENT.md`](TOKENIZER_MASKED_LOSS_INCIDENT.md)
for the full paper trail.

Additional regression tests added for these fixes:

- `test_tokenizer_mse_is_computed_only_on_masked_patches`
- `test_tokenizer_mask_ratio_matches_config`
- `test_tokenizer_grad_norm_does_not_grow_progressively`
- `test_tokenizer_forward_handles_single_frame_input`
- `test_qk_normalization_does_not_use_sqrt_scaling`
- `test_running_rms_floor_prevents_runaway_amplification`

The fixes added `mask_ratio` (default `0.75`) and `rms_floor_ratio`
(default `0.1`) to `DreamerV4FullConfig`, removed the `sqrt(head_dim)`
attention scaling from `transformer.py`, and updated `RunningRMS` in
`tokenizer.py`.

## Evaluation

```bash
python evaluate_transfer.py \
  --algorithm dreamerv4-full \
  --checkpoint results/dreamer-v4-full-checkpoints/dreamerv4_full_latest.pt \
  --manifest results/dreamer-v4-full-checkpoints/manifest.json
```

Hardware deployment uses the same `dreamerv4-full` algorithm name and retains
the tokenizer and dynamics context between control steps.

## Source layout

- `dreamerv4_full/config.py`: architecture and objective configuration.
- `dreamerv4_full/transformer.py`: efficient space-time transformer.
- `dreamerv4_full/tokenizer.py`: causal tokenizer and LPIPS objective.
- `dreamerv4_full/model.py`: shortcut dynamics and MTP heads.
- `dreamerv4_full/agent.py`: the four optimization phases and checkpointing.
- `train_dreamerv4_full.py`: offline Robobo training pipeline.
