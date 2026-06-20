# DreamerV3 Training Analysis

## 1. KL Divergence: kl_dyn == kl_rep (Expected Behavior)

Both `kl_dyn` and `kl_rep` always produce identical numerical values.
This is by design — both compute KL(posterior ‖ prior).

The difference is **gradient flow only** (not logged value):
- `kl_dyn`: posterior detached → gradients train the **prior** (dynamics)
- `kl_rep`: prior detached → gradients train the **posterior** (encoder)

Code: `rssm.py:kl_loss` (~line 170)

### KL value interpretation
- ~1.655 (above free_nats=1.0): healthy — posterior is learning to diverge from prior
- Drops to ~1.02: normal training dynamics, not posterior collapse
- If reported KL is exactly 1.0, the free-nats clamp is active. This only
  suggests posterior collapse if raw KL diagnostics are also near zero.

---

## 2. Blurry Decoded Images

**Root cause**: MSE pixel reconstruction loss averages over possible futures.

Decoder architecture (`world_model.py:67-95`):
- Input: RSSM state (h, z) → 512+1024 = 1536-dim
- FC layer: 1536 → 256*4*4
- 4 ConvTranspose2d layers: 4x4 → 8 → 16 → 32 → 64
- Output: Sigmoid (float [0,1])

This is a known DreamerV3 limitation. Blurry reconstruction is expected
and not automatically fatal. The more important question is whether the
world model predicts food reward, continuation, and latent transitions
accurately.

---

## 3. Food Hallucinations in Reconstructions

Decoded images sometimes show green food blobs where the original
has none.

**Contributing factors** (multiple, not just one):

1. `green_saliency_loss` (`dreamerv3.py:41-52`) with `food_recon_weight=2.0`.
   The saliency weights target-green 20x, but also penalizes false green
   on non-food frames. The 2.0x multiplier creates a strong gradient signal
   that may bias the decoder toward preserving food-like patterns.

2. Small food objects are hard to encode in the stochastic state.

3. MSE decoder uncertainty — the decoder averages over possible futures,
   and food blobs are small high-frequency features.

4. Reward-event oversampling changes the frame distribution toward
   food-containing frames, potentially reducing negative examples.

5. Limited decoder capacity (4 conv-transpose layers from 4x4 to 64x64).

**Assessment**: The saliency loss contributes but is not the sole cause.
Hallucinations likely indicate latent ambiguity, decoder capacity limits,
and dataset bias rather than a direct consequence of the saliency term.

---

## 4. Identified Implementation Issues

### Issue 4.1: `logits_to_value()` — VERIFIED CORRECT (NOT AN ISSUE)

The current implementation (`distributional.py:95-100`):
```python
def logits_to_value(logits):
    bins = _make_bin_centers(logits.device)  # symexp([-20, ..., 20])
    probs = F.softmax(logits, dim=-1)
    return (probs * symexp(bins)).sum(-1)    # E[symexp(Y)]
```

This matches the DreamerV3 paper exactly. The paper defines:
- y = softmax(f(x))^T × B
- B = symexp([-20, ..., +20])

So the value IS: y = Σ p(b) × symexp(b) = E[symexp(Y)]

The paper's comment about "decoupling the size of the gradients from the
size of the targets" refers to the **training loss** (two-hot cross-entropy),
not the value conversion. The loss only depends on probabilities assigned
to bins, not on the continuous values of symexp(bins). The value conversion
for imagination/control is intended to be E[symexp(Y)].

**Verdict**: No issue. Current implementation is correct per the paper.

---

### Issue 4.2: Prefill not counted in global_step (FIXED)

**Status**: FIXED in `train_dreamerv3.py`

Added `agent.global_step += 1` during prefill so that training starts
immediately after prefill completes, eliminating the double warmup.

```python
# During prefill loop:
obs = next_obs
ir_obs = next_ir
steps_prefilled += 1
agent.global_step += 1  # Count prefill in global_step
```

---

### Issue 4.3: Replay ignores short successful episodes (FIXED)

**Status**: FIXED in `replay_buffer.py`

Removed the `ep_len < self.sequence_length` skip. Short episodes are now
sampled with their full length as the sequence length. All samples in a
batch are truncated to the minimum length to avoid padding issues.

```python
# Use full episode length if shorter than sequence_length
seq_len = min(self.sequence_length, ep_len)

# After sampling, truncate batch to minimum length
min_action_len = min(arr.shape[0] for arr in batch_action)
effective_seq_len = min(min_action_len, min_obs_len - 1)
```

---

### Issue 4.4: RSSM applies symlog to learned embeddings (FIXED)

**Status**: FIXED in `rssm.py` and `world_model.py`

Added `obs_is_embedding` flag to RSSM. When `True`, symlog and LayerNorm
are skipped — the observation is already a learned embedding from CNN/MLP encoder.

```python
# rssm.py — observe() now checks flag
if self.obs_is_embedding:
    obs_embed = F.silu(self.obs_embed(obs))
else:
    obs_symlog = torch.sign(obs) * torch.log1p(torch.abs(obs))
    obs_normed = self.obs_norm(obs_symlog)
    obs_embed = F.silu(self.obs_embed(obs_normed))

# world_model.py — passes flag based on encoder mode
self.rssm = RSSM(
    ...,
    obs_is_embedding=(use_multimodal or use_images),
)
```

**Reference implementations** (NM512/dreamerv3-torch, DrunkJin/dreamer-from-scratch):
- Apply symlog to raw inputs BEFORE encoding, not after
- RSSM receives pre-encoded embeddings without symlog

**Before fix**: Symlog applied to CNN features already normalized by LayerNorm + SiLU,
compressing learned representations and hurting small visual features like food.

---

### Issue 4.9: Decoder symlog usage (VERIFIED CORRECT)

The reconstruction loss correctly handles symlog per mode:

**Multimodal/CNN mode** (`dreamerv3.py:303-320`):
- CNN decoder outputs in [0, 1] (Sigmoid activation)
- Loss: `F.mse_loss(obs_pred, obs_target)` — raw MSE against raw images
- This is correct: decoder output and target are in same space

**Vector-only mode** (`dreamerv3.py:332-337`):
- MLP decoder outputs raw values (no activation)
- Loss: `F.mse_loss(obs_pred, symlog(target))` — decoder expected to output in symlog space
- This matches reference implementations: MLP decoder predicts in symlog space

**IR decoder** (`dreamerv3.py:314-315`):
- IR values are bounded in [0, 1] (calibrated)
- Loss: `F.mse_loss(ir_pred, ir_target)` — raw MSE
- Comment: "Calibrated IR is already bounded in [0, 1]; applying symlog would make deployment preprocessing differ from training"

**Reward predictor**: Uses `two_hot_loss` which handles symlog internally — raw rewards passed correctly.

**Verdict**: Decoder symlog usage is correct for all modes.

---

### Issue 4.5: Action flow analysis (NOT AN ISSUE)

The review raised concerns about action mismatch under domain randomization.
After tracing the full flow:

1. `DomainRandomizationWrapper.step()` applies `_randomize_action()`
2. Calls `self.env.step(randomized_action)`
3. `RoboboCompactEnv.step()` passes to `ActionExecutor.execute()`
4. `ActionExecutor` applies smoothing + safety filtering
5. Returns `executed_action` in action_info
6. `RoboboCompactEnv.step()` does `info.update(action_info)`
7. `DomainRandomizationWrapper.step()` adds `randomized_action` to info

The training loop uses:
```python
executed_action = info.get("executed_action", action)
```

This correctly gets the action AFTER smoothing/safety (the actual physical
action). The domain-randomized observation paired with the smoothed action
is the correct training signal — the agent learns "given this noisy
observation, this smoothed action was executed."

**Verdict**: No issue. Action attribution is correct.

---

### Issue 4.6: IR decoder unbounded output (FIXED)

**Status**: FIXED in `world_model.py`

Added Sigmoid activation to bound IR decoder output to [0, 1], matching
the calibrated IR target range.

```python
# Before:
self.ir_decoder = mlp(rssm_state_size, mlp_hidden, ir_dim, mlp_layers)

# After:
self.ir_decoder = nn.Sequential(
    mlp(rssm_state_size, mlp_hidden, ir_dim, mlp_layers),
    nn.Sigmoid(),
)
```

IR targets are NOT symlog'd (line 314: raw [0, 1] values), so the
decoder should output raw [0, 1] to match. Sigmoid ensures this.

---

### Issue 4.7: image_size hardcoded to 64 (LOW PRIORITY)

`CNNEncoder` and `CNNDecoder` assume 64x64 images with hardcoded:
```python
nn.Linear(256 * 4 * 4, embed_dim)  # encoder
self.fc = nn.Sequential(nn.Linear(state_dim, 256 * 4 * 4))  # decoder
```

If `--image-size` is changed, the model will silently produce wrong shapes.

**Fix**: Add assertion `if image_size != 64: raise ValueError(...)`

**Severity**: Low for default, high if changing image size.

---

### Issue 4.8: World model freezing should use try/finally (LOW PRIORITY)

```python
for p in self.world_model.parameters():
    p.requires_grad_(False)
# ... actor/critic updates ...
for p in self.world_model.parameters():
    p.requires_grad_(True)
```

If an exception occurs between these blocks, model stays frozen.

**Fix**: Wrap in try/finally.

**Severity**: Low — robustness improvement.

---

## 5. Summary: Priority-Ordered Fix List

| Priority | Issue | Status | Impact |
|----------|-------|--------|--------|
| ~~MEDIUM-HIGH~~ | ~~Replay ignores short successful episodes~~ | FIXED | Biased sampling, potential crash |
| ~~MEDIUM~~ | ~~Prefill not counted in global_step~~ | FIXED | Double warmup, misleading logs |
| ~~MEDIUM~~ | ~~RSSM applies symlog to learned embeddings~~ | FIXED | Reduced representation quality for visual features |
| MEDIUM | DreamerV4 forward() missing symlog | OPEN | Training/imagination preprocessing mismatch |
| ~~LOW-MEDIUM~~ | ~~IR decoder unbounded output~~ | FIXED | IR prediction instability |
| LOW | image_size hardcoded | OPEN | Breaks if changed |
| LOW | World model freezing lacks try/finally | OPEN | Robustness |

---

## 7. DreamerV4 Issues

### Issue 7.1: Transformer forward() missing symlog (MEDIUM PRIORITY)

DreamerV4's `CausalTransformer` has inconsistent preprocessing:
- `forward()` (training): does NOT apply symlog — `obs_tok = self.obs_embed(observations)`
- `imagine_trajectory()`: DOES apply symlog — `obs_symlog = symlog(h_obs)`

This means training and imagination use different preprocessing. The model
learns to predict in symlog space during imagination, but training targets
are in normalized space (not symlog'd).

**Fix**: Apply symlog in `forward()` to match `imagine_trajectory()`:
```python
# forward() — add symlog before embedding
obs_symlog = symlog(observations)
obs_tok = self.obs_embed(obs_symlog)
```

**File**: `dreamerv4/transformer.py:108-109`
**Severity**: Medium — causes train/inference distribution shift.

---

## 6. DreamerV3 Pipeline Files

### Core Algorithm
| File | Purpose |
|------|---------|
| `dreamerv3/dreamerv3.py` | Agent: ties WM + actor + critic, contains `train_step()` |
| `dreamerv3/rssm.py` | RSSM: prior/posterior networks, KL loss, observe/imagine |
| `dreamerv3/world_model.py` | CNN encoder/decoder, reward/continue heads, `observe_sequence()` |
| `dreamerv3/actor_critic.py` | Squashed Gaussian actor (REINFORCE), distributional critic |
| `dreamerv3/config.py` | All hyperparameters (DreamerV3Config dataclass) |
| `dreamerv3/replay_buffer.py` | Episode-based buffer with reward-event sampling |
| `dreamerv3/optim.py` | LaProp optimizer + adaptive gradient clipping |

### Environment & Infrastructure
| File | Purpose |
|------|---------|
| `rl_robobo_compact_env.py` | Gym env: obs (blob+IR), reward, food randomization |
| `domain_randomization.py` | Sensor/actuator/image randomization wrapper |
| `transfer.py` | Calibration profiles, observation/reward contracts, ActionExecutor |
| `safety_wrapper.py` | Emergency collision avoidance |
| `distributional.py` | Symlog bins, two-hot encoding, `logits_to_value` |

### Entry Points
| File | Purpose |
|------|---------|
| `train_dreamerv3.py` | Training CLI: args, env setup, training loop |
| `run_dreamerv3.sh` | Shell wrapper for `train_dreamerv3.py` |

### Key code references
- KL computation: `rssm.py:kl_loss` (~line 170)
- Food saliency loss: `dreamerv3.py:green_saliency_loss` (line 41)
- World model loss assembly: `dreamerv3.py:train_step` KL section (~line 270)
- Decoder: `world_model.py:CNNDecoder` (line 67)
- Value conversion: `distributional.py:logits_to_value` (line 95) — matches paper's E[symexp(Y)]
- Action flow: `transfer.py:ActionExecutor.execute` (line 266)
- Replay sampling: `replay_buffer.py:sample` (line 120)
