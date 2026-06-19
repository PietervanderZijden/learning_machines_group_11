# DreamerV3 and DreamerV4 Implementation Audit

## Summary

> **DreamerV3 remediation status (June 18, 2026):** The implementation has
> since been updated to correct two-hot decoding, add replay value learning,
> slow-value regularization, multi-state reward-aware imagination starts,
> continuation-weighted losses, LaProp, AGC, paper-aligned discount and policy
> variance bounds, and tests for these behaviors. The architecture remains
> deliberately smaller than the paper and retains Robobo-specific sparse-food
> replay, camera/IR inputs, and food-pixel reconstruction weighting.

The detailed findings below describe the implementation at the time of the
original audit; use the remediation status above when assessing DreamerV3.

This repository contains recognizable implementations of DreamerV3 and
DreamerV4 concepts, but neither implementation currently matches its paper
closely enough to be described as paper-accurate.

- **DreamerV3:** A coherent approximation with several correct core
  components, but important critic objectives, optimizer details, and training
  procedures are missing or changed.
- **DreamerV4:** A substantially simplified interpretation of the paper with
  critical temporal-alignment issues in imagination training.

The existing Dreamer-related test suite passes:

```text
22 passed, 1 warning
```

However, these tests primarily cover tensor shapes, numerical finiteness,
checkpointing, replay behavior, and isolated return calculations. They do not
establish equivalence with the paper algorithms.

## Critical Findings

### 1. Incorrect Symexp Two-Hot Decoding

File:
`catkin_ws/src/learning_machines/src/learning_machines/distributional.py`

The implementation decodes reward and value distributions as:

```python
symexp(sum(probabilities * symlog_bins))
```

The papers define the expectation over the bins in the original value space:

```python
sum(probabilities * symexp(symlog_bins))
```

These expressions are not equivalent because `symexp` is nonlinear. A focused
numerical check gave:

```text
Repository result: 1.783257
Paper expectation: 3.373261
```

This error affects both DreamerV3 and DreamerV4:

- Predicted rewards
- Critic values
- Lambda-return bootstrapping
- Actor advantages
- PMPO advantage signs

This should be corrected before interpreting either agent's training results.

### 2. DreamerV4 Uses Mismatched States and Actions

File:
`catkin_ws/src/learning_machines/src/learning_machines/dreamerv4/dreamerv4_image.py`

During imagination, action `a_t` is sampled from the current latent `s_t`.
However, critic values and behavioral-prior probabilities are evaluated using
`imag_latents[:, 1:]`, corresponding to `s_{t+1}`.

As a result:

- The advantage assigned to `a_t` is based on the wrong state.
- The PMPO likelihood term uses a log probability collected at `s_t`.
- The prior-KL term evaluates the same action at `s_{t+1}`.

The PMPO objective is therefore optimizing temporally mismatched state-action
pairs.

### 3. DreamerV4 Training and Inference Token Layouts Differ

The dynamics model is trained with blocks containing a corrupted target latent
and its corresponding action. During imagination, the implementation:

1. Adds a context block containing the current latent and action.
2. Adds another block for the noisy next latent using the same action.
3. Reads reward and termination predictions from the first block.

This differs from the layout used during dynamics training and likely predicts
rewards and terminations for the wrong timestep. The imagined trajectories
used by PMPO are therefore not generated under the dynamics model's training
convention.

## DreamerV3 Assessment

### Components Implemented Correctly or Approximately

- Recurrent state-space model with deterministic and categorical stochastic
  states
- Straight-through categorical sampling
- One-percent uniform mixture for categorical distributions
- Dynamics and representation KL losses with a one-nat free-bit floor
- Observation reconstruction
- Symexp two-hot reward and value heads, apart from the decoding error above
- Continuation prediction
- Lambda returns
- REINFORCE actor objective
- Percentile-based return-scale normalization
- Entropy regularization
- Zero initialization of reward and value output layers
- Online RSSM filtering using the previously executed action

### Significant Deviations

#### Missing Replay Critic Loss

DreamerV3 applies a critic loss to both:

- Imagined trajectories, with scale `1.0`
- Replay trajectories, with scale `0.3`

The repository trains the critic only on imagined trajectories.

#### Missing Slow-Critic Regularization

The paper regularizes the critic toward an exponentially moving average of
itself. The repository has a target critic for bootstrapping, but does not add
the slow-value regularization loss to critic training.

#### Only One Imagination Start Per Sequence

The implementation starts imagination from the final state of each replay
sequence. DreamerV3 normally starts rollouts from replayed model states
throughout the sequence, providing many more diverse actor-critic training
starts.

#### Non-Paper Advantage Clipping

Normalized advantages are clipped to `[-5, 5]`. This changes the actor
objective and is not part of the paper algorithm.

#### Optimizer and Gradient Clipping

The paper uses:

- LaProp
- Adaptive gradient clipping with coefficient `0.3`
- Learning rate `4e-5`
- Optimizer epsilon `1e-20`

The repository uses separate Adam optimizers and global gradient-norm
clipping.

#### Architecture Differences

- Standard GRU instead of the block-diagonal GRU
- LayerNorm or no normalization where the paper uses RMSNorm
- Smaller and differently scaled networks
- Different continuous-action distribution parameterization

These adaptations may be reasonable for a smaller robotics task, but they are
not paper-equivalent.

#### Discount Mismatch

The paper uses a discount factor of `0.997`. The repository defaults to
`0.994009`.

#### Unused `kl_balance`

The RSSM accepts a `kl_balance` argument but does not use it. The implemented
separate dynamics and representation KL terms otherwise follow the paper's
stop-gradient structure.

#### Replay Sampling Difference

The paper describes uniform replay with an online queue. The repository uses
sparse-reward event balancing. This is an intentional task-specific
modification rather than a DreamerV3 paper feature.

## DreamerV4 Assessment

### Components That Resemble the Paper

- Three broad training stages:
  - Tokenizer pretraining
  - Dynamics and behavior training
  - Imagination reinforcement learning
- Masked image reconstruction
- Tanh latent bottleneck
- X-prediction shortcut objective
- Sampling shortcut step sizes from powers of two
- Signal-level ramp weight `0.9 * tau + 0.1`
- Four shortcut sampling steps during imagination
- Multi-token action and reward prediction
- Distributional reward and value heads
- PMPO-style separation of positive and negative advantages
- Behavioral-prior regularization
- Frozen world model during imagination reinforcement learning

### Major Architectural Deviations

#### Tokenizer Is Not Causal or Temporal

The paper uses a block-causal transformer tokenizer over video sequences with
patch and latent tokens. The repository tokenizer is an independent
per-frame convolutional autoencoder.

It therefore does not provide the paper's temporal compression or causal
frame-by-frame decoding.

#### LPIPS Is Not Implemented

The repository replaces LPIPS with MSE between features produced by the
tokenizer's own trainable encoder. This is not equivalent to LPIPS and allows
the feature metric itself to adapt during optimization.

#### Dynamics Architecture Is Strongly Simplified

The paper's dynamics model uses:

- Multiple spatial latent tokens
- Register tokens
- A two-dimensional space-time transformer
- Sparse temporal attention
- RMSNorm
- RoPE
- SwiGLU
- QK normalization
- Attention-logit soft capping
- Grouped-query attention

The repository uses one vector latent per frame and a standard PyTorch
transformer encoder with learned positional embeddings.

#### Agent Heads Are Outside the Dynamics Transformer

The paper inserts task or agent tokens into the dynamics transformer and
predicts actions, rewards, and values from their hidden states. Other
modalities cannot attend back to these tokens, preventing causal confusion.

The repository trains separate actor, reward, and value MLPs directly on
tokenizer latents. It does not implement agent-token attention structure or
task conditioning.

#### MTP Finetuning Omits Video Prediction

During the paper's agent-finetuning phase, behavior cloning and reward losses
are added while continuing the noisy video prediction objective. The
repository trains MTP heads in a separate phase without continuing the
dynamics loss.

#### Separate Reward Models Are Used

The MTP reward head trained during behavior finetuning is not used during
imagination. Imagined rewards instead come from the dynamics model's original
reward head.

Thus, the reward model optimized during MTP finetuning is disconnected from
the PMPO training signal.

#### Extra Entropy Objective

DreamerV4 replaces DreamerV3 return normalization and entropy regularization
with PMPO plus behavioral-prior KL. The repository adds an entropy term to the
PMPO objective, changing Equation 11 of the paper.

#### Memoryless Deployment

Deployment encodes only the current image and IR readings and applies the
actor directly to that latent. It does not maintain dynamics-transformer
context.

The deployed policy therefore cannot use the temporal world-model state that
DreamerV4 is designed around.

### Shortcut-Objective Details

The x-prediction bootstrap calculation is broadly recognizable. However:

- Flow and bootstrap samples are averaged separately and then added, changing
  their relative weighting from a single expectation over samples.
- The minimum step size is fixed through the module constant rather than
  consistently using configuration.
- The final latent is appended to `corrupted_latents` but is not actually
  consumed by the forward pass.
- Context tokens cached for bootstrap targets are input embeddings, not
  transformer key/value states.

## Test Coverage Gaps

Additional tests should verify:

1. Two-hot decoding against the paper equation.
2. Exact state-action-reward indexing for imagined trajectories.
3. Reward and continuation predictions corresponding to generated next
   states.
4. PMPO likelihood and prior-KL terms using identical states and actions.
5. Shortcut flow and bootstrap equations against manually computed targets.
6. Equivalence between dynamics training and imagination token layouts.
7. DreamerV3 replay critic loss and slow-value regularization.
8. Actor-critic rollout starts from all intended replay states.
9. Deployment action dependence on temporal context.

## Recommended Priority

### Immediate Correctness Fixes

1. Correct the shared two-hot decoder.
2. Define one explicit transition convention for DreamerV4.
3. Align action log probabilities, critic values, rewards, continuations, and
   prior probabilities to the same state indices.
4. Make DreamerV4 imagination use the same token layout as dynamics training.
5. Use the MTP-trained reward head during imagination, or train the dynamics
   reward head during agent finetuning.

### DreamerV3 Paper-Alignment Work

1. Add replay value learning with scale `0.3`.
2. Add slow-critic regularization.
3. Start imagined rollouts from replay states across the sampled sequence.
4. Remove non-paper advantage clipping unless retained as a documented
   adaptation.
5. Implement AGC and a closer optimizer setup.
6. Correct the discount and policy distribution defaults.

### DreamerV4 Paper-Alignment Work

Achieving genuine DreamerV4 fidelity requires more than local fixes. It would
require:

1. A causal temporal tokenizer.
2. Spatial latent and register tokens.
3. A space-time transformer architecture.
4. Agent/task tokens inside the dynamics transformer.
5. Joint video, policy, and reward finetuning.
6. Context-preserving imagination and deployment.

Until those components exist, the implementation should be described as a
DreamerV4-inspired shortcut world-model agent rather than DreamerV4.

## Overall Conclusion

Neither implementation currently reproduces the corresponding paper
algorithm.

DreamerV3 is a practical, task-adapted approximation that can be brought
closer to the paper incrementally. DreamerV4 retains several high-level ideas
from the paper but differs in architecture and training design, and its
current imagination indexing errors must be fixed before results are
meaningful.

## Primary Sources

- [DreamerV3 paper: Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104)
- [DreamerV3 reference implementation](https://github.com/danijar/dreamerv3)
- [DreamerV4 paper: Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527)
- [DreamerV4 project page](https://danijar.com/project/dreamer4/)
