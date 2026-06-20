# Pending Decisions and Action Items

This file tracks open questions, deferred decisions, and planned implementations for the DreamerV4 / Robobo project.
Items are grouped by area. Completed items should be moved to a "Done" section with a date.

## DreamerV4 Training & Data Strategy

1. **[DECISION NEEDED]** Extend `train_dreamerv4_full.py` to mix simulator + real episodes.
   - Add `--hardware-record-dir`, `--hardware-calibration`, `--hardware-sample-ratio`, `--hardware-validation-fraction`, `--dataset-seed`.
   - Reuse `StreamingEpisodeDataset`-style sampling from `train_dreamerv4_image.py`.
   - Ensure latent cache is invalidated when dataset composition changes.

2. **[TIMING]** Start collecting simulator data for DreamerV4 pretraining.
   - Target: 2,000–5,000 episodes using diverse SAC/DreamerV3 checkpoints or scripted/random policies.
   - Requires the simulation environment to be set up first.

3. **[TIMING]** Start collecting real-world data for DreamerV4 fine-tuning.
   - Target: 300–500 episodes.
   - Use diverse DreamerV3/SAC checkpoints as drivers.
   - No domain randomization on real data.

## DreamerV3 Issues & Data Collection

4. ~~**[IMPLEMENT]** Save intermediate DreamerV3 checkpoints for diverse-policy data collection.~~ **DONE**
     - Added `--checkpoint-history` flag to `train_dreamerv3.py`.
     - Step-numbered checkpoints are saved as `dreamerv3_step_<step>.pt` every `--checkpoint-every` steps.
     - Old checkpoints beyond the history limit are automatically deleted.

5. ~~**[INVESTIGATE]** DreamerV3 reconstructions in `evaluate_transfer.py` are blurry and do not match inputs.~~ **INVESTIGATED**
     - The evaluation path (`_dreamerv3_prior_prediction`) predicts the **next frame from the prior** (state + action, no next observation). This is inherently harder than training-time posterior reconstruction.
     - DreamerV3 uses MSE reconstruction loss, which encourages blurry mean predictions.
     - No normalization or architectural bug was found; image values are correctly scaled to `[0, 1]`, decoder ends with `sigmoid`, and policy state updates are correct.
     - To confirm whether the model is simply undertrained vs. broken, compare posterior reconstructions (feed `next_obs` into the model and decode from the posterior state) against the prior predictions currently shown in `evaluate_transfer.py`.
     - Potential improvements: add posterior-reconstruction comparison to `evaluate_transfer.py`, or retrain with a perceptual/SSIM loss term.

6. **[CONFIRM]** Whether DreamerV3 policy checkpoints can drive image-based data collection.
   - DreamerV3 already records image episodes internally, so it is a natural source.
   - SAC uses blob features, so SAC checkpoints may not be directly usable for image-based collection.

## Model Quality & Diagnostics

7. **[AFTER V4 TRAINING]** Evaluate the final DreamerV4-full checkpoint in simulation.
   - Check food-collection performance, episode returns, and behavior stability.

7b. **[IN PROGRESS]** DreamerV4 PMPO policy collapse: imagined returns became increasingly negative (−1 → −25) and positive reward fraction stayed low.
   - Investigated PMPO paper formulation: DreamerV4 does **not** use advantage normalization, entropy regularization, or return normalization. It uses sign-based advantages + reverse KL to a frozen BC prior.
   - Added PMPO hyperparameter CLI flags (`--pmpo-alpha`, `--prior-kl-weight`, `--entropy-weight`, `--imagination-horizon`, `--no-advantage-normalization`) for experimentation, but defaults should remain paper-aligned.
   - Still need to diagnose root cause: poor behavior prior, time penalty in rewards, too-short imagination horizon, or negative reward predictions.

8. **[CONSIDER]** Increase DreamerV4 tokenizer capacity if reconstructions remain blurry.
   - Scale `--model-dim`, `--latent-tokens`, `--latent-channels`, `--tokenizer-layers` as GPU memory allows.
   - Compare against the paper’s ~400M-parameter tokenizer and document the gap.

9. **[CONSIDER]** Tune the tokenizer mask ratio.
   - Current fixed ratio is `0.75` (deviation from paper’s `U(0, 0.9)`).
   - Run short tokenizer experiments with 0.5, 0.6, 0.75, 0.9 and compare validation MSE/LPIPS.

10. ~~**[CONSIDER]** Add separate gradient clipping for the DreamerV4 agent head.~~ **DONE**
     - Added `agent_grad_clip` to `DreamerV4FullConfig`.
     - Modified `_optimization_step` to clip agent/policy/reward/continue parameters separately from dynamics/value.
     - Added `--agent-grad-clip` CLI flag to `train_dreamerv4_full.py`.
     - To use in the next run: `--agent-grad-clip 10.0` (or another value) while keeping `--grad-clip 1.0` for the world model.

## Logging & Experiment Tracking

11. ~~**[IMPLEMENT]** Add Weights & Biases (wandb) logging to `train_dreamerv4_full.py`.~~ **DONE**
     - Logs scalar metrics and validation metrics for all phases.
     - Added `--no-wandb`, `--wandb-project`, `--wandb-entity`, `--wandb-run-name` CLI flags.
     - Logs hyperparameters and dataset manifest metadata.

12. ~~**[IMPLEMENT]** Log tokenizer inputs/outputs to wandb after the tokenizer stage finishes.~~ **DONE**
     - After the tokenizer phase, samples validation frames and logs a side-by-side panel: original | reconstructed | absolute error.

13. **[CONSIDER]** Add richer per-stage diagnostic visualizations for DreamerV4.
     - World model: log latent MSE per step as a histogram, action/reward distributions.
     - FineTune / Imagination: log value distributions, imagined return histograms, prior KL histograms.
     - These can be added incrementally once basic wandb logging is validated.

## Data Collection & Environment

14. ~~**[IMPLEMENT]** Record image + IR episodes from SAC training for DreamerV4 offline training.~~ **DONE**
    - Added `--record-dir`, `--no-record`, `--image-size` to `train_sac.py`.
    - `RoboboSACEnv` now saves `.npz` episodes with `images`, `actions`, `rewards`, `dones`, and `irs` in the `robobo-obs-v2` / `robobo-reward-v4` format.

15. ~~**[IMPLEMENT]** Add `--no-domain-randomization` flag to training scripts.~~ **DONE**
    - `train_sac.py` already had `--domain-randomization` via `BooleanOptionalAction`; verified `--no-domain-randomization` works.
    - Added `--domain-randomization` / `--no-domain-randomization` to `train_dreamerv3.py`.

## Visualization & Tooling

16. **[OPTIONAL]** Add a DreamerV4 decode/reconstruction visualization path to `evaluate_transfer.py`.
     - Currently only DreamerV3 produces decoded images.
     - Requires calling `agent.tokenizer.encode()` and `agent.tokenizer.decode()` with `agent.eval()` enabled.

## Notes

- Do **not** start real-world data collection or heavy V4 evaluation until the current V4 training run finishes.
- Keep this file updated as items are completed or decisions are made.
