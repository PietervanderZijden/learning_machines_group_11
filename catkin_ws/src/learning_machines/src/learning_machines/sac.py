"""Numerically stabilized SAC integration for the Robobo action wrapper."""
from __future__ import annotations

import numpy as np
import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3 import SAC


class StabilizedSAC(SAC):
    """SAC with clipped gradients and action-wrapper-aware replay semantics.

    The replay action remains the policy-requested action. Smoothing, actuator
    randomization, and safety are part of the environment transition. Training
    the critic on post-wrapper actions while the actor outputs pre-wrapper
    actions makes the actor optimize a different action space.
    """

    def __init__(self, *args, max_grad_norm: float = 10.0, **kwargs):
        self.max_grad_norm = float(max_grad_norm)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        self._update_learning_rate(optimizers)

        actor_losses, critic_losses = [], []
        actor_grad_norms, critic_grad_norms = [], []
        q_means, q_abs_maxes, target_means, target_abs_maxes, td_abs_means = (
            [], [], [], [], []
        )
        entropy_means, reward_means, reward_abs_maxes = [], [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            discounts = (
                replay_data.discounts
                if replay_data.discounts is not None
                else self.gamma
            )
            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )
            log_prob = log_prob.reshape(-1, 1)
            ent_coef = self.ent_coef_tensor

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations
                )
                next_q_values = th.cat(
                    self.critic_target(
                        replay_data.next_observations, next_actions
                    ),
                    dim=1,
                )
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values -= ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.critic(
                replay_data.observations, replay_data.actions
            )
            critic_loss = 0.5 * sum(
                F.mse_loss(current_q, target_q_values)
                for current_q in current_q_values
            )
            if not th.isfinite(critic_loss):
                raise RuntimeError("non-finite SAC critic loss")
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            critic_grad_norm = th.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
            self.critic.optimizer.step()

            q_values_pi = th.cat(
                self.critic(replay_data.observations, actions_pi), dim=1
            )
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            if not th.isfinite(actor_loss):
                raise RuntimeError("non-finite SAC actor loss")
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = th.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.max_grad_norm
            )
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.batch_norm_stats,
                    self.batch_norm_stats_target,
                    1.0,
                )

            stacked_q = th.cat(current_q_values, dim=1).detach()
            td_error = current_q_values[0].detach() - target_q_values
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            actor_grad_norms.append(actor_grad_norm.item())
            critic_grad_norms.append(critic_grad_norm.item())
            q_means.append(stacked_q.mean().item())
            q_abs_maxes.append(stacked_q.abs().max().item())
            target_means.append(target_q_values.mean().item())
            target_abs_maxes.append(target_q_values.abs().max().item())
            td_abs_means.append(td_error.abs().mean().item())
            entropy_means.append((-log_prob).mean().item())
            reward_means.append(replay_data.rewards.mean().item())
            reward_abs_maxes.append(replay_data.rewards.abs().max().item())

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", self.ent_coef_tensor.item())
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/actor_grad_norm", np.mean(actor_grad_norms))
        self.logger.record("train/critic_grad_norm", np.mean(critic_grad_norms))
        self.logger.record("train/q_mean", np.mean(q_means))
        self.logger.record("train/q_abs_max", np.mean(q_abs_maxes))
        self.logger.record("train/target_q_mean", np.mean(target_means))
        self.logger.record("train/target_q_abs_max", np.mean(target_abs_maxes))
        self.logger.record("train/td_abs_mean", np.mean(td_abs_means))
        self.logger.record("train/policy_entropy", np.mean(entropy_means))
        self.logger.record("train/replay_reward_mean", np.mean(reward_means))
        self.logger.record(
            "train/replay_reward_abs_max", np.mean(reward_abs_maxes)
        )


# Backward-compatible import name for source files created during the transfer
# refactor. New code should use the accurate class name above.
ExecutedActionSAC = StabilizedSAC
