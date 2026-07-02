'Paper-aligned DreamerV4 training phases and imagination learning.'
from __future__ import annotations

import copy
import contextlib
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from learning_machines.distributional import (
    _make_bin_centers,
    symexp,
    two_hot_loss,
)

from .config import DreamerV4FullConfig
from .model import (
    InteractiveDynamics,
    MLPHead,
    MTPDistributionalHead,
    TanhGaussianMTPPolicy,
    sample_shortcut_schedule,
)
from .optim import (
    adamw_parameter_groups,
    initialize_dreamerv4,
    warmup_cosine_lambda,
)
from .tokenizer import CausalVideoTokenizer, RunningRMS


def finite_grad_norm(parameters, maximum: float) -> float:
    parameters = [p for p in parameters if p.grad is not None]
    for parameter in parameters:
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("non-finite DreamerV4 gradient")
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(parameter.grad.detach().double())
        for parameter in parameters
    ])).item() if parameters else 0.0
    if not np.isfinite(norm):
        raise FloatingPointError("non-finite DreamerV4 gradient norm")
    scale = min(1.0, maximum / (norm + 1e-12))
    if scale < 1:
        for parameter in parameters:
            parameter.grad.mul_(scale)
    return float(norm)


def td_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    continuation: torch.Tensor,
    gamma: float,
    lambda_: float,
) -> torch.Tensor:
    horizon = rewards.shape[1]
    result = torch.empty_like(rewards)
    carry = values[:, -1]
    for index in reversed(range(horizon)):
        bootstrap = (
            (1 - lambda_) * values[:, index + 1] + lambda_ * carry
        )
        carry = rewards[:, index] + gamma * continuation[:, index] * bootstrap
        result[:, index] = carry
    return result


def distribution_value(logits: torch.Tensor) -> torch.Tensor:
    'Decode a symlog two-hot head as used by Dreamer 3/4.'
    bins = _make_bin_centers(logits.device)
    expectation = (logits.float().softmax(-1) * bins).sum(-1)
    return symexp(expectation)


class DreamerV4FullAgent(nn.Module):
    'DreamerV4 with tokenizer, shortcut world model, MTP, and PMPO phases.'

    def __init__(self, cfg: DreamerV4FullConfig, device: str | torch.device = "auto"):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device == "auto" else torch.device(device)
        self.tokenizer = CausalVideoTokenizer(cfg)
        self.dynamics = InteractiveDynamics(cfg)
        self.policy = TanhGaussianMTPPolicy(cfg)
        self.reward = MTPDistributionalHead(cfg)
        self.value = MLPHead(cfg.model_dim, 255)
        self.continue_head = MLPHead(cfg.model_dim, 1)
        initialize_dreamerv4(self.tokenizer, cfg.tokenizer_layers)
        initialize_dreamerv4(self.dynamics, cfg.dynamics_layers)
        initialize_dreamerv4(self.policy, 1)
        initialize_dreamerv4(self.reward, 1)
        initialize_dreamerv4(self.value, 1)
        initialize_dreamerv4(self.continue_head, 1)
        nn.init.zeros_(self.dynamics.latent_output.weight)
        nn.init.zeros_(self.dynamics.latent_output.bias)
        for head in self.reward.heads:
            nn.init.zeros_(head.output.weight)
            nn.init.zeros_(head.output.bias)
        nn.init.zeros_(self.value.output.weight)
        nn.init.zeros_(self.value.output.bias)
        nn.init.zeros_(self.continue_head.output.weight)
        nn.init.constant_(self.continue_head.output.bias, 2.0)
        for head in self.policy.heads:
            nn.init.zeros_(head.output.weight)
            nn.init.zeros_(head.output.bias)
            head.output.bias.data[cfg.action_dim:] = -1.0
        self.prior_policy = copy.deepcopy(self.policy)

        self.world_rms = RunningRMS(cfg.rms_decay, cfg.rms_floor_ratio)
        self.policy_rms = RunningRMS(cfg.rms_decay, cfg.rms_floor_ratio)
        self.reward_rms = RunningRMS(cfg.rms_decay, cfg.rms_floor_ratio)
        self.continue_rms = RunningRMS(cfg.rms_decay, cfg.rms_floor_ratio)

        self.tokenizer_optimizer = torch.optim.AdamW(
            adamw_parameter_groups(
                self.tokenizer.named_parameters(), cfg.weight_decay
            ),
            lr=cfg.learning_rate,
        )
        self.world_optimizer = torch.optim.AdamW(
            adamw_parameter_groups(
                self.dynamics.named_parameters(), cfg.weight_decay
            ),
            lr=cfg.learning_rate,
        )
        finetune_named = (
            [(f"dynamics.{name}", value) for name, value in self.dynamics.named_parameters()]
            + [(f"policy.{name}", value) for name, value in self.policy.named_parameters()]
            + [(f"reward.{name}", value) for name, value in self.reward.named_parameters()]
            + [(f"continue.{name}", value) for name, value in self.continue_head.named_parameters()]
        )
        self.finetune_optimizer = torch.optim.AdamW(
            adamw_parameter_groups(finetune_named, cfg.weight_decay),
            lr=cfg.learning_rate,
        )
        rl_named = (
            [(f"policy.{name}", value) for name, value in self.policy.named_parameters()]
            + [(f"value.{name}", value) for name, value in self.value.named_parameters()]
        )
        self.rl_optimizer = torch.optim.AdamW(
            adamw_parameter_groups(rl_named, cfg.weight_decay),
            lr=cfg.learning_rate,
        )
        self.schedulers: dict[str, torch.optim.lr_scheduler.LambdaLR] = {}
        self.scheduler_totals: dict[str, int] = {}
        scaler_enabled = (
            cfg.mixed_precision
            and self.device.type == "cuda"
            and cfg.mixed_precision_dtype == "float16"
        )
        self.grad_scaler = torch.amp.GradScaler(
            "cuda", enabled=scaler_enabled
        )
        self.to(self.device)

    def configure_schedulers(self, totals: dict[str, int]):
        optimizers = {
            "tokenizer": self.tokenizer_optimizer,
            "world": self.world_optimizer,
            "finetune": self.finetune_optimizer,
            "imagination": self.rl_optimizer,
        }
        self.scheduler_totals = {
            name: max(1, int(totals[name])) for name in optimizers
        }
        self.schedulers = {
            name: torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                warmup_cosine_lambda(
                    self.scheduler_totals[name],
                    self.cfg.warmup_fraction,
                    self.cfg.minimum_lr_ratio,
                ),
            )
            for name, optimizer in optimizers.items()
        }

    @staticmethod
    def _transfer_optimizer_state(
        source: torch.optim.Optimizer,
        target: torch.optim.Optimizer,
        parameters,
    ):
        for parameter in parameters:
            if parameter in source.state:
                target.state[parameter] = copy.deepcopy(source.state[parameter])

    def prepare_agent_finetune(self):
        'Preserve dynamics Adam moments when entering joint finetuning.'
        self._transfer_optimizer_state(
            self.world_optimizer,
            self.finetune_optimizer,
            self.dynamics.parameters(),
        )

    def prepare_imagination(self):
        'Preserve policy Adam moments when entering PMPO.'
        self._transfer_optimizer_state(
            self.finetune_optimizer,
            self.rl_optimizer,
            self.policy.parameters(),
        )

    def _autocast(self):
        if not self.cfg.mixed_precision or self.device.type != "cuda":
            return contextlib.nullcontext()
        dtype = (
            torch.bfloat16
            if self.cfg.mixed_precision_dtype == "bfloat16"
            else torch.float16
        )
        return torch.autocast("cuda", dtype=dtype)

    def _optimization_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        parameters,
        scheduler_name: str,
        agent_parameters=None,
    ) -> tuple[float, float]:
        parameters = list(parameters)
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(optimizer)
        if agent_parameters is not None and self.cfg.agent_grad_clip is not None:
            agent_parameters = list(agent_parameters)
            agent_ids = {id(p) for p in agent_parameters}
            non_agent = [p for p in parameters if id(p) not in agent_ids]
            finite_grad_norm(non_agent, self.cfg.grad_clip)
            norm = finite_grad_norm(agent_parameters, self.cfg.agent_grad_clip)
        else:
            norm = finite_grad_norm(parameters, self.cfg.grad_clip)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()
        scheduler = self.schedulers.get(scheduler_name)
        if scheduler is not None:
            scheduler.step()
        return norm, learning_rate

    def update_tokenizer(
        self, images: torch.Tensor, ir: torch.Tensor | None = None
    ) -> dict[str, float]:
        self.tokenizer_optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = self.tokenizer(images, ir)
        norm, learning_rate = self._optimization_step(
            output["loss"],
            self.tokenizer_optimizer,
            self.tokenizer.parameters(),
            "tokenizer",
        )
        return {
            "tok/loss": output["loss"].item(),
            "tok/mse": output["mse_loss"].item(),
            "tok/lpips": output["lpips_loss"].item(),
            "tok/ir": output["ir_loss"].item(),
            "tok/grad_norm": norm,
            "tok/learning_rate": learning_rate,
        }

    @torch.no_grad()
    def encode(
        self, images: torch.Tensor, ir: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.tokenizer.encode(images.to(self.device), None if ir is None else ir.to(self.device))

    def _world_sequence(
        self, latents: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        'Align each transition action a_t with its arrival latent z_{t+1}.'
        return latents[:, 1:], actions

    def _policy_sequence(
        self, latents: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        'Represent state z_t using the action that arrived at that state.'
        state = latents[:, :-1]
        previous_actions = torch.cat([
            torch.zeros_like(actions[:, :1]),
            actions[:, :-1],
        ], 1)
        return state, previous_actions

    def _shortcut_loss(
        self,
        target: torch.Tensor,
        previous_actions: torch.Tensor,
        mask: torch.Tensor,
        task_ids: torch.Tensor | None,
        action_known: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch, time = target.shape[:2]
        signal, step = sample_shortcut_schedule(
            (batch, time), target.device, self.cfg.shortcut_steps
        )
        noise = torch.randn_like(target)
        expanded_signal = signal[..., None, None]
        corrupted = (1 - expanded_signal) * noise + expanded_signal * target
        prediction = self.dynamics(
            corrupted,
            previous_actions,
            signal,
            step,
            task_ids,
            action_known,
        )
        predicted_x = prediction["latent"]
        d_min = 1 / self.cfg.shortcut_steps
        flow = step <= d_min + 1e-7
        bootstrap = ~flow
        per_item = torch.zeros_like(signal)
        flow_error = (predicted_x - target).square().mean((-1, -2))
        per_item = torch.where(flow, flow_error, per_item)

        if (bootstrap & mask.bool()).any():
            with torch.no_grad():
                half_step = step / 2
                first_x = self.dynamics(
                    corrupted,
                    previous_actions,
                    signal,
                    half_step,
                    task_ids,
                    action_known,
                )["latent"]
                first_velocity = (first_x - corrupted) / (
                    1 - expanded_signal
                ).clamp_min(1e-6)
                midpoint = corrupted + first_velocity * half_step[..., None, None]
                midpoint_signal = signal + half_step
                second_x = self.dynamics(
                    midpoint,
                    previous_actions,
                    midpoint_signal,
                    half_step,
                    task_ids,
                    action_known,
                )["latent"]
                second_velocity = (second_x - midpoint) / (
                    1 - midpoint_signal[..., None, None]
                ).clamp_min(1e-6)
                target_velocity = (first_velocity + second_velocity) / 2
            predicted_velocity = (predicted_x - corrupted) / (
                1 - expanded_signal
            ).clamp_min(1e-6)
            bootstrap_error = (
                (1 - expanded_signal).square()
                * (predicted_velocity - target_velocity).square()
            ).mean((-1, -2))
            per_item = torch.where(bootstrap, bootstrap_error, per_item)

        ramp = 0.9 * signal + 0.1
        loss = (per_item * ramp * mask).sum() / mask.sum().clamp_min(1)
        metrics = {
            "world/loss": loss.detach(),
            "world/flow": (
                flow_error * flow * mask
            ).sum().detach() / (flow * mask).sum().clamp_min(1),
            "world/bootstrap": (
                per_item * bootstrap * mask
            ).sum().detach() / (bootstrap * mask).sum().clamp_min(1),
        }
        return loss, prediction, metrics

    def update_world_model(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor,
        action_known: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.world_optimizer.zero_grad(set_to_none=True)
        target, transition_actions = self._world_sequence(latents, actions)
        with self._autocast():
            loss, _, metrics = self._shortcut_loss(
                target,
                transition_actions,
                mask,
                None,
                action_known,
            )
        norm, learning_rate = self._optimization_step(
            loss,
            self.world_optimizer,
            self.dynamics.parameters(),
            "world",
        )
        return {**{key: value.item() for key, value in metrics.items()},
                "world/grad_norm": norm,
                "world/learning_rate": learning_rate}

    def update_agent_finetune(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
        task_ids: torch.Tensor | None = None,
    ) -> dict[str, float]:
        'Jointly retain shortcut prediction and learn Eq.'
        self.finetune_optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            world_target, transition_actions = self._world_sequence(latents, actions)
            world_loss, _, world_metrics = self._shortcut_loss(
                world_target, transition_actions, mask, task_ids
            )
            policy_state, previous_actions = self._policy_sequence(latents, actions)
            policy_signal, policy_step = sample_shortcut_schedule(
                policy_state.shape[:2],
                policy_state.device,
                self.cfg.shortcut_steps,
            )
            policy_noise = torch.randn_like(policy_state)
            policy_corrupted = (
                (1 - policy_signal[..., None, None]) * policy_noise
                + policy_signal[..., None, None] * policy_state
            )
            hidden = self.dynamics(
                policy_corrupted,
                previous_actions,
                policy_signal,
                policy_step,
                task_ids,
            )["agent"]
            action_terms, reward_terms = [], []
            for distance in range(self.cfg.mtp_length):
                valid = actions.shape[1] - distance
                if valid <= 0:
                    break
                valid_mask = mask[:, :valid]
                state = hidden[:, :valid]
                target_action = actions[:, distance : distance + valid]
                target_reward = rewards[:, distance : distance + valid]
                action_nll = -self.policy.log_prob(
                    state, target_action, distance
                )
                reward_nll = two_hot_loss(
                    self.reward(state, distance),
                    target_reward,
                    reduction="none",
                )
                denominator = valid_mask.sum().clamp_min(1)
                action_terms.append((action_nll * valid_mask).sum() / denominator)
                reward_terms.append((reward_nll * valid_mask).sum() / denominator)
            action_loss = torch.stack(action_terms).sum()
            reward_loss = torch.stack(reward_terms).sum()
            continue_logits = self.continue_head(hidden).squeeze(-1)
            continue_target = 1 - dones.float()
            continue_loss = (
                F.binary_cross_entropy_with_logits(
                    continue_logits, continue_target, reduction="none"
                ) * mask
            ).sum() / mask.sum().clamp_min(1)
            total = (
                self.world_rms.normalize(world_loss)
                + self.policy_rms.normalize(action_loss)
                + self.reward_rms.normalize(reward_loss)
                + self.continue_rms.normalize(continue_loss)
            )
        parameters = [
            p
            for group in self.finetune_optimizer.param_groups
            for p in group["params"]
        ]
        agent_parameters = (
            list(self.policy.parameters())
            + list(self.reward.parameters())
            + list(self.continue_head.parameters())
        )
        norm, learning_rate = self._optimization_step(
            total,
            self.finetune_optimizer,
            parameters,
            "finetune",
            agent_parameters=agent_parameters,
        )
        return {
            **{key: value.item() for key, value in world_metrics.items()},
            "agent/action_nll": action_loss.item(),
            "agent/reward_nll": reward_loss.item(),
            "agent/continue_loss": continue_loss.item(),
            "agent/total_loss": total.item(),
            "agent/grad_norm": norm,
            "agent/learning_rate": learning_rate,
        }

    def freeze_behavior_prior(self):
        self.prior_policy.load_state_dict(self.policy.state_dict())
        self.prior_policy.requires_grad_(False)
        self.prior_policy.eval()

    def _clean_agent_hidden(
        self,
        latents: torch.Tensor,
        previous_actions: torch.Tensor,
        task_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        shape = latents.shape[:2]
        signal = torch.ones(shape, device=latents.device)
        step = torch.full(
            shape, 1 / self.cfg.shortcut_steps, device=latents.device
        )
        return self.dynamics(
            latents,
            previous_actions,
            signal,
            step,
            task_ids,
            torch.ones_like(signal, dtype=torch.bool),
        )["agent"]

    @torch.no_grad()
    def imagine(
        self,
        context_latents: torch.Tensor,
        context_actions: torch.Tensor,
        task_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latents = context_latents[:, -self.cfg.context_length :].clone()
        actions = context_actions[
            :, -(latents.shape[1] - 1) :
        ].clone()
        previous_actions = torch.cat([
            torch.zeros_like(actions[:, :1]),
            actions,
        ], 1)
        rollout = {
            "hidden": [], "actions": [], "rewards": [],
            "continuation": [], "values": [],
        }
        for _ in range(self.cfg.imagination_horizon):
            hidden = self._clean_agent_hidden(
                latents, previous_actions, task_ids
            )[:, -1]
            action, _ = self.policy.sample(hidden)
            rollout["hidden"].append(hidden)
            rollout["actions"].append(action)
            rollout["values"].append(distribution_value(self.value(hidden)))

            context_signal = self.cfg.context_signal
            history_noise = torch.randn_like(latents)
            history_corrupted = (
                (1 - context_signal) * history_noise
                + context_signal * latents
            )
            current = torch.randn(
                latents.shape[0],
                1,
                self.cfg.latent_tokens,
                self.cfg.latent_channels,
                device=latents.device,
            )
            denoise_actions = torch.cat([
                previous_actions, action.unsqueeze(1)
            ], 1)
            for sample_step in range(self.cfg.shortcut_steps):
                tau = sample_step / self.cfg.shortcut_steps
                full_latents = torch.cat([history_corrupted, current], 1)
                signal = torch.full(
                    full_latents.shape[:2],
                    context_signal,
                    device=latents.device,
                )
                signal[:, -1] = tau
                step = torch.full_like(
                    signal, 1 / self.cfg.shortcut_steps
                )
                clean = self.dynamics(
                    full_latents,
                    denoise_actions,
                    signal,
                    step,
                    task_ids,
                    torch.ones_like(signal, dtype=torch.bool),
                )["latent"][:, -1:]
                velocity = (clean - current) / max(1e-6, 1 - tau)
                current = current + velocity / self.cfg.shortcut_steps
            latents = torch.cat([latents, current.clamp(-1, 1)], 1)
            previous_actions = denoise_actions
            if latents.shape[1] > self.cfg.context_length:
                latents = latents[:, -self.cfg.context_length :]
                previous_actions = previous_actions[:, -self.cfg.context_length :]
            next_hidden = self._clean_agent_hidden(
                latents, previous_actions, task_ids
            )[:, -1]
            rollout["rewards"].append(
                distribution_value(self.reward(next_hidden))
            )
            rollout["continuation"].append(
                torch.sigmoid(self.continue_head(next_hidden).squeeze(-1))
            )

        final_hidden = self._clean_agent_hidden(
            latents, previous_actions, task_ids
        )[:, -1]
        rollout["bootstrap"] = distribution_value(self.value(final_hidden))
        return {
            key: torch.stack(value, 1) if isinstance(value, list) else value
            for key, value in rollout.items()
        }

    def update_imagination(
        self,
        context_latents: torch.Tensor,
        context_actions: torch.Tensor,
        task_ids: torch.Tensor | None = None,
    ) -> dict[str, float]:
        self.rl_optimizer.zero_grad(set_to_none=True)
        was_training = self.training
        self.eval()
        rollout = self.imagine(context_latents, context_actions, task_ids)
        if was_training:
            self.train()
        hidden = rollout["hidden"].detach()
        actions = rollout["actions"].detach()
        rewards = rollout["rewards"].detach()
        continuation = rollout["continuation"].detach()
        with self._autocast():
            value_logits = self.value(hidden)
            values = distribution_value(value_logits)
            all_values = torch.cat([
                values, rollout["bootstrap"].detach().unsqueeze(1)
            ], 1)
            returns = td_lambda_returns(
                rewards,
                all_values.detach(),
                continuation,
                self.cfg.gamma,
                self.cfg.lambda_,
            )
            value_loss = two_hot_loss(value_logits, returns.detach())
            advantages = (returns - values).detach()
            if self.cfg.normalize_advantages:
                advantages = (
                    (advantages - advantages.mean())
                    / (advantages.std(unbiased=False) + 1e-8)
                )
            log_prob = self.policy.log_prob(hidden, actions)
            positive = advantages >= 0
            negative = ~positive
            positive_loss = (
                -log_prob[positive].mean()
                if positive.any() else log_prob.new_zeros(())
            )
            negative_loss = (
                log_prob[negative].mean()
                if negative.any() else log_prob.new_zeros(())
            )
            policy_loss = (
                self.cfg.pmpo_alpha * positive_loss
                + (1 - self.cfg.pmpo_alpha) * negative_loss
            )
            prior_kl = self.policy.reverse_kl(hidden, self.prior_policy).mean()
            entropy_term = log_prob.mean()
            actor_loss = (
                policy_loss
                + self.cfg.prior_kl_weight * prior_kl
                + self.cfg.entropy_weight * entropy_term
            )
            total = value_loss + actor_loss
        parameters = list(self.policy.parameters()) + list(self.value.parameters())
        norm, learning_rate = self._optimization_step(
            total,
            self.rl_optimizer,
            parameters,
            "imagination",
            agent_parameters=self.policy.parameters(),
        )
        return {
            "rl/total_loss": total.item(),
            "rl/policy_loss": policy_loss.item(),
            "rl/value_loss": value_loss.item(),
            "rl/prior_kl": prior_kl.item(),
            "rl/return_mean": returns.mean().item(),
            "rl/positive_fraction": positive.float().mean().item(),
            "rl/grad_norm": norm,
            "rl/learning_rate": learning_rate,
        }

    @torch.no_grad()
    def act(
        self,
        images: torch.Tensor,
        ir: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
        deterministic: bool = True,
        task_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.tokenizer.encode(images.to(self.device), None if ir is None else ir.to(self.device))
        if latent.dim() == 3:
            latent = latent.unsqueeze(1)
        batch = latent.shape[0]
        if previous_action is None:
            previous_action = torch.zeros(
                batch, 1, self.cfg.action_dim, device=self.device
            )
        elif previous_action.dim() == 2:
            previous_action = previous_action.unsqueeze(1)
        hidden = self._clean_agent_hidden(
            latent, previous_action.to(self.device), task_ids
        )[:, -1]
        return self.policy.sample(hidden, deterministic)[0]

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save({
            "config": self.cfg.to_dict(),
            "model": self.state_dict(),
            "tokenizer_optimizer": self.tokenizer_optimizer.state_dict(),
            "world_optimizer": self.world_optimizer.state_dict(),
            "finetune_optimizer": self.finetune_optimizer.state_dict(),
            "rl_optimizer": self.rl_optimizer.state_dict(),
            "scheduler_totals": self.scheduler_totals,
            "schedulers": {
                name: scheduler.state_dict()
                for name, scheduler in self.schedulers.items()
            },
            "grad_scaler": self.grad_scaler.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }, temporary)
        temporary.replace(path)

    @classmethod
    def load(
        cls, path: str | Path, device: str | torch.device = "auto"
    ) -> "DreamerV4FullAgent":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        agent = cls(DreamerV4FullConfig(**checkpoint["config"]), device)
        agent.load_state_dict(checkpoint["model"])
        scheduler_totals = checkpoint.get("scheduler_totals", {})
        if scheduler_totals:
            agent.configure_schedulers(scheduler_totals)
        agent.tokenizer_optimizer.load_state_dict(checkpoint["tokenizer_optimizer"])
        agent.world_optimizer.load_state_dict(checkpoint["world_optimizer"])
        agent.finetune_optimizer.load_state_dict(checkpoint["finetune_optimizer"])
        agent.rl_optimizer.load_state_dict(checkpoint["rl_optimizer"])
        if scheduler_totals:
            for name, state in checkpoint.get("schedulers", {}).items():
                if name in agent.schedulers:
                    agent.schedulers[name].load_state_dict(state)
        if checkpoint.get("grad_scaler"):
            agent.grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all([
                state.detach().to("cpu", dtype=torch.uint8)
                for state in checkpoint["cuda_rng_state"]
            ])
        return agent
