import math
from unittest.mock import patch

import pytest
import torch

from learning_machines.dreamerv3.actor_critic import Actor
from learning_machines.dreamerv3.dreamerv3 import green_saliency_loss
from learning_machines.dreamerv3.optim import LaProp, adaptive_clip_grad_
from learning_machines.dreamerv4.dreamerv4_image import finite_clip_grad_norm_
from learning_machines.coppelia_startup import check_coppelia_service
from train_dreamerv3 import dreamer_updates_per_env_step
from train_sac import RoboboSACEnv


def test_dreamer_replay_ratio_converts_transitions_to_update_frequency():
    assert dreamer_updates_per_env_step(512, 32, 50) == pytest.approx(0.32)
    with pytest.raises(ValueError):
        dreamer_updates_per_env_step(-1, 32, 50)


def test_dreamerv3_actor_mean_and_std_are_bounded():
    actor = Actor(
        state_dim=4,
        action_dim=2,
        std_min=0.1,
        std_max=1.0,
        mean_limit=2.5,
    )
    with torch.no_grad():
        actor.net[-1].weight.zero_()
        actor.net[-1].bias.copy_(
            torch.tensor([100.0, -100.0, -100.0, -100.0])
        )
    state = torch.zeros(3, 4)
    deterministic = actor(state, deterministic=True)
    assert deterministic.abs().max().item() <= math.tanh(2.5) + 1e-6
    assert actor.std(state).min().item() >= 0.1 - 1e-6
    assert actor.std(state).max().item() <= 1.0 + 1e-6


def test_dreamerv3_agc_clips_relative_to_parameter_scale():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    parameter.grad = torch.full_like(parameter, 100.0)
    raw_norm = adaptive_clip_grad_([parameter], clipping=0.3)
    assert raw_norm > 100
    parameter_norm = torch.linalg.vector_norm(parameter, dim=1)
    gradient_norm = torch.linalg.vector_norm(parameter.grad, dim=1)
    assert torch.all(gradient_norm <= 0.3 * parameter_norm + 1e-6)


def test_laprop_warmup_and_state_round_trip():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = LaProp([parameter], lr=0.1, warmup_steps=2)
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    first = parameter.detach().clone()
    assert first.item() == pytest.approx(0.95)
    state = optimizer.state_dict()

    restored_parameter = torch.nn.Parameter(first.clone())
    restored = LaProp([restored_parameter], lr=0.1, warmup_steps=2)
    restored.load_state_dict(state)
    restored_parameter.grad = torch.tensor([1.0])
    restored.step()
    assert restored_parameter.item() < first.item()


def test_v4_gradient_clipping_uses_finite_high_precision_norm():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.full_like(parameter, 1e30)
    norm = finite_clip_grad_norm_([parameter], max_norm=1.0)
    assert math.isfinite(norm)
    assert torch.isfinite(parameter.grad).all()
    assert torch.linalg.vector_norm(parameter.grad).item() <= 1.00001


def test_v4_gradient_clipping_rejects_non_finite_gradients():
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.grad = torch.tensor([float("inf")])
    with pytest.raises(FloatingPointError):
        finite_clip_grad_norm_([parameter], max_norm=1.0)


def test_sac_observation_includes_previous_executed_action():
    env = RoboboSACEnv.__new__(RoboboSACEnv)
    env._previous_executed_action = torch.tensor([0.25, -0.5]).numpy()
    obs = env._flatten_obs(
        {
            "blob": torch.tensor([0.5, 0.5, 0.1, 1.0]).numpy(),
            "ir": torch.zeros(8).numpy(),
        }
    )
    assert obs.shape == (14,)
    assert obs[-2:].tolist() == pytest.approx([0.25, -0.5])


def test_green_saliency_loss_penalizes_missing_food_pixels():
    target = torch.zeros(1, 1, 3, 8, 8)
    target[..., 1, 3:5, 3:5] = 1.0
    missing = torch.zeros_like(target)
    reconstructed = target.clone()
    assert green_saliency_loss(reconstructed, target).item() == pytest.approx(0.0)
    assert green_saliency_loss(missing, target).item() > 0.0


def test_coppelia_preflight_rejects_bind_address():
    with pytest.raises(ConnectionError, match="bind address"):
        check_coppelia_service("0.0.0.0", 23000)


def test_coppelia_preflight_accepts_reachable_tcp_service():
    with patch("socket.create_connection") as connect:
        connect.return_value.__enter__.return_value = object()
        check_coppelia_service("127.0.0.1", 23000)
        connect.assert_called_once_with(("127.0.0.1", 23000), timeout=3.0)
