from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from learning_machines.dreamerv4_full import (
    DreamerV4FullAgent,
    DreamerV4FullConfig,
)
from learning_machines.dreamerv4_full.agent import (
    distribution_value,
    td_lambda_returns,
)
from learning_machines.dreamerv4_full.model import sample_shortcut_schedule
from train_dreamerv4_full import sample_latent_batch


def small_config(**overrides):
    values = {
        "image_size": 16,
        "patch_size": 8,
        "ir_dim": 2,
        "action_dim": 2,
        "num_tasks": 2,
        "model_dim": 16,
        "latent_tokens": 4,
        "latent_channels": 4,
        "register_tokens": 2,
        "tokenizer_layers": 2,
        "dynamics_layers": 2,
        "heads": 4,
        "kv_heads": 2,
        "ff_multiplier": 2,
        "temporal_every": 2,
        "context_length": 4,
        "shortcut_steps": 4,
        "mtp_length": 2,
        "imagination_horizon": 2,
        "lpips_weight": 0.0,
    }
    values.update(overrides)
    return DreamerV4FullConfig(**values)


def make_batch(agent, batch=2, transitions=4):
    cfg = agent.cfg
    latents = torch.randn(
        batch,
        transitions + 1,
        cfg.latent_tokens,
        cfg.latent_channels,
        device=agent.device,
    ).tanh()
    actions = torch.rand(
        batch, transitions, cfg.action_dim, device=agent.device
    ) * 2 - 1
    rewards = torch.randn(batch, transitions, device=agent.device)
    dones = torch.zeros(batch, transitions, device=agent.device)
    mask = torch.ones(batch, transitions, device=agent.device)
    return latents, actions, rewards, dones, mask


def test_shortcut_schedule_uses_power_of_two_steps_and_matching_grid():
    signal, step = sample_shortcut_schedule((512,), torch.device("cpu"), 4)
    assert set(step.tolist()) <= {1.0, 0.5, 0.25}
    grid = signal / step
    torch.testing.assert_close(grid, grid.round())
    assert torch.all(signal >= 0)
    assert torch.all(signal < 1)


def test_tokenizer_is_causal_and_preserves_spatial_latent_tokens():
    agent = DreamerV4FullAgent(small_config(), "cpu")
    images = torch.rand(2, 3, 3, 16, 16)
    ir = torch.rand(2, 3, 2)
    changed = images.clone()
    changed[:, -1] = torch.rand_like(changed[:, -1])

    first = agent.tokenizer.encode(images, ir)
    second = agent.tokenizer.encode(changed, ir)

    assert first.shape == (2, 3, 4, 4)
    torch.testing.assert_close(first[:, :-1], second[:, :-1], atol=1e-5, rtol=1e-5)
    reconstruction, ir_reconstruction = agent.tokenizer.decode(first)
    assert reconstruction.shape == images.shape
    assert ir_reconstruction.shape == ir.shape


def test_agent_token_cannot_change_world_prediction():
    agent = DreamerV4FullAgent(small_config(), "cpu")
    cfg = agent.cfg
    latent = torch.randn(2, 3, cfg.latent_tokens, cfg.latent_channels)
    actions = torch.randn(2, 3, cfg.action_dim)
    signal = torch.rand(2, 3)
    step = torch.full((2, 3), 0.25)

    task_zero = agent.dynamics(
        latent, actions, signal, step, torch.zeros(2, dtype=torch.long)
    )
    task_one = agent.dynamics(
        latent, actions, signal, step, torch.ones(2, dtype=torch.long)
    )

    torch.testing.assert_close(task_zero["latent"], task_one["latent"])
    assert not torch.allclose(task_zero["agent"], task_one["agent"])


def test_unknown_action_embedding_is_distinct_from_zero_action():
    agent = DreamerV4FullAgent(small_config(), "cpu")
    cfg = agent.cfg
    latent = torch.randn(1, 2, cfg.latent_tokens, cfg.latent_channels)
    actions = torch.zeros(1, 2, cfg.action_dim)
    signal = torch.full((1, 2), 0.5)
    step = torch.full((1, 2), 0.25)
    with torch.no_grad():
        agent.dynamics.unknown_action.fill_(1.0)
        agent.dynamics.latent_output.weight.normal_(std=0.1)
    known = agent.dynamics(
        latent, actions, signal, step, action_known=torch.ones(1, 2)
    )["latent"]
    unknown = agent.dynamics(
        latent, actions, signal, step, action_known=torch.zeros(1, 2)
    )["latent"]
    assert not torch.allclose(known, unknown)


def test_world_and_policy_sequences_use_distinct_action_alignment():
    agent = DreamerV4FullAgent(small_config(), "cpu")
    latents = torch.arange(5.0).view(1, 5, 1, 1).expand(1, 5, 4, 4)
    actions = torch.arange(8.0).view(1, 4, 2)

    world_latents, world_actions = agent._world_sequence(latents, actions)
    policy_latents, previous_actions = agent._policy_sequence(latents, actions)

    torch.testing.assert_close(world_latents, latents[:, 1:])
    torch.testing.assert_close(world_actions, actions)
    torch.testing.assert_close(policy_latents, latents[:, :-1])
    torch.testing.assert_close(previous_actions[:, 0], torch.zeros(1, 2))
    torch.testing.assert_close(previous_actions[:, 1:], actions[:, :-1])


def test_full_training_phases_have_finite_metrics(tmp_path: Path):
    agent = DreamerV4FullAgent(small_config(), "cpu")
    images = torch.rand(2, 3, 3, 16, 16)
    ir = torch.rand(2, 3, 2)
    tokenizer_metrics = agent.update_tokenizer(images, ir)
    batch = make_batch(agent)
    world_metrics = agent.update_world_model(batch[0], batch[1], batch[4])
    finetune_metrics = agent.update_agent_finetune(*batch)
    agent.freeze_behavior_prior()
    imagination_metrics = agent.update_imagination(batch[0], batch[1])

    for metrics in (
        tokenizer_metrics,
        world_metrics,
        finetune_metrics,
        imagination_metrics,
    ):
        assert all(np.isfinite(value) for value in metrics.values())

    path = tmp_path / "full.pt"
    agent.save(path)
    loaded = DreamerV4FullAgent.load(path, "cpu")
    assert loaded.cfg == agent.cfg
    for left, right in zip(agent.parameters(), loaded.parameters()):
        torch.testing.assert_close(left, right)


def test_td_lambda_matches_one_step_terminal_returns():
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[4.0, 5.0, 6.0]])
    continuation = torch.tensor([[1.0, 0.0]])
    result = td_lambda_returns(
        rewards, values, continuation, gamma=0.5, lambda_=1.0
    )
    expected_last = 2.0
    expected_first = 1.0 + 0.5 * expected_last
    torch.testing.assert_close(result, torch.tensor([[expected_first, expected_last]]))


def test_uniform_distributional_head_decodes_to_zero():
    value = distribution_value(torch.zeros(3, 255))
    torch.testing.assert_close(value, torch.zeros(3), atol=1e-6, rtol=0)


def test_lpips_weights_are_not_stored_in_agent_checkpoint():
    agent = DreamerV4FullAgent(small_config(), "cpu")
    assert not any(
        "lpips._metric" in key or "lpips.metric" in key
        for key in agent.state_dict()
    )


def test_schedulers_and_learning_rates_resume_exactly(tmp_path: Path):
    agent = DreamerV4FullAgent(small_config(warmup_fraction=0.2), "cpu")
    agent.configure_schedulers({
        "tokenizer": 10,
        "world": 10,
        "finetune": 10,
        "imagination": 10,
    })
    batch = make_batch(agent)
    metrics = agent.update_world_model(batch[0], batch[1], batch[4])
    assert metrics["world/learning_rate"] < agent.cfg.learning_rate

    path = tmp_path / "scheduled.pt"
    agent.save(path)
    loaded = DreamerV4FullAgent.load(path, "cpu")
    assert loaded.schedulers["world"].last_epoch == agent.schedulers["world"].last_epoch
    assert (
        loaded.world_optimizer.param_groups[0]["lr"]
        == agent.world_optimizer.param_groups[0]["lr"]
    )


def test_standalone_samples_use_unknown_actions():
    cfg = small_config()
    episode = {
        "latents": torch.randn(5, cfg.latent_tokens, cfg.latent_channels),
        "actions": torch.randn(4, cfg.action_dim),
        "rewards": torch.randn(4),
        "dones": torch.zeros(4),
    }
    batch = sample_latent_batch(
        [episode], 2, 3, torch.device("cpu"), start_frame_fraction=1.0
    )
    mask, action_known = batch[4], batch[5]
    assert torch.all(mask[:, 0] == 1)
    assert torch.all(mask[:, 1:] == 0)
    assert not action_known.any()


def test_tokenizer_masking_does_not_explode_gradients():
    'Masked patches used a zero-initialized mask token, which made RMSNorm.'
    agent = DreamerV4FullAgent(small_config(lpips_weight=0.0), "cpu")
    images = torch.rand(2, 3, 3, 16, 16)
    ir = torch.rand(2, 3, 2)
    agent.tokenizer_optimizer.zero_grad(set_to_none=True)
    output = agent.tokenizer(images, ir)
    output["loss"].backward()
    max_norm = max(
        p.grad.norm().item()
        for p in agent.tokenizer.parameters()
        if p.grad is not None
    )
    assert max_norm < 1e4, f"tokenizer gradient exploded: {max_norm}"

    mask_norm = agent.tokenizer.mask_token.grad.norm().item()
    assert mask_norm < 1e4, f"mask_token gradient exploded: {mask_norm}"


def test_tokenizer_mse_is_computed_only_on_masked_patches():
    'MSE must be averaged only over masked pixels, not all pixels.'
    agent = DreamerV4FullAgent(small_config(lpips_weight=0.0), "cpu")
    images = torch.rand(2, 3, 3, 16, 16)
    ir = torch.rand(2, 3, 2)


    torch.manual_seed(42)
    output = agent.tokenizer(images, ir)
    reconstruction = output["reconstruction"]
    reported_mse = output["mse_loss"].item()


    torch.manual_seed(42)
    latent, patch_mask = agent.tokenizer.encode(
        images, ir, mask_patches=True, return_mask=True
    )
    B, T, C, H, W = reconstruction.shape
    patch = agent.tokenizer.cfg.patch_size
    gh, gw = H // patch, W // patch
    pixel_mask = (
        patch_mask.view(B, T, gh, gw)
        .float()
        .unsqueeze(2)
        .repeat_interleave(patch, dim=-2)
        .repeat_interleave(patch, dim=-1)
    )
    expected_mse = (
        ((reconstruction - images).pow(2) * pixel_mask).sum()
        / pixel_mask.sum().clamp_min(1.0)
    ).item()
    assert np.isclose(reported_mse, expected_mse, rtol=1e-4, atol=1e-6)



    full_mse = F.mse_loss(reconstruction, images).item()
    assert not np.isclose(reported_mse, full_mse, rtol=1e-2, atol=1e-4)


def test_tokenizer_mask_ratio_matches_config():
    'Patch masking uses the configured fixed mask ratio.'
    target_ratio = 0.5
    agent = DreamerV4FullAgent(
        small_config(lpips_weight=0.0, mask_ratio=target_ratio), "cpu"
    )
    images = torch.rand(2, 3, 3, 16, 16)
    ir = torch.rand(2, 3, 2)
    ratios = []
    for _ in range(20):
        _, patch_mask = agent.tokenizer.encode(
            images, ir, mask_patches=True, return_mask=True
        )
        ratios.append(patch_mask.float().mean().item())
    mean_ratio = sum(ratios) / len(ratios)
    assert abs(mean_ratio - target_ratio) < 0.05, (
        f"mask ratio {mean_ratio} far from config {target_ratio}"
    )


def test_tokenizer_grad_norm_does_not_grow_progressively():
    'Masked-only loss keeps the optimizer away from the clip boundary.'
    agent = DreamerV4FullAgent(small_config(lpips_weight=0.0), "cpu")
    images = torch.rand(2, 3, 3, 16, 16)
    ir = torch.rand(2, 3, 2)
    norms = []
    for _ in range(10):
        metrics = agent.update_tokenizer(images, ir)
        norms.append(metrics["tok/grad_norm"])
    assert all(np.isfinite(n) for n in norms)
    assert max(norms) < 50, f"tokenizer grad_norm too large: {max(norms)}"
    increases = sum(1 for i in range(1, len(norms)) if norms[i] > norms[i - 1])
    assert increases <= len(norms) // 2, (
        f"grad_norm trended upward too often: {norms}"
    )


def test_tokenizer_forward_handles_single_frame_input():
    'forward() accepts both 4-D (B, C, H, W) and 5-D (B, T, C, H, W) images.'
    agent = DreamerV4FullAgent(small_config(lpips_weight=0.0), "cpu")
    images_5d = torch.rand(2, 3, 3, 16, 16)
    ir_5d = torch.rand(2, 3, 2)
    images_4d = images_5d[:, 0]
    ir_4d = ir_5d[:, 0]

    out_5d = agent.tokenizer(images_5d, ir_5d)
    out_4d = agent.tokenizer(images_4d, ir_4d)

    assert out_4d["reconstruction"].shape == images_4d.shape
    assert out_4d["latent"].shape == out_5d["latent"][:, 0].shape
    assert np.isfinite(out_4d["loss"].item())


def test_qk_normalization_does_not_use_sqrt_scaling():
    'QK normalization bounds dot products to [-1, 1]; there must not be an.'
    from learning_machines.dreamerv4_full.transformer import GroupedQueryAttention

    dim = 32
    heads = 4
    kv_heads = 2
    module = GroupedQueryAttention(dim, heads, kv_heads)
    x = torch.randn(2, 8, dim)
    positions = torch.arange(8)

    with torch.no_grad():
        batch, length, _ = x.shape
        head_dim = dim // heads
        q = module.q_proj(x).view(batch, length, heads, head_dim)
        k = module.k_proj(x).view(batch, length, kv_heads, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q = F.normalize(q.float(), dim=-1).to(x.dtype)
        k = F.normalize(k.float(), dim=-1).to(x.dtype)
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        logits = torch.matmul(q, k.transpose(-2, -1))
        assert logits.max().item() <= 1.0 + 1e-5
        assert logits.min().item() >= -1.0 - 1e-5


def test_running_rms_floor_prevents_runaway_amplification():
    'RunningRMS should stop shrinking its divisor once loss drops below a.'
    from learning_machines.dreamerv4_full.tokenizer import RunningRMS

    rms = RunningRMS(decay=0.5, floor_ratio=0.1)
    first = torch.tensor(1.0)
    normalized_first = rms.normalize(first)
    torch.testing.assert_close(normalized_first, torch.tensor(1.0), atol=1e-6, rtol=0)


    for _ in range(20):
        normalized = rms.normalize(torch.tensor(0.001))

    torch.testing.assert_close(normalized, torch.tensor(0.01), atol=1e-4, rtol=0)


def test_dynamics_unknown_actions_do_not_explode_gradients():
    'Unknown actions used a zero-initialized embedding, which made RMSNorm.'
    agent = DreamerV4FullAgent(small_config(), "cpu")
    cfg = agent.cfg
    latents = torch.randn(2, 5, cfg.latent_tokens, cfg.latent_channels)
    actions = torch.zeros(2, 5, cfg.action_dim)
    signal = torch.full((2, 5), 0.5)
    step = torch.full((2, 5), 0.25)
    agent.world_optimizer.zero_grad(set_to_none=True)
    output = agent.dynamics(
        latents,
        actions,
        signal,
        step,
        action_known=torch.zeros(2, 5, dtype=torch.bool),
    )
    loss = output["latent"].mean() + output["agent"].mean()
    loss.backward()
    max_norm = max(
        p.grad.norm().item()
        for p in agent.dynamics.parameters()
        if p.grad is not None
    )
    assert max_norm < 1e4, f"dynamics gradient exploded: {max_norm}"
    unknown_norm = agent.dynamics.unknown_action.grad.norm().item()
    assert unknown_norm < 1e4, f"unknown_action gradient exploded: {unknown_norm}"
