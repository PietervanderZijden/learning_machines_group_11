import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from learning_machines.dreamerv4.dreamerv4_image import ImageDreamerV4Agent
from train_dreamerv4_image import (
    StreamingEpisodeDataset,
    TrainingProgress,
    load_recorded_episodes,
    scan_recorded_episodes,
)


def test_dreamerv4_full_training_and_reload_pipeline():
    batch, transitions = 2, 3
    agent = ImageDreamerV4Agent(
        latent_dim=16,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        context_length=8,
        imagination_horizon=2,
        lr=1e-4,
        image_size=64,
        ir_dim=8,
        mtp_length=2,
    ).cpu()
    images = torch.rand(batch * (transitions + 1), 3, 64, 64)
    ir = torch.rand(batch * (transitions + 1), 8)
    tokenizer_metrics = agent.update_tokenizer(images[:4], ir[:4])
    with torch.no_grad():
        latents = agent.tokenizer.encode(images, ir).reshape(batch, transitions + 1, -1)
    actions = torch.rand(batch, transitions, 2) * 2 - 1
    rewards = torch.randn(batch, transitions)
    dones = torch.zeros(batch, transitions)
    mask = torch.ones(batch, transitions)

    metric_groups = [
        tokenizer_metrics,
        agent.update_dynamics(latents, actions, rewards, dones, mask),
        agent.update_mtp_behavior(latents, actions, rewards, mask),
    ]
    agent.freeze_behavior_prior(copy_to_actor=True)
    metric_groups.append(agent.update_actor_critic(latents[:, 0]))
    assert "ac/actor_entropy" in metric_groups[-1]
    for metrics in metric_groups:
        for value in metrics.values():
            if isinstance(value, (int, float, np.number)):
                assert np.isfinite(float(value))

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/agent.pt"
        agent.save(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert "python_rng_state" in checkpoint
        assert "numpy_rng_state" in checkpoint
        assert "torch_rng_state" in checkpoint
        loaded = ImageDreamerV4Agent.load(path, torch.device("cpu"))
        with torch.no_grad():
            action = loaded.act(latents[:, 0], deterministic=True)
        assert action.shape == (batch, 2)
        assert torch.isfinite(action).all()


def test_dreamerv4_dynamics_handles_right_padded_sequences():
    agent = ImageDreamerV4Agent(
        latent_dim=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        context_length=4,
        image_size=64,
    ).cpu()
    latents = torch.randn(2, 5, agent.state_dim)
    actions = torch.randn(2, 4, 2).clamp(-1, 1)
    rewards = torch.zeros(2, 4)
    dones = torch.zeros(2, 4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)

    metrics = agent.update_dynamics(latents, actions, rewards, dones, mask)

    assert all(np.isfinite(value) for value in metrics.values())


def test_dreamerv4_dynamics_rejects_non_right_padded_masks():
    agent = ImageDreamerV4Agent(
        latent_dim=8,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ff_dim=32,
        context_length=4,
        image_size=64,
    ).cpu()
    latents = torch.randn(1, 5, agent.state_dim)
    actions = torch.zeros(1, 4, 2)
    tau = torch.zeros(1, 4)
    d = torch.full((1, 4), 0.25)
    mask = torch.tensor([[1, 0, 1, 0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="right padding"):
        agent.dynamics(latents, actions, tau, d, mask)


def test_recorded_episode_loader_skips_incompatible_legacy_files():
    with tempfile.TemporaryDirectory() as directory:
        episode_dir = Path(directory) / "episodes"
        episode_dir.mkdir()
        common = {
            "images": np.zeros((3, 3, 64, 64), dtype=np.uint8),
            "irs": np.zeros((3, 8), dtype=np.float32),
            "actions": np.zeros((2, 2), dtype=np.float32),
            "rewards": np.zeros(2, dtype=np.float32),
            "dones": np.array([False, True]),
        }
        np.savez_compressed(episode_dir / "legacy.npz", **common)
        np.savez_compressed(
            episode_dir / "valid.npz",
            **common,
            observation_contract=np.array("robobo-obs-v2"),
            reward_contract=np.array("robobo-reward-v4"),
            control_interval_seconds=np.array(0.4),
            calibration_profile=np.array("simulation"),
            phone_tilt=np.array(100),
        )
        episodes = load_recorded_episodes(directory, "simulation")
        assert len(episodes) == 1
        scan = scan_recorded_episodes(directory, "simulation")
        assert len(scan.episodes) == 1
        assert len(scan.skipped) == 1

        np.savez_compressed(
            episode_dir / "old_timing.npz",
            **common,
            observation_contract=np.array("robobo-obs-v2"),
            reward_contract=np.array("robobo-reward-v4"),
            control_interval_seconds=np.array(0.2),
            calibration_profile=np.array("simulation"),
            phone_tilt=np.array(100),
        )
        episodes = load_recorded_episodes(directory, "simulation")
        assert len(episodes) == 1
        scan = scan_recorded_episodes(directory, "simulation")
        assert len(scan.skipped) == 2


def _write_episode(path: Path, calibration: str, value: int = 0):
    np.savez_compressed(
        path,
        images=np.full((4, 3, 8, 8), value, dtype=np.uint8),
        irs=np.full((4, 8), value / 10, dtype=np.float32),
        actions=np.zeros((3, 2), dtype=np.float32),
        rewards=np.zeros(3, dtype=np.float32),
        dones=np.array([False, False, True]),
        observation_contract=np.array("robobo-obs-v2"),
        reward_contract=np.array("robobo-reward-v4"),
        control_interval_seconds=np.array(0.4),
        calibration_profile=np.array(calibration),
        phone_tilt=np.array(100),
    )


def test_streaming_dataset_mixes_sources_without_preloading(tmp_path):
    sim_dir = tmp_path / "sim" / "episodes"
    hw_dir = tmp_path / "hw" / "episodes"
    sim_dir.mkdir(parents=True)
    hw_dir.mkdir(parents=True)
    _write_episode(sim_dir / "ep_0.npz", "simulation", 1)
    _write_episode(hw_dir / "ep_0.npz", "hardware", 2)
    _write_episode(hw_dir / "ep_1.npz", "hardware", 3)
    sim = scan_recorded_episodes(str(sim_dir.parent), "simulation", source="simulation")
    hardware = scan_recorded_episodes(str(hw_dir.parent), "hardware", source="hardware")
    dataset = StreamingEpisodeDataset(
        sim.episodes,
        hardware_train=hardware.episodes[:1],
        hardware_validation=hardware.episodes[1:],
        hardware_sample_ratio=1.0,
    )
    images, ir, fraction = dataset.sample_observation_batch(
        4, True, torch.device("cpu")
    )
    assert images.shape == (4, 3, 8, 8)
    assert ir.shape == (4, 8)
    assert fraction == 1.0
    assert torch.all(images > 0)
    assert dataset.hardware_validation[0] not in dataset.training_refs


def test_hardware_vector_observations_supply_ir(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    np.savez_compressed(
        episode_dir / "hardware.npz",
        images=np.zeros((4, 3, 8, 8), dtype=np.uint8),
        observations=np.tile(
            np.arange(12, dtype=np.float32), (4, 1)
        ),
        actions=np.zeros((3, 2), dtype=np.float32),
        rewards=np.zeros(3, dtype=np.float32),
        dones=np.array([False, False, True]),
        manifest=np.array(
            '{"observation_contract":"robobo-obs-v2",'
            '"reward_contract":"robobo-reward-v4",'
            '"control_interval_seconds":0.4,'
            '"calibration_profile":"hardware","phone_tilt":100}'
        ),
    )
    episodes = load_recorded_episodes(
        str(tmp_path), "hardware", source="hardware"
    )
    assert len(episodes) == 1
    np.testing.assert_array_equal(
        episodes[0]["irs"][0], np.arange(4, 12, dtype=np.float32)
    )


def test_training_progress_round_trip(tmp_path):
    path = tmp_path / "progress.json"
    progress = TrainingProgress(
        tokenizer=10, dynamics=20, mtp=30, pmpo=40, global_log_step=100
    )
    progress.save(path)
    assert TrainingProgress.load(path) == progress
