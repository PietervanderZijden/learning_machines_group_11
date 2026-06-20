import numpy as np
import torch

from learning_machines.dreamerv3.actor_critic import Actor, lambda_return
from learning_machines.dreamerv3.rssm import RSSM
from learning_machines.dreamerv3.replay_buffer import ReplayBuffer
from learning_machines.dreamerv3.config import DreamerV3Config
from learning_machines.distributional import (
    _make_bin_centers,
    logits_to_value,
    symexp,
)
from learning_machines.dreamerv3.dreamerv3 import (
    DreamerV3,
    _cpu_byte_rng_states,
    select_imagination_starts,
)
from learning_machines.dreamerv4.actor_critic import (
    SquashedGaussianActor,
    compute_td_lambda_returns,
)


def test_cuda_rng_states_are_normalized_to_cpu_byte_tensors():
    states = _cpu_byte_rng_states([
        torch.tensor([1, 2, 3], dtype=torch.int64),
        np.array([4, 5, 6], dtype=np.uint8),
    ])

    assert all(state.device.type == "cpu" for state in states)
    assert all(state.dtype == torch.uint8 for state in states)
    torch.testing.assert_close(states[0], torch.tensor([1, 2, 3], dtype=torch.uint8))


def test_v3_lambda_return_bootstraps_terminal_value():
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[10.0, 20.0, 30.0]])
    continue_logits = torch.full((1, 2), 20.0)
    result = lambda_return(rewards, values, continue_logits, gamma=0.5, lam=0.5)
    expected_t1 = 2.0 + 0.5 * 30.0
    expected_t0 = 1.0 + 0.5 * (0.5 * 20.0 + 0.5 * expected_t1)
    torch.testing.assert_close(result, torch.tensor([[expected_t0, expected_t1]]))


def test_v4_lambda_return_uses_zero_initial_gae():
    rewards = torch.tensor([[1.0, 2.0]])
    values = torch.tensor([[10.0, 20.0, 30.0]])
    bootstrap = torch.tensor([[30.0]])
    dones = torch.zeros(1, 2)
    result = compute_td_lambda_returns(
        rewards, values, bootstrap, dones, gamma=0.5, lam=0.5
    )
    delta1 = 2.0 + 0.5 * 30.0 - 20.0
    expected1 = delta1 + 20.0
    delta0 = 1.0 + 0.5 * 20.0 - 10.0
    expected0 = delta0 + 0.5 * 0.5 * delta1 + 10.0
    torch.testing.assert_close(result, torch.tensor([[expected0, expected1]]))


def test_v3_actor_std_bounds_and_finite_score_function_gradients():
    actor = Actor(6, 2)
    state = torch.randn(32, 6)
    action, log_prob = actor.get_action_and_log_prob(state)
    assert action.requires_grad is False
    std = actor.std(state)
    assert float(std.detach().min()) >= torch.exp(torch.tensor(-5.0)).item()
    assert float(std.detach().max()) <= torch.exp(torch.tensor(2.0)).item()
    loss = -log_prob.mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )


def test_twohot_decoder_takes_expectation_in_original_value_space():
    bins = _make_bin_centers(torch.device("cpu"))
    logits = torch.full((255,), -1000.0)
    logits[torch.argmin((bins - 0.0).abs())] = 0.0
    logits[torch.argmin((bins - 2.0).abs())] = 0.0
    probabilities = torch.softmax(logits, dim=-1)
    expected = (probabilities * symexp(bins)).sum()
    torch.testing.assert_close(logits_to_value(logits), expected)


def test_imagination_start_selection_keeps_food_event_states():
    states = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    rewards = torch.zeros(2, 5)
    rewards[0, 3] = 100.0
    rewards[1, 1] = 100.0
    selected = select_imagination_starts(states, rewards, 2, 1.0)
    selected = selected.reshape(2, 2, 3)
    torch.testing.assert_close(selected[0, 0], states[0, 3])
    torch.testing.assert_close(selected[1, 0], states[1, 1])


def test_imagination_start_selection_can_use_all_transition_states():
    states = torch.randn(2, 6, 3)
    rewards = torch.zeros(2, 5)
    selected = select_imagination_starts(states, rewards, 0, 1.0)
    torch.testing.assert_close(selected, states[:, :-1].reshape(-1, 3))


def test_raw_kl_is_reported_without_free_nat_floor():
    rssm = RSSM(
        obs_dim=4,
        action_dim=2,
        deterministic_size=8,
        stochastic_classes=2,
        stochastic_bins=3,
        hidden_size=8,
    )
    logits = torch.zeros(5, 6)
    raw = rssm.raw_kl(logits, logits)
    dyn, rep = rssm.kl_loss(logits, logits, free_nats=1.0)
    torch.testing.assert_close(raw, torch.zeros_like(raw))
    assert dyn.item() == 1.0
    assert rep.item() == 1.0


def test_v4_policy_uses_detached_score_function_samples():
    actor = SquashedGaussianActor(obs_dim=6, act_dim=2)
    action, log_prob = actor(torch.randn(16, 6))
    assert action.requires_grad is False
    (-log_prob.mean()).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )


def test_v3_replay_preserves_terminal_next_observation():
    buffer = ReplayBuffer(3, 2, capacity=20, sequence_length=2)
    buffer.add(np.array([0, 0, 0]), np.array([0, 0]), 0.0, False)
    buffer.add(
        np.array([1, 1, 1]),
        np.array([1, 1]),
        1.0,
        True,
        next_obs=np.array([2, 2, 2]),
    )
    episode = buffer._episodes[0]
    assert episode["obs"].shape == (3, 3)
    batch = buffer.sample(1, torch.device("cpu"))
    torch.testing.assert_close(batch["obs"][0, -1], torch.tensor([2.0, 2.0, 2.0]))


def test_v3_replay_balances_sparse_reward_event_sequences():
    np.random.seed(4)
    buffer = ReplayBuffer(
        3,
        2,
        capacity=1000,
        sequence_length=4,
        reward_event_fraction=1.0,
        reward_event_threshold=1.0,
    )
    obs = np.zeros((21, 3), dtype=np.float32)
    action = np.zeros((20, 2), dtype=np.float32)
    reward = np.zeros(20, dtype=np.float32)
    reward[10] = 100.0
    done = np.zeros(20, dtype=np.float32)
    done[-1] = 1.0
    buffer.add_episode(obs, action, reward, done)
    batch = buffer.sample(16, torch.device("cpu"))
    assert batch["reward_event_sample_fraction"].item() == 1.0
    assert torch.all(torch.max(batch["reward"], dim=1).values >= 100.0)


def test_v3_replay_restores_recent_aligned_recordings(tmp_path):
    episode_dir = tmp_path / "episodes"
    episode_dir.mkdir()
    for index in range(3):
        np.savez_compressed(
            episode_dir / f"ep_{index:06d}.npz",
            images=np.full((4, 3, 8, 8), index, dtype=np.uint8),
            irs=np.full((4, 8), index, dtype=np.float32),
            actions=np.zeros((3, 2), dtype=np.float32),
            rewards=np.zeros(3, dtype=np.float32),
            dones=np.array([False, False, True]),
            observation_contract=np.array("robobo-obs-v2"),
            reward_contract=np.array("robobo-reward-v4"),
            control_interval_seconds=np.array(0.4),
        )
    buffer = ReplayBuffer(
        3, 2, capacity=6, sequence_length=2, obs_shape=(3, 8, 8), ir_dim=8
    )
    restored = buffer.restore_recorded_episodes(tmp_path)
    assert restored == 6
    assert buffer.size == 6
    assert len(buffer._episodes) == 2
    assert buffer._episodes[-1]["obs"][0, 0, 0, 0] == 2 / 255.0


def test_v3_recurrent_context_uses_executed_action():
    config = DreamerV3Config(
        obs_dim=3,
        action_dim=2,
        deterministic_size=8,
        stochastic_classes=2,
        stochastic_bins=2,
        hidden_size=8,
        embed_size=8,
        mlp_hidden=8,
        actor_hidden=8,
        critic_hidden=8,
        use_images=False,
        use_multimodal=False,
        buffer_capacity=20,
    )
    agent = DreamerV3(config, device="cpu")
    executed = np.array([0.25, -0.5], dtype=np.float32)
    agent.set_executed_action(executed)
    torch.testing.assert_close(
        agent._prev_action,
        torch.tensor(executed).reshape(1, 2),
    )


def test_v3_train_step_includes_replay_and_slow_critic_losses():
    config = DreamerV3Config(
        obs_dim=3,
        action_dim=2,
        deterministic_size=8,
        stochastic_classes=2,
        stochastic_bins=2,
        hidden_size=8,
        embed_size=8,
        mlp_hidden=8,
        actor_hidden=8,
        critic_hidden=8,
        use_images=False,
        use_multimodal=False,
        sequence_length=3,
        batch_size=2,
        imagination_horizon=2,
        imagination_starts=2,
        optimizer_warmup=0,
        buffer_capacity=20,
    )
    agent = DreamerV3(config, device="cpu")
    batch = {
        "obs": torch.randn(2, 4, 3),
        "action": torch.rand(2, 3, 2) * 2 - 1,
        "reward": torch.tensor([[0.0, 100.0, 0.0], [0.0, 0.0, 0.0]]),
        "done": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
    }
    metrics = agent.train_step(batch)
    assert metrics["imagination_starts"] == 4
    assert metrics["critic_replay_loss"] > 0
    assert metrics["critic_slow_regularization"] > 0
    assert all(
        np.isfinite(value)
        for value in metrics.values()
        if isinstance(value, (int, float))
    )
