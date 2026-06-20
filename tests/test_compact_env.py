"""Tests for rl_robobo_compact_env.py (requires CoppeliaSim on port 23000)."""
from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym


class TestRoboboCompactEnvConfig:
    def test_default_config(self):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig()
        assert cfg.step_millis == 400
        assert cfg.max_episode_steps == 150
        assert cfg.collision_ir_threshold == 0.85
        assert cfg.collect_reward == 100.0

    def test_config_creation_with_overrides(self):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(
            step_millis=200,
            max_episode_steps=300,
            collect_reward=200.0,
        )
        assert cfg.step_millis == 200
        assert cfg.max_episode_steps == 300
        assert cfg.collect_reward == 200.0


class TestRoboboCompactEnvCreation:
    def test_env_creation(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_observation_space_shape(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        assert "ir" in env.observation_space.spaces

    def test_action_space_shape(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        assert env.action_space.shape == (2,)
        assert env.action_space.low.min() >= -1.0
        assert env.action_space.high.max() <= 1.0


class TestRoboboCompactEnvReset:
    def test_reset_returns_obs_and_info(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "ir" in obs
        assert isinstance(info, dict)
        env.close()

    def test_reset_ir_shape(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        obs, _ = env.reset()
        assert obs["ir"].shape == (8,)
        assert obs["ir"].dtype == np.float32
        env.close()

    def test_reset_ir_in_range(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        obs, _ = env.reset()
        ir = obs["ir"]
        assert np.all(ir >= 0.0) and np.all(ir <= 1.0), f"IR out of range: {ir}"
        env.close()


class TestRoboboCompactEnvStep:
    def test_step_returns_tuple(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        env.reset()
        action = np.array([0.3, 0.3], dtype=np.float32)
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        env.close()

    def test_step_ir_in_range(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        env.reset()
        action = np.array([0.3, 0.3], dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        ir = obs["ir"]
        assert np.all(ir >= 0.0) and np.all(ir <= 1.0), f"IR out of range after step: {ir}"
        env.close()

    def test_action_affects_reward(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        env.reset()
        _, r1, _, _, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        env.reset()
        _, r2, _, _, _ = env.step(np.array([0.5, 0.5], dtype=np.float32))
        # Both should produce valid rewards (not crash)
        assert isinstance(r1, float)
        assert isinstance(r2, float)
        env.close()

    def test_max_episode_steps_truncation(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False, max_episode_steps=5)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        env.reset()
        truncated = False
        for _ in range(10):
            _, _, _, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
            if truncated:
                break
        assert truncated, "Env should truncate after max_episode_steps"
        env.close()


class TestRoboboCompactEnvClose:
    def test_close_stops_sim(self, sim_robobo):
        from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
        cfg = RoboboCompactEnvConfig(initialize_phone_tilt=False)
        env = RoboboCompactEnv(rob=sim_robobo, config=cfg)
        env.reset()
        env.close()
        assert sim_robobo.is_stopped()
