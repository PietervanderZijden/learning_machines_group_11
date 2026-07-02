'Tests for domain_randomization.py (no sim required).'
from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym


class MockEnv(gym.Env):
    'Minimal mock env that produces 3-channel HWC images (like real camera).'
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8),
            "ir": gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
        })
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs = {
            "image": np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8),
            "ir": np.random.rand(8).astype(np.float32),
        }
        return obs, {}

    def step(self, action):
        obs = {
            "image": np.random.randint(0, 255, (3, 64, 64), dtype=np.uint8),
            "ir": np.random.rand(8).astype(np.float32),
        }
        return obs, 0.0, False, False, {}


class TestRandomizationRanges:
    def test_default_creation(self):
        from learning_machines.domain_randomization import RandomizationRanges
        r = RandomizationRanges()
        assert r is not None

    def test_has_expected_fields(self):
        from learning_machines.domain_randomization import RandomizationRanges
        r = RandomizationRanges()
        assert hasattr(r, 'ir_gain')
        assert hasattr(r, 'camera_contrast')


class TestDomainRandomizationWrapper:
    def test_wrapper_creation(self):
        from learning_machines.domain_randomization import DomainRandomizationWrapper
        env = MockEnv()
        wrapper = DomainRandomizationWrapper(env)
        assert wrapper is not None

    def test_wrapper_obs_space(self):
        from learning_machines.domain_randomization import DomainRandomizationWrapper
        env = MockEnv()
        wrapper = DomainRandomizationWrapper(env)
        assert wrapper.observation_space == env.observation_space

    def test_wrapper_action_space(self):
        from learning_machines.domain_randomization import DomainRandomizationWrapper
        env = MockEnv()
        wrapper = DomainRandomizationWrapper(env)
        assert wrapper.action_space == env.action_space

    def test_wrapper_reset(self):
        from learning_machines.domain_randomization import DomainRandomizationWrapper
        env = MockEnv()
        wrapper = DomainRandomizationWrapper(env)
        obs, info = wrapper.reset()
        assert "image" in obs
        assert "ir" in obs

    def test_wrapper_step(self):
        from learning_machines.domain_randomization import DomainRandomizationWrapper
        env = MockEnv()
        wrapper = DomainRandomizationWrapper(env)
        wrapper.reset()
        action = np.array([0.5, -0.5], dtype=np.float32)
        obs, reward, term, trunc, info = wrapper.step(action)
        assert "image" in obs

    def test_wrapper_disabled(self):
        from learning_machines.domain_randomization import DomainRandomizationWrapper
        env = MockEnv()
        wrapper = DomainRandomizationWrapper(env, enabled=False)
        obs, _ = wrapper.reset()
        assert "ir" in obs
