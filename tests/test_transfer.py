"""Tests for transfer.py utility functions (no sim required)."""
from __future__ import annotations

import numpy as np
import pytest


class TestCalibrationProfile:
    def test_default_simulation(self):
        from learning_machines.transfer import default_calibration_profile
        profile = default_calibration_profile("simulation")
        assert profile is not None
        assert len(profile.sensors) == 8

    def test_default_hardware(self):
        from learning_machines.transfer import default_calibration_profile
        profile = default_calibration_profile("hardware")
        assert profile is not None
        assert len(profile.sensors) == 8

    def test_profile_roundtrip(self):
        from learning_machines.transfer import default_calibration_profile
        profile = default_calibration_profile("simulation")
        d = profile.to_dict()
        assert "sensors" in d or "version" in d


class TestTransferReward:
    def test_collecting_food_positive(self):
        from learning_machines.transfer import transfer_reward, RewardConfig
        cfg = RewardConfig()
        r, parts = transfer_reward(
            newly_collected=1, elapsed_delta_seconds=0.4,
            elapsed_seconds=1.0, completed=False, collision=False,
            action_change=0.0, config=cfg,
        )
        assert r > 0.0
        assert parts["collect_reward"] == 100.0

    def test_time_pressure_on_completion(self):
        from learning_machines.transfer import transfer_reward, RewardConfig
        cfg = RewardConfig()
        r_fast, _ = transfer_reward(
            newly_collected=0, elapsed_delta_seconds=0.4,
            elapsed_seconds=1.0, completed=True, collision=False,
            action_change=0.0, config=cfg,
        )
        r_slow, _ = transfer_reward(
            newly_collected=0, elapsed_delta_seconds=0.4,
            elapsed_seconds=50.0, completed=True, collision=False,
            action_change=0.0, config=cfg,
        )
        assert r_fast > r_slow

    def test_collision_penalty(self):
        from learning_machines.transfer import transfer_reward, RewardConfig
        cfg = RewardConfig(collision_penalty=10.0)
        r_no_coll, _ = transfer_reward(
            newly_collected=0, elapsed_delta_seconds=0.4,
            elapsed_seconds=1.0, completed=False, collision=False,
            action_change=0.0, config=cfg,
        )
        r_coll, _ = transfer_reward(
            newly_collected=0, elapsed_delta_seconds=0.4,
            elapsed_seconds=1.0, completed=False, collision=True,
            action_change=0.0, config=cfg,
        )
        assert r_no_coll > r_coll

    def test_completion_bonus(self):
        from learning_machines.transfer import transfer_reward, RewardConfig
        cfg = RewardConfig()
        r_done, _ = transfer_reward(
            newly_collected=3, elapsed_delta_seconds=0.4,
            elapsed_seconds=5.0, completed=True, collision=False,
            action_change=0.0, config=cfg,
        )
        r_not_done, _ = transfer_reward(
            newly_collected=3, elapsed_delta_seconds=0.4,
            elapsed_seconds=5.0, completed=False, collision=False,
            action_change=0.0, config=cfg,
        )
        assert r_done > r_not_done

    def test_action_change_penalty(self):
        from learning_machines.transfer import transfer_reward, RewardConfig
        cfg = RewardConfig(action_change_penalty=5.0)
        r_low, _ = transfer_reward(
            newly_collected=0, elapsed_delta_seconds=0.4,
            elapsed_seconds=1.0, completed=False, collision=False,
            action_change=0.1, config=cfg,
        )
        r_high, _ = transfer_reward(
            newly_collected=0, elapsed_delta_seconds=0.4,
            elapsed_seconds=1.0, completed=False, collision=False,
            action_change=1.0, config=cfg,
        )
        assert r_low > r_high


class TestObservationAdapter:
    def test_creation(self):
        from learning_machines.transfer import ObservationAdapter, default_calibration_profile
        profile = default_calibration_profile("simulation")
        adapter = ObservationAdapter(profile)
        assert adapter is not None


class TestActionExecutor:
    def test_creation(self):
        from learning_machines.transfer import ActionExecutor
        executor = ActionExecutor()
        assert executor is not None


class TestSmoothingConfig:
    def test_defaults(self):
        from learning_machines.transfer import SmoothingConfig
        cfg = SmoothingConfig()
        assert cfg.previous_weight >= 0.0
        assert cfg.requested_weight >= 0.0


class TestBlobProgressPotential:
    def test_no_blob(self):
        from learning_machines.transfer import blob_progress_potential
        r = blob_progress_potential([0.5, 0.5, 0.01, 0.0])
        assert r == 0.0

    def test_centered_blob(self):
        from learning_machines.transfer import blob_progress_potential
        r = blob_progress_potential([0.5, 0.5, 0.01, 1.0])
        assert r > 0.0

    def test_off_center_lower(self):
        from learning_machines.transfer import blob_progress_potential
        r_center = blob_progress_potential([0.5, 0.5, 0.01, 1.0])
        r_edge = blob_progress_potential([0.9, 0.5, 0.01, 1.0])
        assert r_center > r_edge
