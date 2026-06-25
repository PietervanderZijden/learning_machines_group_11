from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from deploy_dreamerv3_reference_hardware import (
    ReliableWheelCommander,
    WheelCommandCancelled,
    load_reference_contract,
    normalize_compiled_state_dict,
    validate_contract_file,
)
from learning_machines.rl_robobo_compact_env import (
    RoboboCompactEnv,
    RoboboCompactEnvConfig,
)
from tests.test_transfer_contract import _FakeRobobo
from showcase_dreamerv3_reference import episode_summary


def _reference_contract():
    return {
        "reward_contract": "robobo-push-phased-dense-v1",
        "max_episode_steps": 200,
        "discount": 0.997,
        "push_time_penalty_per_second": 0.05,
        "push_approach_potential_scale": 2.0,
        "push_goal_potential_offset": 2.0,
        "push_goal_potential_scale": 4.0,
        "push_contact_bonus": 1.0,
        "push_approach_completion_bonus": 5.0,
        "push_goal_completion_bonus": 15.0,
        "action_smoothing": False,
        "pre_action_safety": False,
        "max_action_delta": 2.0,
        "image_size": [96, 96],
        "dyn_hidden": 512,
        "dyn_deter": 1024,
        "units": 512,
        "cnn_depth": 32,
        "cnn_minres": 3,
        "batch_size": 4,
        "batch_length": 48,
        "train_ratio": 128,
    }


class _RetryRobobo:
    def __init__(self):
        self._used_pids = set()
        self.calls = []
        self.refreshes = 0

    def _first_unblocked(self):
        return min(set(range(1, 20)) - self._used_pids)

    def move(self, left, right, millis, blockid=None):
        self._used_pids.add(blockid)
        self.calls.append((left, right, millis, blockid))
        if len(self.calls) >= 2:
            self._used_pids.discard(blockid)
        return blockid

    def is_blocked(self, blockid):
        return blockid in self._used_pids

    def refresh_move_service(self):
        self.refreshes += 1


def test_retry_resends_identical_command_with_fresh_blockid():
    rob = _RetryRobobo()
    commander = ReliableWheelCommander(
        rob, reply_timeout=0.02, poll_interval=0.001, logger=lambda _msg: None
    )

    commander(20, -15, 400)

    assert [call[:3] for call in rob.calls] == [(20, -15, 400)] * 2
    assert rob.calls[0][3] != rob.calls[1][3]
    assert commander.last_retry_count == 1
    assert commander.last_retry_reasons == ["unlock_timeout"]
    assert rob.refreshes == 1


def test_retry_can_be_interrupted_during_missing_unlock():
    rob = _RetryRobobo()
    cancelled = threading.Event()
    commander = ReliableWheelCommander(
        rob,
        reply_timeout=1.0,
        poll_interval=0.001,
        cancel_check=cancelled.is_set,
        logger=lambda _msg: None,
    )
    cancelled.set()

    with pytest.raises(WheelCommandCancelled):
        commander(10, 10, 400)

    assert rob.calls == []


def test_compiled_checkpoint_keys_are_normalized_to_eager_layout():
    state = {
        "_wm._orig_mod.encoder.weight": object(),
        "_task_behavior._orig_mod.actor.weight": object(),
    }

    normalized = normalize_compiled_state_dict(state)

    assert set(normalized) == {
        "_wm.encoder.weight",
        "_task_behavior.actor.weight",
    }


def test_reference_checkpoint_and_sidecar_contract_are_strict(tmp_path):
    contract = _reference_contract()
    checkpoint = {
        "checkpoint_version": 1,
        "agent_state_dict": {},
        "training_step": 100,
        "reward_contract": contract,
    }
    assert load_reference_contract(checkpoint) == contract

    path = tmp_path / "reward_contract.json"
    path.write_text(json.dumps(contract))
    validate_contract_file(contract, path)
    path.write_text(json.dumps({**contract, "image_size": [64, 64]}))
    with pytest.raises(ValueError, match="does not match"):
        validate_contract_file(contract, path)


def test_reference_contract_rejects_non_96_pixel_checkpoint():
    contract = _reference_contract()
    contract["image_size"] = [64, 64]
    checkpoint = {
        "checkpoint_version": 1,
        "agent_state_dict": {},
        "training_step": 100,
        "reward_contract": contract,
    }

    with pytest.raises(ValueError, match="image_size"):
        load_reference_contract(checkpoint)


def test_hardware_push_inference_does_not_require_simulator_geometry():
    rob = _FakeRobobo()
    commands = []
    env = RoboboCompactEnv(
        rob=rob,
        config=RoboboCompactEnvConfig(
            task="push",
            return_image=True,
            image_obs_size=(96, 96),
            initialize_phone_tilt=False,
            reset_settle_time=0.0,
            max_episode_steps=1,
            max_wheel_speed=70,
            action_smoothing=False,
            pre_action_safety=False,
            max_action_delta=2.0,
            randomize_push_layout=False,
            hardware_inference_only=True,
            hardware_wheel_command=lambda left, right, millis: commands.append(
                (left, right, millis)
            ),
        ),
    )

    obs, _info = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(
        np.ones(2, dtype=np.float32)
    )

    assert obs["image"].shape == (3, 96, 96)
    assert next_obs["image"].shape == (3, 96, 96)
    assert commands == [(70, 70, 400)]
    assert reward == 0.0
    assert not terminated
    assert truncated
    np.testing.assert_allclose(info["executed_action"], [1.0, 1.0])


def test_showcase_episode_summary_reports_success_and_timing():
    summary = episode_summary(
        2,
        1.0,
        15,
        True,
        {
            "block_goal_distance": 0.12,
            "collisions": 3,
            "red_block_visible": 1.0,
            "green_goal_visible": 0.0,
        },
    )

    assert summary == {
        "episode": 2,
        "success": True,
        "reward": 1.0,
        "steps": 15,
        "simulated_seconds": 6.0,
        "block_goal_distance": 0.12,
        "collisions": 3,
        "red_block_visible": True,
        "green_goal_visible": False,
    }
