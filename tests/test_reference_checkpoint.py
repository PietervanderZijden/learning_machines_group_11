from __future__ import annotations

import torch
import pytest

from learning_machines.reference_checkpoint import (
    REFERENCE_CHECKPOINT_VERSION,
    collect_optimizer_state_dicts,
    load_optimizer_state_dicts,
    save_checkpoint_atomic,
    validate_checkpoint,
)
from train_dreamerv3_reference_push import (
    REFERENCE_CNN_DEPTH,
    REFERENCE_CNN_MINRES,
    REFERENCE_DYN_DETER,
    REFERENCE_DYN_HIDDEN,
    REFERENCE_IMAGE_SIZE,
    REFERENCE_MLP_UNITS,
)


class _ExplodingDescriptor:
    def __getattr__(self, _name):
        raise RuntimeError("runtime logger object must not be traversed")


class _OptimizerWrapper:
    def __init__(self, parameter):
        self._opt = torch.optim.Adam([parameter], lr=1e-3)


class _Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self._model_opt = _OptimizerWrapper(self.weight)
        self._logger = _ExplodingDescriptor()
        self._config = _ExplodingDescriptor()
        self._dataset = (value for value in ())


def test_reference_push_uses_upper_intermediate_model_defaults():
    assert REFERENCE_IMAGE_SIZE == (96, 96)
    assert REFERENCE_DYN_HIDDEN == 1024
    assert REFERENCE_DYN_DETER == 1024
    assert REFERENCE_MLP_UNITS == 1024
    assert REFERENCE_CNN_DEPTH == 64
    assert REFERENCE_CNN_MINRES == 3


def test_reference_cnn_round_trips_96_pixel_images():
    import networks

    encoder = networks.ConvEncoder(
        (96, 96, 3),
        depth=REFERENCE_CNN_DEPTH,
        minres=REFERENCE_CNN_MINRES,
    )
    encoded = encoder(torch.zeros(1, 1, 96, 96, 3))
    assert encoded.shape[-1] == encoder.outdim

    decoder = networks.ConvDecoder(
        feat_size=32,
        shape=(3, 96, 96),
        depth=REFERENCE_CNN_DEPTH,
        act="ELU",
        minres=REFERENCE_CNN_MINRES,
    )
    decoded = decoder(torch.zeros(1, 1, 32))
    assert decoded.shape == (1, 1, 96, 96, 3)


def test_reference_checkpoint_collects_optimizer_without_traversing_logger():
    states = collect_optimizer_state_dicts(_Agent())

    assert states.keys() == {"_model_opt._opt"}
    assert states["_model_opt._opt"]["param_groups"][0]["lr"] == 1e-3


def test_reference_checkpoint_handles_cycles():
    agent = _Agent()
    agent.cycle = agent

    states = collect_optimizer_state_dicts(agent)

    assert states.keys() == {"_model_opt._opt"}


def test_reference_checkpoint_handles_torch_compiled_modules():
    compiled = torch.compile(_Agent())
    root = torch.nn.Module()
    root._wm = compiled

    states = collect_optimizer_state_dicts(root)

    assert len(states) == 1
    assert next(iter(states)).endswith("._model_opt._opt")


def test_reference_checkpoint_restores_optimizer_state():
    source = _Agent()
    source._model_opt._opt.param_groups[0]["lr"] = 4e-5
    states = collect_optimizer_state_dicts(source)
    target = _Agent()

    load_optimizer_state_dicts(target, states)

    assert target._model_opt._opt.param_groups[0]["lr"] == 4e-5


def test_reference_checkpoint_is_atomic_and_validated(tmp_path):
    contract = {"reward_contract": "robobo-push-sparse-v1"}
    path = tmp_path / "latest.pt"
    checkpoint = {
        "checkpoint_version": REFERENCE_CHECKPOINT_VERSION,
        "agent_state_dict": {"weight": torch.ones(())},
        "optims_state_dict": {},
        "training_step": 4000,
        "reward_contract": contract,
        "wandb_run_id": "run-id",
    }

    save_checkpoint_atomic(checkpoint, path)
    loaded = torch.load(path, weights_only=False)
    validate_checkpoint(
        loaded,
        reward_contract=contract,
        replay_step=4000,
    )

    assert path.exists()
    assert not (tmp_path / ".latest.pt.tmp").exists()


def test_reference_checkpoint_rejects_replay_step_mismatch():
    checkpoint = {
        "checkpoint_version": REFERENCE_CHECKPOINT_VERSION,
        "agent_state_dict": {},
        "optims_state_dict": {},
        "training_step": 4000,
        "reward_contract": {"reward_contract": "robobo-push-sparse-v1"},
    }

    with pytest.raises(ValueError, match="checkpoint/replay step mismatch"):
        validate_checkpoint(
            checkpoint,
            reward_contract=checkpoint["reward_contract"],
            replay_step=3999,
        )


def test_reference_checkpoint_allows_one_partial_episode_of_replay_lag():
    checkpoint = {
        "checkpoint_version": REFERENCE_CHECKPOINT_VERSION,
        "agent_state_dict": {},
        "optims_state_dict": {},
        "training_step": 4000,
        "reward_contract": {"reward_contract": "robobo-push-sparse-v1"},
    }

    validate_checkpoint(
        checkpoint,
        reward_contract=checkpoint["reward_contract"],
        replay_step=3801,
        max_replay_lag=199,
    )
    with pytest.raises(ValueError, match="checkpoint/replay step mismatch"):
        validate_checkpoint(
            checkpoint,
            reward_contract=checkpoint["reward_contract"],
            replay_step=3800,
            max_replay_lag=199,
        )
