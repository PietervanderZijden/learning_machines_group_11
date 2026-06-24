from __future__ import annotations

import torch

from learning_machines.reference_checkpoint import collect_optimizer_state_dicts


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
