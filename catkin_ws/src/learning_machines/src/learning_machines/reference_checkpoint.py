from __future__ import annotations

import os
import pathlib
import types
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

REFERENCE_CHECKPOINT_VERSION = 1


_SKIP_ATTRIBUTES = {
    "_config",
    "_dataset",
    "_logger",
    "_metrics",
}


def collect_optimizer_state_dicts(obj) -> dict[str, dict]:
    """Collect optimizer state without traversing runtime logger/config objects."""
    collected: dict[str, dict] = {}
    visited: set[int] = set()

    def visit(value, path: str) -> None:
        if isinstance(value, torch.optim.Optimizer):
            collected[path] = value.state_dict()
            return
        if id(value) in visited or _is_leaf(value):
            return
        visited.add(id(value))

        if isinstance(value, Mapping):
            for name, child in value.items():
                if isinstance(name, str):
                    visit(child, _join(path, name))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, _join(path, str(index)))
            return

        try:
            attributes = vars(value)
        except (TypeError, ValueError):
            return
        for name, child in tuple(attributes.items()):
            if name in _SKIP_ATTRIBUTES:
                continue
            visit(child, _join(path, name))

    visit(obj, "")
    return collected


def load_optimizer_state_dicts(obj, state_dicts: Mapping[str, dict]) -> None:
    """Restore optimizer states collected by collect_optimizer_state_dicts."""
    for path, state_dict in state_dicts.items():
        current: Any = obj
        for key in path.split("."):
            if isinstance(current, Mapping):
                current = current[key]
            elif isinstance(current, (list, tuple)):
                current = current[int(key)]
            else:
                current = getattr(current, key)
        if not isinstance(current, torch.optim.Optimizer):
            raise TypeError(f"checkpoint path is not an optimizer: {path}")
        current.load_state_dict(state_dict)


def save_checkpoint_atomic(checkpoint: Mapping[str, Any], path: pathlib.Path) -> None:
    """Write a checkpoint without exposing a partially written latest.pt."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(dict(checkpoint), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    reward_contract: Mapping[str, Any],
    replay_step: int,
    max_replay_lag: int = 0,
    max_replay_lead: int = 0,
) -> None:
    required = {
        "checkpoint_version",
        "agent_state_dict",
        "optims_state_dict",
        "training_step",
        "reward_contract",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            "reference checkpoint is missing required fields: "
            + ", ".join(sorted(missing))
        )
    if checkpoint["checkpoint_version"] != REFERENCE_CHECKPOINT_VERSION:
        raise ValueError(
            "unsupported reference checkpoint version: "
            f"{checkpoint['checkpoint_version']}"
        )
    if checkpoint["reward_contract"] != dict(reward_contract):
        raise ValueError(
            "checkpoint reward/action contract does not match this run"
        )
    checkpoint_step = int(checkpoint["training_step"])
    replay_step = int(replay_step)
    replay_lag = checkpoint_step - replay_step
    replay_lead = replay_step - checkpoint_step
    if replay_lag > max_replay_lag or replay_lead > max_replay_lead:
        raise ValueError(
            "checkpoint/replay step mismatch: checkpoint has "
            f"{checkpoint_step} steps but replay has {replay_step}; "
            f"allowed replay lag is {max_replay_lag} and "
            f"allowed replay lead is {max_replay_lead}"
        )


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def _is_leaf(value) -> bool:
    return (
        value is None
        or isinstance(
            value,
            (
                bool,
                int,
                float,
                complex,
                str,
                bytes,
                bytearray,
                np.ndarray,
                np.generic,
                torch.Tensor,
                torch.nn.Parameter,
                types.ModuleType,
                types.FunctionType,
                types.MethodType,
                types.GeneratorType,
                type,
            ),
        )
    )
