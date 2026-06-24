from __future__ import annotations

import types
from collections.abc import Mapping

import numpy as np
import torch


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
