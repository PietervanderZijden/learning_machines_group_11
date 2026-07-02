"""Unit tests for simulator timeout behavior."""

from __future__ import annotations

import time

import pytest

from robobo_interface.simulation import SimulationRobobo, timeout


def test_timeout_returns_result() -> None:
    """Return a callable result before the deadline."""
    assert timeout(lambda: 4, 1) == 4


def test_timeout_raises_without_killing_process() -> None:
    """Raise TimeoutError while leaving the test process alive."""
    with pytest.raises(TimeoutError):
        timeout(lambda: time.sleep(0.1), 0.01)


def test_transition_timeout_uses_configured_duration() -> None:
    """Raise when a state transition exceeds its configured duration."""
    rob = object.__new__(SimulationRobobo)
    rob._timeout_dur = 0.01
    with pytest.raises(RuntimeError, match="within 0.01 seconds"):
        rob._wait_for_state(lambda: False, "start")
