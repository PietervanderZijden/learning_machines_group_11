'Tests for HardwareRobobo improvements.'
from __future__ import annotations

import pytest


class TestSetWheelSpeeds:
    def test_method_exists(self, sim_robobo):
        assert hasattr(sim_robobo, "set_wheel_speeds")
        assert callable(sim_robobo.set_wheel_speeds)

    def test_call_does_not_crash(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.play_simulation()
        try:
            sim_robobo.set_wheel_speeds(0.0, 0.0, duration_s=0.4)
        finally:
            sim_robobo.stop_simulation()


class TestMoveRetry:
    def test_move_works(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.play_simulation()
        try:
            blockid = sim_robobo.move(10, 10, 200)
            assert blockid is not None
            sim_robobo.sleep(0.3)
        finally:
            sim_robobo.stop_simulation()


class TestLazyHardwareImport:
    """Verify deferred hardware dependency loading."""

    def test_hardware_not_imported_at_module_level(self):
        """Keep the hardware module unloaded after importing the package."""
        import importlib
        import sys
        import robobo_interface

        sys.modules.pop("robobo_interface.hardware", None)
        importlib.reload(robobo_interface)
        assert "robobo_interface.hardware" not in sys.modules
