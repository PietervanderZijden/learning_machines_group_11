'Tests for SimulationRobobo improvements.'
from __future__ import annotations

import math
import pytest


class TestConnection:
    def test_connects(self, sim_port: int, sim_ip: str):
        from robobo_interface import SimulationRobobo

        rob = SimulationRobobo(api_port=sim_port, ip_adress=sim_ip)
        assert rob._sim is not None
        rob.stop_simulation()

    def test_stepping_enabled_by_default(self, sim_robobo):
        assert sim_robobo._stepping_enabled is True

    def test_handles_cached(self, sim_robobo):
        assert hasattr(sim_robobo, "_left_motor_joint")
        assert hasattr(sim_robobo, "_right_motor_joint")


class TestConfigureSimulationTiming:
    def test_sets_simulation_time_step(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.configure_simulation_timing()
        sim_dt = sim_robobo._sim.getSimulationTimeStep()
        assert math.isclose(sim_dt, 0.4, abs_tol=1e-6), f"Expected 0.4s sim step, got {sim_dt}"

    def test_sets_physics_time_step(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.configure_simulation_timing()
        physics_dt = float(sim_robobo._sim.getFloatParam(sim_robobo._sim.floatparam_physicstimestep))
        assert math.isclose(physics_dt, 0.005, abs_tol=1e-6), f"Expected 0.005s physics step, got {physics_dt}"

    def test_raises_if_running(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.play_simulation()
        try:
            with pytest.raises(RuntimeError):
                sim_robobo.configure_simulation_timing()
        finally:
            sim_robobo.stop_simulation()


class TestPlayPauseStop:
    def test_play_and_stop(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.play_simulation()
        assert sim_robobo.is_running()
        sim_robobo.stop_simulation()
        assert sim_robobo.is_stopped()

    def test_pause(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.play_simulation()
        assert sim_robobo.is_running()
        sim_robobo.pause_simulation()
        assert sim_robobo.is_paused()
        sim_robobo.stop_simulation()


class TestStepping:
    def test_step_advances_simulation(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.play_simulation()
        t_before = sim_robobo._sim.getSimulationTime()
        sim_robobo._client.step()
        t_after = sim_robobo._sim.getSimulationTime()
        assert t_after > t_before, "Simulation time did not advance after step"
        sim_robobo.stop_simulation()

    def test_step_multiple(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.play_simulation()
        t_before = sim_robobo._sim.getSimulationTime()
        for _ in range(10):
            sim_robobo._client.step()
        t_after = sim_robobo._sim.getSimulationTime()
        assert t_after > t_before
        sim_robobo.stop_simulation()


class TestSleep:
    def test_sleep_advances_sim_time(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.play_simulation()
        t_before = sim_robobo._sim.getSimulationTime()
        sim_robobo.sleep(0.4)
        t_after = sim_robobo._sim.getSimulationTime()
        assert t_after > t_before, "Sim time did not advance during sleep"
        sim_robobo.stop_simulation()

    def test_sleep_raises_when_stopped(self, sim_robobo):
        sim_robobo.stop_simulation()
        with pytest.raises(RuntimeError, match="not running"):
            sim_robobo.sleep(0.1)

    def test_sleep_graceful_during_stop(self, sim_robobo):
        sim_robobo.stop_simulation()
        sim_robobo.play_simulation()
        sim_robobo.stop_simulation()
        assert sim_robobo.is_stopped()


class TestDisplayDisabled:
    def test_display_off_or_headless(self, sim_robobo):
        result = sim_robobo._sim.getBoolParam(
            sim_robobo._sim.boolparam_display_enabled
        )
        assert result is False, "Display should be disabled for headless speed"


class TestPatchFoodContactCallback:
    """Verify food callback correction."""

    def test_food_script_patched(self, sim_robobo):
        'Require both contact-handle orders to check Robobo membership.'
        if sim_robobo._food_script is None:
            pytest.skip("Could not retrieve child script from scene")
        text = sim_robobo._sim.getScriptStringParam(
            sim_robobo._food_script,
            sim_robobo._sim.scriptstringparam_text,
        )
        assert "belongs_to_robobo(inData.handle1)" in text
        assert "belongs_to_robobo(inData.handle2)" in text
