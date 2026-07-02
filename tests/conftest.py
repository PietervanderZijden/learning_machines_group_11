"""Shared fixtures for robobo_interface tests."""
from __future__ import annotations

import os
import pytest


@pytest.fixture(scope="session")
def sim_port() -> int:
    return int(os.getenv("COPPELIA_SIM_PORT", "23000"))


@pytest.fixture(scope="session")
def sim_ip() -> str:
    return os.getenv("COPPELIA_SIM_IP", "127.0.0.1")


@pytest.fixture(scope="session")
def sim_robobo(sim_port: int, sim_ip: str):
    """Provide a connected SimulationRobobo that is stopped between tests.

    The fixture is session-scoped so the ZMQ connection is established only
    once.  The simulation is stopped after each test that may have started it.
    """
    from robobo_interface import SimulationRobobo

    rob = SimulationRobobo(api_port=sim_port, ip_adress=sim_ip)
    yield rob
    if rob.is_running():
        rob.stop_simulation()
