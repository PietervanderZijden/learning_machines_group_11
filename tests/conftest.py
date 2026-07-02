'Shared fixtures for robobo_interface tests.'
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "catkin_ws/src/learning_machines/src"))
sys.path.insert(0, str(ROOT / "catkin_ws/src/robobo_interface/src"))


@pytest.fixture(scope="session")
def sim_port() -> int:
    """Return the configured simulator port."""
    return int(os.getenv("COPPELIA_SIM_PORT", "23000"))


@pytest.fixture(scope="session")
def sim_ip() -> str:
    """Return the configured simulator address."""
    return os.getenv("COPPELIA_SIM_IP", "127.0.0.1")


@pytest.fixture(scope="session")
def sim_robobo(sim_port: int, sim_ip: str):
    'Provide a connected SimulationRobobo that is stopped between tests.'
    from robobo_interface import SimulationRobobo

    rob = SimulationRobobo(api_port=sim_port, ip_adress=sim_ip)
    yield rob
    if rob.is_running():
        rob.stop_simulation()
