"""Fast, explicit CoppeliaSim connection preflight for training commands."""
from __future__ import annotations

import socket


def check_coppelia_service(host: str, port: int, timeout: float = 3.0) -> None:
    if host in {"0.0.0.0", "::"}:
        raise ConnectionError(
            f"COPPELIA_SIM_IP={host!r} is a bind address, not a client "
            "destination. Use 127.0.0.1 for a local simulator."
        )
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return
    except OSError as exc:
        raise ConnectionError(
            f"Cannot reach the CoppeliaSim ZMQ service at {host}:{port}. "
            "Start the scene's ZMQ remote API service and verify the host/port. "
            "For a simulator on this machine use COPPELIA_SIM_IP=127.0.0.1."
        ) from exc
