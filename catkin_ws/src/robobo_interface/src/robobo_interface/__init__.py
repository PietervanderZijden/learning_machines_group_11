from .datatypes import (
    TOLERANCE,
    Emotion,
    SoundEmotion,
    LedColor,
    LedId,
    Acceleration,
    Orientation,
    Position,
    WheelPosition,
)
from .base import IRobobo
from .simulation import SimulationRobobo


def __getattr__(name: str):
    if name == "HardwareRobobo":
        from .hardware import HardwareRobobo
        return HardwareRobobo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = (
    "IRobobo",
    "TOLERANCE",
    "Emotion",
    "SoundEmotion",
    "LedColor",
    "LedId",
    "Acceleration",
    "Orientation",
    "Position",
    "WheelPosition",
    "HardwareRobobo",
    "SimulationRobobo",
)
