import numpy as np

from learning_machines.calibrate_ir import robust_profile
from validate_hardware import (
    calibration_quality,
    initialize_camera_down,
    run_wheel_test,
    summarize_ir,
)


class FakeRobot:
    def __init__(self):
        self.commands = []

    def move(self, left, right, millis):
        self.commands.append((left, right, millis))


def test_ir_summary_reports_all_sensors():
    samples = np.stack([np.arange(8), np.arange(8) + 2.0])
    result = summarize_ir(samples)
    assert len(result) == 8
    assert result["BackL"]["median"] == 1.0
    assert result["FrontLL"]["max"] == 9.0


def test_calibration_quality_detects_weak_sensor():
    open_space = np.zeros((5, 8))
    near = np.full((5, 8), 20.0)
    near[:, 3] = 2.0
    result = calibration_quality(
        {"open_space": open_space, "near_obstacle": near}, minimum_span=5.0
    )
    assert not result["passed"]
    assert result["weak_sensors"] == ["FrontR"]


def test_robust_profile_uses_obstacle_side_percentile_for_inverse_sensor():
    open_space = np.full((10, 8), 100.0)
    near = np.full((10, 8), 20.0)
    near[9, :] = 95.0
    profile = robust_profile(
        {"open_space": open_space, "near_obstacle": near}, "test", "test"
    )
    assert all(sensor.polarity == -1 for sensor in profile.sensors)
    assert all(sensor.near_obstacle == 20.0 for sensor in profile.sensors)


def test_wheel_test_is_bounded_and_stops_between_commands():
    robot = FakeRobot()
    events = run_wheel_test(robot, speed=99, duration_seconds=5.0, sleep=lambda _: None)
    assert len(events) == 4
    assert all(abs(event["left"]) <= 20 and abs(event["right"]) <= 20 for event in events)
    assert all(event["duration_ms"] <= 500 for event in events)
    assert robot.commands[0] == (0, 0, 200)
    assert robot.commands[-1] == (0, 0, 200)
    for index, command in enumerate(robot.commands):
        if command[:2] != (0, 0):
            assert robot.commands[index - 1] == (0, 0, 200)
            assert robot.commands[index + 1] == (0, 0, 200)


def test_camera_is_moved_down_without_blocking_on_unlock(monkeypatch):
    class TiltRobot:
        def __init__(self):
            self.tilt = 50
            self.commands = []
            self._used_pids = {7}

        def read_phone_tilt(self):
            return self.tilt

        def set_phone_tilt(self, target, speed):
            self.commands.append((target, speed))
            self.tilt = target
            return 7

    monkeypatch.setattr("validate_hardware.time.sleep", lambda _: None)
    robot = TiltRobot()
    result = initialize_camera_down(robot, target=100)

    assert robot.commands == [(100, 10)]
    assert result == {"requested": 100, "before": 50, "after": 100}
    assert not robot._used_pids
